"""Small fail-closed primitives for publishing repository artifacts.

The API is intentionally Linux/POSIX specific.  Every authority-bearing path
is relative to an explicitly supplied root, and every walk is performed from
retained directory descriptors with ``O_NOFOLLOW``.  Absolute paths and
``Path.resolve()`` are never used to authorize a mutation.

``publish_bundle_exclusive`` provides *logical* atomicity: ordinary members
are durable before the completion member is linked, and readers must treat a
bundle without its completion member as incomplete.  POSIX cannot make
multiple directory entries visible in one syscall.  POSIX also has no atomic
compare-and-unlink operation, so exact release assumes that all writers with
directory mutation authority cooperate with this protocol.
"""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import stat
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_TEMP_FLAGS = (
    os.O_RDWR
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_APPEND_FLAGS = (
    os.O_WRONLY
    | os.O_APPEND
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_CHUNK_SIZE = 1024 * 1024
_TEMP_ATTEMPTS = 128

# Indirections keep failure injection local to this module's tests.  They are
# private implementation details, not supported extension points.
_open = os.open
_close = os.close
_stat = os.stat
_fstat = os.fstat
_mkdir = os.mkdir
_rmdir = os.rmdir
_link = os.link
_unlink = os.unlink
_write = os.write
_read = os.read
_pread = os.pread
_fsync = os.fsync
_fchmod = os.fchmod


class ArtifactIOError(RuntimeError):
    """Base class for fail-closed artifact I/O errors."""


class UnsupportedPlatformError(ArtifactIOError):
    """Raised before mutation when required POSIX primitives are unavailable."""


class OwnershipChangedError(ArtifactIOError):
    """Raised when a pathname no longer names the claimed filesystem node."""


class RollbackError(ArtifactIOError):
    """A publication failed and at least one owned cleanup could not be proved."""

    def __init__(
        self, primary_error: BaseException, rollback_errors: Sequence[BaseException]
    ) -> None:
        self.primary_error = primary_error
        self.rollback_errors = tuple(rollback_errors)
        super().__init__(
            "artifact publication failed and owned rollback was incomplete: "
            f"{primary_error}; rollback errors="
            + "; ".join(str(error) for error in self.rollback_errors)
        )


class CommitIndeterminateError(ArtifactIOError):
    """A completion member became visible before a later operation failed."""

    def __init__(
        self,
        primary_error: BaseException,
        *,
        published_paths: Sequence[str],
        cleanup_errors: Sequence[BaseException] = (),
    ) -> None:
        self.primary_error = primary_error
        self.published_paths = tuple(published_paths)
        self.cleanup_errors = tuple(cleanup_errors)
        self.completion_visible = True
        self.completion_link_succeeded = True
        message = (
            "bundle completion link succeeded and may have been observed; "
            f"publication state is committed or durability-indeterminate: "
            f"{primary_error}"
        )
        if self.cleanup_errors:
            message += "; cleanup errors=" + "; ".join(
                str(error) for error in self.cleanup_errors
            )
        super().__init__(message)


class PublicationIndeterminateError(ArtifactIOError):
    """A single publication is visible but final cleanup did not complete."""

    def __init__(self, relative_path: str, primary_error: BaseException) -> None:
        self.relative_path = relative_path
        self.primary_error = primary_error
        self.publication_visible = True
        self.publication_link_succeeded = True
        super().__init__(
            f"artifact publication is visible but final state is indeterminate: "
            f"{relative_path}: {primary_error}"
        )


class RemovalIndeterminateError(ArtifactIOError):
    """An owned entry was removed but its parent fsync then failed."""

    def __init__(self, relative_path: str, primary_error: BaseException) -> None:
        self.relative_path = relative_path
        self.primary_error = primary_error
        self.removed = True
        super().__init__(
            f"owned entry was removed but durability is indeterminate: "
            f"{relative_path}: {primary_error}"
        )


@dataclass(frozen=True)
class FileClaim:
    """A stable immutable-file identity and content claim."""

    relative_path: str
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclass(frozen=True)
class OwnedDirectory:
    """Identity of a directory created exclusively by this process.

    Directory size and timestamps are intentionally absent because owned
    children may change them.  Removal additionally requires an empty exact
    dev/inode/type/mode match under the cooperative-writer contract.
    """

    relative_path: str
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class OwnedLog:
    """An exclusively reserved mutable log and its initial empty-file claim."""

    reservation: FileClaim


@dataclass(frozen=True)
class PublishItem:
    """One immutable byte payload in a completion-last publication."""

    relative_path: str
    payload: bytes
    mode: int = 0o644


@dataclass(frozen=True)
class PublishedBundle:
    """Claims returned after the completion member is durable."""

    members: tuple[FileClaim, ...]
    completion: FileClaim


@dataclass
class _ParentHandle:
    parts: tuple[str, ...]
    fd: int


@dataclass
class _StagedFile:
    item: PublishItem
    parent: _ParentHandle
    name: str
    temporary_name: str
    fd: int | None
    sha256: str
    written_device: int | None = None
    written_inode: int | None = None
    written_mode: int | None = None
    written_size: int | None = None
    written_mtime_ns: int | None = None
    temporary_exists: bool = True
    linked: bool = False
    claim: FileClaim | None = None
    transition_close_error: OSError | None = None


def _require_supported_platform() -> None:
    required_constants = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    missing_constants = [name for name in required_constants if not hasattr(os, name)]
    required_dir_fd = (os.open, os.stat, os.mkdir, os.rmdir, os.link, os.unlink)
    missing_dir_fd = [
        function.__name__
        for function in required_dir_fd
        if function not in os.supports_dir_fd
    ]
    missing_nofollow = [
        function.__name__
        for function in (os.stat, os.link)
        if function not in os.supports_follow_symlinks
    ]
    if (
        os.name != "posix"
        or missing_constants
        or missing_dir_fd
        or missing_nofollow
        or not hasattr(os, "pread")
    ):
        raise UnsupportedPlatformError(
            "artifact I/O requires Linux/POSIX openat/linkat/unlinkat, "
            "O_DIRECTORY, O_NOFOLLOW, pread, and directory fsync support; "
            f"missing constants={missing_constants}, dir_fd={missing_dir_fd}, "
            f"nofollow={missing_nofollow}"
        )


def _canonical_root(value: os.PathLike[str] | str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("artifact root must be a nonempty text path")
    if (
        not os.path.isabs(raw)
        or os.path.normpath(raw) != raw
        or (raw.startswith("//") and raw != "/")
        or Path(raw).anchor != "/"
    ):
        raise ValueError("artifact root must be an absolute canonical path")
    return Path(raw)


def _relative_parts(value: os.PathLike[str] | str) -> tuple[str, ...]:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise ValueError("artifact path must be a nonempty root-relative text path")
    parsed = PurePosixPath(raw)
    if (
        parsed.is_absolute()
        or raw in {".", ".."}
        or parsed.as_posix() != raw
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise ValueError("artifact path must be canonical and root-relative")
    return tuple(parsed.parts)


def _validated_mode(value: int, *, directory: bool) -> int:
    if type(value) is not int or not 0 <= value <= 0o777:
        kind = "directory" if directory else "file"
        raise ValueError(f"{kind} mode must contain permission bits only")
    if directory and value & 0o500 != 0o500:
        raise ValueError("directory mode must grant owner read and search access")
    if not directory and value & 0o400 != 0o400:
        raise ValueError("file mode must grant owner read access")
    return value


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return int(value.st_dev), int(value.st_ino), int(value.st_mode)


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _owner_identity(value: os.stat_result) -> tuple[int, int, int]:
    return int(value.st_dev), int(value.st_ino), int(value.st_mode)


def _close_all(*descriptors: int | None) -> list[OSError]:
    """Attempt every close; callers preserve any already-active exception."""

    errors: list[OSError] = []
    seen: set[int] = set()
    for descriptor in descriptors:
        if descriptor is None or descriptor in seen:
            continue
        seen.add(descriptor)
        try:
            _close(descriptor)
        except OSError as error:
            errors.append(error)
    return errors


def _close_finally(
    label: str,
    *descriptors: int | None,
    publication_visible_path: str | None = None,
    removed_path: str | None = None,
) -> None:
    errors = _close_all(*descriptors)
    if errors and sys.exc_info()[0] is None:
        error = ArtifactIOError(
            f"{label} completed but descriptor cleanup failed: "
            + "; ".join(str(error) for error in errors)
        )
        if publication_visible_path is not None:
            raise PublicationIndeterminateError(
                publication_visible_path, error
            ) from error
        if removed_path is not None:
            raise RemovalIndeterminateError(removed_path, error) from error
        raise error


def _open_absolute_directory(path: Path) -> int:
    descriptor: int | None = _open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for component in path.parts[1:]:
            if descriptor is None:  # pragma: no cover - local invariant
                raise AssertionError("root-walk descriptor is absent")
            child = _open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            close_errors = _close_all(descriptor)
            # close(2) failure is state-ambiguous; never retry the old number.
            descriptor = None
            if close_errors:
                child_close_errors = _close_all(child)
                error = ArtifactIOError(
                    "descriptor cleanup failed during artifact root walk: "
                    + "; ".join(str(error) for error in close_errors)
                )
                if child_close_errors:
                    raise RollbackError(error, child_close_errors) from error
                raise error
            descriptor = child
        if descriptor is None:  # pragma: no cover - local invariant
            raise AssertionError("root-walk descriptor is absent")
        state = _fstat(descriptor)
        if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
            raise ArtifactIOError("artifact root is not a direct real directory")
        return descriptor
    except BaseException as error:
        close_errors = _close_all(descriptor)
        if close_errors:
            raise RollbackError(error, close_errors) from error
        raise


def _open_relative_directory(root_fd: int, parts: Sequence[str]) -> int:
    descriptor: int | None = os.dup(root_fd)
    try:
        for component in parts:
            if descriptor is None:  # pragma: no cover - local invariant
                raise AssertionError("parent-walk descriptor is absent")
            child = _open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            close_errors = _close_all(descriptor)
            # close(2) failure is state-ambiguous; never retry the old number.
            descriptor = None
            if close_errors:
                child_close_errors = _close_all(child)
                error = ArtifactIOError(
                    "descriptor cleanup failed during artifact parent walk: "
                    + "; ".join(str(error) for error in close_errors)
                )
                if child_close_errors:
                    raise RollbackError(error, child_close_errors) from error
                raise error
            descriptor = child
        if descriptor is None:  # pragma: no cover - local invariant
            raise AssertionError("parent-walk descriptor is absent")
        state = _fstat(descriptor)
        if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
            raise ArtifactIOError("artifact parent is not a direct real directory")
        return descriptor
    except BaseException as error:
        close_errors = _close_all(descriptor)
        if close_errors:
            raise RollbackError(error, close_errors) from error
        raise


def _verify_root_binding(root: Path, root_fd: int) -> None:
    try:
        probe = _open_absolute_directory(root)
    except OSError as error:
        raise OwnershipChangedError("artifact root path binding changed") from error
    try:
        if _directory_identity(_fstat(probe)) != _directory_identity(_fstat(root_fd)):
            raise OwnershipChangedError("artifact root path binding changed")
    finally:
        _close_finally("artifact root verification", probe)


def _verify_parent_binding(root_fd: int, parent: _ParentHandle) -> None:
    try:
        probe = _open_relative_directory(root_fd, parent.parts)
    except OSError as error:
        raise OwnershipChangedError("artifact parent path binding changed") from error
    try:
        if _directory_identity(_fstat(probe)) != _directory_identity(_fstat(parent.fd)):
            raise OwnershipChangedError("artifact parent path binding changed")
    finally:
        _close_finally("artifact parent verification", probe)


def _open_root(root: os.PathLike[str] | str) -> tuple[Path, int]:
    _require_supported_platform()
    normalized = _canonical_root(root)
    try:
        descriptor = _open_absolute_directory(normalized)
    except OSError as error:
        raise ArtifactIOError(
            "artifact root must be a direct real directory"
        ) from error
    try:
        # Probe directory-fsync support before any public operation is allowed
        # to mutate the namespace rooted here.
        try:
            _fsync(descriptor)
        except OSError as error:
            unsupported = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}
            if error.errno in unsupported:
                raise UnsupportedPlatformError(
                    "artifact root filesystem does not support directory fsync"
                ) from error
            raise
        _verify_root_binding(normalized, descriptor)
        return normalized, descriptor
    except BaseException as error:
        close_errors = _close_all(descriptor)
        if close_errors:
            raise RollbackError(error, close_errors) from error
        raise


def _open_parent(root_fd: int, parts: tuple[str, ...]) -> _ParentHandle:
    try:
        descriptor = _open_relative_directory(root_fd, parts[:-1])
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise ArtifactIOError(
            "artifact parent must be an existing direct real directory"
        ) from error
    parent = _ParentHandle(parts=parts[:-1], fd=descriptor)
    try:
        _verify_parent_binding(root_fd, parent)
        return parent
    except BaseException as error:
        close_errors = _close_all(descriptor)
        if close_errors:
            raise RollbackError(error, close_errors) from error
        raise


def _open_root_and_parent(
    root: os.PathLike[str] | str, parts: tuple[str, ...]
) -> tuple[Path, int, _ParentHandle]:
    root_path, root_fd = _open_root(root)
    try:
        parent = _open_parent(root_fd, parts)
    except BaseException as error:
        close_errors = _close_all(root_fd)
        if close_errors:
            raise RollbackError(error, close_errors) from error
        raise
    return root_path, root_fd, parent


def _stat_optional(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return _stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _raise_if_exists(parent_fd: int, name: str, relative_path: str) -> None:
    if _stat_optional(parent_fd, name) is not None:
        raise FileExistsError(f"refusing to replace artifact: {relative_path}")


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        try:
            written = _write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("artifact write made no progress")
        remaining = remaining[written:]


def _digest_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = _pread(descriptor, min(_CHUNK_SIZE, size - offset), offset)
        if not chunk:
            raise OwnershipChangedError("owned artifact became shorter during hashing")
        digest.update(chunk)
        offset += len(chunk)
    if _pread(descriptor, 1, size):
        raise OwnershipChangedError("owned artifact became longer during hashing")
    return digest.hexdigest()


def _new_temporary(parent_fd: int) -> tuple[str, int]:
    for _attempt in range(_TEMP_ATTEMPTS):
        name = f".artifact-{secrets.token_hex(16)}.tmp"
        try:
            return name, _open(name, _TEMP_FLAGS, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
    raise ArtifactIOError("could not reserve an exclusive artifact temporary file")


def _reserve_stage(parent: _ParentHandle, item: PublishItem) -> _StagedFile:
    if not isinstance(item.payload, bytes):
        raise ValueError("artifact payload must be bytes")
    mode = _validated_mode(item.mode, directory=False)
    temporary_name, descriptor = _new_temporary(parent.fd)
    return _StagedFile(
        item=PublishItem(item.relative_path, item.payload, mode),
        parent=parent,
        name=_relative_parts(item.relative_path)[-1],
        temporary_name=temporary_name,
        fd=descriptor,
        sha256=hashlib.sha256(item.payload).hexdigest(),
    )


def _stage_descriptor(stage: _StagedFile) -> int:
    if stage.fd is None:
        raise OwnershipChangedError("staged artifact descriptor is unavailable")
    return stage.fd


def _record_written_stage_identity(stage: _StagedFile, state: os.stat_result) -> None:
    stage.written_device = int(state.st_dev)
    stage.written_inode = int(state.st_ino)
    stage.written_mode = int(state.st_mode)
    stage.written_size = int(state.st_size)
    stage.written_mtime_ns = int(state.st_mtime_ns)


def _matches_written_stage_identity(stage: _StagedFile, state: os.stat_result) -> bool:
    return stage.written_device is not None and (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_mode),
        int(state.st_size),
        int(state.st_mtime_ns),
    ) == (
        stage.written_device,
        stage.written_inode,
        stage.written_mode,
        stage.written_size,
        stage.written_mtime_ns,
    )


def _write_stage(stage: _StagedFile) -> None:
    descriptor = _stage_descriptor(stage)
    _fchmod(descriptor, stage.item.mode)
    _write_all(descriptor, stage.item.payload)
    _fsync(descriptor)
    state = _fstat(descriptor)
    if (
        not stat.S_ISREG(state.st_mode)
        or stat.S_IMODE(state.st_mode) != stage.item.mode
        or state.st_nlink != 1
        or state.st_size != len(stage.item.payload)
        or _digest_descriptor(descriptor, len(stage.item.payload)) != stage.sha256
    ):
        raise OwnershipChangedError("staged artifact identity or bytes changed")
    _record_written_stage_identity(stage, state)


def _same_owned_stage(stage: _StagedFile, path_state: os.stat_result) -> bool:
    descriptor = stage.fd
    opened_here = False
    if descriptor is None:
        try:
            descriptor = _open(
                stage.name,
                _READ_FLAGS,
                dir_fd=stage.parent.fd,
            )
        except OSError:
            return False
        opened_here = True
    try:
        descriptor_state = _fstat(descriptor)
        if stage.temporary_exists:
            allowed_links = {2 if stage.linked else 1}
        elif stage.linked:
            # Some NFS clients keep one hidden silly-rename link until the
            # descriptor used for staging is closed.  The inode started with
            # one link and this publisher added exactly one public link, so a
            # transient count of two is still an owned stage.
            allowed_links = {1, 2}
        else:
            allowed_links = {0}
        return (
            stat.S_ISREG(path_state.st_mode)
            and _file_identity(path_state) == _file_identity(descriptor_state)
            and path_state.st_nlink in allowed_links
            and _matches_written_stage_identity(stage, path_state)
            and stat.S_IMODE(path_state.st_mode) == stage.item.mode
            and path_state.st_size == len(stage.item.payload)
            and _digest_descriptor(descriptor, len(stage.item.payload)) == stage.sha256
        )
    finally:
        if opened_here:
            _close(descriptor)


def _remove_temporary_exact(stage: _StagedFile) -> None:
    if not stage.temporary_exists:
        return
    state = _stat_optional(stage.parent.fd, stage.temporary_name)
    if state is None:
        stage.temporary_exists = False
        return
    descriptor_state = _fstat(stage.fd) if stage.fd is not None else state
    expected_links = 1 + int(stage.linked)
    if (
        not stat.S_ISREG(state.st_mode)
        or _owner_identity(state) != _owner_identity(descriptor_state)
        or state.st_nlink != expected_links
        or descriptor_state.st_nlink != expected_links
        or (stage.fd is None and not _matches_written_stage_identity(stage, state))
    ):
        raise OwnershipChangedError(
            "temporary artifact was replaced or changed; preserving current entry"
        )
    try:
        _unlink(stage.temporary_name, dir_fd=stage.parent.fd)
    except BaseException:
        # A signal can run after unlinkat removed our name but before this
        # frame records the effect.  Reconcile only an observable absence (or
        # a foreign replacement); never claim or remove that replacement.
        try:
            current = _stat_optional(stage.parent.fd, stage.temporary_name)
            if current is None or not _matches_written_stage_identity(stage, current):
                stage.temporary_exists = False
        except BaseException:
            # Preserve the primary asynchronous/OS failure.  A still-true flag
            # makes the enclosing cleanup retry the exact-owner check.
            pass
        raise
    stage.temporary_exists = False


def _claim_linked_stage(stage: _StagedFile) -> FileClaim:
    path_state = _stat_optional(stage.parent.fd, stage.name)
    if path_state is None:
        raise OwnershipChangedError("published artifact disappeared")
    descriptor = _stage_descriptor(stage)
    descriptor_state = _fstat(descriptor)
    if (
        not stat.S_ISREG(path_state.st_mode)
        or _file_identity(path_state) != _file_identity(descriptor_state)
        or path_state.st_nlink != 1
        or stat.S_IMODE(path_state.st_mode) != stage.item.mode
        or path_state.st_size != len(stage.item.payload)
        or _digest_descriptor(descriptor, len(stage.item.payload)) != stage.sha256
    ):
        raise OwnershipChangedError("published artifact identity or bytes changed")
    claim = FileClaim(
        relative_path=stage.item.relative_path,
        device=int(path_state.st_dev),
        inode=int(path_state.st_ino),
        mode=int(path_state.st_mode),
        link_count=int(path_state.st_nlink),
        size_bytes=int(path_state.st_size),
        mtime_ns=int(path_state.st_mtime_ns),
        ctime_ns=int(path_state.st_ctime_ns),
        sha256=stage.sha256,
    )
    stage.claim = claim
    return claim


def _link_stage(stage: _StagedFile) -> FileClaim:
    try:
        _link(
            stage.temporary_name,
            stage.name,
            src_dir_fd=stage.parent.fd,
            dst_dir_fd=stage.parent.fd,
            follow_symlinks=False,
        )
    except BaseException:
        # A Python signal can run after linkat changed the namespace but before
        # this frame records that fact.  Recover the post-effect state from the
        # retained staging descriptor so completion cannot be misclassified.
        linked_state = _stat_optional(stage.parent.fd, stage.name)
        if linked_state is not None:
            descriptor_state = _fstat(stage.fd)
            if (
                stat.S_ISREG(linked_state.st_mode)
                and _owner_identity(linked_state) == _owner_identity(descriptor_state)
                and linked_state.st_nlink == 2
                and descriptor_state.st_nlink == 2
            ):
                stage.linked = True
        raise
    stage.linked = True
    linked_state = _stat_optional(stage.parent.fd, stage.name)
    if linked_state is None or not _same_owned_stage(stage, linked_state):
        raise OwnershipChangedError("newly linked artifact identity or bytes changed")
    descriptor = _stage_descriptor(stage)
    stage.fd = None
    try:
        _close(descriptor)
    except OSError as error:
        try:
            _fstat(descriptor)
        except OSError as probe_error:
            if probe_error.errno != errno.EBADF:
                raise
            # Linux may report a close error after releasing the descriptor.
            # Retain that report and finish the exact namespace transition so
            # the caller can classify the already-visible publication phase.
            stage.transition_close_error = error
        else:
            stage.fd = descriptor
            raise
    _remove_temporary_exact(stage)
    stage.fd = _open(
        stage.name,
        _READ_FLAGS,
        dir_fd=stage.parent.fd,
    )
    return _claim_linked_stage(stage)


def _rollback_link_exact(stage: _StagedFile) -> None:
    if not stage.linked:
        return
    state = _stat_optional(stage.parent.fd, stage.name)
    if state is None:
        stage.linked = False
        return
    if not _same_owned_stage(stage, state):
        raise OwnershipChangedError(
            "published artifact was replaced or changed; preserving current entry"
        )
    if stage.claim is not None and _file_identity(state) != (
        stage.claim.device,
        stage.claim.inode,
        stage.claim.mode,
        stage.claim.link_count,
        stage.claim.size_bytes,
        stage.claim.mtime_ns,
        stage.claim.ctime_ns,
    ):
        raise OwnershipChangedError(
            "published artifact changed after its claim; preserving current entry"
        )
    _unlink(stage.name, dir_fd=stage.parent.fd)
    stage.linked = False


def _close_stages(stages: Sequence[_StagedFile]) -> list[OSError]:
    errors: list[OSError] = []
    for stage in stages:
        if stage.fd is None:
            continue
        descriptor = stage.fd
        stage.fd = None
        try:
            _close(descriptor)
        except OSError as error:
            errors.append(error)
    return errors


def _close_parents(parents: Sequence[_ParentHandle]) -> list[OSError]:
    errors: list[OSError] = []
    for parent in parents:
        try:
            _close(parent.fd)
        except OSError as error:
            errors.append(error)
    return errors


def snapshot_file(
    root: os.PathLike[str] | str,
    relative_path: os.PathLike[str] | str,
    *,
    capture_bytes: bool = False,
) -> tuple[FileClaim, bytes | None]:
    """Hash a stable regular file without following any path symlink."""

    if type(capture_bytes) is not bool:
        raise ValueError("capture_bytes must be boolean")
    parts = _relative_parts(relative_path)
    normalized_relative = "/".join(parts)
    root_path, root_fd, parent = _open_root_and_parent(root, parts)
    descriptor: int | None = None
    try:
        before_path = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
        if not stat.S_ISREG(before_path.st_mode) or stat.S_ISLNK(before_path.st_mode):
            raise ArtifactIOError("artifact must be a direct regular file")
        descriptor = _open(parts[-1], _READ_FLAGS, dir_fd=parent.fd)
        before_descriptor = _fstat(descriptor)
        if _file_identity(before_descriptor) != _file_identity(before_path):
            raise OwnershipChangedError("artifact changed before descriptor open")
        digest = hashlib.sha256()
        retained = bytearray() if capture_bytes else None
        while True:
            chunk = _read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            if retained is not None:
                retained.extend(chunk)
        after_descriptor = _fstat(descriptor)
        after_path = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
        if {
            _file_identity(before_path),
            _file_identity(before_descriptor),
            _file_identity(after_descriptor),
            _file_identity(after_path),
        } != {_file_identity(before_path)}:
            raise OwnershipChangedError("artifact changed while being hashed")
        _verify_parent_binding(root_fd, parent)
        _verify_root_binding(root_path, root_fd)
        final_descriptor = _fstat(descriptor)
        final_path = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
        if _file_identity(final_descriptor) != _file_identity(
            before_path
        ) or _file_identity(final_path) != _file_identity(before_path):
            raise OwnershipChangedError(
                "artifact changed during final path-binding validation"
            )
        claim = FileClaim(
            relative_path=normalized_relative,
            device=int(after_path.st_dev),
            inode=int(after_path.st_ino),
            mode=int(after_path.st_mode),
            link_count=int(after_path.st_nlink),
            size_bytes=int(after_path.st_size),
            mtime_ns=int(after_path.st_mtime_ns),
            ctime_ns=int(after_path.st_ctime_ns),
            sha256=digest.hexdigest(),
        )
        return claim, None if retained is None else bytes(retained)
    finally:
        _close_finally("stable artifact snapshot", descriptor, parent.fd, root_fd)


def publish_bytes_exclusive(
    root: os.PathLike[str] | str,
    relative_path: os.PathLike[str] | str,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> FileClaim:
    """Durably publish immutable bytes without replacing any directory entry."""

    parts = _relative_parts(relative_path)
    normalized_relative = "/".join(parts)
    item = PublishItem(normalized_relative, payload, mode)
    root_path, root_fd, parent = _open_root_and_parent(root, parts)
    stage: _StagedFile | None = None
    cleanup_errors: list[BaseException] = []
    try:
        _raise_if_exists(parent.fd, parts[-1], normalized_relative)
        _verify_parent_binding(root_fd, parent)
        stage = _reserve_stage(parent, item)
        _write_stage(stage)
        _verify_parent_binding(root_fd, parent)
        claim = _link_stage(stage)
        _fsync(parent.fd)
        _verify_parent_binding(root_fd, parent)
        _verify_root_binding(root_path, root_fd)
        claim = _claim_linked_stage(stage)
        if stage.transition_close_error is not None:
            raise PublicationIndeterminateError(
                normalized_relative, stage.transition_close_error
            )
        return claim
    except PublicationIndeterminateError:
        raise
    except BaseException as error:
        if stage is not None and stage.linked:
            try:
                _rollback_link_exact(stage)
            except BaseException as rollback_error:
                cleanup_errors.append(rollback_error)
        if stage is not None and stage.temporary_exists:
            try:
                _remove_temporary_exact(stage)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if stage is not None:
            try:
                _fsync(parent.fd)
            except BaseException as durability_error:
                cleanup_errors.append(durability_error)
        if cleanup_errors:
            raise RollbackError(error, cleanup_errors) from error
        raise
    finally:
        _close_finally(
            "exclusive artifact publication",
            None if stage is None else stage.fd,
            parent.fd,
            root_fd,
            publication_visible_path=(
                normalized_relative if stage is not None and stage.linked else None
            ),
        )


def acquire_lock_exclusive(
    root: os.PathLike[str] | str,
    relative_path: os.PathLike[str] | str,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> FileClaim:
    """Acquire an immutable no-clobber lock and return its release capability."""

    return publish_bytes_exclusive(root, relative_path, payload, mode=mode)


def _claim_identity(claim: FileClaim) -> tuple[int, ...]:
    return (
        claim.device,
        claim.inode,
        claim.mode,
        claim.link_count,
        claim.size_bytes,
        claim.mtime_ns,
        claim.ctime_ns,
    )


def _release_exact_claim(
    root: os.PathLike[str] | str, owner: FileClaim, *, kind: str
) -> None:
    parts = _relative_parts(owner.relative_path)
    if "/".join(parts) != owner.relative_path:
        raise ValueError(f"{kind} claim path is not canonical")
    root_path, root_fd, parent = _open_root_and_parent(root, parts)
    descriptor: int | None = None
    removed = False
    try:
        before_path = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before_path.st_mode)
            or before_path.st_nlink != 1
            or _file_identity(before_path) != _claim_identity(owner)
        ):
            raise OwnershipChangedError(
                f"{kind} no longer has its exact owned identity"
            )
        descriptor = _open(parts[-1], _READ_FLAGS, dir_fd=parent.fd)
        before_descriptor = _fstat(descriptor)
        if _file_identity(before_descriptor) != _claim_identity(owner):
            raise OwnershipChangedError(f"{kind} changed before release open")
        digest = hashlib.sha256()
        while True:
            chunk = _read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        after_descriptor = _fstat(descriptor)
        after_path = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
        if (
            _file_identity(after_descriptor) != _claim_identity(owner)
            or _file_identity(after_path) != _claim_identity(owner)
            or digest.hexdigest() != owner.sha256
        ):
            raise OwnershipChangedError(
                f"{kind} changed or belongs to another owner; preserving it"
            )
        _verify_parent_binding(root_fd, parent)
        _verify_root_binding(root_path, root_fd)
        # POSIX has no compare-and-unlink primitive.  Under the documented
        # cooperative-writer contract, the immediately preceding identity and
        # digest checks bind this unlink to the owner claim.
        try:
            _unlink(parts[-1], dir_fd=parent.fd)
        except BaseException as error:
            current = _stat_optional(parent.fd, parts[-1])
            if current is None or _owner_identity(current) != (
                owner.device,
                owner.inode,
                owner.mode,
            ):
                removed = True
                raise RemovalIndeterminateError(owner.relative_path, error) from error
            raise
        else:
            removed = True
        try:
            _fsync(parent.fd)
        except BaseException as error:
            raise RemovalIndeterminateError(owner.relative_path, error) from error
    finally:
        _close_finally(
            "exact owned-file release",
            descriptor,
            parent.fd,
            root_fd,
            removed_path=owner.relative_path if removed else None,
        )


def release_lock_exact(root: os.PathLike[str] | str, owner: FileClaim) -> None:
    """Remove only the unchanged, singly linked lock represented by ``owner``."""

    _release_exact_claim(root, owner, kind="lock")


def create_directory_exclusive(
    root: os.PathLike[str] | str,
    relative_path: os.PathLike[str] | str,
    *,
    mode: int = 0o755,
) -> OwnedDirectory:
    """Create one directory under an existing descriptor-walked parent."""

    mode = _validated_mode(mode, directory=True)
    parts = _relative_parts(relative_path)
    normalized_relative = "/".join(parts)
    root_path, root_fd, parent = _open_root_and_parent(root, parts)
    child_fd: int | None = None
    created = False
    owner: OwnedDirectory | None = None
    cleanup_errors: list[BaseException] = []
    try:
        _verify_parent_binding(root_fd, parent)
        try:
            _mkdir(parts[-1], mode=mode, dir_fd=parent.fd)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace artifact directory: {normalized_relative}"
            ) from error
        except BaseException as error:
            if _stat_optional(parent.fd, parts[-1]) is not None:
                raise PublicationIndeterminateError(
                    normalized_relative, error
                ) from error
            raise
        created = True
        child_fd = _open(parts[-1], _DIRECTORY_FLAGS, dir_fd=parent.fd)
        _fchmod(child_fd, mode)
        state = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
        descriptor_state = _fstat(child_fd)
        if (
            not stat.S_ISDIR(state.st_mode)
            or _directory_identity(state) != _directory_identity(descriptor_state)
            or stat.S_IMODE(state.st_mode) != mode
        ):
            raise OwnershipChangedError("created directory identity changed")
        owner = OwnedDirectory(
            relative_path=normalized_relative,
            device=int(state.st_dev),
            inode=int(state.st_ino),
            mode=int(state.st_mode),
        )
        _fsync(child_fd)
        _fsync(parent.fd)
        _verify_parent_binding(root_fd, parent)
        _verify_root_binding(root_path, root_fd)
        final_state = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
        if (
            child_fd is None
            or _directory_identity(final_state) != _directory_identity(_fstat(child_fd))
            or _directory_identity(final_state)
            != (owner.device, owner.inode, owner.mode)
        ):
            raise OwnershipChangedError(
                "created directory changed during final validation"
            )
        return owner
    except BaseException as error:
        if created:
            try:
                state = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
                if child_fd is None:
                    raise OwnershipChangedError(
                        "created directory cannot be identified for rollback"
                    )
                descriptor_state = _fstat(child_fd)
                if (
                    not stat.S_ISDIR(state.st_mode)
                    or _directory_identity(state)
                    != _directory_identity(descriptor_state)
                    or (
                        owner is not None
                        and _directory_identity(state)
                        != (
                            owner.device,
                            owner.inode,
                            owner.mode,
                        )
                    )
                ):
                    raise OwnershipChangedError(
                        "created directory was replaced; preserving current entry"
                    )
                _rmdir(parts[-1], dir_fd=parent.fd)
                _fsync(parent.fd)
            except BaseException as rollback_error:
                cleanup_errors.append(rollback_error)
        if cleanup_errors:
            raise RollbackError(error, cleanup_errors) from error
        raise
    finally:
        _close_finally(
            "exclusive directory creation",
            child_fd,
            parent.fd,
            root_fd,
            publication_visible_path=(
                normalized_relative if owner is not None else None
            ),
        )


def remove_empty_directory_exact(
    root: os.PathLike[str] | str, owner: OwnedDirectory
) -> None:
    """Remove an exact owned directory only while it remains empty."""

    parts = _relative_parts(owner.relative_path)
    root_path, root_fd, parent = _open_root_and_parent(root, parts)
    child_fd: int | None = None
    removed = False
    try:
        state = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
        if not stat.S_ISDIR(state.st_mode) or _directory_identity(state) != (
            owner.device,
            owner.inode,
            owner.mode,
        ):
            raise OwnershipChangedError(
                "owned directory was replaced or changed; preserving current entry"
            )
        child_fd = _open(parts[-1], _DIRECTORY_FLAGS, dir_fd=parent.fd)
        if _directory_identity(_fstat(child_fd)) != (
            owner.device,
            owner.inode,
            owner.mode,
        ):
            raise OwnershipChangedError("owned directory changed before removal")
        _verify_parent_binding(root_fd, parent)
        _verify_root_binding(root_path, root_fd)
        try:
            _rmdir(parts[-1], dir_fd=parent.fd)
        except BaseException as error:
            current = _stat_optional(parent.fd, parts[-1])
            if current is None or _owner_identity(current) != (
                owner.device,
                owner.inode,
                owner.mode,
            ):
                removed = True
                raise RemovalIndeterminateError(owner.relative_path, error) from error
            raise
        else:
            removed = True
        try:
            _fsync(parent.fd)
        except BaseException as error:
            raise RemovalIndeterminateError(owner.relative_path, error) from error
    finally:
        _close_finally(
            "exact directory removal",
            child_fd,
            parent.fd,
            root_fd,
            removed_path=owner.relative_path if removed else None,
        )


def reserve_log_exclusive(
    root: os.PathLike[str] | str,
    relative_path: os.PathLike[str] | str,
    *,
    mode: int = 0o644,
) -> OwnedLog:
    """Reserve a new empty mutable log without replacing any entry."""

    if type(mode) is not int or mode & 0o600 != 0o600:
        raise ValueError("log mode must grant owner read and write access")
    return OwnedLog(
        reservation=publish_bytes_exclusive(root, relative_path, b"", mode=mode)
    )


def _log_owner_matches(owner: OwnedLog, state: os.stat_result) -> bool:
    claim = owner.reservation
    return (
        stat.S_ISREG(state.st_mode)
        and state.st_nlink == 1
        and _owner_identity(state) == (claim.device, claim.inode, claim.mode)
    )


@contextmanager
def open_log_append_exact(
    root: os.PathLike[str] | str, owner: OwnedLog
) -> Iterator[BinaryIO]:
    """Append through the exact reserved inode, never through a replacement."""

    parts = _relative_parts(owner.reservation.relative_path)
    root_path, root_fd, parent = _open_root_and_parent(root, parts)
    descriptor: int | None = None
    handle: BinaryIO | None = None
    body_error: BaseException | None = None
    try:
        before = _stat(parts[-1], dir_fd=parent.fd, follow_symlinks=False)
        if not _log_owner_matches(owner, before):
            raise OwnershipChangedError("reserved log identity changed before append")
        descriptor = _open(parts[-1], _APPEND_FLAGS, dir_fd=parent.fd)
        if _file_identity(_fstat(descriptor)) != _file_identity(before):
            raise OwnershipChangedError("reserved log changed before append open")
        _verify_parent_binding(root_fd, parent)
        _verify_root_binding(root_path, root_fd)
        handle = os.fdopen(descriptor, "ab", buffering=0)
        descriptor = None
        try:
            yield handle
        except BaseException as error:
            body_error = error
            raise
        finally:
            finalization_errors: list[BaseException] = []
            if handle is not None and handle.closed:
                finalization_errors.append(
                    ArtifactIOError(
                        "caller closed the owned log handle before final validation"
                    )
                )
            elif handle is not None:
                try:
                    handle.flush()
                    _fsync(handle.fileno())
                    after_descriptor = _fstat(handle.fileno())
                    try:
                        after_path = _stat(
                            parts[-1], dir_fd=parent.fd, follow_symlinks=False
                        )
                    except OSError as error:
                        raise OwnershipChangedError(
                            "reserved log path disappeared during append"
                        ) from error
                    if not _log_owner_matches(
                        owner, after_descriptor
                    ) or _file_identity(after_descriptor) != _file_identity(after_path):
                        raise OwnershipChangedError(
                            "reserved log path changed during append; replacement "
                            "was not written"
                        )
                    _verify_parent_binding(root_fd, parent)
                    _verify_root_binding(root_path, root_fd)
                    final_descriptor = _fstat(handle.fileno())
                    final_path = _stat(
                        parts[-1], dir_fd=parent.fd, follow_symlinks=False
                    )
                    if not _log_owner_matches(
                        owner, final_descriptor
                    ) or _file_identity(final_descriptor) != _file_identity(final_path):
                        raise OwnershipChangedError(
                            "reserved log changed during final path validation"
                        )
                except BaseException as error:
                    finalization_errors.append(error)
                finally:
                    try:
                        handle.close()
                    except BaseException as error:
                        finalization_errors.append(error)
            if finalization_errors:
                if body_error is not None:
                    raise RollbackError(body_error, finalization_errors) from body_error
                error = finalization_errors[0]
                if len(finalization_errors) > 1:
                    error = RollbackError(error, finalization_errors[1:])
                raise PublicationIndeterminateError(
                    owner.reservation.relative_path, error
                ) from error
    finally:
        _close_finally(
            "owned log append",
            descriptor,
            parent.fd,
            root_fd,
            publication_visible_path=(
                owner.reservation.relative_path if handle is not None else None
            ),
        )


def rollback_unused_log_exact(root: os.PathLike[str] | str, owner: OwnedLog) -> None:
    """Remove only an unchanged, still-empty pre-handoff log reservation."""

    _release_exact_claim(root, owner.reservation, kind="unused log")


def _validated_publish_item(item: PublishItem) -> PublishItem:
    if not isinstance(item, PublishItem):
        raise TypeError("bundle members must be PublishItem instances")
    parts = _relative_parts(item.relative_path)
    if not isinstance(item.payload, bytes):
        raise ValueError("artifact payload must be bytes")
    return PublishItem(
        relative_path="/".join(parts),
        payload=item.payload,
        mode=_validated_mode(item.mode, directory=False),
    )


def publish_bundle_exclusive(
    root: os.PathLike[str] | str,
    members: Sequence[PublishItem],
    *,
    completion: PublishItem,
) -> PublishedBundle:
    """Publish a multi-file bundle with its completion member linked last.

    Any failure before the completion link triggers reverse-order, exact-owned
    rollback.  Once that link succeeds the bundle is externally discoverable;
    subsequent failure is reported as :class:`CommitIndeterminateError` and
    published entries are never rolled back.
    """

    normalized_members = tuple(_validated_publish_item(item) for item in members)
    normalized_completion = _validated_publish_item(completion)
    if not normalized_members:
        raise ValueError("bundle must contain at least one non-completion member")
    paths = [item.relative_path for item in normalized_members]
    paths.append(normalized_completion.relative_path)
    if len(paths) != len(set(paths)):
        raise ValueError("bundle member paths must be distinct")
    ordered_members = tuple(
        sorted(normalized_members, key=lambda item: item.relative_path)
    )

    root_path, root_fd = _open_root(root)
    parents_by_parts: dict[tuple[str, ...], _ParentHandle] = {}
    stages: list[_StagedFile] = []
    cleanup_errors: list[BaseException] = []

    def parent_for(item: PublishItem) -> _ParentHandle:
        parts = _relative_parts(item.relative_path)
        parent_parts = parts[:-1]
        if parent_parts not in parents_by_parts:
            parents_by_parts[parent_parts] = _open_parent(root_fd, parts)
        return parents_by_parts[parent_parts]

    completion_stage: _StagedFile | None = None
    try:
        all_items = (*ordered_members, normalized_completion)
        # A known collision causes zero publication and zero staging files.
        for item in all_items:
            parts = _relative_parts(item.relative_path)
            parent = parent_for(item)
            _raise_if_exists(parent.fd, parts[-1], item.relative_path)
            _verify_parent_binding(root_fd, parent)
        _verify_root_binding(root_path, root_fd)

        for item in all_items:
            stage = _reserve_stage(parent_for(item), item)
            stages.append(stage)
            _write_stage(stage)
            if item.relative_path == normalized_completion.relative_path:
                completion_stage = stage

        member_stages = stages[:-1]
        for stage in member_stages:
            _verify_parent_binding(root_fd, stage.parent)
            _link_stage(stage)

        data_parents = {stage.parent.parts: stage.parent for stage in member_stages}
        for parent_parts in sorted(data_parents):
            parent = data_parents[parent_parts]
            _fsync(parent.fd)
            _verify_parent_binding(root_fd, parent)
        _verify_root_binding(root_path, root_fd)

        if completion_stage is None:  # pragma: no cover - construction invariant
            raise AssertionError("completion stage was not prepared")
        _verify_parent_binding(root_fd, completion_stage.parent)
        for stage in member_stages:
            _claim_linked_stage(stage)
        _link_stage(completion_stage)
        _fsync(completion_stage.parent.fd)
        for parent_parts in sorted(parents_by_parts):
            _verify_parent_binding(root_fd, parents_by_parts[parent_parts])
        _verify_root_binding(root_path, root_fd)
        for stage in stages:
            _claim_linked_stage(stage)
        if completion_stage.claim is None or any(
            stage.claim is None for stage in member_stages
        ):
            raise AssertionError("published bundle is missing an artifact claim")
        transition_close_errors = [
            stage.transition_close_error
            for stage in stages
            if stage.transition_close_error is not None
        ]
        if transition_close_errors:
            raise transition_close_errors[0]
        return PublishedBundle(
            members=tuple(
                stage.claim for stage in member_stages if stage.claim is not None
            ),
            completion=completion_stage.claim,
        )
    except BaseException as error:
        completion_visible = completion_stage is not None and completion_stage.linked
        if not completion_visible:
            for stage in reversed(stages):
                if stage.linked:
                    try:
                        _rollback_link_exact(stage)
                    except BaseException as rollback_error:
                        cleanup_errors.append(rollback_error)
        for stage in stages:
            if stage.temporary_exists:
                try:
                    _remove_temporary_exact(stage)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
        for parent_parts in sorted(parents_by_parts):
            try:
                _fsync(parents_by_parts[parent_parts].fd)
            except BaseException as durability_error:
                cleanup_errors.append(durability_error)
        if completion_visible:
            raise CommitIndeterminateError(
                error,
                published_paths=[
                    stage.item.relative_path for stage in stages if stage.linked
                ],
                cleanup_errors=cleanup_errors,
            ) from error
        if cleanup_errors:
            raise RollbackError(error, cleanup_errors) from error
        raise
    finally:
        close_errors = _close_stages(stages)
        close_errors.extend(_close_parents(list(parents_by_parts.values())))
        try:
            _close(root_fd)
        except OSError as error:
            close_errors.append(error)
        if close_errors and sys.exc_info()[0] is None:
            error = ArtifactIOError(
                "bundle publication completed but descriptor cleanup failed: "
                + "; ".join(str(error) for error in close_errors)
            )
            if completion_stage is not None and completion_stage.linked:
                raise CommitIndeterminateError(
                    error,
                    published_paths=[
                        stage.item.relative_path for stage in stages if stage.linked
                    ],
                    cleanup_errors=close_errors,
                ) from error
            raise error
