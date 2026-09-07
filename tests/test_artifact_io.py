from __future__ import annotations

import dataclasses
import errno
import hashlib
import os
import stat
import threading
from pathlib import Path

import pytest

from scripts import artifact_io


def _temporary_paths(root: Path) -> list[Path]:
    return sorted(root.rglob(".artifact-*.tmp"))


def _fd_count() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


@pytest.mark.parametrize(
    "relative",
    (
        "",
        ".",
        "..",
        "/absolute",
        "a/../b",
        "a/./b",
        "a//b",
        "a/",
        "a\x00b",
    ),
)
def test_canonical_root_relative_paths_are_required(
    tmp_path: Path, relative: str
) -> None:
    with pytest.raises(ValueError, match="canonical|nonempty"):
        artifact_io.publish_bytes_exclusive(tmp_path, relative, b"payload")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("root_value", ("relative", "/tmp/../tmp"))
def test_root_must_be_absolute_and_canonical(tmp_path: Path, root_value: str) -> None:
    del tmp_path
    with pytest.raises(ValueError, match="absolute canonical"):
        artifact_io.publish_bytes_exclusive(root_value, "artifact", b"payload")


def test_root_and_intermediate_symlinks_are_rejected_without_escape(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(artifact_io.ArtifactIOError):
        artifact_io.publish_bytes_exclusive(root_alias, "artifact", b"payload")
    assert not (real_root / "artifact").exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    (real_root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(artifact_io.ArtifactIOError):
        artifact_io.publish_bytes_exclusive(real_root, "linked/artifact", b"payload")
    assert not (outside / "artifact").exists()


def test_root_walk_close_error_is_not_retried_and_leaks_no_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_close = artifact_io._close
    closed_once: set[int] = set()
    injected = False

    def close_then_raise_once(descriptor: int) -> None:
        nonlocal injected
        if not injected:
            injected = True
            closed_once.add(descriptor)
            real_close(descriptor)
            raise OSError(errno.EIO, "injected walk close report")
        assert descriptor not in closed_once
        real_close(descriptor)

    before_fds = _fd_count()
    monkeypatch.setattr(artifact_io, "_close", close_then_raise_once)
    with pytest.raises(artifact_io.ArtifactIOError, match="root walk"):
        artifact_io.publish_bytes_exclusive(tmp_path, "artifact", b"payload")
    assert _fd_count() == before_fds
    assert not (tmp_path / "artifact").exists()


def test_publish_is_exact_durable_and_stably_readable(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    payload = b"complete bytes\n"

    claim = artifact_io.publish_bytes_exclusive(
        tmp_path, "artifacts/result.bin", payload, mode=0o640
    )

    target = tmp_path / "artifacts/result.bin"
    observed = target.stat(follow_symlinks=False)
    assert target.read_bytes() == payload
    assert stat.S_IMODE(observed.st_mode) == 0o640
    assert observed.st_nlink == 1
    assert claim.relative_path == "artifacts/result.bin"
    assert claim.sha256 == hashlib.sha256(payload).hexdigest()
    assert (claim.device, claim.inode) == (observed.st_dev, observed.st_ino)
    snapshot, retained = artifact_io.snapshot_file(
        tmp_path, claim.relative_path, capture_bytes=True
    )
    assert snapshot == claim
    assert retained == payload
    assert _temporary_paths(tmp_path) == []


def test_stage_descriptor_is_closed_before_temp_unlink_for_nfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avoid NFS silly-renaming an unlinked but still-open staging inode."""

    (tmp_path / "out").mkdir()
    real_reserve = artifact_io._reserve_stage
    real_unlink = artifact_io._unlink
    stages: list[artifact_io._StagedFile] = []

    def capture_stage(
        parent: artifact_io._ParentHandle, item: artifact_io.PublishItem
    ) -> artifact_io._StagedFile:
        stage = real_reserve(parent, item)
        stages.append(stage)
        return stage

    def verify_closed_before_unlink(path: str, **kwargs: object) -> None:
        matching = [stage for stage in stages if stage.temporary_name == path]
        if matching:
            assert matching[0].fd is None
        real_unlink(path, **kwargs)

    monkeypatch.setattr(artifact_io, "_reserve_stage", capture_stage)
    monkeypatch.setattr(artifact_io, "_unlink", verify_closed_before_unlink)

    claim = artifact_io.publish_bytes_exclusive(
        tmp_path, "out/artifact", b"nfs-safe publication"
    )

    assert claim.link_count == 1
    assert (tmp_path / "out/artifact").stat().st_nlink == 1
    assert _temporary_paths(tmp_path) == []


@pytest.mark.parametrize(
    "node_kind", ("file", "directory", "symlink", "dangling_symlink", "fifo")
)
def test_publish_never_replaces_any_existing_node(
    tmp_path: Path, node_kind: str
) -> None:
    parent = tmp_path / "out"
    parent.mkdir()
    target = parent / "artifact"
    if node_kind == "file":
        target.write_bytes(b"foreign")
    elif node_kind == "directory":
        target.mkdir()
    elif node_kind == "symlink":
        source = tmp_path / "source"
        source.write_bytes(b"foreign")
        target.symlink_to(source)
    elif node_kind == "dangling_symlink":
        target.symlink_to(tmp_path / "missing")
    else:
        os.mkfifo(target)

    before = target.lstat()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", b"replacement")
    after = target.lstat()
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )
    assert _temporary_paths(tmp_path) == []


def test_two_publishers_have_exactly_one_winner(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    barrier = threading.Barrier(2)
    successes: list[artifact_io.FileClaim] = []
    failures: list[BaseException] = []

    def publish(payload: bytes) -> None:
        barrier.wait()
        try:
            successes.append(
                artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", payload)
            )
        except BaseException as error:
            failures.append(error)

    threads = [
        threading.Thread(target=publish, args=(payload,))
        for payload in (b"first", b"second")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    assert (tmp_path / "out/artifact").read_bytes() in {b"first", b"second"}
    assert _temporary_paths(tmp_path) == []


def test_link_racer_wins_without_clobber_or_temp_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    target = tmp_path / "out/artifact"
    real_link = artifact_io._link

    def raced_link(source: str, destination: str, **kwargs: object) -> None:
        target.write_bytes(b"racer")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(artifact_io, "_link", raced_link)
    with pytest.raises(FileExistsError):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", b"candidate")
    assert target.read_bytes() == b"racer"
    assert _temporary_paths(tmp_path) == []


def test_partial_writes_are_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_write = artifact_io._write

    def partial_write(descriptor: int, payload: object) -> int:
        view = memoryview(payload)  # type: ignore[arg-type]
        return real_write(descriptor, view[: max(1, len(view) // 3)])

    monkeypatch.setattr(artifact_io, "_write", partial_write)
    payload = b"0123456789" * 10
    artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", payload)
    assert (tmp_path / "out/artifact").read_bytes() == payload


def test_partial_write_failure_cleans_owned_temp_and_closes_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_write = artifact_io._write
    calls = 0

    def failing_write(descriptor: int, payload: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, memoryview(payload)[:1])  # type: ignore[arg-type]
        raise OSError(errno.EIO, "injected write failure")

    monkeypatch.setattr(artifact_io, "_write", failing_write)
    before_fds = _fd_count()
    with pytest.raises(OSError, match="injected write failure"):
        artifact_io.publish_bytes_exclusive(
            tmp_path, "out/artifact", b"partial payload"
        )
    assert _fd_count() == before_fds
    assert not (tmp_path / "out/artifact").exists()
    assert _temporary_paths(tmp_path) == []


def test_directory_fsync_capability_is_checked_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()

    def unsupported(_descriptor: int) -> None:
        raise OSError(errno.EINVAL, "directory fsync unsupported")

    monkeypatch.setattr(artifact_io, "_fsync", unsupported)
    with pytest.raises(artifact_io.UnsupportedPlatformError):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", b"payload")
    assert not (tmp_path / "out/artifact").exists()
    assert _temporary_paths(tmp_path) == []


def test_file_fsync_failure_cleans_owned_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_fsync = artifact_io._fsync

    def fail_regular_file(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(errno.EIO, "injected file fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifact_io, "_fsync", fail_regular_file)
    with pytest.raises(OSError, match="injected file fsync failure"):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", b"payload")
    assert not (tmp_path / "out/artifact").exists()
    assert _temporary_paths(tmp_path) == []


def test_fchmod_and_keyboard_interrupt_clean_owned_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()

    def fail_fchmod(_descriptor: int, _mode: int) -> None:
        raise OSError(errno.EPERM, "injected fchmod failure")

    monkeypatch.setattr(artifact_io, "_fchmod", fail_fchmod)
    with pytest.raises(OSError, match="injected fchmod failure"):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/fchmod", b"payload")
    assert _temporary_paths(tmp_path) == []

    monkeypatch.undo()

    def interrupt_write(_descriptor: int, _payload: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(artifact_io, "_write", interrupt_write)
    with pytest.raises(KeyboardInterrupt):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/interrupted", b"payload")
    assert _temporary_paths(tmp_path) == []


def test_link_failure_cleans_owned_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EPERM, "hard links unsupported")

    monkeypatch.setattr(artifact_io, "_link", fail_link)
    with pytest.raises(OSError, match="hard links unsupported"):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", b"payload")
    assert not (tmp_path / "out/artifact").exists()
    assert _temporary_paths(tmp_path) == []


def test_temp_unlink_failure_rolls_back_link_and_then_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_unlink = artifact_io._unlink
    failed = False

    def fail_once(path: str, **kwargs: object) -> None:
        nonlocal failed
        if path.startswith(".artifact-") and not failed:
            failed = True
            raise OSError(errno.EIO, "injected temp unlink failure")
        real_unlink(path, **kwargs)

    monkeypatch.setattr(artifact_io, "_unlink", fail_once)
    with pytest.raises(OSError, match="injected temp unlink failure"):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", b"payload")
    assert not (tmp_path / "out/artifact").exists()
    assert _temporary_paths(tmp_path) == []


def test_parent_swap_during_publish_cannot_escape_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "out"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original = tmp_path / "original"
    real_link = artifact_io._link
    swapped = False

    def swap_then_link(source: str, destination: str, **kwargs: object) -> None:
        nonlocal swapped
        if not swapped:
            parent.rename(original)
            parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(artifact_io, "_link", swap_then_link)
    with pytest.raises(artifact_io.OwnershipChangedError):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", b"payload")
    assert not (outside / "artifact").exists()
    assert not (original / "artifact").exists()
    assert _temporary_paths(original) == []


def test_post_fsync_replacement_prevents_stale_single_file_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    target = tmp_path / "out/artifact"
    original = tmp_path / "out/original"
    real_fsync = artifact_io._fsync
    replaced = False

    def replace_after_directory_fsync(descriptor: int) -> None:
        nonlocal replaced
        real_fsync(descriptor)
        if (
            not replaced
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and target.exists()
        ):
            target.rename(original)
            target.write_bytes(b"foreign")
            replaced = True

    monkeypatch.setattr(artifact_io, "_fsync", replace_after_directory_fsync)
    with pytest.raises(artifact_io.RollbackError):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/artifact", b"owned")
    assert target.read_bytes() == b"foreign"
    assert original.read_bytes() == b"owned"


def test_lock_release_requires_exact_identity_and_digest(tmp_path: Path) -> None:
    (tmp_path / "locks").mkdir()
    payload = b'{"owner":"one"}\n'
    owner = artifact_io.acquire_lock_exclusive(
        tmp_path, "locks/generation.lock", payload
    )
    path = tmp_path / "locks/generation.lock"

    wrong_digest = dataclasses.replace(owner, sha256="0" * 64)
    with pytest.raises(artifact_io.OwnershipChangedError):
        artifact_io.release_lock_exact(tmp_path, wrong_digest)
    assert path.read_bytes() == payload

    alias = tmp_path / "locks/alias"
    os.link(path, alias)
    with pytest.raises(artifact_io.OwnershipChangedError):
        artifact_io.release_lock_exact(tmp_path, owner)
    assert path.read_bytes() == payload
    alias.unlink()
    # Creating/removing a hard link changes ctime, so the original immutable
    # release claim remains invalid even after the alias is gone.
    with pytest.raises(artifact_io.OwnershipChangedError):
        artifact_io.release_lock_exact(tmp_path, owner)
    assert path.read_bytes() == payload


def test_exact_unchanged_lock_is_released(tmp_path: Path) -> None:
    (tmp_path / "locks").mkdir()
    owner = artifact_io.acquire_lock_exclusive(tmp_path, "locks/lock", b"owned")
    artifact_io.release_lock_exact(tmp_path, owner)
    assert not (tmp_path / "locks/lock").exists()


def test_lock_release_fsync_failure_reports_removed_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "locks").mkdir()
    owner = artifact_io.acquire_lock_exclusive(tmp_path, "locks/lock", b"owned")
    path = tmp_path / "locks/lock"
    real_fsync = artifact_io._fsync

    def fail_after_unlink(descriptor: int) -> None:
        if not path.exists():
            raise OSError(errno.EIO, "injected post-unlink fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifact_io, "_fsync", fail_after_unlink)
    with pytest.raises(artifact_io.RemovalIndeterminateError) as captured:
        artifact_io.release_lock_exact(tmp_path, owner)
    assert captured.value.removed is True
    assert not path.exists()


def test_identical_bytes_on_replacement_inode_are_never_released(
    tmp_path: Path,
) -> None:
    (tmp_path / "locks").mkdir()
    payload = b"same bytes"
    owner = artifact_io.acquire_lock_exclusive(tmp_path, "locks/lock", payload)
    lock = tmp_path / "locks/lock"
    original = tmp_path / "locks/original"
    lock.rename(original)
    lock.write_bytes(payload)

    with pytest.raises(artifact_io.OwnershipChangedError):
        artifact_io.release_lock_exact(tmp_path, owner)
    assert lock.read_bytes() == payload
    assert original.read_bytes() == payload


def test_lock_symlink_replacement_is_preserved(tmp_path: Path) -> None:
    (tmp_path / "locks").mkdir()
    owner = artifact_io.acquire_lock_exclusive(tmp_path, "locks/lock", b"owned")
    lock = tmp_path / "locks/lock"
    original = tmp_path / "locks/original"
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"foreign")
    lock.rename(original)
    lock.symlink_to(replacement)

    with pytest.raises(artifact_io.OwnershipChangedError):
        artifact_io.release_lock_exact(tmp_path, owner)
    assert lock.is_symlink()
    assert replacement.read_bytes() == b"foreign"


def test_owned_directory_is_exclusive_and_removed_only_if_exact_and_empty(
    tmp_path: Path,
) -> None:
    (tmp_path / "runs").mkdir()
    owner = artifact_io.create_directory_exclusive(tmp_path, "runs/attempt", mode=0o750)
    run = tmp_path / "runs/attempt"
    assert stat.S_IMODE(run.stat().st_mode) == 0o750
    with pytest.raises(FileExistsError):
        artifact_io.create_directory_exclusive(tmp_path, "runs/attempt")

    child = run / "foreign"
    child.write_bytes(b"keep")
    with pytest.raises(OSError):
        artifact_io.remove_empty_directory_exact(tmp_path, owner)
    assert child.read_bytes() == b"keep"
    child.unlink()
    artifact_io.remove_empty_directory_exact(tmp_path, owner)
    assert not run.exists()


def test_replacement_directory_is_preserved(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    owner = artifact_io.create_directory_exclusive(tmp_path, "runs/attempt")
    run = tmp_path / "runs/attempt"
    original = tmp_path / "runs/original"
    run.rename(original)
    run.mkdir()

    with pytest.raises(artifact_io.OwnershipChangedError):
        artifact_io.remove_empty_directory_exact(tmp_path, owner)
    assert run.is_dir()
    assert original.is_dir()


def test_directory_removal_fsync_failure_is_explicitly_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "runs").mkdir()
    owner = artifact_io.create_directory_exclusive(tmp_path, "runs/attempt")
    run = tmp_path / "runs/attempt"
    real_fsync = artifact_io._fsync

    def fail_after_removal(descriptor: int) -> None:
        if not run.exists():
            raise OSError(errno.EIO, "injected post-rmdir fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifact_io, "_fsync", fail_after_removal)
    with pytest.raises(artifact_io.RemovalIndeterminateError) as captured:
        artifact_io.remove_empty_directory_exact(tmp_path, owner)
    assert captured.value.removed is True
    assert not run.exists()


def test_directory_creation_failure_rolls_back_exact_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "runs").mkdir()
    real_fsync = artifact_io._fsync
    directory_calls = 0

    def fail_child_fsync(descriptor: int) -> None:
        nonlocal directory_calls
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_calls += 1
            if directory_calls == 2:
                raise OSError(errno.EIO, "injected child directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifact_io, "_fsync", fail_child_fsync)
    with pytest.raises(OSError, match="injected child directory fsync failure"):
        artifact_io.create_directory_exclusive(tmp_path, "runs/attempt")
    assert not (tmp_path / "runs/attempt").exists()


def test_log_append_is_bound_to_reserved_inode(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    owner = artifact_io.reserve_log_exclusive(tmp_path, "logs/run.log")
    path = tmp_path / "logs/run.log"
    with artifact_io.open_log_append_exact(tmp_path, owner) as handle:
        handle.write(b"line one\n")
    assert path.read_bytes() == b"line one\n"

    with pytest.raises(artifact_io.OwnershipChangedError):
        artifact_io.rollback_unused_log_exact(tmp_path, owner)
    assert path.read_bytes() == b"line one\n"


def test_log_replacement_before_append_is_preserved(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    owner = artifact_io.reserve_log_exclusive(tmp_path, "logs/run.log")
    path = tmp_path / "logs/run.log"
    path.rename(tmp_path / "logs/original")
    path.write_bytes(b"foreign")

    with pytest.raises(artifact_io.OwnershipChangedError):
        with artifact_io.open_log_append_exact(tmp_path, owner):
            pytest.fail("replacement log must not be opened")
    assert path.read_bytes() == b"foreign"


def test_log_replacement_during_append_never_receives_bytes(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    owner = artifact_io.reserve_log_exclusive(tmp_path, "logs/run.log")
    path = tmp_path / "logs/run.log"
    original = tmp_path / "logs/original"

    with pytest.raises(artifact_io.PublicationIndeterminateError) as captured:
        with artifact_io.open_log_append_exact(tmp_path, owner) as handle:
            path.rename(original)
            path.write_bytes(b"foreign")
            handle.write(b"owned stream")
    assert isinstance(captured.value.primary_error, artifact_io.OwnershipChangedError)
    assert path.read_bytes() == b"foreign"
    assert original.read_bytes() == b"owned stream"


def test_log_body_and_finalization_failures_are_both_retained(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    owner = artifact_io.reserve_log_exclusive(tmp_path, "logs/run.log")
    path = tmp_path / "logs/run.log"

    with pytest.raises(artifact_io.RollbackError) as captured:
        with artifact_io.open_log_append_exact(tmp_path, owner) as handle:
            path.rename(tmp_path / "logs/original")
            path.write_bytes(b"foreign")
            handle.write(b"owned")
            raise ValueError("body failed")
    assert isinstance(captured.value.primary_error, ValueError)
    assert any(
        isinstance(error, artifact_io.OwnershipChangedError)
        for error in captured.value.rollback_errors
    )
    assert path.read_bytes() == b"foreign"


def test_log_handle_cannot_be_closed_before_final_validation(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    owner = artifact_io.reserve_log_exclusive(tmp_path, "logs/run.log")
    with pytest.raises(artifact_io.ArtifactIOError, match="caller closed"):
        with artifact_io.open_log_append_exact(tmp_path, owner) as handle:
            handle.close()


def test_unused_log_rollback_is_exact(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    owner = artifact_io.reserve_log_exclusive(tmp_path, "logs/run.log")
    artifact_io.rollback_unused_log_exact(tmp_path, owner)
    assert not (tmp_path / "logs/run.log").exists()


def test_owned_node_modes_must_leave_capabilities_usable(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="owner read"):
        artifact_io.publish_bytes_exclusive(
            tmp_path, "out/unreadable", b"payload", mode=0o200
        )
    with pytest.raises(ValueError, match="read and write"):
        artifact_io.reserve_log_exclusive(tmp_path, "out/unwritable.log", mode=0o400)
    with pytest.raises(ValueError, match="owner read and search"):
        artifact_io.create_directory_exclusive(tmp_path, "out/unsearchable", mode=0o600)
    assert list((tmp_path / "out").iterdir()) == []


def test_bundle_stages_everything_and_links_completion_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    events: list[tuple[str, str]] = []
    real_link = artifact_io._link
    real_fsync = artifact_io._fsync

    def record_link(source: str, destination: str, **kwargs: object) -> None:
        events.append(("link", destination))
        real_link(source, destination, **kwargs)

    def record_fsync(descriptor: int) -> None:
        kind = "dir" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    monkeypatch.setattr(artifact_io, "_link", record_link)
    monkeypatch.setattr(artifact_io, "_fsync", record_fsync)
    bundle = artifact_io.publish_bundle_exclusive(
        tmp_path,
        [
            artifact_io.PublishItem("two/b.json", b"b"),
            artifact_io.PublishItem("one/a.json", b"a"),
        ],
        completion=artifact_io.PublishItem("one/complete.json", b"done"),
    )

    link_events = [event for event in events if event[0] == "link"]
    assert link_events == [
        ("link", "a.json"),
        ("link", "b.json"),
        ("link", "complete.json"),
    ]
    first_link = events.index(("link", "a.json"))
    second_link = events.index(("link", "b.json"))
    completion_link = events.index(("link", "complete.json"))
    assert sum(event == ("fsync", "file") for event in events[:first_link]) == 3
    assert ("fsync", "dir") in events[second_link + 1 : completion_link]
    assert [claim.relative_path for claim in bundle.members] == [
        "one/a.json",
        "two/b.json",
    ]
    assert bundle.completion.relative_path == "one/complete.json"
    assert _temporary_paths(tmp_path) == []


def test_bundle_preexisting_member_causes_zero_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out/member").write_bytes(b"foreign")

    def forbidden_temp(_parent_fd: int) -> tuple[str, int]:
        pytest.fail("preflight collision must happen before staging")

    monkeypatch.setattr(artifact_io, "_new_temporary", forbidden_temp)
    with pytest.raises(FileExistsError):
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [artifact_io.PublishItem("out/member", b"owned")],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert (tmp_path / "out/member").read_bytes() == b"foreign"
    assert not (tmp_path / "out/complete").exists()


def test_bundle_late_member_collision_rolls_back_only_owned_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_link = artifact_io._link
    calls = 0

    def collide_on_second(source: str, destination: str, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            (tmp_path / "out/b").write_bytes(b"racer")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(artifact_io, "_link", collide_on_second)
    with pytest.raises(FileExistsError):
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [
                artifact_io.PublishItem("out/a", b"a"),
                artifact_io.PublishItem("out/b", b"b"),
            ],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert not (tmp_path / "out/a").exists()
    assert (tmp_path / "out/b").read_bytes() == b"racer"
    assert not (tmp_path / "out/complete").exists()
    assert _temporary_paths(tmp_path) == []


def test_bundle_completion_collision_rolls_back_all_owned_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_link = artifact_io._link

    def collide_on_completion(source: str, destination: str, **kwargs: object) -> None:
        if destination == "complete":
            (tmp_path / "out/complete").write_bytes(b"foreign completion")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(artifact_io, "_link", collide_on_completion)
    with pytest.raises(FileExistsError):
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [artifact_io.PublishItem("out/a", b"a")],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert not (tmp_path / "out/a").exists()
    assert (tmp_path / "out/complete").read_bytes() == b"foreign completion"
    assert _temporary_paths(tmp_path) == []


def test_bundle_preserves_replaced_member_and_reports_incomplete_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_link = artifact_io._link
    calls = 0

    def replace_then_collide(source: str, destination: str, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            first = tmp_path / "out/a"
            first.rename(tmp_path / "out/original-owned-a")
            first.write_bytes(b"foreign replacement")
            (tmp_path / "out/b").write_bytes(b"racer")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(artifact_io, "_link", replace_then_collide)
    with pytest.raises(artifact_io.RollbackError) as captured:
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [
                artifact_io.PublishItem("out/a", b"a"),
                artifact_io.PublishItem("out/b", b"b"),
            ],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert isinstance(captured.value.primary_error, FileExistsError)
    assert captured.value.rollback_errors
    assert (tmp_path / "out/a").read_bytes() == b"foreign replacement"
    assert (tmp_path / "out/b").read_bytes() == b"racer"
    assert not (tmp_path / "out/complete").exists()


def test_bundle_rollback_continues_after_owned_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_link = artifact_io._link
    real_unlink = artifact_io._unlink

    def collide_on_completion(source: str, destination: str, **kwargs: object) -> None:
        if destination == "complete":
            (tmp_path / "out/complete").write_bytes(b"foreign")
        real_link(source, destination, **kwargs)

    def fail_one_owned_rollback(path: str, **kwargs: object) -> None:
        if path == "b":
            raise OSError(errno.EIO, "injected rollback unlink failure")
        real_unlink(path, **kwargs)

    monkeypatch.setattr(artifact_io, "_link", collide_on_completion)
    monkeypatch.setattr(artifact_io, "_unlink", fail_one_owned_rollback)
    with pytest.raises(artifact_io.RollbackError) as captured:
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [
                artifact_io.PublishItem("out/a", b"a"),
                artifact_io.PublishItem("out/b", b"b"),
            ],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert isinstance(captured.value.primary_error, FileExistsError)
    assert not (tmp_path / "out/a").exists()
    assert (tmp_path / "out/b").read_bytes() == b"b"
    assert (tmp_path / "out/complete").read_bytes() == b"foreign"


def test_bundle_member_changed_during_data_fsync_blocks_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_fsync = artifact_io._fsync
    changed = False

    def mutate_data_member(descriptor: int) -> None:
        nonlocal changed
        real_fsync(descriptor)
        member = tmp_path / "out/a"
        if (
            not changed
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and member.exists()
        ):
            member.write_bytes(b"changed")
            changed = True

    monkeypatch.setattr(artifact_io, "_fsync", mutate_data_member)
    with pytest.raises(artifact_io.RollbackError):
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [
                artifact_io.PublishItem("out/a", b"a"),
                artifact_io.PublishItem("out/b", b"b"),
            ],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert (tmp_path / "out/a").read_bytes() == b"changed"
    assert not (tmp_path / "out/b").exists()
    assert not (tmp_path / "out/complete").exists()


def test_bundle_never_rolls_back_after_completion_becomes_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_link = artifact_io._link
    real_fsync = artifact_io._fsync
    completion_linked = False
    failed = False

    def observe_completion(source: str, destination: str, **kwargs: object) -> None:
        nonlocal completion_linked
        real_link(source, destination, **kwargs)
        if destination == "complete":
            completion_linked = True

    def fail_after_completion(descriptor: int) -> None:
        nonlocal failed
        if (
            completion_linked
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and not failed
        ):
            failed = True
            raise OSError(errno.EIO, "injected completion-directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifact_io, "_link", observe_completion)
    monkeypatch.setattr(artifact_io, "_fsync", fail_after_completion)
    with pytest.raises(artifact_io.CommitIndeterminateError) as captured:
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [artifact_io.PublishItem("out/a", b"a")],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert captured.value.completion_visible is True
    assert (tmp_path / "out/a").read_bytes() == b"a"
    assert (tmp_path / "out/complete").read_bytes() == b"done"
    assert _temporary_paths(tmp_path) == []


@pytest.mark.parametrize("interrupt_at", ("member", "completion"))
def test_bundle_recovers_link_effect_before_async_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_at: str
) -> None:
    (tmp_path / "out").mkdir()
    real_link = artifact_io._link

    def link_then_interrupt(source: str, destination: str, **kwargs: object) -> None:
        real_link(source, destination, **kwargs)
        expected = "member" if interrupt_at == "member" else "complete"
        if destination == expected:
            raise KeyboardInterrupt

    monkeypatch.setattr(artifact_io, "_link", link_then_interrupt)
    if interrupt_at == "member":
        with pytest.raises(KeyboardInterrupt):
            artifact_io.publish_bundle_exclusive(
                tmp_path,
                [artifact_io.PublishItem("out/member", b"member")],
                completion=artifact_io.PublishItem("out/complete", b"done"),
            )
        assert not (tmp_path / "out/member").exists()
        assert not (tmp_path / "out/complete").exists()
    else:
        with pytest.raises(artifact_io.CommitIndeterminateError) as captured:
            artifact_io.publish_bundle_exclusive(
                tmp_path,
                [artifact_io.PublishItem("out/member", b"member")],
                completion=artifact_io.PublishItem("out/complete", b"done"),
            )
        assert isinstance(captured.value.primary_error, KeyboardInterrupt)
        assert (tmp_path / "out/member").read_bytes() == b"member"
        assert (tmp_path / "out/complete").read_bytes() == b"done"
    assert _temporary_paths(tmp_path) == []


def test_single_recovers_temporary_unlink_effect_before_async_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_unlink = artifact_io._unlink
    interrupted = False

    def unlink_then_interrupt(path: str, **kwargs: object) -> None:
        nonlocal interrupted
        real_unlink(path, **kwargs)
        if path.startswith(".artifact-") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(artifact_io, "_unlink", unlink_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/member", b"member")
    assert interrupted
    assert not (tmp_path / "out/member").exists()
    assert _temporary_paths(tmp_path) == []


def test_completion_changed_during_final_fsync_is_commit_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_fsync = artifact_io._fsync
    changed = False

    def mutate_completion(descriptor: int) -> None:
        nonlocal changed
        real_fsync(descriptor)
        completion = tmp_path / "out/complete"
        if (
            not changed
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and completion.exists()
        ):
            completion.write_bytes(b"changed completion")
            changed = True

    monkeypatch.setattr(artifact_io, "_fsync", mutate_completion)
    with pytest.raises(artifact_io.CommitIndeterminateError):
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [artifact_io.PublishItem("out/a", b"a")],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert (tmp_path / "out/a").read_bytes() == b"a"
    assert (tmp_path / "out/complete").read_bytes() == b"changed completion"


def test_completion_disappearance_after_link_keeps_committed_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_fsync = artifact_io._fsync
    removed = False

    def remove_completion(descriptor: int) -> None:
        nonlocal removed
        real_fsync(descriptor)
        completion = tmp_path / "out/complete"
        if (
            not removed
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and completion.exists()
        ):
            completion.unlink()
            removed = True

    monkeypatch.setattr(artifact_io, "_fsync", remove_completion)
    with pytest.raises(artifact_io.CommitIndeterminateError) as captured:
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [artifact_io.PublishItem("out/a", b"a")],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert captured.value.completion_link_succeeded is True
    assert (tmp_path / "out/a").read_bytes() == b"a"
    assert not (tmp_path / "out/complete").exists()


def test_single_and_bundle_final_close_failures_keep_visible_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out").mkdir()
    real_reserve = artifact_io._reserve_stage
    real_close = artifact_io._close
    stage_fds: list[int] = []
    failed_fd: int | None = None

    def capture_stage(
        parent: artifact_io._ParentHandle, item: artifact_io.PublishItem
    ) -> artifact_io._StagedFile:
        stage = real_reserve(parent, item)
        stage_fds.append(stage.fd)
        return stage

    def close_then_report(descriptor: int) -> None:
        nonlocal failed_fd
        real_close(descriptor)
        if descriptor in stage_fds and failed_fd is None:
            failed_fd = descriptor
            raise OSError(errno.EIO, "injected final close report")

    monkeypatch.setattr(artifact_io, "_reserve_stage", capture_stage)
    monkeypatch.setattr(artifact_io, "_close", close_then_report)
    with pytest.raises(artifact_io.PublicationIndeterminateError):
        artifact_io.publish_bytes_exclusive(tmp_path, "out/single", b"single")
    assert (tmp_path / "out/single").read_bytes() == b"single"

    stage_fds.clear()
    failed_fd = None
    with pytest.raises(artifact_io.CommitIndeterminateError):
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [artifact_io.PublishItem("out/member", b"member")],
            completion=artifact_io.PublishItem("out/complete", b"done"),
        )
    assert (tmp_path / "out/member").read_bytes() == b"member"
    assert (tmp_path / "out/complete").read_bytes() == b"done"


def test_bundle_rejects_empty_duplicate_and_completion_aliases(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    completion = artifact_io.PublishItem("out/complete", b"done")
    with pytest.raises(ValueError, match="at least one"):
        artifact_io.publish_bundle_exclusive(tmp_path, [], completion=completion)
    with pytest.raises(ValueError, match="distinct"):
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [
                artifact_io.PublishItem("out/a", b"one"),
                artifact_io.PublishItem("out/a", b"two"),
            ],
            completion=completion,
        )
    with pytest.raises(ValueError, match="distinct"):
        artifact_io.publish_bundle_exclusive(
            tmp_path,
            [artifact_io.PublishItem("out/complete", b"member")],
            completion=completion,
        )
    assert list((tmp_path / "out").iterdir()) == []
