"""Prepare the selection-bound 1,000-update R/S/E scale-up registry.

This is a two-revision, CPU-only publication workflow:

``materialize-configs`` runs at the clean pushed framework revision R4 and
exclusively writes the three selected-design resolved configs.  After those
files alone are committed and pushed as R5, ``freeze-registry`` validates the
entire screen/selection history, config reconstruction, and source closure,
then exclusively writes the registry candidate.  That registry alone is later
committed as R6 before any scale-up run starts.

Neither command inventories GPUs or launches training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.udlm import verify_optimization_screen as screen  # noqa: E402
from scripts.udlm import verify_scale_up_registry as verifier  # noqa: E402


SCREEN_REGISTRY_PATH = (
    "experiments/udlm/protocols/optimization_screen_registry_v2.json"
)
SCREEN_ARTIFACT_PATHS = {
    "scheduler_evidence": "experiments/udlm/screens/scheduler_evidence.json",
    "scheduler_selection": "experiments/udlm/screens/scheduler_selection.json",
    "conditioning_evidence": "experiments/udlm/screens/conditioning_evidence.json",
    "conditioning_selection": "experiments/udlm/screens/conditioning_selection.json",
}
HEX_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class ScaleUpPreparationError(RuntimeError):
    """Raised when a publication boundary is incomplete or ambiguous."""


def _run_git(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _require_clean_pushed_source() -> str:
    if _run_git(["status", "--porcelain=v1", "--untracked-files=all"]).stdout:
        raise ScaleUpPreparationError("worktree must be completely clean")
    revision = _run_git(["rev-parse", "HEAD"]).stdout.strip()
    upstream = _run_git(["rev-parse", "@{upstream}"]).stdout.strip()
    if HEX_REVISION.fullmatch(revision) is None:
        raise ScaleUpPreparationError("HEAD is not a full Git revision")
    if revision != upstream:
        raise ScaleUpPreparationError("HEAD is not pushed exactly to its upstream")
    return revision


def _single_parent(revision: str) -> str:
    fields = _run_git(["rev-list", "--parents", "-n", "1", revision]).stdout.split()
    if len(fields) != 2 or fields[0] != revision:
        raise ScaleUpPreparationError(f"{revision} is not a single-parent commit")
    return fields[1]


def _require_exact_pushed_revision(revision: str) -> None:
    if HEX_REVISION.fullmatch(revision) is None:
        raise ScaleUpPreparationError("expected source revision is invalid")
    head = _run_git(["rev-parse", "HEAD"]).stdout.strip()
    upstream = _run_git(["rev-parse", "@{upstream}"]).stdout.strip()
    if head != revision or upstream != revision:
        raise ScaleUpPreparationError(
            "HEAD or upstream changed during exclusive publication"
        )


def _git_blob_bytes(revision: str, relative_path: str) -> bytes:
    object_name = f"{revision}:{relative_path}"
    try:
        kind = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "cat-file", "-t", object_name],
            check=True,
            capture_output=True,
        ).stdout.strip()
        if kind != b"blob":
            raise ScaleUpPreparationError(f"Git object is not a blob: {object_name}")
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "cat-file", "blob", object_name],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ScaleUpPreparationError(f"missing Git blob: {object_name}") from error


def _git_blob_absent(revision: str, relative_path: str) -> bool:
    if HEX_REVISION.fullmatch(revision) is None:
        raise ScaleUpPreparationError("historical absence revision is invalid")
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or path.as_posix() != relative_path
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ScaleUpPreparationError("historical absence path is invalid")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "ls-tree",
            "-r",
            "--full-tree",
            "--name-only",
            "-z",
            revision,
            "--",
            relative_path,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ScaleUpPreparationError("Git blob-absence query was indeterminate")
    payload = result.stdout
    if payload and not payload.endswith(b"\0"):
        raise ScaleUpPreparationError("Git blob-absence query was truncated")
    if payload:
        entries = tuple(os.fsdecode(item) for item in payload[:-1].split(b"\0"))
        if any(not entry for entry in entries) or len(entries) != len(set(entries)):
            raise ScaleUpPreparationError("Git blob-absence query was malformed")
    return not payload


def _blob_reference_from_git(revision: str, relative_path: str) -> dict[str, Any]:
    payload = _git_blob_bytes(revision, relative_path)
    if not payload:
        raise ScaleUpPreparationError(f"committed blob is empty: {relative_path}")
    live = REPOSITORY_ROOT.joinpath(*PurePosixPath(relative_path).parts)
    expected_live = REPOSITORY_ROOT.resolve(strict=True).joinpath(
        *PurePosixPath(relative_path).parts
    )
    if (
        not live.is_file()
        or live.is_symlink()
        or live.resolve(strict=True) != expected_live
        or live.read_bytes() != payload
    ):
        raise ScaleUpPreparationError(
            f"live file differs from committed blob: {relative_path}"
        )
    return {
        "root": "repository",
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _json_reference_from_git(
    revision: str, relative_path: str, *, expected_schema: int | None = None
) -> dict[str, Any]:
    reference = _blob_reference_from_git(revision, relative_path)
    payload = _git_blob_bytes(revision, relative_path)
    parsed = screen.strict_json_loads(payload, label=relative_path)
    if not isinstance(parsed, Mapping):
        raise ScaleUpPreparationError(f"JSON blob is not an object: {relative_path}")
    schema_version = parsed.get("schema_version")
    if type(schema_version) is not int or schema_version < 1:
        raise ScaleUpPreparationError(f"JSON blob has no positive schema: {relative_path}")
    if expected_schema is not None and schema_version != expected_schema:
        raise ScaleUpPreparationError(f"JSON blob schema differs: {relative_path}")
    return {
        **reference,
        "schema_version": schema_version,
        "canonical_sha256": verifier.canonical_json_sha256(parsed),
    }


def _config_reference_from_git(revision: str, relative_path: str) -> dict[str, Any]:
    reference = _blob_reference_from_git(revision, relative_path)
    payload = _git_blob_bytes(revision, relative_path)
    parsed = screen.strict_json_loads(payload, label=relative_path)
    if not isinstance(parsed, Mapping):
        raise ScaleUpPreparationError(f"config is not a JSON object: {relative_path}")
    try:
        verifier._require_deterministic_json_bytes(
            payload, parsed, label=relative_path
        )
    except verifier.ScaleUpValidationError as error:
        raise ScaleUpPreparationError(str(error)) from error
    return {
        **reference,
        "canonical_sha256": verifier.canonical_json_sha256(parsed),
    }


def _screen_authority(selection_revision: str) -> dict[str, Any]:
    authority = {
        "screen_registry": _json_reference_from_git(
            selection_revision,
            SCREEN_REGISTRY_PATH,
            expected_schema=screen.REGISTRY_SCHEMA_VERSION,
        )
    }
    for name, path in SCREEN_ARTIFACT_PATHS.items():
        authority[name] = _json_reference_from_git(
            selection_revision,
            path,
            expected_schema=(
                screen.EVIDENCE_SCHEMA_VERSION
                if name.endswith("evidence")
                else screen.SELECTION_SCHEMA_VERSION
            ),
        )
    return authority


def _validate_framework_transition(
    *, framework_revision: str
) -> tuple[str, list[str]]:
    selection_revision = _single_parent(framework_revision)
    paths = sorted(verifier.git_changed_paths_loader(selection_revision, framework_revision))
    if not verifier.FRAMEWORK_REQUIRED_PATHS <= set(paths) or any(
        not verifier._framework_path_allowed(path) for path in paths
    ):
        raise ScaleUpPreparationError(
            "R4 must contain the four scale-up scripts and only allowed compatibility/tests"
        )
    verifier._validate_revision_edge(
        parent=selection_revision,
        child=framework_revision,
        expected_paths=frozenset(paths),
        label="selection-to-framework publication",
        git_ancestor_checker=screen.git_ancestor_checker,
        git_sole_parent_checker=screen.git_sole_parent_checker,
        git_pushed_checker=screen.git_pushed_checker,
        git_diff_checker=screen.git_diff_checker,
        changed_paths_loader=verifier.git_changed_paths_loader,
    )
    for required in verifier.FRAMEWORK_REQUIRED_PATHS:
        if not _git_blob_absent(selection_revision, required):
            raise ScaleUpPreparationError(
                f"scale-up framework path already existed at R3: {required}"
            )
    return selection_revision, paths


def _resolve_selections(
    *, selection_revision: str
) -> tuple[
    dict[str, Any],
    screen.ValidatedRegistry,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    authority = _screen_authority(selection_revision)
    publication_stub = {"selection_revision": selection_revision}
    try:
        registry, scheduler, conditioning = verifier._validate_screen_authority(
            authority,
            publication=publication_stub,
            loader=screen.local_blob_loader,
            git_blob_loader=screen.git_blob_loader,
            git_ancestor_checker=screen.git_ancestor_checker,
            git_sole_parent_checker=screen.git_sole_parent_checker,
            git_tree_paths_loader=screen.git_tree_paths_loader,
            git_pushed_checker=screen.git_pushed_checker,
            git_diff_checker=screen.git_diff_checker,
            changed_paths_loader=verifier.git_changed_paths_loader,
        )
    except Exception as error:
        raise ScaleUpPreparationError("screen selections failed recomputation") from error
    selected = {
        "scheduler_arm_id": scheduler["selected_arm_id"],
        "conditioning_arm_id": conditioning["selected_arm_id"],
    }
    scheduler_arm = screen._arm(
        screen._stage(registry, "scheduler"), str(selected["scheduler_arm_id"])
    )
    conditioner_arm = screen._arm(
        screen._stage(registry, "conditioning"),
        str(selected["conditioning_arm_id"]),
    )
    entry = screen._registered_config(
        conditioner_arm, scheduler_arm_id=str(selected["scheduler_arm_id"])
    )
    config = entry.get("parsed_config")
    if not isinstance(config, Mapping):
        raise ScaleUpPreparationError("selected screen config is unavailable")
    selected.update(
        {
            "scheduler": scheduler_arm["scheduler"],
            "conditioner": conditioner_arm["conditioner"],
            "screen_config": entry["config"],
        }
    )
    return authority, registry, scheduler, conditioning, config


def _config_directory(gpu_count: int) -> Path:
    return REPOSITORY_ROOT / verifier.CONFIG_DIRECTORY_TEMPLATE.format(
        gpu_count=gpu_count
    )


def _config_paths(gpu_count: int) -> dict[str, str]:
    directory = PurePosixPath(
        verifier.CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count)
    )
    return {
        variant: (directory / verifier.CONFIG_FILENAMES[variant]).as_posix()
        for variant in verifier.EXPECTED_VARIANTS
    }


def _member_records(
    *, gpu_count: int, config_revision: str, config_references: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selected_suffix = config_revision[:12]
    records: list[dict[str, Any]] = []
    for position, (variant, slug) in enumerate(
        zip(verifier.EXPECTED_VARIANTS, verifier.EXPECTED_SLUGS, strict=True)
    ):
        run_name = f"scaleup-w{gpu_count}-{slug}-{selected_suffix}"
        records.append(
            {
                "position": position,
                "slug": slug,
                "training_variant": variant,
                "prior_variant": verifier.EXPECTED_PRIORS[variant],
                "comparison_role": verifier.EXPECTED_ROLES[variant],
                "run_name": run_name,
                "output_directory": f"output/udlm/{run_name}",
                "config": dict(config_references[variant]),
            }
        )
    return records


def _candidate_documents(
    *,
    selected_config: Mapping[str, Any],
    gpu_count: int,
    global_batch_size: int,
    micro_batch_size: int,
    num_workers: int,
    config_revision_for_names: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    paths = _config_paths(gpu_count)
    placeholder_references = {
        variant: {
            "root": "repository",
            "relative_path": path,
            "sha256": "0" * 64,
            "size_bytes": 1,
            "canonical_sha256": "0" * 64,
        }
        for variant, path in paths.items()
    }
    members = _member_records(
        gpu_count=gpu_count,
        config_revision=config_revision_for_names,
        config_references=placeholder_references,
    )
    documents: dict[str, dict[str, Any]] = {}
    for member in members:
        variant = str(member["training_variant"])
        documents[variant] = verifier.derive_scale_up_config(
            selected_config,
            training_variant=variant,
            gpu_count=gpu_count,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            num_workers=num_workers,
            output_directory=str(member["output_directory"]),
        )
    hashes = {verifier.matched_config_sha256(config) for config in documents.values()}
    if len(hashes) != 1:
        raise ScaleUpPreparationError("derived R/S/E configs are not matched")
    return documents, members


def _publish_bytes_exclusive(path: Path, payload: bytes, *, label: str) -> None:
    """Publish below direct repository directories without following symlinks."""

    repository = Path(os.path.abspath(os.fspath(REPOSITORY_ROOT)))
    normalized = Path(os.path.abspath(os.fspath(path)))
    if (
        normalized != path
        or normalized == repository
        or not normalized.is_relative_to(repository)
    ):
        raise ScaleUpPreparationError(f"{label} output escapes the repository")
    relative = normalized.relative_to(repository)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd: int | None = None
    temporary_name: str | None = None
    try:
        try:
            directory_fd = os.open(repository, directory_flags)
        except OSError as error:
            raise ScaleUpPreparationError(
                "repository root must be a direct real directory"
            ) from error
        for component in relative.parent.parts:
            try:
                child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                try:
                    child_fd = os.open(
                        component, directory_flags, dir_fd=directory_fd
                    )
                except OSError as error:
                    raise ScaleUpPreparationError(
                        f"{label} parent must be a direct real directory"
                    ) from error
            except OSError as error:
                raise ScaleUpPreparationError(
                    f"{label} parent must be a direct real directory"
                ) from error
            os.close(directory_fd)
            directory_fd = child_fd
        try:
            os.stat(relative.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to replace {label}: {normalized}")

        descriptor: int | None = None
        for _attempt in range(100):
            candidate = f".{relative.name}.{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o644,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:  # pragma: no cover
            raise ScaleUpPreparationError(
                f"could not reserve a temporary {label} publication"
            )
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                relative.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace {label}: {normalized}"
            ) from error
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_name = None
        os.fsync(directory_fd)
        published = os.stat(
            relative.name, dir_fd=directory_fd, follow_symlinks=False
        )
        live = os.stat(normalized, follow_symlinks=False)
        expected_parent = repository.resolve(strict=True).joinpath(
            *relative.parent.parts
        )
        if (
            not stat.S_ISREG(published.st_mode)
            or (published.st_dev, published.st_ino) != (live.st_dev, live.st_ino)
            or normalized.parent.resolve(strict=True) != expected_parent
        ):
            raise ScaleUpPreparationError(
                f"{label} publication path changed during exclusive write"
            )
    finally:
        if directory_fd is not None:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            os.close(directory_fd)


def _require_exact_untracked(paths: Sequence[str]) -> None:
    status = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    ).stdout
    expected = b"".join(f"?? {path}\0".encode() for path in sorted(paths))
    # Git returns lexical order for these non-pathspec status entries.
    if status != expected:
        raise ScaleUpPreparationError(
            "publication is not the exact declared untracked-file set"
        )


def _validate_fresh_namespaces(gpu_count: int, *, revisions: Sequence[str]) -> None:
    selected_directory = verifier.CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count)
    for revision in revisions:
        if screen.git_tree_paths_loader(revision, PurePosixPath(selected_directory)):
            raise ScaleUpPreparationError(
                "selected scale-up config family already exists in publication history"
            )
    for candidate in range(1, verifier.MAX_GPU_COUNT + 1):
        directory = verifier.CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=candidate)
        if candidate != gpu_count and any(
            screen.git_tree_paths_loader(revision, PurePosixPath(directory))
            for revision in revisions
        ):
            raise ScaleUpPreparationError(
                "an unselected GPU-count config family exists in publication history"
            )
        live = REPOSITORY_ROOT / directory
        if os.path.lexists(live):
            raise FileExistsError(f"scale-up config directory already exists: {live}")
    for candidate in range(1, verifier.MAX_GPU_COUNT + 1):
        registry_relative = verifier.REGISTRY_RELATIVE_PATH_TEMPLATE.format(
            gpu_count=candidate
        )
        if any(
            not _git_blob_absent(revision, registry_relative)
            for revision in revisions
        ):
            raise ScaleUpPreparationError(
                "a scale-up registry family already exists in publication history"
            )
        registry_path = REPOSITORY_ROOT / registry_relative
        if os.path.lexists(registry_path):
            raise FileExistsError(f"scale-up registry already exists: {registry_path}")


def materialize_configs(gpu_count: int) -> dict[str, Any]:
    """Exclusively publish the three R5 config candidates at clean pushed R4."""

    gpu_count = verifier.validate_gpu_count(gpu_count)
    micro_batch_size = 2
    global_batch_size = verifier.default_global_batch_size(
        gpu_count, micro_batch_size=micro_batch_size
    )
    num_workers = 1
    verifier.exact_accumulation_steps(
        global_batch_size, micro_batch_size, gpu_count
    )
    framework_revision = _require_clean_pushed_source()
    selection_revision, framework_paths = _validate_framework_transition(
        framework_revision=framework_revision
    )
    _validate_fresh_namespaces(
        gpu_count, revisions=(selection_revision, framework_revision)
    )
    _authority, _registry, scheduler, conditioning, selected_config = (
        _resolve_selections(selection_revision=selection_revision)
    )
    documents, members = _candidate_documents(
        selected_config=selected_config,
        gpu_count=gpu_count,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        num_workers=num_workers,
        config_revision_for_names=framework_revision,
    )
    paths = _config_paths(gpu_count)
    payloads = {
        variant: verifier.json_bytes(document)
        for variant, document in documents.items()
    }
    if _require_clean_pushed_source() != framework_revision:
        raise ScaleUpPreparationError("source changed while configs were composed")
    for variant in verifier.EXPECTED_VARIANTS:
        _publish_bytes_exclusive(
            REPOSITORY_ROOT / paths[variant],
            payloads[variant],
            label=f"{variant} scale-up config",
        )
    _require_exact_untracked(tuple(paths.values()))
    _require_exact_pushed_revision(framework_revision)
    for variant, relative_path in paths.items():
        retained = screen._read_stable_file(
            REPOSITORY_ROOT / relative_path, retain=True
        )
        if not isinstance(retained, bytes) or retained != payloads[variant]:
            raise ScaleUpPreparationError(
                f"published scale-up config bytes changed: {relative_path}"
            )
    _require_exact_untracked(tuple(paths.values()))
    _require_exact_pushed_revision(framework_revision)
    return {
        "status": "three_scale_up_configs_materialized_no_gpu_operation",
        "selection_revision": selection_revision,
        "framework_revision": framework_revision,
        "framework_paths": framework_paths,
        "gpu_count": gpu_count,
        "global_batch_size": global_batch_size,
        "micro_batch_size_per_process": micro_batch_size,
        "accumulate_grad_batches": verifier.exact_accumulation_steps(
            global_batch_size, micro_batch_size, gpu_count
        ),
        "selected_scheduler_arm_id": scheduler["selected_arm_id"],
        "selected_conditioning_arm_id": conditioning["selected_arm_id"],
        "members": [
            {
                "position": member["position"],
                "training_variant": member["training_variant"],
                "relative_path": paths[str(member["training_variant"])],
                "sha256": hashlib.sha256(
                    payloads[str(member["training_variant"])]
                ).hexdigest(),
                "canonical_sha256": verifier.canonical_json_sha256(
                    documents[str(member["training_variant"])]
                ),
            }
            for member in members
        ],
        "gpu_probe_performed": False,
        "training_launched": False,
    }


def _load_committed_configs(
    *,
    config_revision: str,
    gpu_count: int,
    selected_config: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    int,
]:
    paths = _config_paths(gpu_count)
    directory = _config_directory(gpu_count)
    if not directory.is_dir() or directory.is_symlink():
        raise ScaleUpPreparationError("committed config directory is unavailable")
    if {entry.name for entry in directory.iterdir()} != set(
        verifier.CONFIG_FILENAMES.values()
    ):
        raise ScaleUpPreparationError("config directory is not the exact three-file set")
    references: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for variant, path in paths.items():
        references[variant] = _config_reference_from_git(config_revision, path)
        payload = _git_blob_bytes(config_revision, path)
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ScaleUpPreparationError(f"config is invalid JSON: {path}") from error
        if not isinstance(document, dict):
            raise ScaleUpPreparationError(f"config is not an object: {path}")
        documents[variant] = document
    r_config = documents["udlm"]
    try:
        gpu_count_from_config = r_config["trainer"]["devices"]
        global_batch = r_config["loader"]["global_batch_size"]
        micro_batch = r_config["loader"]["batch_size"]
        num_workers = r_config["loader"]["num_workers"]
    except (KeyError, TypeError) as error:
        raise ScaleUpPreparationError("committed R config lacks batch controls") from error
    if type(gpu_count_from_config) is not int or gpu_count_from_config != gpu_count:
        raise ScaleUpPreparationError("committed config world size differs")
    verifier.exact_accumulation_steps(global_batch, micro_batch, gpu_count)
    expected_documents, placeholder_members = _candidate_documents(
        selected_config=selected_config,
        gpu_count=gpu_count,
        global_batch_size=global_batch,
        micro_batch_size=micro_batch,
        num_workers=num_workers,
        config_revision_for_names=_single_parent(config_revision),
    )
    if not screen._exact_json_equal(documents, expected_documents):
        raise ScaleUpPreparationError("committed configs are not reproducible")
    members = _member_records(
        gpu_count=gpu_count,
        config_revision=_single_parent(config_revision),
        config_references=references,
    )
    # The placeholder records determine exactly the same names/outputs.
    if any(
        left["output_directory"] != right["output_directory"]
        for left, right in zip(members, placeholder_members, strict=True)
    ):
        raise AssertionError("member name reconstruction diverged")
    return (
        references,
        documents,
        members,
        int(global_batch),
        int(micro_batch),
        int(num_workers),
    )


def _assert_config_transition(
    *,
    framework_revision: str,
    config_revision: str,
    gpu_count: int,
) -> list[str]:
    expected = sorted(_config_paths(gpu_count).values())
    verifier._validate_revision_edge(
        parent=framework_revision,
        child=config_revision,
        expected_paths=frozenset(expected),
        label="framework-to-config publication",
        git_ancestor_checker=screen.git_ancestor_checker,
        git_sole_parent_checker=screen.git_sole_parent_checker,
        git_pushed_checker=screen.git_pushed_checker,
        git_diff_checker=screen.git_diff_checker,
        changed_paths_loader=verifier.git_changed_paths_loader,
    )
    for path in expected:
        if not _git_blob_absent(framework_revision, path):
            raise ScaleUpPreparationError(f"config existed before R5: {path}")
    return expected


def _build_registry_document(
    *,
    selection_revision: str,
    framework_revision: str,
    config_revision: str,
    framework_paths: Sequence[str],
    config_paths: Sequence[str],
    screen_authority: Mapping[str, Any],
    screen_registry: screen.ValidatedRegistry,
    scheduler_selection: Mapping[str, Any],
    conditioning_selection: Mapping[str, Any],
    selected_config: Mapping[str, Any],
    gpu_count: int,
    global_batch_size: int,
    micro_batch_size: int,
    num_workers: int,
    documents: Mapping[str, Mapping[str, Any]],
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scheduler_arm = screen._arm(
        screen._stage(screen_registry, "scheduler"),
        str(scheduler_selection["selected_arm_id"]),
    )
    conditioner_arm = screen._arm(
        screen._stage(screen_registry, "conditioning"),
        str(conditioning_selection["selected_arm_id"]),
    )
    selected_entry = screen._registered_config(
        conditioner_arm,
        scheduler_arm_id=str(scheduler_selection["selected_arm_id"]),
    )
    common_hashes = {
        verifier.matched_config_sha256(config) for config in documents.values()
    }
    if len(common_hashes) != 1:
        raise ScaleUpPreparationError("committed configs lack one matched digest")
    source_references = [
        _blob_reference_from_git(config_revision, path)
        for path in verifier.SOURCE_PATHS
    ]
    candidate = {
        "schema_version": verifier.REGISTRY_SCHEMA_VERSION,
        "registry_id": verifier.EXPECTED_REGISTRY_ID,
        "status": verifier.EXPECTED_STATUS,
        "claim_scope": verifier.EXPECTED_CLAIM_SCOPE,
        "firewall": {
            "generation_metrics_allowed": False,
            "superiority_evidence_eligible": False,
            "candidate_lock_eligible": False,
            "failed_or_missing_receipt_policy": "incomplete_no_panel",
            "unregistered_attempts_allowed": False,
        },
        "publication": {
            "selection_revision": selection_revision,
            "framework_revision": framework_revision,
            "config_revision": config_revision,
            "framework_paths": list(framework_paths),
            "config_paths": list(config_paths),
        },
        "source": {
            "revision": config_revision,
            "clean": True,
            "pushed": True,
            "blobs": source_references,
        },
        "screen_authority": dict(screen_authority),
        "selected_design": {
            "scheduler_arm_id": scheduler_selection["selected_arm_id"],
            "conditioning_arm_id": conditioning_selection["selected_arm_id"],
            "scheduler": scheduler_arm["scheduler"],
            "conditioner": conditioner_arm["conditioner"],
            "screen_config": selected_entry["config"],
        },
        "common_training": {
            "optimizer_updates": verifier.EXPECTED_MAX_STEPS,
            "training_seed": verifier.EXPECTED_TRAINING_SEED,
            "gpu_count": gpu_count,
            "global_batch_size": global_batch_size,
            "micro_batch_size_per_process": micro_batch_size,
            "accumulate_grad_batches": verifier.exact_accumulation_steps(
                global_batch_size, micro_batch_size, gpu_count
            ),
            "effective_global_batch_size": global_batch_size,
            "num_workers": num_workers,
            "exclude_special_tokens": False,
            "reseed_after_model_initialization": True,
            "initialization": dict(
                screen_registry.data["common_training"]["initialization"]
            ),
            "gpu_safety_policy": {
                "max_utilization_percent": verifier.MAX_SAFE_UTILIZATION_PERCENT,
                "utilization_comparison": "strictly_less_than",
                "min_free_memory_mib": verifier.MIN_SAFE_FREE_MEMORY_MIB,
                "active_compute_processes_allowed": True,
                "compute_mode_prohibited_allowed": False,
            },
            "common_resolved_config_sha256": next(iter(common_hashes)),
            "artifact_schema_versions": {
                "launch_manifest": 2,
                "runtime_config": 2,
                "training_summary": 5,
                "exit_receipt": 5,
                "predecessor_receipt_binding": 1,
                "matched_panel": 2,
            },
        },
        "members": [dict(member) for member in members],
    }
    plain = verifier._plain_json(candidate)
    if not isinstance(plain, dict):  # pragma: no cover - fixed object above
        raise AssertionError("scale-up registry candidate is not an object")
    return plain


def freeze_registry(gpu_count: int) -> dict[str, Any]:
    """Exclusively publish the verifier-approved registry candidate at pushed R5."""

    gpu_count = verifier.validate_gpu_count(gpu_count)
    config_revision = _require_clean_pushed_source()
    framework_revision = _single_parent(config_revision)
    selection_revision, framework_paths = _validate_framework_transition(
        framework_revision=framework_revision
    )
    config_paths = _assert_config_transition(
        framework_revision=framework_revision,
        config_revision=config_revision,
        gpu_count=gpu_count,
    )
    authority, screen_registry, scheduler, conditioning, selected_config = (
        _resolve_selections(selection_revision=selection_revision)
    )
    (
        _references,
        documents,
        members,
        global_batch,
        micro_batch,
        num_workers,
    ) = _load_committed_configs(
        config_revision=config_revision,
        gpu_count=gpu_count,
        selected_config=selected_config,
    )
    for member in members:
        output = REPOSITORY_ROOT / str(member["output_directory"])
        if os.path.lexists(output):
            raise FileExistsError(f"registered output already exists: {output}")
    registry_relative = verifier.REGISTRY_RELATIVE_PATH_TEMPLATE.format(
        gpu_count=gpu_count
    )
    registry_path = REPOSITORY_ROOT / registry_relative
    if not _git_blob_absent(config_revision, registry_relative) or os.path.lexists(
        registry_path
    ):
        raise FileExistsError("scale-up registry path is not fresh at R5")
    candidate = _build_registry_document(
        selection_revision=selection_revision,
        framework_revision=framework_revision,
        config_revision=config_revision,
        framework_paths=framework_paths,
        config_paths=config_paths,
        screen_authority=authority,
        screen_registry=screen_registry,
        scheduler_selection=scheduler,
        conditioning_selection=conditioning,
        selected_config=selected_config,
        gpu_count=gpu_count,
        global_batch_size=global_batch,
        micro_batch_size=micro_batch,
        num_workers=num_workers,
        documents=documents,
        members=members,
    )
    del selected_config
    payload = verifier.json_bytes(candidate)
    raw_sha = hashlib.sha256(payload).hexdigest()
    canonical_sha = verifier.canonical_json_sha256(candidate)
    verifier.load_validated_registry(
        payload,
        relative_path=registry_relative,
        expected_raw_sha256=raw_sha,
        expected_canonical_sha256=canonical_sha,
    )
    if _require_clean_pushed_source() != config_revision:
        raise ScaleUpPreparationError("source changed while registry was validated")
    _publish_bytes_exclusive(registry_path, payload, label="scale-up registry")
    _require_exact_untracked((registry_relative,))
    _require_exact_pushed_revision(config_revision)
    retained_registry = screen._read_stable_file(registry_path, retain=True)
    if not isinstance(retained_registry, bytes) or retained_registry != payload:
        raise ScaleUpPreparationError("published registry bytes changed")
    _require_exact_untracked((registry_relative,))
    _require_exact_pushed_revision(config_revision)
    return {
        "status": "selection_bound_registry_frozen_as_only_r6_candidate",
        "selection_revision": selection_revision,
        "framework_revision": framework_revision,
        "config_revision": config_revision,
        "gpu_count": gpu_count,
        "registry_relative_path": registry_relative,
        "registry_sha256": raw_sha,
        "registry_canonical_sha256": canonical_sha,
        "registry_size_bytes": len(payload),
        "selected_scheduler_arm_id": scheduler["selected_arm_id"],
        "selected_conditioning_arm_id": conditioning["selected_arm_id"],
        "config_count": len(members),
        "gpu_probe_performed": False,
        "training_launched": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize-configs")
    materialize.add_argument("--gpu-count", type=int, choices=range(1, 5), required=True)
    freeze = subparsers.add_parser("freeze-registry")
    freeze.add_argument("--gpu-count", type=int, choices=range(1, 5), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "materialize-configs":
        result = materialize_configs(args.gpu_count)
    else:
        result = freeze_registry(args.gpu_count)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
