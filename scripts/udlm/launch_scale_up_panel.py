"""Launch the next member of a frozen selection-bound R/S/E scale-up panel.

Only registry pins and ``--dry-run`` are accepted.  The registry fixes the
selected scheduler/conditioner, three resolved configs, world size, exact
global-batch arithmetic, 1,000-update budget, seed, warm start, output names,
and the strict ``utilization < 10%`` GPU policy.  Each invocation validates R6
publication and the complete predecessor chain, then previews or launches the
first missing R -> S -> E member.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.udlm import launch_optimization_screen as screen_launcher  # noqa: E402
from scripts.udlm import launch_train_pilot as pilot  # noqa: E402
from scripts.udlm import prepare_scale_up_registry as preparer  # noqa: E402
from scripts.udlm import verify_optimization_screen as screen  # noqa: E402
from scripts.udlm import verify_scale_up_registry as verifier  # noqa: E402


# Preserve the producer's schema-2 purpose string.  The explicit
# ``selection_bound_scale_up`` object distinguishes this registered workflow.
PURPOSE = "bounded UDLM training pilot"
RECEIPT_BASENAME = "pilot_exit_status.json"


@dataclass(frozen=True)
class ScaleUpLaunchPlan:
    """All immutable, CPU-resolved inputs for the next scale-up member."""

    registry: verifier.ValidatedScaleUpRegistry
    source_revision: str
    position: int
    member: Mapping[str, Any]
    run_dir: Path
    log_path: Path
    session_name: str
    checkpoint: Path
    checkpoint_sha256: str
    command: list[str]
    resolved_config: dict[str, object]
    resolved_config_sha256: str
    argv_sha256: str
    matched_panel_spec: dict[str, object]
    matched_panel_spec_sha256: str
    selection_bound_scale_up: dict[str, Any]
    predecessor_receipt: Path | None
    predecessor_receipt_binding: dict[str, object]

    @property
    def training_variant(self) -> str:
        return str(self.member["training_variant"])


def _stable_bytes(path: Path, *, label: str) -> bytes:
    try:
        loaded = screen._read_stable_file(path, retain=True)
    except Exception as error:
        raise RuntimeError(f"cannot read stable {label}: {path}") from error
    if not isinstance(loaded, bytes):
        raise RuntimeError(f"{label} bytes were not retained")
    return loaded


def _open_direct_repository_parent(
    path: Path, *, label: str
) -> tuple[int, Path, str]:
    """Open an existing direct parent by descriptor without following symlinks."""

    repository = Path(os.path.abspath(os.fspath(REPOSITORY_ROOT)))
    normalized = Path(os.path.abspath(os.fspath(path)))
    if (
        normalized != path
        or normalized == repository
        or not normalized.is_relative_to(repository)
    ):
        raise RuntimeError(f"{label} escapes the repository")
    relative = normalized.relative_to(repository)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(repository, flags)
    except OSError as error:
        raise RuntimeError("repository root must be a direct real directory") from error
    try:
        for component in relative.parent.parts:
            try:
                child_fd = os.open(component, flags, dir_fd=directory_fd)
            except OSError as error:
                raise RuntimeError(
                    f"{label} parent must be an existing direct real directory"
                ) from error
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd, normalized, relative.name
    except BaseException:
        os.close(directory_fd)
        raise


def _create_direct_directory_exclusive(path: Path, *, label: str) -> None:
    directory_fd, normalized, name = _open_direct_repository_parent(
        path, label=label
    )
    try:
        try:
            os.mkdir(name, mode=0o755, dir_fd=directory_fd)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace {label}: {normalized}") from error
        state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        live = os.stat(normalized, follow_symlinks=False)
        if (
            not stat.S_ISDIR(state.st_mode)
            or stat.S_ISLNK(state.st_mode)
            or (state.st_dev, state.st_ino) != (live.st_dev, live.st_ino)
        ):
            raise RuntimeError(f"{label} path changed during reservation")
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _safe_publish_bytes_exclusive(path: Path, payload: bytes, *, label: str) -> str:
    if Path(os.path.abspath(os.fspath(preparer.REPOSITORY_ROOT))) != Path(
        os.path.abspath(os.fspath(REPOSITORY_ROOT))
    ):
        raise AssertionError("scale-up publisher repository roots diverged")
    preparer._publish_bytes_exclusive(path, payload, label=label)
    return hashlib.sha256(payload).hexdigest()


def _acquire_scale_up_training_job_lock(
    *, source_revision: str, run_name: str, training_variant: str
) -> tuple[Path, dict[str, object], str]:
    source_revision = verifier._revision(source_revision, "lock source revision")
    if pilot.RUN_NAME_PATTERN.fullmatch(run_name) is None:
        raise ValueError("run_name is invalid")
    training_variant = pilot.validate_training_variant(training_variant)
    lock_path = REPOSITORY_ROOT / "output" / "udlm" / ".single_training_job.lock"
    lock_record = {
        "schema_version": pilot.TRAINING_JOB_LOCK_SCHEMA_VERSION,
        "status": "held",
        "purpose": "enforce_one_R_S_E_pilot_training_job_at_a_time",
        "source_revision": source_revision,
        "run_name": run_name,
        "training_variant": training_variant,
        "owner_token": secrets.token_hex(32),
        "launcher_pid_at_acquisition": os.getpid(),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_process_exit_does_not_make_lock_stale": True,
        "stale_lock_policy": "fail_closed_and_require_manual_review",
        "release_policy": (
            "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure"
        ),
    }
    payload = (
        json.dumps(lock_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        digest = _safe_publish_bytes_exclusive(
            lock_path, payload, label="single scale-up training-job lock"
        )
    except FileExistsError as error:
        raise RuntimeError(
            "another or stale training-job lock exists; fail closed and review "
            f"it manually before launch: {lock_path}"
        ) from error
    return lock_path, lock_record, digest


def _release_exact_scale_up_training_job_lock(
    path: Path, *, expected_sha256: str
) -> None:
    expected_sha256 = verifier._sha256(expected_sha256, "expected lock digest")
    directory_fd, _normalized, name = _open_direct_repository_parent(
        path, label="training-job lock"
    )
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError("training-job lock is not a single-link regular file")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        if pilot._stable_stat_identity(os.fstat(descriptor)) != (
            pilot._stable_stat_identity(before)
        ):
            raise RuntimeError("training-job lock changed before exact release")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            pilot._stable_stat_identity(after) != pilot._stable_stat_identity(before)
            or pilot._stable_stat_identity(current)
            != pilot._stable_stat_identity(before)
            or digest.hexdigest() != expected_sha256
        ):
            raise RuntimeError("refusing to release a changed training-job lock")
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def load_registry(
    path: Path, *, expected_raw_sha256: str, expected_canonical_sha256: str
) -> verifier.ValidatedScaleUpRegistry:
    path = path.resolve(strict=True)
    repository = REPOSITORY_ROOT.resolve(strict=True)
    try:
        relative = path.relative_to(repository).as_posix()
    except ValueError as error:
        raise verifier.ScaleUpValidationError(
            "scale-up registry must be inside the repository"
        ) from error
    validated = verifier.load_validated_registry(
        _stable_bytes(path, label="scale-up registry"),
        relative_path=relative,
        expected_raw_sha256=expected_raw_sha256,
        expected_canonical_sha256=expected_canonical_sha256,
    )
    gpu_count = validated.data["common_training"]["gpu_count"]
    expected_path = verifier.REGISTRY_RELATIVE_PATH_TEMPLATE.format(
        gpu_count=gpu_count
    )
    if relative != expected_path:
        raise verifier.ScaleUpValidationError(
            "scale-up registry path differs from its frozen world size"
        )
    return validated


def require_registry_publication(
    registry: verifier.ValidatedScaleUpRegistry, *, current_revision: str
) -> None:
    """Require HEAD to be exact pushed R6: the registry-only child of R5."""

    current_revision = verifier._revision(current_revision, "launch source revision")
    config_revision = str(registry.data["publication"]["config_revision"])
    path = registry.relative_path.as_posix()
    verifier._validate_revision_edge(
        parent=config_revision,
        child=current_revision,
        expected_paths=frozenset({path}),
        label="config-to-registry publication",
        git_ancestor_checker=screen.git_ancestor_checker,
        git_sole_parent_checker=screen.git_sole_parent_checker,
        git_pushed_checker=screen.git_pushed_checker,
        git_diff_checker=screen.git_diff_checker,
        changed_paths_loader=verifier.git_changed_paths_loader,
    )
    verifier._git_blob_absent(
        config_revision,
        path,
        git_tree_paths_loader=screen.git_tree_paths_loader,
        label="scale-up registry",
    )
    committed = screen.git_blob_loader(current_revision, PurePosixPath(path))
    if (
        len(committed) != registry.raw_size_bytes
        or hashlib.sha256(committed).hexdigest() != registry.raw_sha256
    ):
        raise verifier.ScaleUpValidationError(
            "R6 registry blob differs from the launched registry"
        )


def _registered_output_path(relative_path: str) -> Path:
    text = verifier._relative_path(relative_path, "registered output directory")
    candidate = REPOSITORY_ROOT.joinpath(*PurePosixPath(text).parts)
    resolved = candidate.resolve(strict=False)
    root = REPOSITORY_ROOT.resolve(strict=True)
    if resolved == root or root not in resolved.parents:
        raise verifier.ScaleUpValidationError(
            "registered output directory escapes the repository"
        )
    return candidate


def run_paths(
    registry: verifier.ValidatedScaleUpRegistry,
) -> tuple[tuple[Mapping[str, Any], Path, Path], ...]:
    result = []
    for member in registry.data["members"]:
        run_dir = _registered_output_path(str(member["output_directory"]))
        result.append((member, run_dir, run_dir / RECEIPT_BASENAME))
    return tuple(result)


def _require_direct_directory(path: Path, *, label: str) -> None:
    try:
        state = path.stat(follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"cannot inspect {label}: {path}") from error
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise RuntimeError(f"{label} must be a direct real directory: {path}")


def _require_successful_receipt(path: Path, *, label: str) -> None:
    if not os.path.lexists(path):
        raise RuntimeError(f"{label} exists without a completion receipt")
    snapshot, payload = pilot.stable_repository_artifact_snapshot(
        path, suffix=".json", label=f"{label} completion receipt"
    )
    del snapshot
    receipt = pilot.strict_json_loads(payload, label=f"{label} completion receipt")
    if not isinstance(receipt, Mapping):
        raise RuntimeError(f"{label} completion receipt is not an object")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != pilot.PILOT_EXIT_STATUS_SCHEMA_VERSION
        or receipt.get("status") != "completed"
        or receipt.get("overall_status") != "completed"
        or type(receipt.get("process_exit_status")) is not int
        or receipt.get("process_exit_status") != 0
    ):
        raise RuntimeError(
            f"{label} receipt is failed or malformed; preserve the namespace and "
            "restart the full panel at a new clean pushed publication chain"
        )


def completed_prefix(
    paths: tuple[tuple[Mapping[str, Any], Path, Path], ...],
    *,
    registry: verifier.ValidatedScaleUpRegistry | None = None,
    source_revision: str | None = None,
) -> int:
    """Count successful leading members and reject every R/S/E gap."""

    completed = 0
    missing_seen = False
    for position, (member, run_dir, receipt) in enumerate(paths):
        exists = os.path.lexists(run_dir)
        log_path = REPOSITORY_ROOT / "output" / "logs" / f"{run_dir.name}.log"
        if not exists:
            if os.path.lexists(log_path):
                raise RuntimeError(
                    "scale-up log exists without its registered run directory: "
                    f"{log_path}"
                )
            missing_seen = True
            continue
        if missing_seen:
            raise RuntimeError(
                "scale-up run directories exist out of R -> S -> E order: "
                f"{run_dir}"
            )
        label = f"{member['slug'].upper()} scale-up run"
        _require_direct_directory(run_dir, label=label)
        _require_successful_receipt(receipt, label=label)
        if registry is not None:
            manifest_path = run_dir / "launch_manifest.json"
            manifest_payload = _stable_bytes(
                manifest_path, label=f"{label} launch manifest"
            )
            manifest = pilot.strict_json_loads(
                manifest_payload, label=f"{label} launch manifest"
            )
            if not isinstance(manifest, Mapping):
                raise RuntimeError(f"{label} launch manifest is not an object")
            verifier.validate_manifest_binding(
                manifest.get("selection_bound_scale_up"),
                registry=registry,
                position=position,
            )
            exact = {
                "run_name": member["run_name"],
                "training_variant": member["training_variant"],
                "resolved_training_config_sha256": member["config"][
                    "canonical_sha256"
                ],
                "matched_panel_variant_position": position,
            }
            if source_revision is not None:
                exact.update(
                    {
                        "git_sha": source_revision,
                        "source_revision_before_final_gpu_probe": source_revision,
                    }
                )
            for field, expected in exact.items():
                if manifest.get(field) != expected:
                    raise RuntimeError(
                        f"{label} launch manifest {field} differs from registry"
                    )
        completed += 1
    return completed


def _read_member_config(member: Mapping[str, Any]) -> dict[str, object]:
    reference = member["config"]
    path = REPOSITORY_ROOT.joinpath(
        *PurePosixPath(str(reference["relative_path"])).parts
    )
    payload = _stable_bytes(path, label="registered member config")
    if (
        len(payload) != reference["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != reference["sha256"]
    ):
        raise verifier.ScaleUpValidationError("registered member config bytes changed")
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise verifier.ScaleUpValidationError(
            "registered member config is not JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise verifier.ScaleUpValidationError(
            "registered member config is not an object"
        )
    return parsed


def build_registered_training_command(
    *, member: Mapping[str, Any], run_dir: Path, gpu_count: int
) -> tuple[list[str], dict[str, object], str]:
    """Compose argv back to the exact committed resolved-config bytes."""

    gpu_count = pilot.validate_gpu_count(gpu_count)
    training_variant = pilot.validate_training_variant(
        str(member["training_variant"])
    )
    config_name = str(pilot.TRAINING_VARIANTS[training_variant]["config_name"])
    target = _read_member_config(member)
    command = [
        str(pilot._python_executable()),
        "-u",
        str(REPOSITORY_ROOT / "scripts" / "train.py"),
        "--config-name",
        config_name,
        *screen_launcher._resolved_config_leaf_overrides(target),
        f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    resolved, digest = pilot.compose_resolved_training_config(
        config_name=config_name,
        overrides=command[5:],
        gpu_count=gpu_count,
    )
    if resolved != target or digest != member["config"]["canonical_sha256"]:
        raise verifier.ScaleUpValidationError(
            "scale-up training argv does not reproduce the registered config"
        )
    return command, resolved, digest


def build_matched_panel_spec(
    *,
    registry: verifier.ValidatedScaleUpRegistry,
    source_revision: str,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> tuple[dict[str, object], str]:
    common = registry.data["common_training"]
    specification, digest = pilot.build_matched_panel_spec(
        source_revision=source_revision,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        gpu_count=int(common["gpu_count"]),
        max_steps=verifier.EXPECTED_MAX_STEPS,
        global_batch_size=int(common["global_batch_size"]),
        micro_batch_size=int(common["micro_batch_size_per_process"]),
        num_workers=int(common["num_workers"]),
        seed=verifier.EXPECTED_TRAINING_SEED,
        exclude_special_tokens=False,
        max_utilization_percent=verifier.MAX_SAFE_UTILIZATION_PERCENT,
        min_free_memory_mib=verifier.MIN_SAFE_FREE_MEMORY_MIB,
        common_resolved_config_sha256=str(
            common["common_resolved_config_sha256"]
        ),
    )
    audit = pilot.verify_pilot_empirical_uniform_mix_audit()
    contract = specification["common_training_contract"]
    if (
        contract["empirical_uniform_mix_audit"] != audit
        or contract["empirical_uniform_mix"] != pilot.PILOT_EMPIRICAL_UNIFORM_MIX
    ):
        raise verifier.ScaleUpValidationError(
            "matched panel is not bound to the verified prior-floor audit"
        )
    return specification, digest


def build_selection_bound_scale_up_binding(
    plan_registry: verifier.ValidatedScaleUpRegistry, *, position: int
) -> dict[str, Any]:
    binding = verifier.expected_manifest_binding(plan_registry, position=position)
    pilot.validate_selection_bound_scale_up(
        binding,
        expected_training_variant=verifier.EXPECTED_VARIANTS[position],
        expected_position=position,
        expected_world_size=int(
            plan_registry.data["common_training"]["gpu_count"]
        ),
        expected_resolved_config_sha256=str(
            plan_registry.data["members"][position]["config"]["canonical_sha256"]
        ),
    )
    return verifier.validate_manifest_binding(
        binding, registry=plan_registry, position=position
    )


def build_launch_plan(
    *,
    registry: verifier.ValidatedScaleUpRegistry,
    source_revision: str,
    position: int,
    predecessor_receipt: Path | None,
) -> ScaleUpLaunchPlan:
    position = verifier._integer(position, "scale-up position", minimum=0, maximum=2)
    member = registry.data["members"][position]
    common = registry.data["common_training"]
    gpu_count = pilot.validate_gpu_count(int(common["gpu_count"]))
    run_dir = _registered_output_path(str(member["output_directory"]))
    if run_dir.name != member["run_name"]:
        raise verifier.ScaleUpValidationError(
            "registered run name differs from output directory"
        )
    command, resolved, digest = build_registered_training_command(
        member=member, run_dir=run_dir, gpu_count=gpu_count
    )
    if verifier.matched_config_sha256(resolved) != common[
        "common_resolved_config_sha256"
    ]:
        raise verifier.ScaleUpValidationError(
            "registered member differs from the common matched config"
        )
    checkpoint_ref = common["initialization"]["checkpoint"]
    if checkpoint_ref["root"] != "project":
        raise verifier.ScaleUpValidationError(
            "scale-up checkpoint must be project-rooted"
        )
    checkpoint = screen._root_path(
        "project", PurePosixPath(str(checkpoint_ref["relative_path"]))
    ).resolve(strict=True)
    if pilot.sha256_file(checkpoint) != checkpoint_ref["sha256"]:
        raise verifier.ScaleUpValidationError(
            "live MDLM initialization checkpoint differs from the registry"
        )
    matched_spec, matched_digest = build_matched_panel_spec(
        registry=registry,
        source_revision=source_revision,
        checkpoint=checkpoint,
        checkpoint_sha256=str(checkpoint_ref["sha256"]),
    )
    selection_binding = build_selection_bound_scale_up_binding(
        registry, position=position
    )
    predecessor_binding = pilot.build_predecessor_receipt_binding(
        training_variant=str(member["training_variant"]),
        explicit_genesis=position == 0,
        predecessor_receipt_path=predecessor_receipt,
        matched_panel_spec=matched_spec,
        matched_panel_spec_sha256=matched_digest,
        selection_bound_scale_up=selection_binding,
    )
    session_name = f"genmol_{member['training_variant']}_{member['run_name']}"
    if pilot.RUN_NAME_PATTERN.fullmatch(session_name) is None:
        raise verifier.ScaleUpValidationError("registered tmux session name is invalid")
    return ScaleUpLaunchPlan(
        registry=registry,
        source_revision=source_revision,
        position=position,
        member=member,
        run_dir=run_dir,
        log_path=REPOSITORY_ROOT / "output" / "logs" / f"{member['run_name']}.log",
        session_name=session_name,
        checkpoint=checkpoint,
        checkpoint_sha256=str(checkpoint_ref["sha256"]),
        command=command,
        resolved_config=resolved,
        resolved_config_sha256=digest,
        argv_sha256=pilot.training_argv_sha256(command),
        matched_panel_spec=matched_spec,
        matched_panel_spec_sha256=matched_digest,
        selection_bound_scale_up=selection_binding,
        predecessor_receipt=predecessor_receipt,
        predecessor_receipt_binding=predecessor_binding,
    )


def dry_run_preview(plan: ScaleUpLaunchPlan) -> dict[str, Any]:
    """Return the exact preflight without GPU/tmux/output mutation."""

    return {
        "schema_version": 1,
        "status": "dry_run_preflight_completed_no_launch",
        "project_launch_artifact_mutation_performed": False,
        "training_job_lock_acquired": False,
        "gpu_probe_performed": False,
        "tmux_operation_performed": False,
        "source_revision": plan.source_revision,
        "registry": plan.registry.reference,
        "position": plan.position,
        "arm_id": verifier.EXPECTED_SLUGS[plan.position].upper(),
        "training_variant": plan.training_variant,
        "selected_scheduler_arm_id": plan.registry.scheduler_selection[
            "selected_arm_id"
        ],
        "selected_conditioning_arm_id": plan.registry.conditioning_selection[
            "selected_arm_id"
        ],
        "predicted_run_directory": str(plan.run_dir),
        "predicted_log_path": str(plan.log_path),
        "predicted_tmux_session": plan.session_name,
        "training_argv": plan.command,
        "training_argv_sha256": plan.argv_sha256,
        "resolved_training_config": plan.resolved_config,
        "resolved_training_config_sha256": plan.resolved_config_sha256,
        "matched_panel_spec": plan.matched_panel_spec,
        "matched_panel_spec_sha256": plan.matched_panel_spec_sha256,
        "selection_bound_scale_up": plan.selection_bound_scale_up,
        "predecessor_receipt_binding": plan.predecessor_receipt_binding,
    }


def _select_idle_gpus(
    states: list[pilot.GPUState],
    *,
    gpu_count: int,
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> tuple[pilot.GPUState, ...]:
    """Select 1..4 GPUs through the imported hardened pilot helper."""

    return pilot.select_idle_gpus(
        states,
        gpu_count=gpu_count,
        max_utilization_percent=max_utilization_percent,
        min_free_memory_mib=min_free_memory_mib,
    )


def _launch_locked(
    plan: ScaleUpLaunchPlan,
    *,
    lock_path: Path,
    lock_record: dict[str, object],
    lock_sha256: str,
) -> tuple[bytes, str, Path]:
    """Probe, publish, and hand one registry member to detached tmux."""

    common = plan.registry.data["common_training"]
    gpu_count = int(common["gpu_count"])
    max_utilization = int(
        common["gpu_safety_policy"]["max_utilization_percent"]
    )
    min_free_memory = int(common["gpu_safety_policy"]["min_free_memory_mib"])
    pilot.validate_safety_thresholds(max_utilization, min_free_memory)
    pilot.validate_pilot_output_parents(
        run_dir=plan.run_dir, log_path=plan.log_path, create_missing=False
    )
    pilot._require_predecessor_receipt_before_lock(
        plan.predecessor_receipt_binding,
        current_lock_acquired_at_utc=str(lock_record["acquired_at_utc"]),
    )
    inventory = pilot.probe_all_gpus()
    inventory_completed = datetime.now(timezone.utc).isoformat()
    initially_selected = _select_idle_gpus(
        inventory,
        gpu_count=gpu_count,
        max_utilization_percent=max_utilization,
        min_free_memory_mib=min_free_memory,
    )
    runtime_path = plan.run_dir / "runtime_config.json"
    summary_path = plan.run_dir / "training_summary.json"
    receipt_path = pilot.validate_pilot_exit_receipt_path(
        plan.run_dir / RECEIPT_BASENAME
    )
    checkpoint_path = (
        plan.run_dir / "checkpoints" / f"{verifier.EXPECTED_MAX_STEPS}.ckpt"
    )
    manifest_path = plan.run_dir / "launch_manifest.json"
    source_before_probe = pilot.require_pushed_commit()
    if source_before_probe != plan.source_revision:
        raise RuntimeError("source revision changed before final GPU probe")
    pilot.revalidate_predecessor_receipt_binding(
        plan.predecessor_receipt_binding,
        training_variant=plan.training_variant,
        matched_panel_spec=plan.matched_panel_spec,
        matched_panel_spec_sha256=plan.matched_panel_spec_sha256,
        selection_bound_scale_up=plan.selection_bound_scale_up,
    )
    gpu_states = pilot.reprobe_selected_gpus(
        initially_selected,
        max_utilization_percent=max_utilization,
        min_free_memory_mib=min_free_memory,
    )
    final_probe_completed = datetime.now(timezone.utc).isoformat()
    manifest_created = datetime.now(timezone.utc).isoformat()
    pilot.revalidate_predecessor_chain_artifact_identities(
        plan.predecessor_receipt_binding,
        current_manifest_created_at_utc=manifest_created,
    )
    selected_uuids = [state.uuid for state in gpu_states]
    selected_json = json.dumps(selected_uuids, separators=(",", ":"))
    _create_direct_directory_exclusive(
        plan.run_dir, label="scale-up run directory"
    )
    _safe_publish_bytes_exclusive(
        plan.log_path, b"", label="scale-up training log"
    )
    _create_direct_directory_exclusive(
        plan.run_dir / "hydra", label="scale-up Hydra directory"
    )
    _create_direct_directory_exclusive(
        plan.run_dir / "checkpoints", label="scale-up checkpoint directory"
    )
    output_directory_binding = pilot.build_output_directory_binding(
        run_dir=plan.run_dir,
        log_path=plan.log_path,
    )
    variant = pilot.TRAINING_VARIANTS[plan.training_variant]
    manifest = {
        "launch_manifest_schema_version": pilot.LAUNCH_MANIFEST_SCHEMA_VERSION,
        "created_at": manifest_created,
        "purpose": PURPOSE,
        "gpu_selection_schema_version": 2,
        "git_sha": plan.source_revision,
        "source_revision_before_final_gpu_probe": source_before_probe,
        "run_name": plan.member["run_name"],
        "training_variant": plan.training_variant,
        "hydra_config_name": variant["config_name"],
        "udlm_prior_variant": variant["prior_variant"],
        "udlm_comparison_role": variant["comparison_role"],
        "selection_bound_scale_up": plan.selection_bound_scale_up,
        "output_directory_binding": output_directory_binding,
        "matched_panel_spec": plan.matched_panel_spec,
        "matched_panel_spec_sha256": plan.matched_panel_spec_sha256,
        "matched_panel_variant_position": plan.position,
        "predecessor_receipt_binding": plan.predecessor_receipt_binding,
        "single_training_job_lock": {
            "path": str(lock_path),
            "sha256": lock_sha256,
            "record": lock_record,
            "acquired_before_any_gpu_probe": True,
            "stale_lock_policy": "fail_closed_and_require_manual_review",
            "release_owner": "pilot_exit_receipt_writer_after_publication",
        },
        "tmux_session": plan.session_name,
        "user_requested_gpu_count": gpu_count,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": inventory_completed,
        "gpu_inventory_at_selection": [asdict(state) for state in inventory],
        "initially_selected_gpu_states": [
            asdict(state) for state in initially_selected
        ],
        "logical_cuda_devices": list(range(gpu_count)),
        "physical_gpu_indices": [state.physical_index for state in gpu_states],
        "cuda_visible_device_uuids": selected_uuids,
        "final_uuid_probes_completed_at_utc": final_probe_completed,
        "gpu_states_at_final_uuid_probe": [asdict(state) for state in gpu_states],
        "gpu_safety_policy": {
            "max_utilization_percent": max_utilization,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": min_free_memory,
            "active_compute_processes_allowed": pilot.ACTIVE_COMPUTE_PROCESSES_ALLOWED,
            "compute_mode_prohibited_allowed": False,
        },
        "training_argv": plan.command,
        "training_argv_sha256": plan.argv_sha256,
        "resolved_training_config": plan.resolved_config,
        "resolved_training_config_sha256": plan.resolved_config_sha256,
        "runtime_config_path": str(runtime_path),
        "training_summary_path": str(summary_path),
        "training_summary_schema_version": pilot.TRAINING_SUMMARY_SCHEMA_VERSION,
        "pilot_exit_status_path": str(receipt_path),
        "pilot_exit_status_schema_version": pilot.PILOT_EXIT_STATUS_SCHEMA_VERSION,
        "expected_final_checkpoint_path": str(checkpoint_path),
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_raw_sha256_transport": (
            "passed_out_of_band_to_training_and_receipt_to_avoid_self_hash"
        ),
        "completion_contract": {
            "status_at_launch": "pending",
            "complete_only_if_valid_training_summary_exists": True,
            "complete_only_if_successful_exit_receipt_exists": True,
            "valid_training_summary_and_successful_exit_receipt_both_required": True,
            "missing_summary_after_tmux_exit_means": "incomplete",
            "absent_exit_receipt_means": "incomplete",
            "successful_exit_receipt_requires": {
                "training_exit_status": 0,
                "tee_exit_status": 0,
                "valid_launch_bound_training_summary": True,
                "exact_launch_manifest_still_matches": True,
                "clean_pushed_source_at_receipt": True,
                "predecessor_receipt_binding_unchanged_and_valid": True,
            },
            "training_job_lock_release": (
                "after_exit_receipt_publication_for_completed_or_failed_pipeline"
            ),
        },
        "log_path": str(plan.log_path),
        "log_reserved_exclusively_before_manifest": True,
        "checkpoint": str(plan.checkpoint),
        "checkpoint_sha256": plan.checkpoint_sha256,
        "seed": verifier.EXPECTED_TRAINING_SEED,
        "max_steps": verifier.EXPECTED_MAX_STEPS,
        "global_batch_size": common["global_batch_size"],
        "micro_batch_size_per_process": common["micro_batch_size_per_process"],
        "accumulate_grad_batches": common["accumulate_grad_batches"],
        "effective_global_batch_size": common["effective_global_batch_size"],
        "exclude_special_tokens": False,
        "dry_run": False,
    }
    pilot.validate_selection_bound_scale_up(
        manifest["selection_bound_scale_up"],
        expected_training_variant=plan.training_variant,
        expected_position=plan.position,
        expected_world_size=gpu_count,
        expected_resolved_config_sha256=plan.resolved_config_sha256,
    )
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_sha = _safe_publish_bytes_exclusive(
        manifest_path, manifest_bytes, label="scale-up launch manifest"
    )
    environment_command, _environment = pilot.build_child_environment_command(
        command=plan.command,
        source_revision=plan.source_revision,
        resolved_config_sha256=plan.resolved_config_sha256,
        runtime_config_path=runtime_path,
        training_summary_path=summary_path,
        final_checkpoint_path=checkpoint_path,
        launch_manifest_path=manifest_path,
        launch_manifest_sha256=manifest_sha,
        expected_max_steps=verifier.EXPECTED_MAX_STEPS,
        expected_world_size=gpu_count,
        visible_uuids=",".join(selected_uuids),
        seed=verifier.EXPECTED_TRAINING_SEED,
    )
    shell_command = pilot.build_tmux_shell_command(
        environment_command,
        log_path=plan.log_path,
        training_summary_path=summary_path,
        exit_receipt_path=receipt_path,
        expected_source_revision=plan.source_revision,
        expected_config_sha256=plan.resolved_config_sha256,
        expected_argv_sha256=plan.argv_sha256,
        expected_summary_schema_version=pilot.TRAINING_SUMMARY_SCHEMA_VERSION,
        expected_max_steps=verifier.EXPECTED_MAX_STEPS,
        expected_world_size=gpu_count,
        expected_final_checkpoint_path=checkpoint_path,
        expected_launch_manifest_path=manifest_path,
        expected_launch_manifest_sha256=manifest_sha,
        expected_selected_gpu_uuids_json=selected_json,
        expected_training_job_lock_path=lock_path,
        expected_training_job_lock_sha256=lock_sha256,
        expected_initialization_checkpoint_sha256=plan.checkpoint_sha256,
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            plan.session_name,
            "-c",
            str(REPOSITORY_ROOT),
            "bash",
            "-lc",
            shell_command,
        ],
        check=True,
    )
    return manifest_bytes, manifest_sha, plan.log_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-canonical-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    registry = load_registry(
        args.registry,
        expected_raw_sha256=args.expected_registry_sha256,
        expected_canonical_sha256=args.expected_registry_canonical_sha256,
    )
    source_revision = pilot.require_pushed_commit()
    require_registry_publication(registry, current_revision=source_revision)
    paths = run_paths(registry)
    for _member, run_dir, _receipt in paths:
        pilot.validate_pilot_output_parents(
            run_dir=run_dir,
            log_path=REPOSITORY_ROOT / "output" / "logs" / f"{run_dir.name}.log",
            create_missing=False,
        )
    position = completed_prefix(
        paths, registry=registry, source_revision=source_revision
    )
    if position == len(paths):
        from scripts.udlm.validate_scale_up_panel import validate_scale_up_panel

        result = validate_scale_up_panel(
            paths[-1][2],
            registry=registry,
            expected_source_revision=source_revision,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    member, run_dir, _receipt = paths[position]
    log_path = REPOSITORY_ROOT / "output" / "logs" / f"{run_dir.name}.log"
    if os.path.lexists(run_dir) or os.path.lexists(log_path):
        raise FileExistsError("refusing to overwrite a scale-up launch namespace")
    predecessor = None if position == 0 else paths[position - 1][2]
    plan = build_launch_plan(
        registry=registry,
        source_revision=source_revision,
        position=position,
        predecessor_receipt=predecessor,
    )
    if args.dry_run:
        print(json.dumps(dry_run_preview(plan), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if pilot.tmux_session_exists(plan.session_name):
        raise RuntimeError(f"tmux session already exists: {plan.session_name}")
    pilot.validate_pilot_output_parents(
        run_dir=plan.run_dir, log_path=plan.log_path, create_missing=True
    )
    lock_path, lock_record, lock_sha = _acquire_scale_up_training_job_lock(
        source_revision=source_revision,
        run_name=str(member["run_name"]),
        training_variant=str(member["training_variant"]),
    )
    try:
        manifest_bytes, manifest_sha, launched_log = _launch_locked(
            plan,
            lock_path=lock_path,
            lock_record=lock_record,
            lock_sha256=lock_sha,
        )
    except BaseException as launch_error:
        try:
            handed_off = pilot.tmux_session_exists(plan.session_name)
        except BaseException as verification_error:
            raise RuntimeError(
                "scale-up launch failed with indeterminate tmux handoff; lock retained"
            ) from verification_error
        if handed_off:
            raise RuntimeError(
                "scale-up launch raised after tmux may have accepted the job; "
                "lock retained fail-closed"
            ) from launch_error
        try:
            _release_exact_scale_up_training_job_lock(
                lock_path, expected_sha256=lock_sha
            )
        except Exception as release_error:
            raise RuntimeError(
                "pre-handoff scale-up failure could not release the exact lock"
            ) from release_error
        raise
    print(manifest_bytes.decode("utf-8"), end="")
    print(f"launch manifest SHA-256: {manifest_sha}")
    print(f"launched tmux session {plan.session_name}; log: {launched_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
