"""Validate the completed selection-bound 1,000-update R/S/E panel.

The validator consumes E's terminal receipt and the already validated frozen
registry.  It independently reconstructs every command/config, validates all
schema-5 training artifacts and the recursive receipt chain, and requires the
same schema-1 selection authority to advance exactly R -> S -> E.  It performs
no GPU queries and grants no generation, ranking, or superiority eligibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.udlm import launch_scale_up_panel as launcher  # noqa: E402
from scripts.udlm import launch_train_pilot as pilot  # noqa: E402
from scripts.udlm import verify_optimization_screen as screen  # noqa: E402
from scripts.udlm import verify_scale_up_registry as verifier  # noqa: E402
from scripts.udlm import write_pilot_evidence as evidence_writer  # noqa: E402


EVIDENCE_SCHEMA_VERSION = 1


class ScaleUpPanelValidationError(ValueError):
    """Raised when a retained member or recursive binding is not exact."""


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScaleUpPanelValidationError(f"{label} must be an object")
    return value


def _exact(observed: object, expected: object, *, label: str) -> None:
    if not screen._exact_json_equal(observed, expected):
        raise ScaleUpPanelValidationError(
            f"{label} differs: expected {expected!r}, observed {observed!r}"
        )


def _validated_receipt_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    candidate = Path(os.path.abspath(candidate))
    root = REPOSITORY_ROOT.resolve(strict=True)
    try:
        relative = candidate.relative_to(root / "output" / "udlm")
    except ValueError as error:
        raise ScaleUpPanelValidationError(
            "scale-up receipt must be inside output/udlm"
        ) from error
    if (
        len(relative.parts) != 2
        or relative.name != launcher.RECEIPT_BASENAME
        or pilot.RUN_NAME_PATTERN.fullmatch(relative.parts[0]) is None
    ):
        raise ScaleUpPanelValidationError(
            "scale-up receipt must be output/udlm/<run-name>/pilot_exit_status.json"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ScaleUpPanelValidationError(
            f"scale-up receipt is unavailable: {candidate}"
        ) from error
    if resolved != candidate:
        raise ScaleUpPanelValidationError(
            "scale-up receipt path must not traverse symlinks"
        )
    return candidate


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        artifact = evidence_writer.read_stable_regular_file(path, label=label)
        if artifact.payload is None:
            raise ScaleUpPanelValidationError(f"{label} payload was not retained")
        parsed = evidence_writer.strict_json_loads(artifact.payload, label=label)
    except (OSError, ValueError) as error:
        if isinstance(error, ScaleUpPanelValidationError):
            raise
        raise ScaleUpPanelValidationError(f"cannot validate {label}: {error}") from error
    return dict(_mapping(parsed, label=label)), artifact.snapshot()


def _structural_checkpoint(receipt: Mapping[str, Any]) -> dict[str, object]:
    expected = _mapping(receipt.get("expected_contract"), label="receipt contract")
    checkpoint = _mapping(
        receipt.get("final_checkpoint"), label="receipt checkpoint evidence"
    )
    snapshot = _mapping(
        checkpoint.get("artifact"), label="receipt checkpoint snapshot"
    )
    return {
        "checkpoint": {
            "path": expected.get("final_checkpoint_path"),
            "sha256": snapshot.get("sha256"),
            "size_bytes": snapshot.get("size_bytes"),
            "global_step": expected.get("max_steps"),
        }
    }


def _relative(path: Path, *, label: str) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise ScaleUpPanelValidationError(
            f"{label} must remain inside the repository"
        ) from error


def _validate_gpu_states(
    manifest: Mapping[str, Any], *, gpu_count: int, variant: str
) -> None:
    _exact(
        manifest.get("gpu_selection_schema_version"),
        2,
        label=f"{variant} GPU selection schema",
    )
    _exact(
        manifest.get("gpu_selection_method"),
        "dynamic_idle_discovery",
        label=f"{variant} GPU selection method",
    )
    _exact(
        manifest.get("gpu_inventory_scope"),
        "all_nvidia_gpus",
        label=f"{variant} GPU inventory scope",
    )
    try:
        inventory_time = pilot._utc_timestamp(
            manifest.get("inventory_snapshot_completed_at_utc"),
            label=f"{variant} GPU inventory timestamp",
        )
        final_probe_time = pilot._utc_timestamp(
            manifest.get("final_uuid_probes_completed_at_utc"),
            label=f"{variant} final GPU probe timestamp",
        )
        manifest_time = pilot._utc_timestamp(
            manifest.get("created_at"), label=f"{variant} manifest timestamp"
        )
    except ValueError as error:
        raise ScaleUpPanelValidationError(
            f"{variant} GPU timestamp evidence is invalid: {error}"
        ) from error
    if not inventory_time <= final_probe_time <= manifest_time:
        raise ScaleUpPanelValidationError(
            f"{variant} GPU probe/manifest chronology is invalid"
        )

    inventory = manifest.get("gpu_inventory_at_selection")
    initially_selected = manifest.get("initially_selected_gpu_states")
    final_states = manifest.get("gpu_states_at_final_uuid_probe")
    selected_uuids = manifest.get("cuda_visible_device_uuids")
    if (
        not isinstance(inventory, list)
        or not inventory
        or not isinstance(initially_selected, list)
        or not isinstance(final_states, list)
        or not isinstance(selected_uuids, list)
        or len(initially_selected) != gpu_count
        or len(final_states) != gpu_count
        or len(selected_uuids) != gpu_count
        or len(set(selected_uuids)) != gpu_count
        or any(
            not isinstance(uuid, str) or not uuid.startswith("GPU-")
            for uuid in selected_uuids
        )
    ):
        raise ScaleUpPanelValidationError(
            f"{variant} retained invalid GPU inventory/selection arrays"
        )
    try:
        inventory_records = [
            pilot._validate_gpu_state_record(
                state, label=f"{variant} inventory GPU {index}"
            )
            for index, state in enumerate(inventory)
        ]
        initial_records = [
            pilot._validate_gpu_state_record(
                state, label=f"{variant} initial GPU {index}"
            )
            for index, state in enumerate(initially_selected)
        ]
        final_records = [
            pilot._validate_gpu_state_record(
                state, label=f"{variant} final GPU {index}"
            )
            for index, state in enumerate(final_states)
        ]
    except ValueError as error:
        raise ScaleUpPanelValidationError(
            f"{variant} GPU state/process evidence is invalid: {error}"
        ) from error

    inventory_uuids = [str(state["uuid"]) for state in inventory_records]
    inventory_indices = [int(state["physical_index"]) for state in inventory_records]
    if (
        len(set(inventory_uuids)) != len(inventory_uuids)
        or len(set(inventory_indices)) != len(inventory_indices)
        or inventory_indices != sorted(inventory_indices)
        or any(
            state["memory_total_mib"] <= 0
            for state in (*inventory_records, *initial_records, *final_records)
        )
    ):
        raise ScaleUpPanelValidationError(
            f"{variant} GPU inventory identities are invalid"
        )
    inventory_by_uuid = {
        str(state["uuid"]): state for state in inventory_records
    }
    if any(uuid not in inventory_by_uuid for uuid in selected_uuids):
        raise ScaleUpPanelValidationError(
            f"{variant} selected GPU is absent from the full inventory"
        )
    inventory_states = [
        pilot.GPUState(
            physical_index=int(state["physical_index"]),
            uuid=str(state["uuid"]),
            name=str(state["name"]),
            memory_used_mib=int(state["memory_used_mib"]),
            memory_total_mib=int(state["memory_total_mib"]),
            utilization_percent=int(state["utilization_percent"]),
            compute_mode=str(state["compute_mode"]),
            compute_processes=tuple(state["compute_processes"]),
        )
        for state in inventory_records
    ]
    try:
        expected_selection = pilot.select_idle_gpus(
            inventory_states,
            gpu_count=gpu_count,
            max_utilization_percent=verifier.MAX_SAFE_UTILIZATION_PERCENT,
            min_free_memory_mib=verifier.MIN_SAFE_FREE_MEMORY_MIB,
        )
    except (ValueError, RuntimeError) as error:
        raise ScaleUpPanelValidationError(
            f"{variant} full GPU inventory cannot reproduce an idle selection"
        ) from error
    _exact(
        [(state.uuid, state.physical_index) for state in expected_selection],
        [(state["uuid"], state["physical_index"]) for state in initial_records],
        label=f"{variant} deterministic GPU selection",
    )
    for label, records in (("initial", initial_records), ("final", final_records)):
        _exact(
            [state["uuid"] for state in records],
            selected_uuids,
            label=f"{variant} {label} GPU UUID order",
        )
    _exact(
        initial_records,
        [inventory_by_uuid[uuid] for uuid in selected_uuids],
        label=f"{variant} initially selected inventory rows",
    )
    _exact(
        [(state["uuid"], state["physical_index"]) for state in final_records],
        [(state["uuid"], state["physical_index"]) for state in initial_records],
        label=f"{variant} initial/final GPU identity sequence",
    )
    final_indices = [state["physical_index"] for state in final_records]
    if len(set(final_indices)) != gpu_count:
        raise ScaleUpPanelValidationError(
            f"{variant} final physical GPU identities are duplicated"
        )
    _exact(
        manifest.get("logical_cuda_devices"),
        list(range(gpu_count)),
        label=f"{variant} logical CUDA device mapping",
    )
    _exact(
        selected_uuids,
        [state["uuid"] for state in final_records],
        label=f"{variant} visible GPU UUIDs",
    )
    _exact(
        manifest.get("physical_gpu_indices"),
        final_indices,
        label=f"{variant} physical GPU indices",
    )
    for state in (*initial_records, *final_records):
        if (
            state["utilization_percent"] >= verifier.MAX_SAFE_UTILIZATION_PERCENT
            or state["memory_total_mib"] - state["memory_used_mib"]
            < verifier.MIN_SAFE_FREE_MEMORY_MIB
            or str(state["compute_mode"]).strip().lower() == "prohibited"
        ):
            raise ScaleUpPanelValidationError(
                f"{variant} retained a GPU outside the frozen idle policy"
            )


def _validate_member(
    *,
    registry: verifier.ValidatedScaleUpRegistry,
    source_revision: str,
    position: int,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    receipt_snapshot: Mapping[str, Any],
    previous_receipt_path: Path | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[evidence_writer.StableArtifact, ...],
]:
    member = registry.data["members"][position]
    common = registry.data["common_training"]
    variant = str(member["training_variant"])
    run_dir = receipt_path.parent
    manifest_path = run_dir / "launch_manifest.json"
    summary_path = run_dir / "training_summary.json"
    checkpoint_path = (
        run_dir / "checkpoints" / f"{verifier.EXPECTED_MAX_STEPS}.ckpt"
    )
    try:
        _recorded_at, retained_artifacts = (
            evidence_writer.validate_successful_training_receipt(
            receipt,
            receipt_path=receipt_path,
            structural=_structural_checkpoint(receipt),
        )
        )
    except (OSError, ValueError) as error:
        raise ScaleUpPanelValidationError(
            f"{variant} receipt or recursive artifacts are invalid: {error}"
        ) from error
    summary_artifact, manifest_artifact, runtime_artifact, checkpoint_artifact = (
        retained_artifacts
    )
    expected_artifact_paths = (
        summary_path,
        manifest_path,
        run_dir / "runtime_config.json",
        checkpoint_path,
    )
    if tuple(artifact.path for artifact in retained_artifacts) != expected_artifact_paths:
        raise ScaleUpPanelValidationError(
            f"{variant} semantic validator retained unexpected artifact paths"
        )
    if summary_artifact.payload is None or manifest_artifact.payload is None:
        raise ScaleUpPanelValidationError(
            f"{variant} semantic validator did not retain JSON artifact bytes"
        )
    try:
        manifest = dict(
            _mapping(
                evidence_writer.strict_json_loads(
                    manifest_artifact.payload, label=f"{variant} launch manifest"
                ),
                label=f"{variant} launch manifest",
            )
        )
        summary = dict(
            _mapping(
                evidence_writer.strict_json_loads(
                    summary_artifact.payload, label=f"{variant} training summary"
                ),
                label=f"{variant} training summary",
            )
        )
    except ValueError as error:
        raise ScaleUpPanelValidationError(
            f"{variant} retained semantic artifacts are invalid: {error}"
        ) from error
    manifest_snapshot = manifest_artifact.snapshot()

    expected_contract = _mapping(
        receipt.get("expected_contract"), label=f"{variant} receipt contract"
    )
    expected_values = {
        "source_revision": source_revision,
        "resolved_training_config_sha256": member["config"]["canonical_sha256"],
        "max_steps": verifier.EXPECTED_MAX_STEPS,
        "world_size": common["gpu_count"],
        "launch_manifest_path": str(manifest_path),
        "training_summary_path": str(summary_path),
        "final_checkpoint_path": str(checkpoint_path),
        "initialization_checkpoint_sha256": common["initialization"]["checkpoint"][
            "sha256"
        ],
    }
    for key, expected in expected_values.items():
        _exact(
            expected_contract.get(key), expected, label=f"{variant} receipt {key}"
        )

    command, resolved, resolved_sha = launcher.build_registered_training_command(
        member=member,
        run_dir=run_dir,
        gpu_count=int(common["gpu_count"]),
    )
    argv_sha = pilot.training_argv_sha256(command)
    checkpoint_ref = common["initialization"]["checkpoint"]
    initialization_checkpoint = screen._root_path(
        str(checkpoint_ref["root"]),
        PurePosixPath(str(checkpoint_ref["relative_path"])),
    ).resolve(strict=True)
    panel, panel_sha = launcher.build_matched_panel_spec(
        registry=registry,
        source_revision=source_revision,
        checkpoint=initialization_checkpoint,
        checkpoint_sha256=str(checkpoint_ref["sha256"]),
    )
    binding = verifier.expected_manifest_binding(registry, position=position)
    verifier.validate_manifest_binding(binding, registry=registry, position=position)
    try:
        pilot.validate_selection_bound_scale_up(
            manifest.get("selection_bound_scale_up"),
            expected_training_variant=variant,
            expected_position=position,
            expected_world_size=int(common["gpu_count"]),
            expected_resolved_config_sha256=resolved_sha,
        )
    except ValueError as error:
        raise ScaleUpPanelValidationError(
            f"{variant} selection-bound authority is invalid: {error}"
        ) from error
    _exact(
        manifest.get("selection_bound_scale_up"),
        binding,
        label=f"{variant} registry manifest binding",
    )
    manifest_values = {
        "purpose": launcher.PURPOSE,
        "git_sha": source_revision,
        "source_revision_before_final_gpu_probe": source_revision,
        "run_name": member["run_name"],
        "training_variant": variant,
        "hydra_config_name": pilot.TRAINING_VARIANTS[variant]["config_name"],
        "udlm_prior_variant": member["prior_variant"],
        "udlm_comparison_role": member["comparison_role"],
        "matched_panel_spec": panel,
        "matched_panel_spec_sha256": panel_sha,
        "matched_panel_variant_position": position,
        "user_requested_gpu_count": common["gpu_count"],
        "checkpoint": str(initialization_checkpoint),
        "checkpoint_sha256": checkpoint_ref["sha256"],
        "seed": verifier.EXPECTED_TRAINING_SEED,
        "max_steps": verifier.EXPECTED_MAX_STEPS,
        "global_batch_size": common["global_batch_size"],
        "micro_batch_size_per_process": common["micro_batch_size_per_process"],
        "accumulate_grad_batches": common["accumulate_grad_batches"],
        "effective_global_batch_size": common["effective_global_batch_size"],
        "exclude_special_tokens": False,
        "dry_run": False,
        "launch_manifest_path": str(manifest_path),
        "training_summary_path": str(summary_path),
        "pilot_exit_status_path": str(receipt_path),
        "expected_final_checkpoint_path": str(checkpoint_path),
        "training_argv": command,
        "training_argv_sha256": argv_sha,
        "resolved_training_config": resolved,
        "resolved_training_config_sha256": resolved_sha,
    }
    for key, expected in manifest_values.items():
        _exact(manifest.get(key), expected, label=f"{variant} manifest {key}")
    _exact(
        receipt_path.parent.name,
        member["run_name"],
        label=f"{variant} run-directory name",
    )
    expected_safety = dict(common["gpu_safety_policy"])
    _exact(
        manifest.get("gpu_safety_policy"),
        expected_safety,
        label=f"{variant} GPU safety policy",
    )
    _validate_gpu_states(
        manifest, gpu_count=int(common["gpu_count"]), variant=variant
    )
    _exact(
        expected_contract.get("training_argv_sha256"),
        argv_sha,
        label=f"{variant} receipt argv digest",
    )
    _exact(
        expected_contract.get("launch_manifest_sha256"),
        manifest_snapshot["sha256"],
        label=f"{variant} receipt manifest digest",
    )

    accounting = _mapping(
        summary.get("training_accounting"), label=f"{variant} training accounting"
    )
    accounting_values = {
        "training_seed": verifier.EXPECTED_TRAINING_SEED,
        "optimizer_updates": verifier.EXPECTED_MAX_STEPS,
        "world_size": common["gpu_count"],
        "micro_batch_size_per_rank": common["micro_batch_size_per_process"],
        "accumulate_grad_batches": common["accumulate_grad_batches"],
        "effective_global_examples_per_optimizer_step": common[
            "effective_global_batch_size"
        ],
        "total_requested_example_exposures": (
            verifier.EXPECTED_MAX_STEPS * common["effective_global_batch_size"]
        ),
    }
    for key, expected in accounting_values.items():
        _exact(accounting.get(key), expected, label=f"{variant} accounting {key}")
    _exact(
        summary.get("source_revision"),
        source_revision,
        label=f"{variant} summary source",
    )
    _exact(
        summary.get("resolved_training_config_sha256"),
        resolved_sha,
        label=f"{variant} summary config digest",
    )
    _exact(
        summary.get("training_argv_sha256"),
        argv_sha,
        label=f"{variant} summary argv digest",
    )
    if summary.get("conditioning_gradient_audit") is not None:
        raise ScaleUpPanelValidationError(
            f"{variant} scale-up summary must not claim screen gradient evidence"
        )
    if summary.get("screen_initialization_state_audit") is not None:
        raise ScaleUpPanelValidationError(
            f"{variant} scale-up summary must not claim screen initialization evidence"
        )

    predecessor = _mapping(
        receipt.get("predecessor_receipt_binding"),
        label=f"{variant} predecessor binding",
    )
    _exact(
        manifest.get("predecessor_receipt_binding"),
        predecessor,
        label=f"{variant} manifest/receipt predecessor binding",
    )
    try:
        pilot.revalidate_predecessor_receipt_binding(
            dict(predecessor),
            training_variant=variant,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha,
            selection_bound_scale_up=binding,
        )
    except (OSError, ValueError) as error:
        raise ScaleUpPanelValidationError(
            f"{variant} predecessor chain is invalid: {error}"
        ) from error
    if position == 0:
        _exact(
            predecessor.get("state"),
            "explicit_genesis_no_predecessor",
            label="R genesis state",
        )
        _exact(
            predecessor.get("receipt_artifact"), None, label="R predecessor receipt"
        )
    else:
        _exact(
            predecessor.get("state"),
            "validated_successful_predecessor",
            label=f"{variant} predecessor state",
        )
        claim = _mapping(
            predecessor.get("receipt_artifact"),
            label=f"{variant} predecessor receipt claim",
        )
        _exact(
            claim.get("path"),
            str(previous_receipt_path),
            label=f"{variant} predecessor receipt path",
        )
    return manifest, summary, dict(receipt_snapshot), retained_artifacts


def validate_scale_up_panel(
    terminal_receipt_path: Path,
    *,
    registry: verifier.ValidatedScaleUpRegistry,
    expected_source_revision: str | None = None,
) -> dict[str, object]:
    """Validate and normalize the exact terminal-E scale-up receipt chain."""

    current_revision = pilot.require_pushed_commit()
    if expected_source_revision is not None:
        expected_source_revision = verifier._revision(
            expected_source_revision, "expected scale-up source revision"
        )
        _exact(
            current_revision,
            expected_source_revision,
            label="scale-up source revision",
        )
    launcher.require_registry_publication(
        registry, current_revision=current_revision
    )
    paths = launcher.run_paths(registry)
    receipt_path = _validated_receipt_path(Path(terminal_receipt_path))
    expected_terminal = paths[-1][2]
    _exact(receipt_path, expected_terminal, label="terminal E receipt path")

    records: list[
        tuple[
            Path,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            tuple[evidence_writer.StableArtifact, ...],
        ]
    ] = []
    for position, (_member, _run_dir, member_receipt_path) in enumerate(paths):
        member_receipt_path = _validated_receipt_path(member_receipt_path)
        receipt, snapshot = _read_json(
            member_receipt_path,
            label=f"{verifier.EXPECTED_SLUGS[position].upper()} scale-up receipt",
        )
        manifest, summary, retained_snapshot, retained_artifacts = _validate_member(
            registry=registry,
            source_revision=current_revision,
            position=position,
            receipt_path=member_receipt_path,
            receipt=receipt,
            receipt_snapshot=snapshot,
            previous_receipt_path=(None if position == 0 else paths[position - 1][2]),
        )
        if retained_snapshot != snapshot:
            raise ScaleUpPanelValidationError("receipt snapshot changed during validation")
        records.append(
            (
                member_receipt_path,
                receipt,
                snapshot,
                manifest,
                summary,
                retained_artifacts,
            )
        )

    # Close the read/validation race across every receipt and joined artifact.
    for position, (
        path,
        original,
        snapshot,
        _manifest,
        _summary,
        retained_artifacts,
    ) in enumerate(records):
        retained, retained_snapshot = _read_json(
            path,
            label=f"retained {verifier.EXPECTED_SLUGS[position].upper()} receipt",
        )
        if not screen._exact_json_equal(retained, original) or not (
            screen._exact_json_equal(retained_snapshot, snapshot)
        ):
            raise ScaleUpPanelValidationError(
                "scale-up receipt changed during panel validation"
            )
        for artifact in retained_artifacts:
            current = evidence_writer.read_stable_regular_file(
                artifact.path,
                label=(
                    f"retained {verifier.EXPECTED_SLUGS[position].upper()} "
                    f"{artifact.path.name}"
                ),
                capture_payload=artifact.payload is not None,
                project_scope=artifact.project_scope,
            )
            if (
                not screen._exact_json_equal(current.snapshot(), artifact.snapshot())
                or current.payload != artifact.payload
            ):
                raise ScaleUpPanelValidationError(
                    "scale-up joined artifact changed during panel validation"
                )

    receipt_members: list[dict[str, object]] = []
    checkpoint_members: list[dict[str, object]] = []
    for position, (
        path,
        receipt,
        snapshot,
        manifest,
        _summary,
        _retained_artifacts,
    ) in enumerate(records):
        member = registry.data["members"][position]
        checkpoint = _mapping(
            receipt.get("final_checkpoint"),
            label=f"{member['training_variant']} checkpoint evidence",
        )
        checkpoint_snapshot = _mapping(
            checkpoint.get("artifact"),
            label=f"{member['training_variant']} checkpoint snapshot",
        )
        checkpoint_path = Path(str(checkpoint_snapshot["path"]))
        expected_checkpoint = (
            path.parent
            / "checkpoints"
            / f"{verifier.EXPECTED_MAX_STEPS}.ckpt"
        )
        _exact(
            checkpoint_path,
            expected_checkpoint,
            label=f"{member['training_variant']} checkpoint path",
        )
        receipt_members.append(
            {
                "position": position,
                "arm_id": verifier.EXPECTED_SLUGS[position].upper(),
                "training_variant": member["training_variant"],
                "run_name": member["run_name"],
                "relative_path": _relative(path, label="member receipt"),
                "sha256": snapshot["sha256"],
                "size_bytes": snapshot["size_bytes"],
                "schema_version": receipt["schema_version"],
                "recorded_at_utc": receipt["recorded_at_utc"],
            }
        )
        checkpoint_members.append(
            {
                "position": position,
                "arm_id": verifier.EXPECTED_SLUGS[position].upper(),
                "training_variant": member["training_variant"],
                "run_name": manifest["run_name"],
                "relative_path": _relative(
                    checkpoint_path, label="member checkpoint"
                ),
                "sha256": checkpoint_snapshot["sha256"],
                "size_bytes": checkpoint_snapshot["size_bytes"],
                "global_step": verifier.EXPECTED_MAX_STEPS,
            }
        )

    terminal_receipt, terminal_snapshot, terminal_manifest = (
        records[-1][1],
        records[-1][2],
        records[-1][3],
    )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "validated",
        "claim_scope": "selection_bound_training_health_and_provenance_only",
        "registry": registry.reference,
        "source_revision": current_revision,
        "gpu_count": registry.data["common_training"]["gpu_count"],
        "optimizer_updates_per_member": verifier.EXPECTED_MAX_STEPS,
        "selected_scheduler_arm_id": registry.scheduler_selection["selected_arm_id"],
        "selected_conditioning_arm_id": registry.conditioning_selection[
            "selected_arm_id"
        ],
        "matched_panel_spec_sha256": terminal_manifest[
            "matched_panel_spec_sha256"
        ],
        "terminal_receipt": {
            "root": "repository",
            "relative_path": _relative(receipt_path, label="terminal receipt"),
            "sha256": terminal_snapshot["sha256"],
            "size_bytes": terminal_snapshot["size_bytes"],
            "schema_version": terminal_receipt["schema_version"],
            "canonical_sha256": evidence_writer.canonical_json_sha256(
                terminal_receipt
            ),
            "training_variant": verifier.EXPECTED_VARIANTS[-1],
            "position": 2,
        },
        "receipt_members": receipt_members,
        "checkpoint_members": checkpoint_members,
        "eligibility": {
            "generation": False,
            "ranking": False,
            "superiority": False,
            "candidate_lock": False,
            "registered_evaluation_input": True,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-canonical-sha256", required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--expected-source-revision")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    registry = launcher.load_registry(
        args.registry,
        expected_raw_sha256=args.expected_registry_sha256,
        expected_canonical_sha256=args.expected_registry_canonical_sha256,
    )
    result = validate_scale_up_panel(
        args.terminal_receipt,
        registry=registry,
        expected_source_revision=args.expected_source_revision,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
