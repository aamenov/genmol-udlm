from __future__ import annotations

import copy

import pytest

from scripts.udlm import validate_scale_up_panel as panel


def _state(index: int, *, utilization: int = 9) -> dict[str, object]:
    return {
        "physical_index": index,
        "uuid": f"GPU-{index}",
        "name": "test",
        "memory_used_mib": 10_000,
        "memory_total_mib": 80_000,
        "utilization_percent": utilization,
        "compute_mode": "Default",
        "compute_processes": [
            {"pid": 100 + index, "process_name": "peer", "used_memory_mib": 100}
        ],
    }


def _manifest() -> dict[str, object]:
    initial = [_state(2), _state(4)]
    final = copy.deepcopy(initial)
    return {
        "gpu_selection_schema_version": 2,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": "2026-09-06T00:00:00+00:00",
        "final_uuid_probes_completed_at_utc": "2026-09-06T00:00:01+00:00",
        "created_at": "2026-09-06T00:00:02+00:00",
        "gpu_inventory_at_selection": copy.deepcopy(initial),
        "initially_selected_gpu_states": initial,
        "gpu_states_at_final_uuid_probe": final,
        "logical_cuda_devices": [0, 1],
        "cuda_visible_device_uuids": ["GPU-2", "GPU-4"],
        "physical_gpu_indices": [2, 4],
    }


def test_gpu_evidence_accepts_active_process_telemetry_below_ten_percent() -> None:
    panel._validate_gpu_states(_manifest(), gpu_count=2, variant="udlm")


def test_gpu_evidence_rejects_utilization_equal_to_ten_percent() -> None:
    manifest = _manifest()
    manifest["gpu_states_at_final_uuid_probe"][0]["utilization_percent"] = 10
    with pytest.raises(panel.ScaleUpPanelValidationError, match="idle policy"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")


def test_gpu_evidence_rejects_visible_uuid_reordering() -> None:
    manifest = _manifest()
    manifest["cuda_visible_device_uuids"] = ["GPU-4", "GPU-2"]
    with pytest.raises(panel.ScaleUpPanelValidationError, match="GPU UUID order"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")


def test_gpu_evidence_rejects_initial_final_identity_change() -> None:
    manifest = _manifest()
    manifest["gpu_states_at_final_uuid_probe"][0]["physical_index"] = 8
    manifest["physical_gpu_indices"] = [8, 4]
    with pytest.raises(
        panel.ScaleUpPanelValidationError,
        match="initial/final GPU identity sequence",
    ):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("memory_used_mib", -1),
        ("compute_mode", " Prohibited "),
    ),
)
def test_gpu_evidence_rejects_invalid_terminal_state_fields(
    field: str, value: object
) -> None:
    manifest = _manifest()
    manifest["gpu_states_at_final_uuid_probe"][0][field] = value
    with pytest.raises(panel.ScaleUpPanelValidationError):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")


def test_gpu_evidence_rejects_extra_state_and_process_fields() -> None:
    for target in ("gpu_states_at_final_uuid_probe", "gpu_inventory_at_selection"):
        manifest = _manifest()
        manifest[target][0]["unexpected"] = True
        with pytest.raises(panel.ScaleUpPanelValidationError, match="state/process"):
            panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")

    manifest = _manifest()
    manifest["gpu_states_at_final_uuid_probe"][0]["compute_processes"][0][
        "unexpected"
    ] = True
    with pytest.raises(panel.ScaleUpPanelValidationError, match="state/process"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")


@pytest.mark.parametrize("mutation", ("null_memory", "duplicate_pid"))
def test_gpu_evidence_rejects_nonproducer_process_rows(mutation: str) -> None:
    manifest = _manifest()
    for field in (
        "gpu_inventory_at_selection",
        "initially_selected_gpu_states",
        "gpu_states_at_final_uuid_probe",
    ):
        process_rows = manifest[field][0]["compute_processes"]
        if mutation == "null_memory":
            process_rows[0]["used_memory_mib"] = None
        else:
            process_rows.append(copy.deepcopy(process_rows[0]))
    with pytest.raises(panel.ScaleUpPanelValidationError, match="state/process"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")


def test_gpu_evidence_rejects_inventory_mapping_and_chronology_changes() -> None:
    manifest = _manifest()
    manifest["logical_cuda_devices"] = [1, 0]
    with pytest.raises(panel.ScaleUpPanelValidationError, match="logical CUDA"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")

    manifest = _manifest()
    manifest["initially_selected_gpu_states"][0]["name"] = "different"
    with pytest.raises(panel.ScaleUpPanelValidationError, match="inventory rows"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")

    manifest = _manifest()
    manifest["final_uuid_probes_completed_at_utc"] = "2026-09-06T00:00:03+00:00"
    with pytest.raises(panel.ScaleUpPanelValidationError, match="chronology"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")


def test_gpu_evidence_replays_deterministic_selection_and_positive_memory() -> None:
    manifest = _manifest()
    better = _state(0, utilization=0)
    better["memory_used_mib"] = 0
    manifest["gpu_inventory_at_selection"].insert(0, better)
    with pytest.raises(panel.ScaleUpPanelValidationError, match="deterministic GPU"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")

    manifest = _manifest()
    manifest["gpu_inventory_at_selection"][0]["memory_used_mib"] = 0
    manifest["gpu_inventory_at_selection"][0]["memory_total_mib"] = 0
    with pytest.raises(panel.ScaleUpPanelValidationError, match="inventory identities"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")


def test_gpu_evidence_rejects_nonexact_method_and_boolean_schema() -> None:
    manifest = _manifest()
    manifest["gpu_selection_method"] = "index_zero"
    with pytest.raises(panel.ScaleUpPanelValidationError, match="selection method"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")

    manifest = _manifest()
    manifest["gpu_selection_schema_version"] = True
    with pytest.raises(panel.ScaleUpPanelValidationError, match="selection schema"):
        panel._validate_gpu_states(manifest, gpu_count=2, variant="udlm")
