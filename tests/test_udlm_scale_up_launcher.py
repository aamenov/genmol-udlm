from __future__ import annotations

import hashlib

import pytest

from scripts.udlm import launch_scale_up_panel as launcher
from scripts.udlm import launch_train_pilot as pilot
from scripts.udlm import verify_scale_up_registry as scale


def _gpu(index: int, *, utilization: int, free_mib: int) -> pilot.GPUState:
    total = 80_000
    return pilot.GPUState(
        physical_index=index,
        uuid=f"GPU-{index}",
        name="test",
        memory_used_mib=total - free_mib,
        memory_total_mib=total,
        utilization_percent=utilization,
        compute_mode="Default",
        compute_processes=({"pid": 10, "process_name": "peer", "used_memory_mib": 1},),
    )


def test_dynamic_selector_supports_three_and_uses_strict_utilization_threshold() -> None:
    states = [
        _gpu(0, utilization=10, free_mib=79_000),
        _gpu(1, utilization=9, free_mib=70_000),
        _gpu(2, utilization=0, free_mib=60_000),
        _gpu(3, utilization=1, free_mib=50_000),
    ]
    selected = launcher._select_idle_gpus(
        states,
        gpu_count=3,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )
    assert [state.physical_index for state in selected] == [1, 2, 3]
    assert all(state.physical_index != 0 for state in selected)


def test_dynamic_selector_supports_four_gpus() -> None:
    states = [_gpu(index, utilization=index, free_mib=70_000 - index) for index in range(4)]
    selected = launcher._select_idle_gpus(
        states,
        gpu_count=4,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )
    assert len(selected) == 4
    assert {state.uuid for state in selected} == {f"GPU-{index}" for index in range(4)}


def _registry(payload: bytes) -> scale.ValidatedScaleUpRegistry:
    return scale.ValidatedScaleUpRegistry(
        data={
            "publication": {"config_revision": "a" * 40},
            "common_training": {"gpu_count": 2},
        },
        relative_path=scale.PurePosixPath(
            "experiments/udlm/protocols/selection_bound_scale_up_registry_gpu2.json"
        ),
        raw_sha256=hashlib.sha256(payload).hexdigest(),
        raw_size_bytes=len(payload),
        canonical_sha256="c" * 64,
        screen_registry=None,  # type: ignore[arg-type]
        scheduler_selection={},
        conditioning_selection={},
    )


def test_registry_launch_requires_exact_registry_only_r6_transition(monkeypatch) -> None:
    payload = b"registry-bytes"
    registry = _registry(payload)
    path = registry.relative_path.as_posix()
    monkeypatch.setattr(launcher.screen, "git_ancestor_checker", lambda a, b: True)
    monkeypatch.setattr(launcher.screen, "git_sole_parent_checker", lambda a, b: True)
    monkeypatch.setattr(launcher.screen, "git_pushed_checker", lambda revision: True)
    monkeypatch.setattr(
        launcher.screen, "git_diff_checker", lambda a, b, allowed: allowed == {path}
    )
    monkeypatch.setattr(
        launcher.verifier,
        "git_changed_paths_loader",
        lambda a, b: frozenset({path}),
    )
    monkeypatch.setattr(launcher.verifier, "_git_blob_absent", lambda *a, **k: None)
    monkeypatch.setattr(launcher.screen, "git_blob_loader", lambda revision, p: payload)
    launcher.require_registry_publication(registry, current_revision="b" * 40)

    monkeypatch.setattr(
        launcher.verifier,
        "git_changed_paths_loader",
        lambda a, b: frozenset({path, "scripts/train.py"}),
    )
    with pytest.raises(scale.ScaleUpValidationError, match="changed paths"):
        launcher.require_registry_publication(registry, current_revision="b" * 40)


def test_registry_launch_rejects_committed_blob_mismatch(monkeypatch) -> None:
    registry = _registry(b"registry-bytes")
    path = registry.relative_path.as_posix()
    monkeypatch.setattr(launcher.screen, "git_ancestor_checker", lambda a, b: True)
    monkeypatch.setattr(launcher.screen, "git_sole_parent_checker", lambda a, b: True)
    monkeypatch.setattr(launcher.screen, "git_pushed_checker", lambda revision: True)
    monkeypatch.setattr(launcher.screen, "git_diff_checker", lambda a, b, allowed: True)
    monkeypatch.setattr(
        launcher.verifier,
        "git_changed_paths_loader",
        lambda a, b: frozenset({path}),
    )
    monkeypatch.setattr(launcher.verifier, "_git_blob_absent", lambda *a, **k: None)
    monkeypatch.setattr(
        launcher.screen, "git_blob_loader", lambda revision, p: b"tampered"
    )
    with pytest.raises(scale.ScaleUpValidationError, match="blob differs"):
        launcher.require_registry_publication(registry, current_revision="b" * 40)


def test_scale_launch_mutations_reject_swapped_output_parent(
    tmp_path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    udlm = repository / "output" / "udlm"
    logs = repository / "output" / "logs"
    outside = tmp_path / "outside"
    udlm.mkdir(parents=True)
    logs.mkdir()
    outside.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(launcher.preparer, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(pilot, "REPOSITORY_ROOT", repository)

    pilot.validate_pilot_output_parents(
        run_dir=launcher.REPOSITORY_ROOT / "output" / "udlm" / "scale-test",
        log_path=launcher.REPOSITORY_ROOT / "output" / "logs" / "scale-test.log",
        create_missing=False,
    )
    udlm.rename(repository / "output" / "udlm-original")
    udlm.symlink_to(outside, target_is_directory=True)
    run_dir = udlm / "scale-test"
    with pytest.raises(RuntimeError, match="direct real directory"):
        launcher._create_direct_directory_exclusive(
            run_dir, label="scale-up run directory"
        )
    with pytest.raises(launcher.preparer.ScaleUpPreparationError):
        launcher._safe_publish_bytes_exclusive(
            run_dir / "launch_manifest.json", b"{}\n", label="manifest"
        )
    with pytest.raises(launcher.preparer.ScaleUpPreparationError):
        launcher._acquire_scale_up_training_job_lock(
            source_revision="a" * 40,
            run_name="scale-test",
            training_variant="udlm",
        )
    assert not (outside / "scale-test").exists()
    assert not (outside / ".single_training_job.lock").exists()

    logs.rename(repository / "output" / "logs-original")
    logs.symlink_to(outside, target_is_directory=True)
    with pytest.raises(launcher.preparer.ScaleUpPreparationError):
        launcher._safe_publish_bytes_exclusive(
            logs / "scale-test.log", b"", label="scale-up log"
        )
    assert not (outside / "scale-test.log").exists()
