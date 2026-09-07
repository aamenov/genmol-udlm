"""CPU checks for the bounded throughput launch plan and shared GPU exclusion."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from scripts import artifact_io
from scripts.udlm import launch_engineering_training as launcher
from scripts.udlm.launch_train_pilot import GPUState


GENERATION_LEASE = "output/.single_generation_job.lock"
TRAINING_LEASE = "output/udlm/.single_training_job.lock"


def _gpu(*, index=2, utilization=9, free_mib=30_000, processes=()):
    return GPUState(
        physical_index=index,
        uuid=f"GPU-fixture-{index}",
        name="NVIDIA A6000",
        memory_used_mib=49_140 - free_mib,
        memory_total_mib=49_140,
        utilization_percent=utilization,
        compute_mode="Default",
        compute_processes=processes,
    )


@pytest.mark.parametrize("count", [0, 3, -1, True, False, 1.5, "1", None])
def test_plan_and_selection_reject_invalid_gpu_counts(count):
    with pytest.raises((ValueError, TypeError)):
        launcher.build_plan(count)
    with pytest.raises((ValueError, TypeError)):
        launcher.select_gpus([_gpu()], count)


@pytest.mark.parametrize("count,accumulation", [(1, 8), (2, 4)])
def test_resolved_plan_retains_global_batch_and_fresh_mdlm_ema_start(
    count, accumulation
):
    plan = launcher.build_plan(count)
    config = plan["config"]
    assert plan["gpu_count"] == config["trainer"]["devices"] == count
    assert config["trainer"]["num_nodes"] == 1
    assert config["trainer"]["max_steps"] == 20
    assert config["seed"] == 1400
    assert config["loader"]["global_batch_size"] == 128
    assert config["loader"]["batch_size"] == 16
    assert config["trainer"]["accumulate_grad_batches"] == accumulation
    assert count * 16 * accumulation == 128
    assert config["training"]["diffusion"] == "udlm"
    assert config["training"]["udlm"]["prior_variant"] == "empirical_frequency"
    assert config["training"]["udlm"]["empirical_uniform_mix"] == 0.0002
    assert config["training"]["udlm"]["exclude_special_tokens"] is False
    assert config["training"]["udlm"]["conditioning_variant"] == "film_adaln"
    assert config["training"]["init_from_mdlm_ema"] is True
    assert Path(config["training"]["init_from_mdlm_checkpoint"]).name == "50000.ckpt"
    assert len(config["training"]["init_from_mdlm_checkpoint_sha256"]) == 64
    assert config["trainer"]["detect_anomaly"] is False
    assert config["training"]["reseed_after_model_initialization"] is False
    assert config["training"]["pilot_fail_on_nonfinite_loss"] is False
    assert config["callback"]["every_n_train_steps"] == 20
    assert Path(plan["output_relative"]).is_relative_to("output/udlm")
    assert Path(plan["log_relative"]).is_relative_to("output/logs")
    argv = plan["training_argv"]
    assert argv[1] == "-u" and Path(argv[2]).name == "train.py"
    assert argv[argv.index("--config-name") + 1] == "udlm_e_throughput20"
    overrides = argv[argv.index("--config-name") + 2 :]
    assert f"trainer.devices={count}" in overrides
    assert f"trainer.accumulate_grad_batches={accumulation}" in overrides
    assert not any("resume" in item or "ckpt_path=" in item for item in overrides)


def test_plan_does_not_inherit_mutation_from_previous_composition():
    first = launcher.build_plan(1)
    first["config"]["training"]["udlm"]["prior_variant"] = "forged"
    first["config"]["loader"]["global_batch_size"] = 1
    second = launcher.build_plan(2)
    assert (
        second["config"]["training"]["udlm"]["prior_variant"] == "empirical_frequency"
    )
    assert second["config"]["loader"]["global_batch_size"] == 128
    assert second["config"]["trainer"]["accumulate_grad_batches"] == 4


def test_nine_percent_with_active_compute_and_exact_memory_floor_is_eligible():
    active = _gpu(processes=({"pid": 123, "used_memory_mib": 19_140},))
    ten_percent = _gpu(index=5, utilization=10, free_mib=49_000)
    selected = launcher.select_gpus([ten_percent, active], 1)
    assert list(selected) == [active]
    assert selected[0].physical_index == 2
    assert selected[0].compute_processes == active.compute_processes


@pytest.mark.parametrize(
    "state",
    [
        _gpu(utilization=10),
        _gpu(utilization=11),
        _gpu(free_mib=29_999),
        replace(_gpu(), compute_mode="Prohibited"),
    ],
)
def test_rejected_gpu_never_satisfies_requested_capacity(state):
    with pytest.raises((ValueError, RuntimeError)):
        launcher.select_gpus([state], 1)


def test_two_gpu_request_requires_two_distinct_eligible_devices():
    allowed = [_gpu(index=2), _gpu(index=6, utilization=0)]
    rejected = _gpu(index=0, utilization=10)
    selected = launcher.select_gpus([rejected, *allowed], 2)
    assert {state.uuid for state in selected} == {state.uuid for state in allowed}
    with pytest.raises((ValueError, RuntimeError)):
        launcher.select_gpus([allowed[0], rejected], 2)


@pytest.mark.parametrize("duplicate_field", ["uuid", "physical_index"])
def test_duplicate_gpu_identity_is_rejected(duplicate_field):
    first, second = _gpu(index=2), _gpu(index=6)
    second = replace(second, **{duplicate_field: getattr(first, duplicate_field)})
    with pytest.raises((ValueError, RuntimeError)):
        launcher.select_gpus([first, second], 2)


def _lease_root(tmp_path):
    (tmp_path / "output/udlm").mkdir(parents=True)
    return tmp_path


def _identity():
    return {"purpose": "cpu-test", "source_revision": "a" * 40, "run_id": "test"}


def test_both_leases_are_owned_and_exactly_released(tmp_path):
    root = _lease_root(tmp_path)
    claims = launcher.acquire_leases(root, _identity())
    assert len(claims) == 2
    assert all(isinstance(claim, artifact_io.FileClaim) for claim in claims)
    assert {claim.relative_path for claim in claims} == {
        GENERATION_LEASE,
        TRAINING_LEASE,
    }
    for claim in claims:
        path = root / claim.relative_path
        assert path.stat().st_ino == claim.inode
        assert path.read_bytes()
    launcher.release_leases(root, claims)
    assert not (root / GENERATION_LEASE).exists()
    assert not (root / TRAINING_LEASE).exists()


@pytest.mark.parametrize("occupied", [GENERATION_LEASE, TRAINING_LEASE])
def test_existing_generation_or_training_lease_blocks_launch_without_clobber(
    tmp_path, occupied
):
    root = _lease_root(tmp_path)
    existing = root / occupied
    existing.write_bytes(b"preexisting owner\n")
    before = existing.stat()
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        launcher.acquire_leases(root, _identity())
    assert existing.read_bytes() == b"preexisting owner\n"
    assert existing.stat().st_ino == before.st_ino
    other = TRAINING_LEASE if occupied == GENERATION_LEASE else GENERATION_LEASE
    assert not (
        root / other
    ).exists(), "a failed pair acquisition must roll back its own lease"


@pytest.mark.parametrize("occupied", [GENERATION_LEASE, TRAINING_LEASE])
def test_symlink_lease_collision_preserves_target_and_rolls_back_owned_peer(
    tmp_path, occupied
):
    root = _lease_root(tmp_path)
    target = root / "foreign-owner.json"
    target.write_bytes(b"foreign content\n")
    path = root / occupied
    path.symlink_to(target)
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        launcher.acquire_leases(root, _identity())
    assert path.is_symlink()
    assert target.read_bytes() == b"foreign content\n"
    other = TRAINING_LEASE if occupied == GENERATION_LEASE else GENERATION_LEASE
    assert not (root / other).exists()


@pytest.mark.parametrize("replaced", [GENERATION_LEASE, TRAINING_LEASE])
def test_release_preserves_same_bytes_replacement_inode(tmp_path, replaced):
    root = _lease_root(tmp_path)
    claims = launcher.acquire_leases(root, _identity())
    path = root / replaced
    owned_payload = path.read_bytes()
    path.rename(root / "old-owned-lock")
    path.write_bytes(owned_payload)
    replacement_inode = path.stat().st_ino
    with pytest.raises((artifact_io.OwnershipChangedError, RuntimeError)):
        launcher.release_leases(root, claims)
    assert path.read_bytes() == owned_payload
    assert path.stat().st_ino == replacement_inode


def test_dry_run_never_queries_gpus_acquires_leases_or_starts_training(
    monkeypatch, capsys
):
    source = {"head": "a" * 40, "upstream": "a" * 40}
    monkeypatch.setattr(
        launcher.benchmark, "_require_clean_pushed_source", lambda: source
    )
    monkeypatch.setattr(launcher, "verify_checkpoint_input", lambda plan: {})

    def forbidden(*args, **kwargs):
        pytest.fail("CPU dry-run touched a live execution resource")

    monkeypatch.setattr(launcher.audited, "probe_all_gpus", forbidden)
    monkeypatch.setattr(launcher.audited, "probe_gpu_uuid", forbidden)
    monkeypatch.setattr(launcher, "acquire_leases", forbidden)
    monkeypatch.setattr(launcher, "execute", forbidden)
    monkeypatch.setattr(launcher.benchmark, "_require_tmux_for_execution", forbidden)
    assert launcher.main(["--gpu-count", "2", "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["gpu_queries"] == preview["artifact_mutations"] == 0
    assert preview["source"] == source
    assert preview["plan"]["gpu_count"] == 2
    assert preview["plan"]["config"]["loader"]["global_batch_size"] == 128


def test_child_environment_exposes_only_selected_uuids_without_stale_rank(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-old-physical-zero")
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "GENMOL_TRAIN_STALE",
        "SLURM_JOB_ID",
    ):
        monkeypatch.setenv(name, "stale")
    env = launcher.child_environment(["GPU-fixture-2", "GPU-fixture-6"])
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-fixture-2,GPU-fixture-6"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["PYTHONHASHSEED"] == "1400"
    assert env["PYTHONNOUSERSITE"] == "1"
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "GENMOL_TRAIN_STALE",
        "SLURM_JOB_ID",
    ):
        assert name not in env


def test_final_recheck_records_rejected_uuid_before_refusing_launch(monkeypatch):
    selected = (_gpu(index=2), _gpu(index=6))
    monkeypatch.setattr(
        launcher.audited,
        "probe_gpu_uuid",
        lambda uuid: replace(
            next(state for state in selected if state.uuid == uuid),
            utilization_percent=10,
        ),
    )
    events = []
    with pytest.raises(RuntimeError, match="final GPU probe rejected"):
        launcher.recheck_gpus(selected, events.append)
    assert len(events) == 2
    assert {event["requested_uuid"] for event in events} == {
        state.uuid for state in selected
    }
    assert all(event["gpus"][0]["utilization_percent"] == 10 for event in events)


@pytest.mark.parametrize(
    "outcome", ["valid", "missing_checkpoint", "nonzero", "nonfinite"]
)
def test_controller_cpu_child_terminal_evidence_and_resource_release(
    tmp_path, monkeypatch, outcome
):
    plan = launcher.build_plan(1)
    source = {"head": "a" * 40, "upstream": "a" * 40}
    plan["config"] = {"test_configuration": "actual CPU child"}
    checkpoint = tmp_path / plan["output_relative"] / "checkpoints/20.ckpt"
    child = """
import json, os, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
outcome, checkpoint, config = sys.argv[1:]
if outcome == 'nonzero':
    raise SystemExit(7)
if outcome != 'missing_checkpoint':
    import torch
    from omegaconf import OmegaConf
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True)
    weight = torch.tensor([float('nan') if outcome == 'nonfinite' else 1.0], device='cpu')
    torch.save({
        'global_step': 20,
        'hyper_parameters': {'config': OmegaConf.create(json.loads(config))},
        'state_dict': {'weight': weight},
        'optimizer_states': [{'state': {0: {'moment': torch.zeros(1, device='cpu')}}}],
        'ema': {'num_updates': 20, 'shadow_params': [torch.ones(1, device='cpu')]},
    }, checkpoint)
print('CPU integration child completed')
"""
    plan["training_argv"] = [
        sys.executable,
        "-c",
        child,
        outcome,
        str(checkpoint),
        json.dumps(plan["config"]),
    ]
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "build_plan", lambda count: plan)
    monkeypatch.setattr(
        launcher, "verify_checkpoint_input", lambda _plan: {"sha256": "b" * 64}
    )
    monkeypatch.setattr(
        launcher.benchmark, "_require_clean_pushed_source", lambda: source
    )
    monkeypatch.setattr(launcher.audited, "probe_all_gpus", lambda: [_gpu()])
    monkeypatch.setattr(launcher.audited, "probe_gpu_uuid", lambda _uuid: _gpu())
    result = launcher.execute(plan, source)
    directory = tmp_path / plan["output_relative"]
    terminal = json.loads((directory / "terminal_manifest.json").read_text())
    launch = json.loads((directory / "launch_manifest.json").read_text())
    assert result == (0 if outcome == "valid" else 1)
    assert terminal["status"] == ("completed" if outcome == "valid" else "failed")
    assert terminal["training_return_code"] == (7 if outcome == "nonzero" else 0)
    assert terminal["training_subprocess_seconds"] > 0
    assert terminal["leases_release_authorized"] is True
    assert not (tmp_path / GENERATION_LEASE).exists()
    assert not (tmp_path / TRAINING_LEASE).exists()
    assert launch["selected_gpu_uuids"] == [_gpu().uuid]
    assert len(launch["probes"]) == 2
    telemetry = [
        json.loads(line)
        for line in (directory / "gpu_telemetry.jsonl").read_text().splitlines()
    ]
    assert {event["phase"] for event in telemetry} >= {
        "selection",
        "immediately_before_launch",
    }
    for relative, digest in terminal["artifact_hashes"].items():
        assert artifact_io.snapshot_file(tmp_path, relative)[0].sha256 == digest
    if outcome == "valid":
        assert terminal["checkpoint"]["global_step"] == 20
        assert terminal["checkpoint"]["finite_tensor_count"] == 3
        assert terminal["completed_example_exposures"] == 2560
        assert terminal["end_to_end_training_examples_per_second"] > 0
    elif outcome == "nonfinite":
        assert "non-finite tensors" in terminal["error"]
    if outcome != "valid":
        assert terminal["completed_example_exposures"] is None
        assert terminal["end_to_end_training_examples_per_second"] is None
    with pytest.raises(FileExistsError):
        launcher.execute(plan, source)
