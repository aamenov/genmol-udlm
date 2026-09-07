"""Synthetic CPU campaign lifecycle tests; no GPU inventory or oracle calls."""

from __future__ import annotations

import copy
import hashlib
import json
import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from scripts.udlm import launch_pmo_campaign as controller
from test_udlm_denoiser_benchmark import _checkpoint


SOURCE = {"head": "a" * 40, "upstream": "a" * 40}


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller.resources, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        controller.resources, "_require_clean_pushed_source", lambda: SOURCE
    )
    for directory in ("scripts", "inputs", "oracle"):
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts/artifact_io.py").write_text("synthetic pinned source\n")
    (tmp_path / "oracle/gsk3b_current.pkl").write_bytes(
        b"synthetic oracle file; never loaded"
    )
    (tmp_path / "inputs/vocabulary.csv").write_text(
        "frag,score,size\n[1*]CC,0.9,2\n[1*]CO,0.8,2\n"
    )
    checkpoint = _checkpoint()
    torch.save(checkpoint, tmp_path / "inputs/model.ckpt")

    def digest(path):
        return hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()

    config = {
        "checkpoint_sha256": digest("inputs/model.ckpt"),
        "diffusion_type": "udlm",
        "parameterization": "x0_denoiser",
        "softmax_temp": 0.5,
        "randomness": 0,
        "min_add_len": 18,
        "num_steps": 128,
        "inference_eps": 1e-5,
        "exclude_special_tokens": False,
        "prior_variant": "schedule_uniform",
        "prior_metadata_sha256": controller.benchmark._canonical_json_sha256(
            checkpoint["udlm_prior_metadata"]
        ),
        "raw_loo_top_p": 1.0,
    }
    (tmp_path / "inputs/sampling.yaml").write_text(yaml.safe_dump(config))
    entry = {
        "id": "arm_seed2300",
        "seed": 2300,
        "checkpoint": "inputs/model.ckpt",
        "checkpoint_sha256": digest("inputs/model.ckpt"),
        "sampling_config": "inputs/sampling.yaml",
        "sampling_config_sha256": digest("inputs/sampling.yaml"),
        "vocabulary": "inputs/vocabulary.csv",
        "vocabulary_sha256": digest("inputs/vocabulary.csv"),
        "max_oracle_calls": 2000,
    }
    panel = {
        "schema_version": 1,
        "campaign_id": "synthetic",
        "oracle": "gsk3b",
        "scientific_status": "synthetic CPU controller test; no benchmark evidence",
        "output_root": "output/udlm/synthetic_pmo",
        "log_root": "output/logs/synthetic_pmo",
        "child_timeout_seconds": 10,
        "capacity_wait_seconds": 0,
        "runner": {
            "max_iterations": 5000,
            "population_size": 100,
            "warmup": 1000,
            "legacy_warmup_off_by_one": True,
            "reporting_frequency": 100,
            "checkpoint_every": 100,
            "guidance_scale": 2,
            "min_mol_size": 20,
            "max_mol_size": 40,
        },
        "entries": [entry],
        "input_files": [
            {
                "path": "oracle/gsk3b_current.pkl",
                "sha256": digest("oracle/gsk3b_current.pkl"),
            }
        ],
    }

    def write(value=None):
        (tmp_path / "panel.json").write_bytes(
            controller.encode(panel if value is None else value)
        )
        return digest("panel.json")

    def plan(count=1):
        return controller.build_plan("panel.json", write(), count)

    return SimpleNamespace(
        root=tmp_path, panel=panel, write=write, plan=plan, digest=digest
    )


def gpu(index=7, *, utilization=9, free=30000, uuid=None, processes=()):
    return controller.resources.GPUState(
        index=index,
        uuid=uuid or f"GPU-synthetic-{index}",
        name="synthetic",
        memory_used_mib=80000 - free,
        memory_total_mib=80000,
        utilization_percent=utilization,
        compute_mode="Default",
        compute_processes=processes,
    )


@pytest.mark.parametrize("count", [True, False, 0, 3, 1.5])
def test_gpu_count_rejects_boolean_or_out_of_scope(study, count):
    with pytest.raises(ValueError, match="gpu-count"):
        study.plan(count)


def test_admission_uses_strict_threshold_and_allows_existing_processes():
    active = gpu(6, processes=({"pid": 999, "used_memory_mib": 100},))
    assert controller.select_gpus(
        [gpu(2, utilization=10), gpu(3, free=29999), active], 1
    ) == [active]
    with pytest.raises(ValueError, match="duplicate"):
        controller.select_gpus([active, gpu(5, uuid=active.uuid)], 2)
    with pytest.raises(ValueError, match="duplicate"):
        controller.select_gpus([active, gpu(6, uuid="GPU-other")], 2)


def test_build_plan_uses_real_cli_and_checkpoint_metadata(study):
    plan = study.plan(2)
    job = plan["jobs"][0]
    command = job["command"]
    assert command[command.index("--oracle") + 1] == "gsk3b"
    assert command[command.index("--gamma") + 1] == "0"
    assert command[command.index("--variant") + 1] == "released"
    assert "--resume" not in command
    assert "--legacy-warmup-off-by-one" in command
    assert "--durable-events" in command
    assert job["config"]["softmax_temp"] == 0.5
    assert job["config"]["max_oracle_calls"] == 2000
    assert (
        job["checkpoint"]["udlm_denoiser_metadata"]
        == controller.benchmark.UDLM_DENOISER_METADATA
    )
    assert job["checkpoint"]["byte_identity_verified_before_and_after_load"] is True
    assert len(plan["inputs"]) == 5
    assert job["run_relative"].endswith("arm_seed2300/gsk3b/released/seed_2300")


@pytest.mark.parametrize(
    "alter",
    [
        lambda panel: panel.update(child_timeout_seconds=float("inf")),
        lambda panel: panel.update(child_timeout_seconds=True),
        lambda panel: panel.update(capacity_wait_seconds=21601),
        lambda panel: panel.update(oracle="missing_oracle"),
        lambda panel: panel["runner"].update(legacy_warmup_off_by_one=1),
        lambda panel: panel["runner"].update(max_iterations=1000),
        lambda panel: panel["entries"][0].update(max_oracle_calls=1000),
        lambda panel: panel["entries"][0].update(seed=True),
        lambda panel: panel["entries"].append(copy.deepcopy(panel["entries"][0])),
        lambda panel: panel.update(output_root="output/udlm/../other"),
        lambda panel: panel.update(input_files=[]),
    ],
)
def test_invalid_finite_panel_rejected_before_resources(study, monkeypatch, alter):
    alter(study.panel)
    probe, launch = Mock(), Mock()
    monkeypatch.setattr(controller.resources, "_snapshot", probe)
    monkeypatch.setattr(controller.subprocess, "Popen", launch)
    with pytest.raises((ValueError, TypeError)):
        study.plan()
    probe.assert_not_called()
    launch.assert_not_called()


def test_non_gsk_task_uses_declared_auxiliary_receipt(study):
    study.panel["oracle"] = "fexofenadine_mpo"
    study.panel["input_files"] = [
        {
            "path": "scripts/artifact_io.py",
            "sha256": study.digest("scripts/artifact_io.py"),
        }
    ]
    plan = study.plan()
    assert plan["jobs"][0]["config"]["oracle"] == "fexofenadine_mpo"
    assert "/fexofenadine_mpo/released/" in plan["jobs"][0]["run_relative"]


def test_late_entry_and_auxiliary_input_drift_fail_preflight(study):
    later = copy.deepcopy(study.panel["entries"][0])
    later.update(id="late", seed=2301, checkpoint_sha256="f" * 64)
    study.panel["entries"].append(later)
    with pytest.raises(ValueError, match="SHA-256"):
        study.plan()
    study.panel["entries"].pop()
    plan = study.plan()
    (study.root / "oracle/gsk3b_current.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        controller.validate_inputs(plan)


def test_symlink_input_rejected(study):
    original = study.root / "inputs/vocabulary.csv"
    payload = original.read_bytes()
    original.unlink()
    (study.root / "inputs/other.csv").write_bytes(payload)
    original.symlink_to("other.csv")
    with pytest.raises(controller.artifact_io.ArtifactIOError):
        study.plan()


def test_dry_run_never_probes_reserves_or_starts(study, monkeypatch, capsys):
    monkeypatch.setattr(
        controller.resources, "_require_project_virtual_environment", lambda: None
    )
    forbidden = Mock(side_effect=AssertionError("dry-run touched execution"))
    for name in (
        "_snapshot",
        "_acquire_generation_lease",
        "_require_tmux_for_execution",
    ):
        monkeypatch.setattr(controller.resources, name, forbidden)
    monkeypatch.setattr(controller.subprocess, "Popen", forbidden)
    assert (
        controller.main(
            [
                "--panel",
                "panel.json",
                "--panel-sha256",
                study.write(),
                "--gpu-count",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["gpu_count"] == 2
    forbidden.assert_not_called()
    assert not (study.root / "output").exists()


def test_absolute_auxiliary_inputs_stay_within_project(study, monkeypatch):
    project = study.root.parent
    monkeypatch.setattr(controller.resources, "_project_root", lambda: project)
    path = study.root / "scripts/artifact_io.py"
    study.panel["input_files"].append(
        {"path": str(path), "sha256": study.digest("scripts/artifact_io.py")}
    )
    plan = study.plan()
    assert any(record["path"] == str(path) for record in plan["inputs"])
    with pytest.raises(ValueError, match="inside the project"):
        controller.fingerprint("/etc/passwd")


def test_absolute_shared_checkpoint_is_allowed_inside_project(study):
    entry = study.panel["entries"][0]
    entry["checkpoint"] = str(study.root / entry["checkpoint"])
    plan = study.plan()
    assert plan["jobs"][0]["checkpoint"]["path"] == entry["checkpoint"]


@pytest.fixture
def runtime(study, monkeypatch):
    clock = SimpleNamespace(now=0.0, peak_concurrency=0)
    processes, launched, signals = {}, [], []
    outcomes = []
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        controller.time,
        "sleep",
        lambda seconds: setattr(clock, "now", clock.now + seconds),
    )
    monkeypatch.setattr(controller.resources, "_snapshot", lambda: [gpu(6), gpu(7)])
    monkeypatch.setattr(
        controller.resources,
        "_probe_gpu",
        lambda uuid: gpu(int(uuid.rsplit("-", 1)[1])),
    )
    monkeypatch.setattr(
        controller,
        "validate_completion",
        lambda job, source: {"oracle_calls": job["entry"]["max_oracle_calls"]},
    )

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 9000 + len(launched)
            self.returncode = None
            self.polls = 0
            self.group_alive = True
            self.target = outcomes.pop(0) if outcomes else 0
            launched.append((self, command, kwargs))
            processes[self.pid] = self
            clock.peak_concurrency = max(
                clock.peak_concurrency, sum(p.group_alive for p in processes.values())
            )

        def poll(self):
            self.polls += 1
            if self.returncode is None and self.target is not None and self.polls >= 2:
                self.returncode = self.target
                self.group_alive = False
            return self.returncode

    def killpg(pid, sig):
        assert pid in processes, "must never signal another process group"
        signals.append((pid, sig))
        if sig == signal.SIGKILL or not getattr(processes[pid], "ignore_term", False):
            processes[pid].group_alive = False
            processes[pid].returncode = -sig

    monkeypatch.setattr(controller.subprocess, "Popen", Process)
    monkeypatch.setattr(
        controller.cleanup,
        "process_group_exists",
        lambda pid: processes[pid].group_alive,
    )
    monkeypatch.setattr(controller.os, "killpg", killpg)
    return SimpleNamespace(
        clock=clock,
        launched=launched,
        outcomes=outcomes,
        signals=signals,
        processes=processes,
    )


def terminal(study):
    return json.loads(
        (
            study.root / study.panel["output_root"] / "terminal_manifest.json"
        ).read_bytes()
    )


def test_two_gpu_waves_are_uuid_mapped_and_never_overlap(study, runtime):
    for number in range(1, 4):
        entry = copy.deepcopy(study.panel["entries"][0])
        entry.update(id=f"arm_{number}", seed=2300 + number)
        study.panel["entries"].append(entry)
    assert controller.execute(study.plan(2), SOURCE) == 0
    assert len(runtime.launched) == 4
    assert runtime.clock.peak_concurrency == 2
    result = terminal(study)
    assert [job["status"] for job in result["jobs"]] == ["completed"] * 4
    for _, command, kwargs in runtime.launched:
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] in (
            "GPU-synthetic-6",
            "GPU-synthetic-7",
        )
        assert command[command.index("--device") + 1] == "cuda:0"
        assert kwargs["env"]["PYTHONHASHSEED"] == "0"
        assert all(
            kwargs["env"][name] == "1"
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        )
    assert result["final_input_validation"] == "unchanged"
    assert result["lease_release_authorized"] is True
    assert not (
        study.root / controller.resources.GENERATION_LEASE_RELATIVE_PATH
    ).exists()
    assert runtime.signals == []


def test_launch_time_threshold_race_fails_without_starting(study, runtime, monkeypatch):
    monkeypatch.setattr(
        controller.resources, "_probe_gpu", lambda _: gpu(6, utilization=10)
    )
    assert controller.execute(study.plan(), SOURCE) == 1
    assert runtime.launched == []
    result = terminal(study)
    assert "final GPU probe" in result["error"]
    assert result["telemetry_records"] == 2


def test_bounded_capacity_wait_before_first_launch(study, runtime, monkeypatch):
    study.panel["capacity_wait_seconds"] = 60
    inventories = iter([[gpu(6, utilization=10)], [gpu(6)]])
    monkeypatch.setattr(controller.resources, "_snapshot", lambda: next(inventories))
    assert controller.execute(study.plan(), SOURCE) == 0
    assert len(runtime.launched) == 1
    assert runtime.clock.now >= 30


def test_capacity_timeout_has_no_child_or_retry(study, runtime, monkeypatch):
    study.panel["capacity_wait_seconds"] = 60
    monkeypatch.setattr(
        controller.resources, "_snapshot", lambda: [gpu(utilization=10)]
    )
    assert controller.execute(study.plan(), SOURCE) == 1
    assert runtime.launched == []
    assert runtime.clock.now == 60
    assert "capacity" in terminal(study)["error"]


def test_child_failure_stops_peer_and_never_launches_next_wave(study, runtime):
    for number in range(1, 3):
        entry = copy.deepcopy(study.panel["entries"][0])
        entry.update(id=f"arm_{number}", seed=2300 + number)
        study.panel["entries"].append(entry)
    runtime.outcomes[:] = [7, None]
    assert controller.execute(study.plan(2), SOURCE) == 1
    assert len(runtime.launched) == 2
    assert runtime.signals == [(9001, signal.SIGTERM)]
    assert "exited 7" in terminal(study)["error"]


def test_timeout_escalates_only_owned_process_group(study, runtime, monkeypatch):
    runtime.outcomes[:] = [None]
    original = controller.subprocess.Popen

    def ignore_term(*args, **kwargs):
        process = original(*args, **kwargs)
        process.ignore_term = True
        return process

    monkeypatch.setattr(controller.subprocess, "Popen", ignore_term)
    assert controller.execute(study.plan(), SOURCE) == 1
    result = terminal(study)
    assert result["jobs"][0]["timed_out"] is True
    assert runtime.signals == [(9000, signal.SIGTERM), (9000, signal.SIGKILL)]
    assert result["jobs"][0]["cleanup"]["group_present_at_end"] is False
    assert result["lease_release_authorized"] is True


def test_source_change_terminates_owned_child(study, runtime, monkeypatch):
    runtime.outcomes[:] = [None]

    def source():
        return {"head": "b" * 40, "upstream": "b" * 40} if runtime.launched else SOURCE

    monkeypatch.setattr(controller.resources, "_require_clean_pushed_source", source)
    assert controller.execute(study.plan(), SOURCE) == 1
    assert runtime.signals == [(9000, signal.SIGTERM)]
    assert "source changed" in terminal(study)["error"]


def test_foreign_lease_is_never_removed(study, runtime):
    plan = study.plan()
    (study.root / "output").mkdir()
    lock = study.root / controller.resources.GENERATION_LEASE_RELATIVE_PATH
    lock.write_bytes(b"foreign lease")
    with pytest.raises(RuntimeError, match="generation lease"):
        controller.execute(plan, SOURCE)
    assert lock.read_bytes() == b"foreign lease"
    assert runtime.launched == []


def test_unreaped_owned_group_retains_exact_lease(study, runtime, monkeypatch):
    runtime.outcomes[:] = [None]
    monkeypatch.setattr(
        controller.os,
        "killpg",
        Mock(side_effect=PermissionError("synthetic signal denial")),
    )
    assert controller.execute(study.plan(), SOURCE) == 1
    result = terminal(study)
    assert result["lease_release_authorized"] is False
    assert (study.root / controller.resources.GENERATION_LEASE_RELATIVE_PATH).exists()
    assert "PermissionError" in result["jobs"][0]["cleanup"]["error"]


def test_replaced_lease_is_preserved_on_failure(study, runtime, monkeypatch):
    runtime.outcomes[:] = [None]
    original = controller.subprocess.Popen
    lock = study.root / controller.resources.GENERATION_LEASE_RELATIVE_PATH

    def replace_lease(*args, **kwargs):
        process = original(*args, **kwargs)
        lock.write_bytes(b"foreign replacement")
        return process

    monkeypatch.setattr(controller.subprocess, "Popen", replace_lease)
    assert controller.execute(study.plan(), SOURCE) == 1
    assert lock.read_bytes() == b"foreign replacement"
    assert terminal(study)["lease_release_authorized"] is False
    assert runtime.signals == [(9000, signal.SIGTERM)]


def test_second_wave_capacity_loss_never_waits_or_retries(study, runtime, monkeypatch):
    study.panel["capacity_wait_seconds"] = 60
    entry = copy.deepcopy(study.panel["entries"][0])
    entry.update(id="second", seed=2301)
    study.panel["entries"].append(entry)
    inventories = iter([[gpu(6)], [gpu(6, utilization=10)]])
    monkeypatch.setattr(controller.resources, "_snapshot", lambda: next(inventories))
    assert controller.execute(study.plan(), SOURCE) == 1
    assert len(runtime.launched) == 1
    assert runtime.clock.now < 60
    assert terminal(study)["unlaunched_entry_ids"] == ["second"]


def write_success(job):
    run = controller.ROOT / job["run_relative"]
    (run / "state").mkdir(parents=True)
    config = job["config"]
    run_id = (
        f"{config['experiment_id']}:{config['oracle']}:released:seed{config['seed']}"
    )
    receipt = {
        "oracle_call_protocol": dict(controller.udlm_sampling.ORACLE_CALL_PROTOCOL),
        "contract": config["pmo_sampling"],
        "checkpoint": job["checkpoint"],
        "implementation_inputs": {},
        "inference_weights": {
            "source": "ema",
            "ema_applied": True,
            "ema": {"shadow_parameter_count": 1, "decay": 0.9, "num_updates": 20},
        },
    }
    manifest = {
        "status": "completed",
        "run_id": run_id,
        "config": config,
        "config_sha256": job["config_sha256"],
        "model": {"sha256": job["checkpoint"]["sha256"]},
        "oracle_calls": 2000,
        "oracle_budget": 2000,
        "seed": config["seed"],
        "task": config["oracle"],
        "variant": "released",
        "extra": {"git": {"commit": SOURCE["head"]}, "sampling": receipt},
    }
    summary = {
        "status": "completed",
        "run_id": run_id,
        "checkpoint_consistent": True,
        "config_sha256": job["config_sha256"],
        "model_sha256": job["checkpoint"]["sha256"],
        "scores": {"all_charged_molecules": {"oracle_calls": 2000}},
        "sampling": {"identity": receipt, "observed": {}},
    }
    (run / "manifest.json").write_bytes(controller.encode(manifest))
    (run / "summary.json").write_bytes(controller.encode(summary))
    (run / "events.jsonl").write_bytes(b"synthetic saved events; no scoring\n")
    (run / "state/latest.pkl").write_bytes(
        b"synthetic opaque state; never deserialized"
    )
    return run, manifest, summary


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "count",
        "slot",
        "source",
        "config",
        "checkpoint",
        "ema",
        "status",
        "oracle_protocol",
    ],
)
def test_saved_completion_must_bind_declared_request(study, tamper):
    job = study.plan()["jobs"][0]
    run, manifest, summary = write_success(job)
    if tamper == "count":
        summary["scores"]["all_charged_molecules"]["oracle_calls"] = 1999
    elif tamper == "slot":
        manifest["run_id"] = summary["run_id"] = "another:run"
    elif tamper == "source":
        manifest["extra"]["git"]["commit"] = "b" * 40
    elif tamper == "config":
        manifest["config"] = {**manifest["config"], "seed": 9999}
    elif tamper == "checkpoint":
        summary["model_sha256"] = "f" * 64
    elif tamper == "ema":
        summary["sampling"]["identity"]["inference_weights"]["ema_applied"] = False
    elif tamper == "status":
        summary["status"] = "max_iterations_reached"
    elif tamper == "oracle_protocol":
        del manifest["extra"]["sampling"]["oracle_call_protocol"]
    (run / "manifest.json").write_bytes(controller.encode(manifest))
    (run / "summary.json").write_bytes(controller.encode(summary))
    if tamper:
        with pytest.raises((RuntimeError, ValueError)):
            controller.validate_completion(job, SOURCE)
    else:
        result = controller.validate_completion(job, SOURCE)
        assert result["oracle_calls"] == 2000
        assert set(result["artifacts"]) == {
            "manifest.json",
            "summary.json",
            "events.jsonl",
            "state/latest.pkl",
        }
