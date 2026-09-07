"""CPU checks for a separate attempt after a capacity-only prelaunch failure."""

import copy
import hashlib
import json
import shutil

import pytest
import yaml

from scripts.udlm import launch_mask_prior_followup_training as launcher
from test_udlm_mask_prior_training import study as original_study


@pytest.fixture
def study(tmp_path):
    root = original_study.__wrapped__(tmp_path)
    for relative in (
        launcher.PROTOCOL,
        *[entry["relative_path"] for entry in launcher.FAILURE_EVIDENCE.values()],
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(launcher.ROOT / relative, target)
    return root


def write_json(path, value):
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_followup_changes_only_output_and_capacity_policy(study):
    baseline = launcher.original.build_plan(2, root=study)
    plan = launcher.build_plan(2, root=study)
    expected = copy.deepcopy(baseline["config"])
    expected["callback"]["dirpath"] = str(study / launcher.OUTPUT / "checkpoints")
    assert plan["config"] == expected
    assert plan["checkpoint_path"] == baseline["checkpoint_path"]
    assert plan["checkpoint_sha256"] == baseline["checkpoint_sha256"]
    assert (
        plan["expected_prior_metadata_sha256"]
        == baseline["expected_prior_metadata_sha256"]
    )
    assert plan["example_exposures"] == 128000
    assert plan["config"]["training"]["init_from_mdlm_ema"] is True
    assert plan["config"]["seed"] == 1500
    assert plan["gpu_count"] == 2
    assert plan["output_relative"] != baseline["output_relative"]
    for key, value in launcher.WAIT_POLICY.items():
        assert plan[key] == plan["protocol"][key] == value
        assert key not in baseline
    assert not any(
        "resume" in part or "ckpt_path=" in part for part in plan["training_argv"]
    )


@pytest.mark.parametrize("name", ["terminal", "request"])
def test_original_failure_bytes_are_immutable(study, name):
    path = study / launcher.FAILURE_EVIDENCE[name]["relative_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        launcher.build_plan(2, root=study)


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_pid", 123),
        ("launch_sha256", "a" * 64),
        ("training_return_code", 0),
        ("checkpoint", {"global_step": 1}),
        ("completed_example_exposures", 0),
        ("leases_release_authorized", False),
        ("status", "completed"),
        ("error", "RuntimeError: unrelated failure"),
        ("source", {"head": "f" * 40, "upstream": "f" * 40}),
    ],
)
def test_training_or_different_failure_cannot_be_relabelled(
    study, monkeypatch, field, value
):
    references = copy.deepcopy(launcher.FAILURE_EVIDENCE)
    path = study / references["terminal"]["relative_path"]
    terminal = json.loads(path.read_text())
    terminal[field] = value
    references["terminal"]["sha256"] = write_json(path, terminal)
    protocol_path = study / launcher.PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    protocol["prelaunch_failure_evidence"] = references
    write_json(protocol_path, protocol)
    monkeypatch.setattr(launcher, "FAILURE_EVIDENCE", references)
    with pytest.raises(ValueError, match="prelaunch-only failure"):
        launcher.build_plan(2, root=study)


@pytest.mark.parametrize("change", ["seed", "lambda", "batch"])
def test_followup_training_drift_is_rejected_even_with_updated_config_hash(
    study, change
):
    path = study / launcher.CONFIG
    config = yaml.safe_load(path.read_text())
    if change == "seed":
        config["seed"] = 1501
    elif change == "lambda":
        config["training"] = {"udlm": {"mask_mixture_weight": 0.8}}
    else:
        config["loader"] = {"batch_size": 8}
    path.write_text(yaml.safe_dump(config))
    protocol_path = study / launcher.PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    protocol["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="beyond its output directory"):
        launcher.build_plan(2, root=study)


@pytest.mark.parametrize(
    "field,value",
    [("gpu_availability_wait_seconds", 3600), ("gpu_availability_poll_seconds", 60)],
)
def test_capacity_policy_cannot_be_changed(study, field, value):
    path = study / launcher.PROTOCOL
    protocol = json.loads(path.read_text())
    protocol[field] = value
    write_json(path, protocol)
    with pytest.raises(ValueError, match="fixed V11b"):
        launcher.build_plan(2, root=study)


def setup_main(study, monkeypatch):
    plan = launcher.build_plan(2, root=study)
    monkeypatch.setattr(launcher, "ROOT", study)
    monkeypatch.setattr(launcher, "build_plan", lambda count: plan)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(
        launcher.engine.benchmark,
        "_require_clean_pushed_source",
        lambda: {"head": "a" * 40, "upstream": "a" * 40},
    )
    return plan


def test_cpu_preview_never_waits_probes_or_creates_attempt(study, monkeypatch, capsys):
    plan = setup_main(study, monkeypatch)
    checked = []
    monkeypatch.setattr(launcher.engine, "verify_checkpoint_input", checked.append)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run touched live resources")

    for name in ("execute", "wait_for_gpu_capacity", "acquire_leases"):
        monkeypatch.setattr(launcher.engine, name, forbidden)
    monkeypatch.setattr(launcher.engine.audited, "probe_all_gpus", forbidden)
    before = sorted(str(path.relative_to(study)) for path in study.rglob("*"))
    assert launcher.main(["--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["plan"] == plan
    assert preview["gpu_queries"] == preview["artifact_mutations"] == 0
    assert checked == [plan]
    assert before == sorted(str(path.relative_to(study)) for path in study.rglob("*"))


def test_live_entry_delegates_one_attempt_and_input_validation_to_wait_engine(
    study, monkeypatch
):
    plan = setup_main(study, monkeypatch)
    monkeypatch.setattr(launcher.engine, "CANONICAL_ROOT", study)
    monkeypatch.setattr(
        launcher.engine.benchmark, "_require_tmux_for_execution", lambda: None
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def no_repeated_hash(*args, **kwargs):
        raise AssertionError("live entry must leave input validation to the engine")

    monkeypatch.setattr(launcher.engine, "verify_checkpoint_input", no_repeated_hash)
    calls = []

    def execute(actual, source, *, plan_builder):
        calls.append((actual, plan_builder))
        return 7

    monkeypatch.setattr(launcher.engine, "execute", execute)
    assert launcher.main([]) == 7
    assert calls == [(plan, launcher.build_plan)]
