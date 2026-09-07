"""CPU-only comparison, evidence and lifecycle checks for the fixed V11 arm."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from scripts.udlm import launch_mask_prior_training as launcher
from test_udlm_engineering_training import _gpu


def write_json(path, value):
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def study(tmp_path):
    root = launcher.ROOT
    shutil.copytree(root / "configs", tmp_path / "configs")
    references = [
        launcher.PROTOCOL,
        launcher.original.PROTOCOL,
        launcher.PRIOR["frequency_artifact"],
        *[entry["relative_path"] for entry in launcher.EVIDENCE.values()],
    ]
    for relative in references:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, path)
    return tmp_path


def test_new_arm_matches_completed_ce_except_prior_and_output(study):
    baseline = launcher.original.build_plans(2, root=study)[1]
    original_config = copy.deepcopy(baseline["config"])
    plan = launcher.build_plan(2, root=study)
    expected = copy.deepcopy(original_config)
    expected["training"]["udlm"].update(
        prior_variant="mask_rich_empirical", mask_mixture_weight=0.9
    )
    expected["callback"]["dirpath"] = str(study / launcher.OUTPUT / "checkpoints")
    assert plan["config"] == expected
    assert baseline["config"] == original_config
    assert plan["checkpoint_path"] == baseline["checkpoint_path"]
    assert plan["checkpoint_sha256"] == baseline["checkpoint_sha256"]
    assert plan["config"]["training"]["init_from_mdlm_ema"] is True
    assert (
        plan["gpu_count"]
        * plan["config"]["loader"]["batch_size"]
        * plan["config"]["trainer"]["accumulate_grad_batches"]
        == 128
    )
    assert plan["config"]["seed"] == 1500
    assert plan["config"]["trainer"]["max_steps"] == 1000
    assert plan["example_exposures"] == 128000
    assert plan["attempt_id"] == "mask_ce_1000_b128_w2"
    assert "campaign_relative" not in plan
    assert plan["expected_prior_metadata_sha256"] == launcher.PRIOR["metadata_sha256"]
    argv = plan["training_argv"]
    assert argv[4] == Path(launcher.CONFIG).stem
    assert "trainer.devices=2" in argv and "trainer.accumulate_grad_batches=4" in argv
    assert f"hydra.run.dir={study / launcher.OUTPUT / 'hydra'}" in argv
    assert not any(
        "resume" in argument or "ckpt_path=" in argument for argument in argv
    )
    assert launcher.engine.canonical_digest(argv) == plan["argv_sha256"]


@pytest.mark.parametrize("count", [1, 3, True, 2.0, "2", None])
def test_training_gpu_count_must_match_historical_ce(count):
    with pytest.raises(ValueError, match="same two-GPU"):
        launcher.build_plan(count)


@pytest.mark.parametrize("name", list(launcher.EVIDENCE))
def test_changed_historical_evidence_is_rejected(study, name):
    path = study / launcher.EVIDENCE[name]["relative_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        launcher.build_plan(2, root=study)


@pytest.mark.parametrize("change", ["seed", "mask_weight", "corruption_alphabet"])
def test_changed_training_settings_rejected_even_with_new_config_hash(study, change):
    path = study / launcher.CONFIG
    config = yaml.safe_load(path.read_text())
    if change == "seed":
        config["seed"] = 1501
    elif change == "mask_weight":
        config["training"]["udlm"]["mask_mixture_weight"] = 0.8
    else:
        config["training"]["udlm"]["exclude_special_tokens"] = True
    path.write_text(yaml.safe_dump(config))
    protocol_path = study / launcher.PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    protocol["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="beyond its prior and output"):
        launcher.build_plan(2, root=study)


@pytest.mark.parametrize(
    "field,value", [("mask_mixture_weight", 0.8), ("metadata_sha256", "0" * 64)]
)
def test_protocol_cannot_select_another_prior(study, field, value):
    path = study / launcher.PROTOCOL
    protocol = json.loads(path.read_text())
    protocol["prior"][field] = value
    write_json(path, protocol)
    with pytest.raises(ValueError, match="fixed V11"):
        launcher.build_plan(2, root=study)


@pytest.mark.parametrize("mutation", ["failed", "source", "exposures", "prior_config"])
def test_control_semantics_checked_beyond_its_supplied_hash(
    study, monkeypatch, mutation
):
    evidence = copy.deepcopy(launcher.EVIDENCE)
    path = study / evidence["v8b_ce_terminal"]["relative_path"]
    terminal = json.loads(path.read_text())
    if mutation == "failed":
        terminal["status"] = "failed"
    elif mutation == "source":
        terminal["source"]["head"] = "f" * 40
    elif mutation == "exposures":
        terminal["completed_example_exposures"] = 16000
    else:
        terminal["plan"]["config"]["training"]["udlm"]["empirical_uniform_mix"] = 0.01
        terminal["plan"]["config_sha256"] = launcher.engine.canonical_digest(
            terminal["plan"]["config"]
        )
    evidence["v8b_ce_terminal"]["sha256"] = write_json(path, terminal)
    protocol_path = study / launcher.PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    protocol["evidence"] = evidence
    write_json(protocol_path, protocol)
    monkeypatch.setattr(launcher, "EVIDENCE", evidence)
    with pytest.raises(ValueError, match="historical control"):
        launcher.build_plan(2, root=study)


def test_cleanup_requires_reviewed_bounded_grace(monkeypatch):
    launcher.require_cleanup_grace()
    monkeypatch.setattr(launcher.engine, "PROCESS_GROUP_EXIT_GRACE_SECONDS", 0.0)
    with pytest.raises(RuntimeError, match="reviewed bounded"):
        launcher.require_cleanup_grace()


def prepare_main(study, monkeypatch):
    plan = launcher.build_plan(2, root=study)
    source = {"head": "a" * 40, "upstream": "a" * 40}
    monkeypatch.setattr(launcher, "ROOT", study)
    monkeypatch.setattr(launcher, "build_plan", lambda count: plan)
    monkeypatch.setattr(
        launcher.engine.benchmark, "_require_clean_pushed_source", lambda: source
    )
    monkeypatch.setattr(launcher.engine, "verify_checkpoint_input", lambda plan: None)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)
    return plan, source


def test_dry_run_is_cpu_only_without_artifact_mutation(study, monkeypatch, capsys):
    plan, _ = prepare_main(study, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preview touched live resources")

    monkeypatch.setattr(launcher.engine, "execute", forbidden)
    monkeypatch.setattr(launcher.engine, "acquire_leases", forbidden)
    monkeypatch.setattr(launcher.engine.audited, "probe_all_gpus", forbidden)
    monkeypatch.setattr(
        launcher.engine.benchmark, "_require_tmux_for_execution", forbidden
    )
    before = sorted(str(path.relative_to(study)) for path in study.rglob("*"))
    assert launcher.main(["--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["plan"] == plan
    assert result["gpu_queries"] == result["artifact_mutations"] == 0
    assert before == sorted(str(path.relative_to(study)) for path in study.rglob("*"))


def test_live_entry_uses_its_own_revalidated_plan_builder(study, monkeypatch):
    plan, source = prepare_main(study, monkeypatch)
    monkeypatch.setattr(launcher.engine, "CANONICAL_ROOT", study)
    monkeypatch.setattr(
        launcher.engine.benchmark, "_require_tmux_for_execution", lambda: None
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    received = []

    def execute(actual_plan, actual_source, *, plan_builder):
        received.append((actual_plan, actual_source, plan_builder))
        return 7

    monkeypatch.setattr(launcher.engine, "execute", execute)
    assert launcher.main([]) == 7
    assert received == [(plan, source, launcher.build_plan)]


@pytest.mark.parametrize(
    "prior_hash", [None, "0" * 64, launcher.PRIOR["metadata_sha256"]]
)
def test_controller_only_accepts_the_prospective_prior_identity(
    study, monkeypatch, prior_hash
):
    plan = launcher.build_plan(2, root=study)
    source = {"head": "a" * 40, "upstream": "a" * 40}
    engine = launcher.engine
    monkeypatch.setattr(engine, "ROOT", study)
    monkeypatch.setattr(
        engine, "verify_checkpoint_input", lambda plan: {"sha256": "b" * 64}
    )
    monkeypatch.setattr(
        engine.benchmark, "_require_clean_pushed_source", lambda: source
    )
    gpus = [_gpu(index=2), _gpu(index=6)]
    monkeypatch.setattr(engine.audited, "probe_all_gpus", lambda: gpus)
    monkeypatch.setattr(
        engine.audited,
        "probe_gpu_uuid",
        lambda uuid: next(g for g in gpus if g.uuid == uuid),
    )
    monkeypatch.setattr(engine, "process_group_exists", lambda pid: False)
    child = SimpleNamespace(pid=12345, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(engine.subprocess, "Popen", lambda *args, **kwargs: child)
    validated = {"global_step": 1000, "sha256": "c" * 64}
    if prior_hash is not None:
        validated["udlm_prior_metadata_sha256"] = prior_hash
    monkeypatch.setattr(
        engine, "validate_checkpoint_output", lambda *a, **kw: validated
    )
    accepted = prior_hash == launcher.PRIOR["metadata_sha256"]
    assert engine.execute(plan, source, plan_builder=lambda count: plan) == (
        0 if accepted else 1
    )
    terminal = json.loads(
        (study / launcher.OUTPUT / "terminal_manifest.json").read_text()
    )
    assert terminal["status"] == ("completed" if accepted else "failed")
    assert terminal["leases_release_authorized"] is True
    assert terminal["completed_example_exposures"] == (128000 if accepted else None)
    if not accepted:
        assert terminal["checkpoint"] is None
        assert terminal["end_to_end_training_examples_per_second"] is None
        assert "prior differs" in terminal["error"]
    assert all(not (study / relative).exists() for relative in engine.LEASE_PATHS)
    with pytest.raises(FileExistsError):
        engine.execute(plan, source, plan_builder=lambda count: plan)
