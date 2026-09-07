"""CPU checks for the matched objective plans and bounded sequential execution."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import artifact_io
from scripts.udlm import launch_engineering_training as engine
from scripts.udlm import launch_objective_training as launcher
from test_udlm_engineering_training import _gpu


@pytest.mark.parametrize("count,accumulation", [(1, 8), (2, 4)])
def test_matching_plans_have_only_objective_and_output_differences(count, accumulation):
    ct, ce = launcher.build_plans(count)
    assert [ct["arm_id"], ce["arm_id"]] == ["ct", "ce"]
    assert launcher.common_config(ct["config"]) == launcher.common_config(ce["config"])
    assert ct["matched_common_config_sha256"] == ce["matched_common_config_sha256"]
    for plan in (ct, ce):
        config = plan["config"]
        assert config["seed"] == 1500
        assert config["trainer"]["max_steps"] == 1000
        assert config["trainer"]["devices"] == count
        assert config["trainer"]["accumulate_grad_batches"] == accumulation
        assert count * config["loader"]["batch_size"] * accumulation == 128
        assert plan["example_exposures"] == 128000
        assert config["training"]["udlm"]["mask_all_special_tokens"] is True
        assert config["training"]["udlm"]["exclude_special_tokens"] is False
        assert config["training"]["init_from_mdlm_ema"] is True
        assert config["callback"]["every_n_train_steps"] == 1000
        assert (
            Path(config["training"]["init_from_mdlm_checkpoint"]).name == "50000.ckpt"
        )
    assert ct["config"]["training"]["udlm"]["parameterization"] == "raw_loo"
    assert ce["config"]["training"]["udlm"]["parameterization"] == "x0_denoiser"


@pytest.mark.parametrize("count", [1, 2])
def test_v7_plan_exactly_matches_pre_refactor_implementation(count):
    source = subprocess.run(
        ["git", "show", "6442dd1:scripts/udlm/launch_engineering_training.py"],
        cwd=engine.ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    namespace = {"__name__": "historical_v7_reference", "__file__": engine.__file__}
    exec(compile(source, engine.__file__, "exec"), namespace)
    assert namespace["build_plan"](count) == engine.build_plan(count)


@pytest.mark.parametrize("failure_arm", [None, "ct", "ce"])
def test_campaign_runs_sequentially_or_stops_after_first_failure(
    tmp_path, monkeypatch, failure_arm
):
    plans = launcher.build_plans(2)
    source = {"head": "a" * 40, "upstream": "a" * 40}
    calls = []
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(
        engine.benchmark, "_require_clean_pushed_source", lambda: source
    )

    def run(plan, actual_source, *, plan_builder):
        assert actual_source == source
        assert plan_builder(2) == plan
        calls.append(plan["arm_id"])
        failed = plan["arm_id"] == failure_arm
        path = tmp_path / plan["output_relative"]
        path.mkdir()
        receipt = {
            "status": "failed" if failed else "completed",
            "plan": plan,
            "source": source,
            "training_return_code": 7 if failed else 0,
            "completed_example_exposures": None if failed else 128000,
            "leases_release_authorized": True,
        }
        artifact_io.publish_bytes_exclusive(
            tmp_path,
            f"{plan['output_relative']}/terminal_manifest.json",
            engine.encode(receipt),
        )
        return 1 if failed else 0

    monkeypatch.setattr(engine, "execute", run)
    result = launcher.execute_campaign(plans, source)
    assert calls == (["ct"] if failure_arm == "ct" else ["ct", "ce"])
    assert result == (0 if failure_arm is None else 1)
    terminal_path = tmp_path / plans[0]["campaign_relative"] / "campaign_terminal.json"
    terminal = json.loads(terminal_path.read_text())
    assert terminal["status"] == ("completed" if failure_arm is None else "failed")
    if failure_arm == "ct":
        assert terminal["arms"][1]["status"] == "not_completed_after_campaign_stop"
    for arm in terminal["arms"]:
        if "terminal_relative" in arm:
            assert (
                artifact_io.snapshot_file(tmp_path, arm["terminal_relative"])[0].sha256
                == arm["terminal_sha256"]
            )
    with pytest.raises(FileExistsError):
        launcher.execute_campaign(plans, source)


def test_shared_engine_honors_1000_update_budget_and_arm_seed(tmp_path, monkeypatch):
    plan = launcher.build_plans(1)[0]
    plan["config"] = {"seed": 1500}
    checkpoint = tmp_path / plan["output_relative"] / "checkpoints/1000.ckpt"
    source = {"head": "a" * 40, "upstream": "a" * 40}
    child = """
import os, sys
from pathlib import Path
assert os.environ['PYTHONHASHSEED'] == '1500'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
path = Path(sys.argv[1]); path.parent.mkdir(parents=True)
torch.save({'global_step': 1000, 'hyper_parameters': {'config': {'seed': 1500}},
 'state_dict': {'weight': torch.ones(1, device='cpu')},
 'optimizer_states': [{'state': {'moment': torch.zeros(1, device='cpu')}}],
 'ema': {'num_updates': 1000, 'shadow_params': [torch.ones(1, device='cpu')]}}, path)
"""
    plan["training_argv"] = [sys.executable, "-c", child, str(checkpoint)]
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(
        engine, "verify_checkpoint_input", lambda _plan: {"sha256": "b" * 64}
    )
    monkeypatch.setattr(
        engine.benchmark, "_require_clean_pushed_source", lambda: source
    )
    monkeypatch.setattr(engine.audited, "probe_all_gpus", lambda: [_gpu()])
    monkeypatch.setattr(engine.audited, "probe_gpu_uuid", lambda _uuid: _gpu())
    revalidations = []

    def builder(count):
        revalidations.append(count)
        return plan

    assert engine.execute(plan, source, plan_builder=builder) == 0
    assert revalidations == [1]
    terminal = json.loads(
        (tmp_path / plan["output_relative"] / "terminal_manifest.json").read_text()
    )
    assert terminal["completed_example_exposures"] == 128000
    assert terminal["checkpoint"]["global_step"] == 1000
    assert (
        terminal["end_to_end_training_examples_per_second"]
        == 128000 / terminal["training_subprocess_seconds"]
    )
    assert not any((tmp_path / relative).exists() for relative in engine.LEASE_PATHS)


@pytest.mark.parametrize(
    "mutation", [None, "missing_metadata", "wrong_marker", "raw_config"]
)
def test_checkpoint_completion_requires_ce_inference_identity(
    tmp_path, monkeypatch, mutation
):
    import torch
    from genmol.model import UDLM_DENOISER_METADATA

    config = {"training": {"udlm": {"parameterization": "x0_denoiser"}}}
    payload = {
        "global_step": 1000,
        "hyper_parameters": {"config": config},
        "state_dict": {
            "weight": torch.ones(1),
            "_udlm_denoiser_ce_version": torch.tensor(1, dtype=torch.int64),
        },
        "optimizer_states": [{"state": {"moment": torch.zeros(1)}}],
        "ema": {"num_updates": 1000, "shadow_params": [torch.ones(1)]},
        "udlm_denoiser_metadata": dict(UDLM_DENOISER_METADATA),
    }
    if mutation == "missing_metadata":
        del payload["udlm_denoiser_metadata"]
    elif mutation == "wrong_marker":
        payload["state_dict"]["_udlm_denoiser_ce_version"] = torch.tensor(1.0)
    elif mutation == "raw_config":
        config["training"]["udlm"]["parameterization"] = "raw_loo"
    path = tmp_path / "1000.ckpt"
    torch.save(payload, path)
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    if mutation is None:
        result = engine.validate_checkpoint_output(path, config, expected_steps=1000)
        assert result["global_step"] == 1000
    else:
        with pytest.raises(RuntimeError, match="denoiser"):
            engine.validate_checkpoint_output(path, config, expected_steps=1000)
