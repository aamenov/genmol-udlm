import pytest

from scripts.udlm.launch_followthrough_learning import build_plan


@pytest.mark.parametrize("gpus", [1, 2])
def test_learning_arms_match_exposure_and_fresh_schedule(gpus):
    plans = [build_plan(arm, gpus) for arm in ("ct", "ce", "mdlm")]
    assert len({p["checkpoint_sha256"] for p in plans}) == 1
    for plan in plans:
        config = plan["config"]
        assert config["trainer"]["max_steps"] == 4000
        assert config["optim"]["scheduler"]["horizon_updates"] == 4000
        assert config["loader"]["batch_size"] * gpus * config["trainer"]["accumulate_grad_batches"] == 128
        assert config["training"]["init_from_mdlm_ema"]
        assert plan["example_exposures"] == 512000
    assert plans[-1]["config"]["training"]["diffusion"] == "mdlm"
    assert plans[1]["config"]["training"]["udlm"]["parameterization"] == "x0_denoiser"


@pytest.mark.parametrize("arm,gpus", [("unknown", 2), ("ct", 0), ("ce", 3)])
def test_unsupported_arm_or_gpu_count_rejected(arm, gpus):
    with pytest.raises(ValueError):
        build_plan(arm, gpus)


def test_configs_pass_the_actual_manual_startup_guard():
    import subprocess
    import sys
    # A fresh interpreter reproduces the entrypoint's resolver-registration order.
    code = '''
from scripts import train as entrypoint
from scripts.udlm.launch_followthrough_learning import build_plan
from omegaconf import OmegaConf
assert entrypoint._PILOT_CONTRACT is None
for arm in ("ct", "ce", "mdlm"):
    config = OmegaConf.create(build_plan(arm, 2)["config"])
    assert entrypoint._reseed_training_rng_after_model_initialization(config, "warm_start") is None
    assert entrypoint._validate_and_record_pilot_config(config) is None
    assert config.training.pilot_fail_on_nonfinite_loss is False
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
