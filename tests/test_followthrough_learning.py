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
