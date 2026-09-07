"""A successful child exit cannot accept a changed MASK-rich checkpoint law."""

from dataclasses import asdict

import pytest
import torch

import test_udlm_mask_rich_benchmark as prior_cases
from scripts.exps.denovo import benchmark
from scripts.udlm import launch_engineering_training as engine


def completed_checkpoint():
    value = prior_cases.checkpoint(0.9, ce=True)
    value["global_step"] = 1000
    value["optimizer_states"] = [{"state": {"moment": torch.zeros(1)}}]
    value["ema"] = {"num_updates": 1000, "shadow_params": [torch.ones(1)]}
    return value


def validate(value, tmp_path, monkeypatch):
    path = tmp_path / "1000.ckpt"
    torch.save(value, path)
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    return engine.validate_checkpoint_output(
        path, value["hyper_parameters"]["config"], expected_steps=1000
    )


def test_completed_ce_checkpoint_binds_exact_mask_prior(tmp_path, monkeypatch):
    value = completed_checkpoint()
    result = validate(value, tmp_path, monkeypatch)
    claim, _ = engine.artifact_io.snapshot_file(tmp_path, "1000.ckpt")
    assert result == {
        **asdict(claim),
        "global_step": 1000,
        "finite_tensor_count": 7,
        "udlm_prior_metadata_sha256": engine.canonical_digest(
            value[benchmark.UDLM_PRIOR_CHECKPOINT_KEY]
        ),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_metadata",
        "missing_marker",
        "wrong_marker",
        "float_marker",
        "base_hash",
        "final_hash",
        "actual_prior",
        "active_mapping",
        "config_lambda",
        "config_floor",
        "config_sampling_eps",
        "config_noise_eps",
        "config_antithetic",
        "config_exclusion",
        "config_vocab",
        "boolean_lambda",
        "boolean_antithetic",
        "missing_ce_metadata",
        "wrong_ce_marker",
        "nonfinite_optimizer",
        "wrong_ema_updates",
    ],
)
def test_completed_checkpoint_rejects_prior_or_training_identity_change(
    tmp_path, monkeypatch, mutation
):
    value = completed_checkpoint()
    metadata = value[benchmark.UDLM_PRIOR_CHECKPOINT_KEY]
    state = value["state_dict"]
    config = value["hyper_parameters"]["config"]
    training = config["training"]
    udlm = training["udlm"]
    if mutation == "missing_metadata":
        del value[benchmark.UDLM_PRIOR_CHECKPOINT_KEY]
    elif mutation == "missing_marker":
        del state[benchmark.UDLM_MASK_RICH_STATE_KEY]
    elif mutation == "wrong_marker":
        state[benchmark.UDLM_MASK_RICH_STATE_KEY] = torch.tensor(
            0.8, dtype=torch.float64
        ).view(torch.int64)
    elif mutation == "float_marker":
        state[benchmark.UDLM_MASK_RICH_STATE_KEY] = torch.tensor(0.9)
    elif mutation == "base_hash":
        metadata["base_stationary_probs_sha256"] = "0" * 64
    elif mutation == "final_hash":
        metadata["stationary_probs_sha256"] = "0" * 64
    elif mutation == "actual_prior":
        # Preserve positivity and total mass while changing the actual law.
        state["mdlm.stationary_probs"][4] -= 0.01
        state["mdlm.stationary_probs"][5] += 0.01
    elif mutation == "active_mapping":
        state["mdlm.token_to_diffusion_index"][4] = 5
    elif mutation == "config_lambda":
        udlm["mask_mixture_weight"] = 0.8
    elif mutation == "config_floor":
        udlm["empirical_uniform_mix"] = 0.001
    elif mutation == "config_sampling_eps":
        training["sampling_eps"] = 0.002
    elif mutation == "config_noise_eps":
        udlm["noise_eps"] = 0.002
    elif mutation == "config_antithetic":
        training["antithetic_sampling"] = False
    elif mutation == "config_exclusion":
        udlm["exclude_special_tokens"] = True
    elif mutation == "config_vocab":
        config["model"]["vocab_size"] = 1879
    elif mutation == "boolean_lambda":
        udlm["mask_mixture_weight"] = True
    elif mutation == "boolean_antithetic":
        training["antithetic_sampling"] = 1
    elif mutation == "missing_ce_metadata":
        del value["udlm_denoiser_metadata"]
    elif mutation == "wrong_ce_marker":
        state["_udlm_denoiser_ce_version"] = torch.tensor(1.0)
    elif mutation == "nonfinite_optimizer":
        value["optimizer_states"][0]["state"]["moment"][0] = float("nan")
    elif mutation == "wrong_ema_updates":
        value["ema"]["num_updates"] = 999
    else:
        raise AssertionError(mutation)
    # Configuration mutations affect both the saved and launch configuration:
    # rejection must come from identity validation, not config inequality.
    with pytest.raises((ValueError, RuntimeError)):
        validate(value, tmp_path, monkeypatch)


def test_legacy_checkpoint_audit_return_shape_stays_unchanged(tmp_path, monkeypatch):
    value = {
        "global_step": 1000,
        "hyper_parameters": {"config": {"training": {"udlm": {}}}},
        "state_dict": {"weight": torch.ones(1)},
        "optimizer_states": [{"state": {"moment": torch.zeros(1)}}],
        "ema": {"num_updates": 1000, "shadow_params": [torch.ones(1)]},
    }
    result = validate(value, tmp_path, monkeypatch)
    claim, _ = engine.artifact_io.snapshot_file(tmp_path, "1000.ckpt")
    assert result == {
        **asdict(claim),
        "global_step": 1000,
        "finite_tensor_count": 3,
    }
