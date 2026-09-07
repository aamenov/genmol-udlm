"""CPU checks for the opt-in MASK mixture and its separate checkpoint identity."""

import copy
import hashlib
import json
import math
from dataclasses import FrozenInstanceError

import pytest
import torch
from omegaconf import OmegaConf

import genmol.model as model_module
from test_udlm_model import (
    _Tokenizer,
    _config,
    _frequency_payload,
    _install_frequency_artifact,
    _run_checkpoint_save_hook,
)


@pytest.fixture(autouse=True)
def local_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(model_module, "get_tokenizer", lambda: _Tokenizer())
    _install_frequency_artifact(monkeypatch, tmp_path, _frequency_payload())


def mask_config(weight=0.9, *, ce=False):
    config = _config(diffusion="udlm", prior_variant="mask_rich_empirical")
    config.training.udlm.mask_mixture_weight = weight
    config.training.udlm.mask_all_special_tokens = True
    if ce:
        config.training.udlm.parameterization = "x0_denoiser"
    return config


@pytest.mark.parametrize("weight", [0.0, 0.9, 0.99, math.nextafter(1.0, 0.0)])
def test_mixture_matches_canonical_empirical_base_and_retains_full_support(weight):
    base = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="empirical_frequency")
    )
    config = mask_config(weight)
    original_config = OmegaConf.to_container(config)
    model = model_module.GenMol(config)
    prior = model.mdlm.stationary_probs
    expected = (1 - weight) * base.mdlm.stationary_probs
    expected[model.mask_index] += weight
    expected /= expected.sum()
    torch.testing.assert_close(prior, expected, rtol=0, atol=2e-16)
    assert torch.all(prior > 0) and torch.isfinite(prior).all()
    assert prior.dtype == torch.float64
    assert model.udlm_prior_metadata.mask_token_id == model.mask_index == 4
    assert model.udlm_prior_metadata.mask_mixture_weight == weight
    assert (
        model.udlm_prior_metadata.base_stationary_probs_sha256
        == base.udlm_prior_metadata.stationary_probs_sha256
    )
    assert (
        model.udlm_prior_metadata.stationary_probs_sha256
        == model_module._canonical_sequence_sha256(prior.tolist())
    )
    assert OmegaConf.to_container(config) == original_config
    if weight == 0:
        assert torch.equal(prior, base.mdlm.stationary_probs)


def test_noncontiguous_active_mapping_uses_full_mask_token_id():
    process, metadata = model_module._build_udlm_process(
        variant="mask_rich_empirical",
        model_vocab_size=11,
        excluded_token_ids=(0, 2, 3),
        sampling_eps=1e-3,
        noise_eps=1e-3,
        antithetic_sampling=True,
        empirical_uniform_mix=0.01,
        tokenizer=_Tokenizer(),
        mask_mixture_weight=0.9,
    )
    assert metadata.mask_token_id == 4
    assert process.diffusion_token_ids[1].item() == 4
    assert process.stationary_probs[1].item() > 0.9
    assert process.token_to_diffusion_index[4].item() == 1
    assert torch.all(process.stationary_probs > 0)


@pytest.mark.parametrize(
    "weight", [None, True, False, -0.1, 1.0, 1.1, float("nan"), float("inf"), "0.9"]
)
def test_mask_mixture_requires_explicit_finite_weight_with_full_support(weight):
    with pytest.raises(ValueError, match="mask_mixture_weight"):
        model_module.GenMol(mask_config(weight))


def test_mask_must_be_an_active_token_and_configuration_must_select_new_variant():
    config = mask_config()
    config.training.udlm.exclude_special_tokens = True
    with pytest.raises(ValueError, match="MASK in the active"):
        model_module.GenMol(config)
    config = mask_config()
    config.training.udlm.prior_variant = "empirical_frequency"
    with pytest.raises(ValueError, match="requires prior_variant"):
        model_module.GenMol(config)
    config = mask_config()
    config.training.diffusion = "mdlm"
    with pytest.raises(ValueError, match="requires training.diffusion"):
        model_module.GenMol(config)


@pytest.mark.parametrize("weight", [None, 0, 1])
def test_existing_empirical_smoothing_remains_required(weight):
    config = mask_config()
    config.training.udlm.empirical_uniform_mix = weight
    with pytest.raises(ValueError, match="empirical_uniform_mix"):
        model_module.GenMol(config)


@pytest.mark.parametrize(
    "variant,digest",
    [
        (
            "release_uniform",
            "343200007e4627a3ac3ada03951b0923761e18060d9c7f35105e5aa4cddcc2cc",
        ),
        (
            "schedule_uniform",
            "2d2849e5bd59256f65b1b0a2c624ea19f3cbe79653609401872e0b7abeafc5d0",
        ),
        (
            "empirical_frequency",
            "667d2d4b201520c6e090cebb05fa8b5a1f1a550f88c55c358389735036b9386f",
        ),
    ],
)
def test_historical_metadata_matches_prechange_byte_digest(variant, digest):
    # Captured from c17e848 with this existing eleven-token fixture before edits.
    config = _config(diffusion="udlm", prior_variant=variant)
    snapshot = OmegaConf.to_container(config)
    model = model_module.GenMol(config)
    metadata = model.udlm_prior_metadata.to_dict()
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == digest
    assert model_module.UDLM_MASK_RICH_STATE_KEY not in model.state_dict()
    assert "mask_mixture_weight" not in metadata
    assert OmegaConf.to_container(config) == snapshot


def test_new_prior_marker_does_not_change_backbone_initialization_or_rng():
    torch.manual_seed(41)
    base = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="empirical_frequency")
    )
    after_base = torch.get_rng_state().clone()
    torch.manual_seed(41)
    model = model_module.GenMol(mask_config())
    assert torch.equal(torch.get_rng_state(), after_base)
    for name, value in base.backbone.state_dict().items():
        assert torch.equal(model.backbone.state_dict()[name], value)
    assert set(model.state_dict()) - set(base.state_dict()) == {
        model_module.UDLM_MASK_RICH_STATE_KEY
    }


def test_metadata_is_frozen_and_marker_survives_mixed_precision_casts():
    model = model_module.GenMol(mask_config())
    with pytest.raises(FrozenInstanceError):
        model.udlm_prior_metadata.mask_mixture_weight = 0.8
    marker = model.state_dict()[model_module.UDLM_MASK_RICH_STATE_KEY].clone()
    for dtype in (torch.bfloat16, torch.float32, torch.float16):
        model.to(dtype=dtype)
        model._validate_runtime_udlm_prior_identity()
        assert torch.equal(
            model.state_dict()[model_module.UDLM_MASK_RICH_STATE_KEY], marker
        )
        assert marker.dtype == torch.int64
        assert model.mdlm.stationary_probs.dtype == torch.float64


@pytest.mark.parametrize("ce", [False, True])
def test_matching_new_checkpoint_roundtrip_and_metadata_tampering(monkeypatch, ce):
    model = model_module.GenMol(mask_config(ce=ce))
    checkpoint = _run_checkpoint_save_hook(monkeypatch, model)
    monkeypatch.setattr(model_module, "fast_forward_info", lambda checkpoint: (0, 0))
    restored = model_module.GenMol(mask_config(ce=ce))
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    for field, value in (
        ("mask_mixture_weight", 0.8),
        ("mask_token_id", 5),
        ("base_stationary_probs_sha256", "0" * 64),
    ):
        changed = copy.deepcopy(checkpoint)
        changed[model_module.UDLM_PRIOR_CHECKPOINT_KEY][field] = value
        with pytest.raises(ValueError, match="prior metadata"):
            restored.on_load_checkpoint(changed)


@pytest.mark.parametrize("strict", [False, True])
def test_lambda_zero_cannot_relabel_empirical_state_or_checkpoint(monkeypatch, strict):
    base = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="empirical_frequency")
    )
    model = model_module.GenMol(mask_config(0.0))
    assert torch.equal(base.mdlm.stationary_probs, model.mdlm.stationary_probs)
    with pytest.raises(ValueError, match="exact mixture marker"):
        model.load_state_dict(base.state_dict(), strict=strict)
    with pytest.raises(ValueError, match="cannot be loaded as another"):
        base.load_state_dict(model.state_dict(), strict=strict)
    checkpoint = _run_checkpoint_save_hook(monkeypatch, base)
    with pytest.raises(ValueError, match="prior metadata"):
        model.on_load_checkpoint(checkpoint)


def test_changed_lambda_marker_or_runtime_config_is_rejected():
    model = model_module.GenMol(mask_config())
    changed = copy.deepcopy(model.state_dict())
    changed[model_module.UDLM_MASK_RICH_STATE_KEY] = torch.tensor(
        0.8, dtype=torch.float64
    ).view(torch.int64)
    with pytest.raises(ValueError, match="exact mixture marker"):
        model.load_state_dict(changed, strict=False)
    model.config.training.udlm.mask_mixture_weight = 0.8
    with pytest.raises(RuntimeError, match="configuration changed"):
        model._validate_runtime_udlm_prior_identity()


def test_ce_training_keeps_clean_controls_immutable_and_masks_gradients(monkeypatch):
    model = model_module.GenMol(mask_config(ce=True))
    model.log = lambda *args, **kwargs: None
    ids = torch.tensor([[1, 5, 4, 6, 0, 2, 3]])
    attention = torch.tensor([[True, True, True, True, True, True, False]])
    original_attention = attention.clone()
    expected_mask = torch.tensor([[False, True, False, True, False, False, False]])
    logits = torch.randn(1, 7, 11, requires_grad=True)

    def corrupt(x, t, mutable_mask):
        assert torch.equal(mutable_mask, expected_mask)
        return torch.where(mutable_mask, torch.full_like(x, model.mask_index), x)

    monkeypatch.setattr(model.mdlm, "forward_process", corrupt)
    monkeypatch.setattr(
        model.mdlm, "loss", lambda *a, **kw: pytest.fail("CE must not call CT loss")
    )
    monkeypatch.setattr(model, "forward", lambda *a, **kw: logits)
    loss = model.training_step({"input_ids": ids, "attention_mask": attention}, 0)
    loss.backward()
    assert torch.equal(attention, original_attention)
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[~expected_mask]) == 0
    expected_gradient = torch.softmax(logits.detach()[expected_mask], -1)
    expected_gradient[torch.arange(2), ids[expected_mask]] -= 1
    torch.testing.assert_close(logits.grad[expected_mask], expected_gradient / 2)


def test_ce_to_loo_recovers_oracle_with_mask_rich_prior_and_mask_observation():
    model = model_module.GenMol(mask_config(ce=True))
    process = model.mdlm
    xt = torch.tensor([[4, 5]])
    t, s = torch.tensor([0.7], dtype=torch.float64), torch.tensor(
        [0.2], dtype=torch.float64
    )
    alpha_t, alpha_s = process.alpha(t).item(), process.alpha(s).item()
    pi = process.stationary_probs
    r = torch.arange(1, 12, dtype=torch.float64)
    r /= r.sum()
    likelihood = (1 - alpha_t) * pi[
        xt[..., None]
    ] + alpha_t * torch.nn.functional.one_hot(xt, 11).to(torch.float64)
    d = r * likelihood
    d /= d.sum(-1, keepdim=True)
    converted = model.sampling_logits(
        d.log(), xt, t, mutable_mask=torch.ones_like(xt, dtype=torch.bool)
    )
    torch.testing.assert_close(
        process.clean_log_probs(converted).exp(),
        r.expand(1, 2, 11),
        atol=1e-12,
        rtol=1e-12,
    )
    probabilities = process.posterior_probs(converted, xt, t, s)
    for position, observed in enumerate(xt[0].tolist()):
        transition = (
            (1 - alpha_t / alpha_s) * pi[observed] * torch.ones(11, dtype=torch.float64)
        )
        transition[observed] += alpha_t / alpha_s
        expected = transition * (alpha_s * r + (1 - alpha_s) * pi)
        expected /= expected.sum()
        torch.testing.assert_close(
            probabilities[0, position], expected, atol=1e-12, rtol=1e-12
        )
