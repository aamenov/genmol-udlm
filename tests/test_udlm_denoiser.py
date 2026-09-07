"""CPU proofs of CE interpretation, masking, checkpoint identity and sampling."""

import copy
from pathlib import Path

import lightning
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import genmol.model as model_module
import genmol.sampler as sampler_module
from genmol.denoiser import clean_denoiser_loss, denoiser_to_loo_logits
from genmol.diffusion import ContinuousCategoricalDiffusion, ContinuousUniformDiffusion
from test_udlm_model import (
    _Tokenizer,
    _config,
    _frequency_payload,
    _install_frequency_artifact,
    _run_checkpoint_save_hook,
)
from test_udlm_sampler import _UDLMModel, _sampler


@pytest.fixture(autouse=True)
def tokenizer(monkeypatch):
    monkeypatch.setattr(model_module, "get_tokenizer", lambda: _Tokenizer())


def ce_config():
    config = _config(
        diffusion="udlm", prior_variant="schedule_uniform", exclude_special=True
    )
    config.training.udlm.parameterization = "x0_denoiser"
    return config


@pytest.mark.parametrize("prior", [[1 / 3] * 3, [0.72, 0.23, 0.05]])
@pytest.mark.parametrize("times", [(0.8, 0.3), (0.9, 0.0), (1e-5, 1e-6)])
def test_conversion_recovers_loo_and_exact_posterior_mixture(prior, times):
    process = ContinuousCategoricalDiffusion(5, prior, excluded_token_ids=[1, 3])
    t, s = [torch.tensor([value], dtype=torch.float64) for value in times]
    xt = torch.tensor([[0, 2, 4]])
    r = torch.tensor(
        [[[0.2, 0.3, 0.5], [0.7, 0.1, 0.2], [0.15, 0.6, 0.25]]], dtype=torch.float64
    )
    alpha_t, alpha_s = process.alpha(t).item(), process.alpha(s).item()
    pi = torch.tensor(prior, dtype=torch.float64)
    pi /= pi.sum()
    current = torch.arange(3)
    likelihood = (1 - alpha_t) * pi[current, None] + alpha_t * torch.eye(
        3, dtype=torch.float64
    )
    d = r * likelihood
    d /= d.sum(-1, keepdim=True)
    logits = torch.full((1, 3, 5), 13.0, dtype=torch.float64)
    logits[..., process.diffusion_token_ids] = d.log()
    converted = denoiser_to_loo_logits(process, logits, xt, t)
    torch.testing.assert_close(
        process.clean_log_probs(converted).exp(), r, atol=1e-11, rtol=1e-11
    )
    observed = process.posterior_probs(converted, xt, t, s)
    # Independently mix each normalized bridge for a specified clean token.
    transition = (1 - alpha_t / alpha_s) * pi[current, None] + (
        alpha_t / alpha_s
    ) * torch.eye(3, dtype=torch.float64)
    clean_to_s = (
        alpha_s * torch.eye(3, dtype=torch.float64) + (1 - alpha_s) * pi[None, :]
    )
    mixture = torch.stack(
        [
            sum(
                d[0, k, j] * transition[k] * clean_to_s[j] / likelihood[k, j]
                for j in range(3)
            )
            for k in range(3)
        ]
    )[None]
    torch.testing.assert_close(observed, mixture, atol=1e-11, rtol=1e-11)
    assert torch.equal(converted[..., [1, 3]], logits[..., [1, 3]])


def test_conversion_preserves_small_prior_in_log_space_and_immutable_context():
    process = ContinuousCategoricalDiffusion(
        4, [1.0, 1e-100], excluded_token_ids=[0, 3]
    )
    logits = torch.zeros(1, 3, 4, dtype=torch.float32)
    xt = torch.tensor([[0, 2, 3]])
    mask = torch.tensor([[False, True, False]])
    t = torch.tensor([1e-5])
    converted = denoiser_to_loo_logits(process, logits, xt, t, mutable_mask=mask)
    assert torch.isfinite(converted).all()
    assert converted[0, 1, 1] > 230
    assert torch.equal(converted[~mask], logits[~mask])
    assert torch.count_nonzero(logits) == 0


@pytest.mark.parametrize("time", [0.0, -0.1, 1.1, float("nan")])
def test_conversion_rejects_invalid_time(time):
    process = ContinuousCategoricalDiffusion(2, [0.7, 0.3])
    with pytest.raises(ValueError, match="0 < t"):
        denoiser_to_loo_logits(
            process,
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, dtype=torch.long),
            torch.tensor([time]),
        )


def test_ce_reduction_and_gradient_masking_include_unchanged_tokens():
    process = ContinuousCategoricalDiffusion(
        5, [0.7, 0.3], excluded_token_ids=[0, 1, 4]
    )
    logits = torch.tensor(
        [
            [
                [9.0, 8.0, 0.2, -0.3, 7.0],
                [2.0, 3.0, 1.0, 2.0, 4.0],
                [8.0, 7.0, 0.5, -0.4, 9.0],
            ],
            [
                [2.0, 3.0, 0.1, 0.9, 4.0],
                [2.0, 3.0, 0.6, -0.2, 4.0],
                [2.0, 3.0, 1.0, 2.0, 4.0],
            ],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 2, 3], [2, 3, 4]])
    mask = torch.tensor([[False, True, False], [True, True, False]])
    local = clean_denoiser_loss(process, logits, targets, mask=mask)
    global_loss = clean_denoiser_loss(
        process, logits, targets, mask=mask, global_mean=True
    )
    selected = F.cross_entropy(
        logits[mask][:, [2, 3]], targets[mask] - 2, reduction="none"
    )
    torch.testing.assert_close(local, torch.stack([selected[0], selected[1:].mean()]))
    torch.testing.assert_close(global_loss, selected.mean())
    global_loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert torch.count_nonzero(logits.grad[..., [0, 1, 4]]) == 0
    assert torch.all(logits.grad[mask][:, [2, 3]].abs() > 0)


def test_ce_rejects_empty_or_excluded_selected_targets_and_legacy_process():
    process = ContinuousCategoricalDiffusion(3, [0.5, 0.5], excluded_token_ids=[0])
    logits, ids = torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="active-alphabet"):
        clean_denoiser_loss(
            process, logits, ids, mask=torch.ones_like(ids, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="no active"):
        clean_denoiser_loss(
            process, logits, ids, mask=torch.zeros_like(ids, dtype=torch.bool)
        )
    with pytest.raises(TypeError, match="schedule-consistent"):
        clean_denoiser_loss(
            ContinuousUniformDiffusion(3),
            logits,
            ids,
            mask=torch.ones_like(ids, dtype=torch.bool),
        )


def test_model_training_step_uses_ce_and_preserves_control_tokens(monkeypatch):
    model = model_module.GenMol(ce_config())
    model.log = lambda *args, **kwargs: None
    model.mdlm.loss = lambda *args, **kwargs: pytest.fail("CE must not call CT loss")
    # Force identity corruption: unchanged tokens still contribute CE gradient.
    monkeypatch.setattr(model.mdlm, "forward_process", lambda x, t, mutable_mask: x)
    ids = torch.tensor([[1, 5, 6, 0, 4, 2, 3]])
    attention = torch.tensor([[True, True, True, True, True, True, False]])
    original_attention = attention.clone()
    mask = model.diffusion_token_mask(ids, attention)
    assert torch.equal(
        mask, torch.tensor([[False, True, True, False, False, False, False]])
    )
    assert torch.equal(attention, original_attention)
    loss = model.training_step({"input_ids": ids, "attention_mask": attention}, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.bert.embeddings.word_embeddings.weight.grad.norm() > 0


def test_old_default_is_identical_and_does_not_mutate_config_or_rng():
    old_config = _config(diffusion="udlm", prior_variant="schedule_uniform")
    old_snapshot = OmegaConf.to_container(old_config)
    explicit_config = copy.deepcopy(old_config)
    explicit_config.training.udlm.parameterization = "raw_loo"
    torch.manual_seed(141)
    old = model_module.GenMol(old_config)
    old_rng = torch.random.get_rng_state()
    torch.manual_seed(141)
    explicit = model_module.GenMol(explicit_config)
    assert torch.equal(old_rng, torch.random.get_rng_state())
    assert OmegaConf.to_container(old_config) == old_snapshot
    assert old.state_dict().keys() == explicit.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    logits = torch.randn(1, 2, 11)
    before = torch.random.get_rng_state()
    assert old.sampling_logits(logits, None, None) is logits
    assert torch.equal(before, torch.random.get_rng_state())


def test_ce_and_ct_start_with_identical_backbone_conditioning_and_rng():
    config = ce_config()
    torch.manual_seed(901)
    ce = model_module.GenMol(config)
    ce_rng = torch.random.get_rng_state()
    ct_config = copy.deepcopy(config)
    del ct_config.training.udlm.parameterization
    torch.manual_seed(901)
    ct = model_module.GenMol(ct_config)
    assert torch.equal(ce_rng, torch.random.get_rng_state())
    for key, value in ce.backbone.state_dict().items():
        assert torch.equal(value, ct.backbone.state_dict()[key])


def test_ce_empirical_prior_uses_verified_artifact(monkeypatch, tmp_path):
    _install_frequency_artifact(monkeypatch, tmp_path, _frequency_payload())
    config = ce_config()
    config.training.udlm.prior_variant = "empirical_frequency"
    model = model_module.GenMol(config)
    assert model.udlm_prior_metadata.variant == "empirical_frequency"
    assert model.udlm_parameterization == "x0_denoiser"
    assert model.mdlm.stationary_probs.max() > model.mdlm.stationary_probs.min()


@pytest.mark.parametrize(
    "diffusion,prior", [("mdlm", "schedule_uniform"), ("udlm", "release_uniform")]
)
def test_ce_rejects_unsupported_process(diffusion, prior):
    config = ce_config()
    config.training.diffusion = diffusion
    config.training.udlm.prior_variant = prior
    with pytest.raises(ValueError, match="schedule-consistent"):
        model_module.GenMol(config)


def test_checkpoint_round_trip_and_cross_parameterization_rejection(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(model_module, "fast_forward_info", lambda checkpoint: (0, 0))
    model = model_module.GenMol(ce_config())
    checkpoint = _run_checkpoint_save_hook(monkeypatch, model)
    checkpoint.update(
        {
            "hyper_parameters": {"config": model.config},
            "pytorch-lightning_version": lightning.__version__,
        }
    )
    checkpoint_path = tmp_path / "ce.ckpt"
    torch.save(checkpoint, checkpoint_path)
    loaded = model_module.GenMol.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    assert loaded.udlm_parameterization == "x0_denoiser"
    assert (
        checkpoint[model_module.UDLM_DENOISER_CHECKPOINT_KEY]["objective"]
        == "clean_token_cross_entropy"
    )
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    raw_config = copy.deepcopy(model.config)
    del raw_config.training.udlm.parameterization
    raw = model_module.GenMol(raw_config)
    raw_checkpoint = _run_checkpoint_save_hook(monkeypatch, raw)
    with pytest.raises(ValueError, match="metadata"):
        model.on_load_checkpoint(raw_checkpoint)
    with pytest.raises(ValueError, match="raw_loo"):
        raw.on_load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="marker"):
        model.load_state_dict(raw.state_dict(), strict=False)
    with pytest.raises(ValueError, match="raw_loo"):
        raw.load_state_dict(model.state_dict(), strict=False)
    with pytest.raises(ValueError, match="raw_loo"):
        model_module.GenMol.load_from_checkpoint(
            checkpoint_path, config=raw_config, map_location="cpu"
        )


def test_ce_sampler_converts_both_fresh_predictor_and_gibbs_logits(monkeypatch):
    model = _UDLMModel()
    model.udlm_parameterization = "x0_denoiser"
    process = ContinuousCategoricalDiffusion(
        9, [0.1, 0.2, 0.3, 0.15, 0.25], excluded_token_ids=[1, 2, 3, 4]
    )
    model.sampling_logits = lambda logits, xt, t, mutable_mask: denoiser_to_loo_logits(
        process, logits, xt, t, mutable_mask=mutable_mask
    )
    sampler = _sampler(model, process, "udlm")
    recorded = []
    original_step = process.step
    original_corrector = sampler_module.random_scan_gibbs_step

    def check(logits, xt, t, mask, kind):
        model_xt, _, model_t = model.calls[-1]
        assert torch.equal(model_xt, xt) and torch.equal(model_t, t)
        raw = torch.zeros(*xt.shape, 9)
        expected = denoiser_to_loo_logits(process, raw, xt, t, mutable_mask=mask)
        torch.testing.assert_close(logits, expected)
        assert not torch.equal(logits[mask], raw[mask])
        recorded.append(kind)

    def step(logits, xt, t, s, **kwargs):
        check(logits, xt, t, kwargs["mutable_mask"], "predictor")
        assert kwargs["temperature"] == 0.5 and kwargs["raw_loo_top_p"] == 0.8
        return original_step(logits, xt, t, s, **kwargs)

    def corrector(diffusion, logits, xt, s, **kwargs):
        check(logits, xt, s, kwargs["mutable_mask"], "corrector")
        assert kwargs["temperature"] == 0.5 and kwargs["raw_loo_top_p"] == 0.8
        return original_corrector(diffusion, logits, xt, s, **kwargs)

    monkeypatch.setattr(process, "step", step)
    monkeypatch.setattr(sampler_module, "random_scan_gibbs_step", corrector)
    ids = torch.tensor([[1, 4, 4, 2, 3]])
    result = sampler.generate(
        ids,
        num_steps=4,
        gibbs_corrector=True,
        softmax_temp=0.5,
        raw_loo_top_p=0.8,
        return_token_ids=True,
    )
    assert recorded == ["predictor", "corrector", "predictor", "corrector"]
    assert len(model.calls) == 4
    assert torch.equal(ids[:, [0, 3, 4]], result[:, [0, 3, 4]])


def test_ce_hydra_config_is_explicit_and_composes_without_data_or_checkpoint():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        version_base=None,
    ):
        config = compose(config_name="udlm_ce")
    assert config.training.udlm.parameterization == "x0_denoiser"
    assert config.training.udlm.prior_variant == "schedule_uniform"
    assert config.training.udlm.exclude_special_tokens is True
