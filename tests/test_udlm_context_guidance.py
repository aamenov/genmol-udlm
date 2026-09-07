"""Synthetic CPU categorical sampler checks; no checkpoints or chemistry."""

import random

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from genmol import context_guidance as guidance
from genmol.denoiser import denoiser_to_loo_logits
from genmol.diffusion import ContinuousCategoricalDiffusion
from genmol.model import GenMol, _build_udlm_process
from genmol.sampler import Sampler, _inference_weights_receipt
from scripts.udlm.audit_context_guidance import guide_exact, reverse_mixture


class SyntheticTokenizer:
    all_special_ids = [0, 1, 2, 3, 4, 7]
    mask_token_id, bos_token_id, eos_token_id, pad_token_id, unk_token_id = (
        4,
        1,
        2,
        3,
        0,
    )


class SyntheticBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, x, attention_mask, *, t):
        self.calls.append((x.clone(), attention_mask.clone(), t.clone()))
        classes = torch.arange(9, dtype=torch.float64).view(1, 1, 9)
        center = (x * attention_mask).sum(-1).remainder(9)[:, None, None]
        return (-(classes - center).square() / 8 + classes * t[:, None, None]).expand(
            *x.shape, 9
        )


class SyntheticModel(torch.nn.Module):
    """Real process/identity validation, deterministic synthetic predictions."""

    _validate_runtime_udlm_prior_identity = GenMol._validate_runtime_udlm_prior_identity
    _validate_udlm_parameterization_state_dict = (
        GenMol._validate_udlm_parameterization_state_dict
    )
    sampling_logits = GenMol.sampling_logits
    diffusion_type = "udlm"
    device = torch.device("cpu")
    mask_index, bos_index, eos_index, pad_index = 4, 1, 2, 3

    def __init__(self, parameterization="raw_loo"):
        super().__init__()
        self.tokenizer = SyntheticTokenizer()
        self.backbone = SyntheticBackbone()
        self.udlm_parameterization = parameterization
        self.config = OmegaConf.create(
            dict(
                training=dict(
                    udlm=dict(
                        parameterization=parameterization,
                        prior_variant="schedule_uniform",
                        sampling_steps=3,
                        inference_eps=1e-5,
                    )
                )
            )
        )
        self.mdlm, self.udlm_prior_metadata = _build_udlm_process(
            variant="schedule_uniform",
            model_vocab_size=9,
            excluded_token_ids=(1, 2, 3),
            sampling_eps=0.001,
            noise_eps=0.001,
            antithetic_sampling=True,
            empirical_uniform_mix=0.0002,
            tokenizer=self.tokenizer,
        )
        if parameterization == "x0_denoiser":
            self.register_buffer(
                "_udlm_denoiser_ce_version", torch.tensor(1, dtype=torch.int64)
            )
        self.eval()

    def forward(self, x, attention_mask=None, t=None):
        return self.backbone(x, attention_mask, t=t)


def make_sampler(parameterization="raw_loo"):
    sampler = Sampler.__new__(Sampler)
    sampler.model = SyntheticModel(parameterization)
    sampler.mdlm = sampler.model.mdlm
    sampler.diffusion_type, sampler.pad_index = "udlm", 3
    sampler._inference_weights = _inference_weights_receipt(
        "ema", True, {"shadow_parameter_count": 1, "decay": 0.999, "num_updates": 1000}
    )
    return sampler


def configuration(**changes):
    value = dict(
        schema_version=1,
        method="posterior_context",
        gamma=1.0,
        scale=2.0,
        context_seed=419,
        predictor_steps=3,
        execution="serial",
    )
    value.update(changes)
    return value


def inputs():
    return torch.tensor([[1, 8, 4, 2, 3], [1, 4, 6, 2, 3], [1, 4, 7, 2, 3]])


def seed_all():
    random.seed(987)
    np.random.seed(987)
    torch.manual_seed(987)


def rng_state():
    numpy = np.random.get_state()
    return (
        random.getstate(),
        (numpy[0], numpy[1].tobytes(), *numpy[2:]),
        torch.get_rng_state().clone(),
    )


@pytest.mark.parametrize("parameterization", ["raw_loo", "x0_denoiser"])
@pytest.mark.parametrize(
    "option,reason",
    [
        ({"gamma": 0}, "gamma_zero"),
        ({"scale": 1}, "scale_one"),
        ({"gamma": 0.1}, "no_selected_context"),
    ],
)
def test_noop_ids_and_all_rng_states_exactly_match_historical_generate(
    parameterization, option, reason, monkeypatch
):
    old, new = make_sampler(parameterization), make_sampler(parameterization)
    seed_all()
    expected = old.generate(
        inputs(), softmax_temp=1.0, randomness=0, num_steps=3, return_token_ids=True
    )
    expected_rng = rng_state()
    seed_all()
    monkeypatch.setattr(
        torch,
        "Generator",
        lambda *a, **k: pytest.fail("No context RNG on an identity path"),
    )
    output = new.generate_context_guided(inputs(), config=configuration(**option))
    assert torch.equal(expected, output["token_ids"])
    actual_rng = rng_state()
    assert actual_rng[:2] == expected_rng[:2] and torch.equal(
        actual_rng[2], expected_rng[2]
    )
    receipt = output["receipt"]
    assert receipt["observations"]["noop_reason"] == reason
    assert receipt["observations"]["backbone_batch_sizes"] == [3] * 3
    assert receipt["observations"]["candidate_equivalent_evaluations"] == 9
    assert receipt["context_policy"]["subset_draws"] == 0
    assert receipt["context_policy"]["selected_counts"] == [0, 0, 0]


@pytest.mark.parametrize("parameterization", ["raw_loo", "x0_denoiser"])
@pytest.mark.parametrize("execution", ["serial", "packed"])
def test_active_guidance_context_clamping_support_and_observed_batch_cost(
    parameterization, execution, monkeypatch
):
    sampler = make_sampler(parameterization)
    original = inputs()
    laws = []
    draw = torch.multinomial

    def capture(probabilities, *args, **kwargs):
        if probabilities.ndim == 2 and probabilities.shape[0] == original.numel():
            laws.append(probabilities.clone())
        return draw(probabilities, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", capture)
    seed_all()
    before_py, before_np, _ = rng_state()
    output = sampler.generate_context_guided(
        original, config=configuration(execution=execution)
    )
    calls = sampler.model.backbone.calls
    receipt = output["receipt"]
    assert receipt["context_policy"]["eligible_counts"] == [1, 1, 0]
    assert receipt["context_policy"]["selected_counts"] == [1, 1, 0]
    assert receipt["context_policy"]["subset_draws"] == 6
    assert receipt["observations"]["backbone_batch_sizes"] == (
        [6] * 3 if execution == "packed" else [3] * 6
    )
    assert receipt["observations"]["candidate_equivalent_evaluations"] == 18
    assert receipt["observations"]["evaluations_per_candidate"] == 6
    assert receipt["identity"]["checkpoint_bytes_revalidated"] is False
    assert before_py == rng_state()[0] and before_np == rng_state()[1]
    assert torch.equal(output["token_ids"][original != 4], original[original != 4])
    assert torch.equal(original, inputs())
    for step in range(3):
        if execution == "packed":
            c, u = calls[step][0].split(3)
            attention_c, attention_u = calls[step][1].split(3)
            t_c, t_u = calls[step][2].split(3)
        else:
            c, attention_c, t_c = calls[2 * step]
            u, attention_u, t_u = calls[2 * step + 1]
        assert torch.equal(c[original == 4], u[original == 4])
        assert torch.equal(attention_c, attention_u) and torch.equal(t_c, t_u)
        expected = c.clone()
        expected[0, 1] = expected[1, 2] = 4
        assert torch.equal(u, expected)
        assert u[2, 2] == 7  # extra tokenizer control must never be hidden
    assert len(laws) == 3
    for law in laws:
        assert (
            law.dtype == torch.float64
            and torch.isfinite(law).all()
            and torch.all(law > 0)
        )
        torch.testing.assert_close(
            law.sum(-1),
            torch.ones(law.shape[0], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


def test_context_subset_rng_is_separate_and_reproducible():
    x = torch.tensor([[1, 5, 6, 8, 4, 2], [1, 8, 6, 5, 4, 2]])
    config = configuration(gamma=0.5, predictor_steps=5)
    outputs = []
    for sample_seed in (71, 72):
        torch.manual_seed(sample_seed)
        outputs.append(make_sampler().generate_context_guided(x, config=config))
    assert (
        outputs[0]["receipt"]["context_policy"]["selection_sha256"]
        == outputs[1]["receipt"]["context_policy"]["selection_sha256"]
    )
    changed = make_sampler().generate_context_guided(
        x, config=configuration(gamma=0.5, predictor_steps=5, context_seed=420)
    )
    assert (
        changed["receipt"]["context_policy"]["selection_sha256"]
        != outputs[0]["receipt"]["context_policy"]["selection_sha256"]
    )


def test_packed_and_serial_reverse_laws_agree_for_deterministic_synthetic_predictor(
    monkeypatch,
):
    seen = []
    original = guidance.blend_reverse_laws

    def capture(*args):
        value = original(*args)
        seen.append(value)
        return value

    monkeypatch.setattr(guidance, "blend_reverse_laws", capture)
    for execution in ("serial", "packed"):
        seed_all()
        make_sampler("x0_denoiser").generate_context_guided(
            inputs(), config=configuration(execution=execution)
        )
    for serial, packed in zip(seen[:3], seen[3:]):
        torch.testing.assert_close(serial, packed, atol=1e-12, rtol=1e-12)


def test_actual_ce_posterior_blend_matches_exact_fraction_counterexample():
    from fractions import Fraction as F

    pi = [F(1, 5), F(1, 2), F(3, 10)]
    d_c, d_u = [F(3, 5), F(3, 10), F(1, 10)], [F(1, 5), F(1, 5), F(3, 5)]
    process = ContinuousCategoricalDiffusion(
        5, [float(p) for p in pi], excluded_token_ids=[1, 3]
    )
    t = torch.tensor([(1 - 1 / 3) / (1 - process.noise_eps)], dtype=torch.float64)
    s = torch.tensor([(1 - 2 / 3) / (1 - process.noise_eps)], dtype=torch.float64)
    x, editable = torch.tensor([[0, 1]]), torch.tensor([[True, False]])
    laws = []
    for d in (d_c, d_u):
        logits = torch.full((1, 2, 5), float("nan"), dtype=torch.float64)
        logits[0, 0, process.diffusion_token_ids] = torch.tensor(
            [float(v) for v in d], dtype=torch.float64
        ).log()
        laws.append(
            guidance.posterior_law(process, logits, x, t, s, editable, "x0_denoiser")
        )
    result = guidance.blend_reverse_laws(*laws, 2)[0, 0]
    expected = guide_exact(
        reverse_mixture(d_c, pi, 0, F(2, 3), F(1, 3)),
        reverse_mixture(d_u, pi, 0, F(2, 3), F(1, 3)),
        2,
    )
    torch.testing.assert_close(
        result,
        torch.tensor([float(v) for v in expected], dtype=torch.float64),
        atol=1e-14,
        rtol=1e-14,
    )


def test_ce_and_raw_branches_receive_original_current_tokens(monkeypatch):
    sampler = make_sampler("x0_denoiser")
    observed = []
    import genmol.denoiser as denoiser

    def convert(process, logits, x, t, **kwargs):
        observed.append((x.clone(), t.clone(), kwargs["mutable_mask"].clone()))
        return denoiser_to_loo_logits(process, logits, x, t, **kwargs)

    monkeypatch.setattr(denoiser, "denoiser_to_loo_logits", convert)
    sampler.generate_context_guided(inputs(), config=configuration())
    assert len(observed) == 6
    for first, second in zip(observed[::2], observed[1::2]):
        assert all(torch.equal(a, b) for a, b in zip(first, second))
        assert first[0][0, 1] == 8 and first[0][1, 2] == 6
        assert torch.equal(first[2], inputs() == 4)


@pytest.mark.parametrize("problem", ["nonfinite", "zero", "underflow"])
def test_numerical_support_loss_is_rejected_without_flooring(problem):
    c = torch.tensor([[[0.5, 0.25, 0.25]]], dtype=torch.float64)
    u = torch.tensor([[[0.2, 0.4, 0.4]]], dtype=torch.float64)
    scale = 2
    if problem == "nonfinite":
        c[..., 0] = float("nan")
    elif problem == "zero":
        u = torch.tensor([[[0.0, 0.5, 0.5]]], dtype=torch.float64)
    else:
        scale = 2000
    with pytest.raises(ValueError, match="finite positive support"):
        guidance.blend_reverse_laws(c, u, scale)


def test_nonfinite_editable_logits_fail_and_remove_observation_hook():
    sampler = make_sampler()
    handle = sampler.model.backbone.register_forward_hook(
        lambda module, args, value: torch.full_like(value, float("nan"))
    )
    before = len(sampler.model.backbone._forward_pre_hooks)
    with pytest.raises(ValueError, match="nonfinite active predictor"):
        sampler.generate_context_guided(inputs(), config=configuration())
    assert len(sampler.model.backbone._forward_pre_hooks) == before
    handle.remove()


def test_training_mode_and_mutated_runtime_prior_fail_before_rng_or_prediction():
    for problem in ("training", "prior"):
        sampler = make_sampler()
        if problem == "training":
            sampler.model.backbone.train()
        else:
            sampler.mdlm.stationary_probs[0] += 0.01
        before = torch.get_rng_state().clone()
        with pytest.raises((ValueError, RuntimeError)):
            sampler.generate_context_guided(inputs(), config=configuration())
        assert not sampler.model.backbone.calls and torch.equal(
            before, torch.get_rng_state()
        )


def test_unused_excluded_sentinel_underflow_does_not_reject_valid_editable_laws():
    process = ContinuousCategoricalDiffusion(
        4, [1e-320, 0.5, 0.5], excluded_token_ids=[1]
    )
    logits = torch.zeros((1, 2, 4), dtype=torch.float64)
    x, editable = torch.tensor([[1, 2]]), torch.tensor([[False, True]])
    t, s = (
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.499999], dtype=torch.float64),
    )
    raw = process.posterior_probs(logits, torch.tensor([[0, 2]]), t, s)
    assert torch.any(raw[0, 0] == 0) and torch.all(raw[0, 1] > 0)
    law = guidance.posterior_law(process, logits, x, t, s, editable, "raw_loo")
    torch.testing.assert_close(
        law[0, 0], torch.full((3,), 1 / 3, dtype=torch.float64), atol=0, rtol=0
    )
    torch.testing.assert_close(law[0, 1], raw[0, 1], atol=0, rtol=0)


@pytest.mark.parametrize(
    "ema", [None, {}, {"shadow_parameter_count": 1, "decay": 0.99, "num_updates": True}]
)
def test_missing_or_malformed_ema_metadata_is_not_advertised(ema):
    sampler = make_sampler()
    sampler._inference_weights = _inference_weights_receipt("ema", True, ema)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="EMA"):
        sampler.generate_context_guided(inputs(), config=configuration())
    assert not sampler.model.backbone.calls and torch.equal(
        before, torch.get_rng_state()
    )
