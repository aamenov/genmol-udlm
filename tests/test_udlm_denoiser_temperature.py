"""CPU oracle checks for the opt-in clean-posterior temperature hypothesis."""

import pytest
import torch

from genmol.denoiser import denoiser_to_loo_logits
from genmol.diffusion import ContinuousCategoricalDiffusion
from genmol.model import GenMol
from test_udlm_sampler import (
    _ContextualToyModel,
    _MDLMModel,
    _MDLMProcess,
    _UDLMModel,
    _UDLMProcess,
    _historical_predictor_only,
    _sampler,
)


def posterior_mixture(prior, denoiser, current, t, s, noise_eps):
    """Enumerate clean j and earlier y using independently normalized bridges."""
    prior = torch.tensor(prior, dtype=torch.float64)
    prior /= prior.sum()
    alpha_t = 1 - (1 - noise_eps) * t
    alpha_s = 1 - (1 - noise_eps) * s
    identity = torch.eye(len(prior), dtype=torch.float64)
    q_t = alpha_t * identity + (1 - alpha_t) * prior[None]
    q_s = alpha_s * identity + (1 - alpha_s) * prior[None]
    q_st = alpha_t / alpha_s * identity + (1 - alpha_t / alpha_s) * prior[None]
    return sum(
        denoiser[j] * q_s[j] * q_st[:, current] / q_t[j, current]
        for j in range(len(prior))
    )


@pytest.mark.parametrize("temperature", [0.5, 1.0])
@pytest.mark.parametrize("prior", [[1 / 3] * 3, [0.9, 0.0999, 0.0001]])
@pytest.mark.parametrize("times", [(0.8, 0.3), (0.5, 0.0), (1e-5, 1e-6)])
def test_direct_clean_temperature_matches_independent_posterior_mixture(
    temperature, prior, times
):
    process = ContinuousCategoricalDiffusion(5, prior, excluded_token_ids=[1, 3])
    t, s = (torch.tensor([v], dtype=torch.float64) for v in times)
    current = torch.tensor([[0, 2, 4]])
    d = torch.tensor(
        [[0.001, 0.009, 0.99], [0.2, 0.3, 0.5], [0.1, 0.85, 0.05]], dtype=torch.float64
    )
    logits = torch.full((1, 3, 5), 17.0, dtype=torch.float64)
    logits[..., process.diffusion_token_ids] = d.log()
    original = logits.clone()
    converted = denoiser_to_loo_logits(
        process, logits, current, t, denoiser_temperature=temperature
    )
    observed = process.posterior_probs(converted, current, t, s, temperature=1.0)
    tempered_d = d.pow(1 / temperature)
    tempered_d /= tempered_d.sum(-1, keepdim=True)
    expected = torch.stack(
        [
            posterior_mixture(prior, tempered_d[k], k, *times, process.noise_eps)
            for k in range(3)
        ]
    )[None]
    torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-10)
    assert torch.equal(logits, original)
    assert torch.equal(converted[..., [1, 3]], logits[..., [1, 3]])
    old = denoiser_to_loo_logits(process, logits, current, t)
    if temperature == 1:
        assert torch.equal(converted, old)
        assert torch.equal(observed, process.posterior_probs(old, current, t, s))


def test_old_implied_denoiser_contains_inverse_likelihood_factor():
    prior = [0.9, 0.0999, 0.0001]
    process = ContinuousCategoricalDiffusion(3, prior)
    d = torch.tensor([0.001, 0.009, 0.99], dtype=torch.float64)
    current = torch.tensor([[2]])
    t, s = (
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.4], dtype=torch.float64),
    )
    old = denoiser_to_loo_logits(process, d.log()[None, None], current, t)
    likelihood = torch.full_like(d, (1 - process.alpha(t).item()) * prior[2])
    likelihood[2] += process.alpha(t).item()
    implied = process.clean_log_probs(old, temperature=0.5).exp()[0, 0] * likelihood
    implied /= implied.sum()
    formula = d.square() / likelihood
    formula /= formula.sum()
    torch.testing.assert_close(implied, formula, atol=1e-12, rtol=1e-12)
    new = denoiser_to_loo_logits(
        process, d.log()[None, None], current, t, denoiser_temperature=0.5
    )
    old_p = process.posterior_probs(old, current, t, s, temperature=0.5)
    new_p = process.posterior_probs(new, current, t, s, temperature=1)
    assert (old_p - new_p).abs().max() > 0.1


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_conversion_preserves_immutable_and_excluded_logits_and_promotes(dtype):
    process = ContinuousCategoricalDiffusion(
        5, [0.2, 0.8], excluded_token_ids=[0, 3, 4]
    )
    logits = torch.arange(15).reshape(1, 3, 5).to(dtype)
    inputs = torch.tensor([[0, 2, 4]])
    mask = torch.tensor([[False, True, False]])
    original = logits.clone()
    converted = denoiser_to_loo_logits(
        process,
        logits,
        inputs,
        torch.tensor([0.7]),
        mutable_mask=mask,
        denoiser_temperature=0.3,
    )
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert converted.dtype == expected_dtype
    assert torch.equal(converted[~mask], logits[~mask].to(expected_dtype))
    assert torch.equal(
        converted[..., [0, 3, 4]], logits[..., [0, 3, 4]].to(expected_dtype)
    )
    assert torch.equal(logits, original)
    expected = logits.to(expected_dtype).clone()
    expected[mask] /= 0.3
    expected = denoiser_to_loo_logits(
        process, expected, inputs, torch.tensor([0.7]), mutable_mask=mask
    )
    torch.testing.assert_close(
        converted[..., [1, 2]], expected[..., [1, 2]], atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "temperature", [True, False, "1", None, 0, -1, float("inf"), float("nan")]
)
def test_denoiser_temperature_rejects_invalid_scalars(temperature):
    process = ContinuousCategoricalDiffusion(2, [0.5, 0.5])
    with pytest.raises(ValueError, match="denoiser_temperature"):
        denoiser_to_loo_logits(
            process,
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, dtype=torch.long),
            torch.tensor([0.5]),
            denoiser_temperature=temperature,
        )


class CEModel(_ContextualToyModel):
    udlm_parameterization = "x0_denoiser"
    sampling_logits = GenMol.sampling_logits

    def __init__(self, process):
        super().__init__()
        self.mdlm = process


def ce_process(nonuniform):
    prior = torch.arange(1, 10, dtype=torch.float64) if nonuniform else torch.ones(9)
    return ContinuousCategoricalDiffusion(9, prior / prior.sum())


class HistoricalCEForward(CEModel):
    """Feed the preexisting predictor loop the historical converted CE logits."""

    def __call__(self, x, attention_mask, t=None):
        logits = super().__call__(x, attention_mask, t=t)
        return denoiser_to_loo_logits(
            self.mdlm, logits, x, t, mutable_mask=self.original_editable
        )


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("temperature", [0.5, 1.0])
@pytest.mark.parametrize("explicit_default", [False, True])
def test_default_ce_predictor_keeps_historical_ids_and_rng(
    nonuniform, temperature, explicit_default
):
    process = ce_process(nonuniform)
    inputs = torch.tensor([[1, 4, 4, 2, 3], [1, 4, 4, 4, 2]])
    historical = HistoricalCEForward(process)
    historical.original_editable = inputs == 4
    torch.manual_seed(771)
    expected = _historical_predictor_only(
        historical, process, inputs, 7, temperature, 1.0
    )
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(771)
    model = CEModel(process)
    actual = _sampler(model, process, "udlm").generate(
        inputs,
        num_steps=7,
        softmax_temp=temperature,
        return_token_ids=True,
        **({"temperature_space": "raw_loo"} if explicit_default else {}),
    )
    assert torch.equal(expected, actual)
    assert torch.equal(expected_rng, torch.random.get_rng_state())


@pytest.mark.parametrize("nonuniform", [False, True])
def test_temperature_one_new_mode_preserves_ids_rng_context_and_nfe(nonuniform):
    process = ce_process(nonuniform)
    inputs = torch.tensor([[1, 4, 4, 8, 2, 3], [1, 8, 4, 4, 2, 3]])
    results = []
    for space in ("raw_loo", "x0_denoiser"):
        model = CEModel(process)
        torch.manual_seed(1907)
        output = _sampler(model, process, "udlm").generate(
            inputs,
            num_steps=6,
            softmax_temp=1,
            temperature_space=space,
            return_token_ids=True,
        )
        assert len(model.calls) == 6
        assert torch.equal(output[inputs != 4], inputs[inputs != 4])
        results.append((output, torch.random.get_rng_state().clone()))
    assert all(torch.equal(old, new) for old, new in zip(*results))


def test_sampler_uses_current_time_clean_conversion_then_unit_bridge(monkeypatch):
    process = ce_process(True)
    model = CEModel(process)
    original_step = process.step
    observed = []
    inputs = torch.tensor([[1, 4, 4, 2, 3]])
    original_conversion = model.sampling_logits

    def conversion(logits, xt, t, *, mutable_mask, denoiser_temperature):
        assert denoiser_temperature == 0.5
        assert torch.equal(xt, model.calls[-1][0])
        assert torch.equal(t, model.calls[-1][2])
        raw = logits.clone()
        raw[mutable_mask] /= 0.5
        expected = denoiser_to_loo_logits(
            process, raw, xt, t, mutable_mask=mutable_mask
        )
        result = original_conversion(
            logits,
            xt,
            t,
            mutable_mask=mutable_mask,
            denoiser_temperature=denoiser_temperature,
        )
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        observed.append("clean_conversion")
        return result

    def step(logits, xt, t, s, **kwargs):
        assert kwargs["temperature"] == 1.0 and kwargs["raw_loo_top_p"] == 1.0
        observed.append("unit_bridge")
        return original_step(logits, xt, t, s, **kwargs)

    monkeypatch.setattr(model, "sampling_logits", conversion)
    monkeypatch.setattr(process, "step", step)
    _sampler(model, process, "udlm").generate(
        inputs,
        num_steps=3,
        softmax_temp=0.5,
        temperature_space="x0_denoiser",
        return_token_ids=True,
    )
    assert observed == ["clean_conversion", "unit_bridge"] * 3


@pytest.mark.parametrize("mode", [None, True, 0, "posterior", "X0_DENOISER", []])
def test_unknown_temperature_space_fails_before_prior_and_model(mode):
    model, process = _UDLMModel(), _UDLMProcess()
    process.sample_prior = lambda *args, **kwargs: pytest.fail("must reject before RNG")
    with pytest.raises(ValueError, match="temperature_space"):
        _sampler(model, process, "udlm").generate(
            torch.tensor([[1, 4, 2]]), temperature_space=mode
        )
    assert not model.calls


@pytest.mark.parametrize("unsupported", ["mdlm", "raw_loo", "top_p", "gibbs"])
def test_new_mode_rejects_unsupported_combinations_before_sampling(unsupported):
    if unsupported == "mdlm":
        model, process, diffusion = _MDLMModel(), _MDLMProcess(), "mdlm"
    else:
        model, process, diffusion = _UDLMModel(), _UDLMProcess(), "udlm"
    model.udlm_parameterization = (
        "raw_loo" if unsupported == "raw_loo" else "x0_denoiser"
    )
    process.sample_prior = lambda *args, **kwargs: pytest.fail("must reject before RNG")
    with pytest.raises(ValueError, match="temperature_space=x0_denoiser"):
        _sampler(model, process, diffusion).generate(
            torch.tensor([[1, 4, 2]]),
            temperature_space="x0_denoiser",
            raw_loo_top_p=0.9 if unsupported == "top_p" else 1.0,
            gibbs_corrector=unsupported == "gibbs",
            num_steps=4,
        )
    assert not model.calls
