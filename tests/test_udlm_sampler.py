import pytest
import torch
from omegaconf import OmegaConf

import genmol.sampler as sampler_module
from genmol.diffusion import ContinuousCategoricalDiffusion, ContinuousUniformDiffusion


class _Tokenizer:
    def batch_decode(self, values, skip_special_tokens=True):
        assert skip_special_tokens
        return ["decoded" for _ in values]


class _UDLMModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.mask_index = 4
        self.bos_index = 1
        self.eos_index = 2
        self.tokenizer = _Tokenizer()
        self.config = OmegaConf.create(
            {
                "training": {
                    "udlm": {"sampling_steps": 3, "inference_eps": 1e-5},
                    "use_bracket_safe": False,
                }
            }
        )
        self.calls = []

    def __call__(self, x, attention_mask, t=None):
        self.calls.append((x.clone(), attention_mask.clone(), t.clone()))
        return torch.zeros(*x.shape, 9)


class _UDLMProcess:
    def __init__(self):
        self.steps = []

    def sample_prior(self, shape, device=None):
        return torch.full(shape, 6, dtype=torch.long, device=device)

    def step(
        self,
        logits,
        x,
        t,
        s,
        *,
        mutable_mask,
        temperature,
        raw_loo_top_p,
    ):
        self.steps.append(
            (
                x.clone(),
                t.clone(),
                s.clone(),
                mutable_mask.clone(),
                temperature,
                raw_loo_top_p,
            )
        )
        return torch.where(mutable_mask, torch.full_like(x, 7), x)


class _MDLMModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.mask_index = 4
        self.bos_index = 1
        self.eos_index = 2
        self.tokenizer = _Tokenizer()
        self.config = OmegaConf.create({"training": {"use_bracket_safe": False}})
        self.calls = []

    def __call__(self, x, attention_mask=None):
        self.calls.append((x.clone(), attention_mask.clone()))
        return torch.zeros(*x.shape, 9)


class _MDLMProcess:
    def __init__(self):
        self.steps = []

    def get_num_steps_confidence(self, x):
        return 3

    def step_confidence(self, logits, x, step, num_steps, softmax_temp, randomness):
        self.steps.append((step, num_steps, softmax_temp, randomness))
        return x


def _sampler(model, process, diffusion_type):
    sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
    sampler.model = model
    sampler.pad_index = 3
    sampler.mdlm = process
    sampler.diffusion_type = diffusion_type
    return sampler


@pytest.fixture(autouse=True)
def _identity_decoder(monkeypatch):
    monkeypatch.setattr(sampler_module, "safe_to_smiles", lambda value, fix=True: value)


def test_udlm_sampling_uses_fixed_time_grid_and_clamps_context():
    model = _UDLMModel()
    process = _UDLMProcess()
    sampler = _sampler(model, process, "udlm")
    x = torch.tensor([[1, 4, 4, 2, 3]])

    samples = sampler.generate(x, softmax_temp=0.7, raw_loo_top_p=0.95, num_steps=3)

    assert samples == ["decoded"]
    assert len(model.calls) == 3
    assert len(process.steps) == 3
    expected_editable = torch.tensor([[False, True, True, False, False]])
    for model_input, attention_mask, _ in model.calls:
        assert model_input[0, 0] == 1
        assert model_input[0, 3] == 2
        assert model_input[0, 4] == 3
        assert torch.equal(
            attention_mask, torch.tensor([[True, True, True, True, False]])
        )
    for _, _, _, editable, temperature, raw_loo_top_p in process.steps:
        assert torch.equal(editable, expected_editable)
        assert temperature == pytest.approx(0.7)
        assert raw_loo_top_p == pytest.approx(0.95)
    assert process.steps[0][1].item() == 1.0
    assert process.steps[-1][2].item() == pytest.approx(1e-5)


@pytest.mark.parametrize("diffusion_type", ["mdlm", "udlm"])
def test_sampler_can_return_raw_token_ids_before_chemical_decoding(diffusion_type):
    if diffusion_type == "udlm":
        model, process = _UDLMModel(), _UDLMProcess()
    else:
        model, process = _MDLMModel(), _MDLMProcess()
    sampler = _sampler(model, process, diffusion_type)
    inputs = torch.tensor([[1, 4, 4, 2]])

    token_ids = sampler.generate(inputs, return_token_ids=True)

    assert isinstance(token_ids, torch.Tensor)
    assert token_ids.shape == inputs.shape
    if diffusion_type == "udlm":
        assert torch.equal(token_ids, torch.tensor([[1, 7, 7, 2]]))
    else:
        assert torch.equal(token_ids, inputs)


def test_udlm_rejects_mdlm_specific_molecular_context_guidance():
    sampler = _sampler(_UDLMModel(), _UDLMProcess(), "udlm")

    with pytest.raises(ValueError, match="posterior-space"):
        sampler.generate(torch.tensor([[1, 4, 2]]), gamma=0.5, w=2)


def test_udlm_missing_sampling_step_config_uses_official_128_step_control():
    model = _UDLMModel()
    del model.config.training.udlm.sampling_steps
    process = _UDLMProcess()
    sampler = _sampler(model, process, "udlm")

    sampler.generate(torch.tensor([[1, 4, 2]]))

    assert len(process.steps) == 128


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("softmax_temp", True),
        ("softmax_temp", "1.0"),
        ("softmax_temp", 0.0),
        ("softmax_temp", float("nan")),
        ("raw_loo_top_p", True),
        ("raw_loo_top_p", "1.0"),
        ("raw_loo_top_p", 0.0),
        ("raw_loo_top_p", 1.01),
        ("raw_loo_top_p", float("inf")),
    ],
)
def test_udlm_sampling_scalars_are_validated_before_prior_sampling(keyword, value):
    model = _UDLMModel()
    process = _UDLMProcess()
    sampler = _sampler(model, process, "udlm")

    with pytest.raises(ValueError, match=keyword):
        sampler.generate(torch.tensor([[1, 4, 2]]), **{keyword: value})

    assert process.steps == []
    assert model.calls == []


def test_mdlm_confidence_loop_contract_is_preserved():
    model = _MDLMModel()
    process = _MDLMProcess()
    sampler = _sampler(model, process, "mdlm")

    samples = sampler.generate(
        torch.tensor([[1, 4, 4, 2]]),
        softmax_temp=0.5,
        randomness=0.25,
    )

    assert samples == ["decoded"]
    assert len(model.calls) == 3
    assert process.steps == [
        (0, 3, 0.5, 0.25),
        (1, 3, 0.5, 0.25),
        (2, 3, 0.5, 0.25),
    ]


def test_insert_mask_consumes_resident_length_distribution(monkeypatch):
    sampler = _sampler(_MDLMModel(), _MDLMProcess(), "mdlm")
    sampler.length_distribution = (8,)
    sampler.model.mask_index = 4

    def fail_if_reopened(*_args, **_kwargs):
        raise AssertionError("the sampler must not reopen data/len.pk")

    monkeypatch.setattr(sampler_module, "open", fail_if_reopened, raising=False)
    output = sampler._insert_mask(
        torch.tensor([[sampler.model.bos_index, sampler.model.eos_index]]),
        num_samples=2,
        min_add_len=1,
    )

    assert output.shape == (2, 8)
    assert torch.all(output[:, 0] == sampler.model.bos_index)
    assert torch.all(output[:, -1] == sampler.model.eos_index)
    assert torch.all(output[:, 1:-1] == sampler.model.mask_index)


@pytest.mark.parametrize("configured_budget", [False, True])
def test_gibbs_corrector_spy_uses_fresh_predictor_state_and_time_at_fixed_nfe(
    monkeypatch, configured_budget
):
    class CountingModel(_UDLMModel):
        def __call__(self, x, attention_mask, t=None):
            super().__call__(x, attention_mask, t)
            return torch.full((*x.shape, 9), float(len(self.calls)))

    model, process = CountingModel(), _UDLMProcess()
    model.config.training.udlm.sampling_steps = 6
    sampler = _sampler(model, process, "udlm")
    inputs = torch.tensor([[1, 4, 4, 8, 2, 3], [1, 8, 8, 8, 2, 3]])
    original = inputs.clone()
    editable = inputs == 4
    corrector_calls = []

    def corrector_spy(
        diffusion, logits, xt, s, *, mutable_mask, temperature, raw_loo_top_p
    ):
        assert diffusion is process
        assert torch.all(logits == 2 * (len(corrector_calls) + 1))
        assert torch.equal(mutable_mask, editable)
        assert temperature == 0.7 and raw_loo_top_p == 0.95
        assert torch.equal(xt, torch.where(editable, 7, original))
        assert torch.equal(s, process.steps[-1][2])
        result = xt.clone()
        result[0, 1] = 8
        corrector_calls.append((xt.clone(), s.clone(), result.clone()))
        return result

    monkeypatch.setattr(sampler_module, "random_scan_gibbs_step", corrector_spy)
    budget_argument = {} if configured_budget else {"num_steps": 6}
    result = sampler.generate(
        inputs,
        softmax_temp=0.7,
        raw_loo_top_p=0.95,
        gibbs_corrector=True,
        return_token_ids=True,
        **budget_argument,
    )
    assert len(model.calls) == 6
    assert len(process.steps) == len(corrector_calls) == 3
    grid = torch.linspace(1.0, 1e-5, 4)
    for index, (_, time, corrected) in enumerate(corrector_calls):
        predictor_call, corrector_call = model.calls[2 * index : 2 * index + 2]
        assert torch.equal(predictor_call[2], grid[index].expand(2))
        assert torch.equal(corrector_call[2], grid[index + 1].expand(2))
        assert torch.equal(corrector_call[0], torch.where(editable, 7, original))
        assert torch.equal(time, corrector_call[2])
        assert torch.equal(predictor_call[1], inputs != 3)
        assert torch.equal(corrector_call[1], inputs != 3)
        if index < 2:
            assert torch.equal(model.calls[2 * index + 2][0], corrected)
    assert torch.equal(result[~editable], inputs[~editable])
    assert torch.equal(inputs, original)


class _ContextualToyModel(_UDLMModel):
    def __call__(self, x, attention_mask, t=None):
        super().__call__(x, attention_mask, t)
        classes = torch.arange(9, dtype=torch.float32).view(1, 1, 9)
        return -((classes - x.unsqueeze(-1)) ** 2) / 8 + classes * t[:, None, None]


def _real_process(nonuniform):
    return (
        ContinuousCategoricalDiffusion(9, torch.arange(1, 10, dtype=torch.float64) / 45)
        if nonuniform
        else ContinuousUniformDiffusion(9)
    )


def _historical_predictor_only(model, process, inputs, budget, temperature, top_p):
    """Literal pre-corrector transition sequence retained as an RNG regression oracle."""
    attention = inputs != 3
    editable = inputs == model.mask_index
    x = torch.where(editable, process.sample_prior(inputs.shape), inputs)
    times = torch.linspace(1.0, 1e-5, budget + 1, dtype=torch.float32)
    for index in range(budget):
        t, s = times[index].expand(len(x)), times[index + 1].expand(len(x))
        logits = model(x, attention, t=t)
        x = process.step(
            logits,
            x,
            t,
            s,
            mutable_mask=editable,
            temperature=temperature,
            raw_loo_top_p=top_p,
        )
    return x


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("explicit_false", [False, True])
def test_gibbs_default_matches_historical_sampled_ids_and_rng_state(
    monkeypatch, nonuniform, explicit_false
):
    def forbidden_corrector(*args, **kwargs):
        raise AssertionError("default path must not call the corrector")

    monkeypatch.setattr(sampler_module, "random_scan_gibbs_step", forbidden_corrector)
    process = _real_process(nonuniform)
    inputs = torch.tensor([[1, 4, 4, 2, 3], [1, 4, 4, 4, 2]])
    torch.manual_seed(991)
    start = torch.random.get_rng_state().clone()
    expected = _historical_predictor_only(
        _ContextualToyModel(), process, inputs, 7, 0.7, 0.95
    )
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(start)
    sampler = _sampler(_ContextualToyModel(), process, "udlm")
    extra = {"gibbs_corrector": False} if explicit_false else {}
    actual = sampler.generate(
        inputs,
        num_steps=7,
        softmax_temp=0.7,
        raw_loo_top_p=0.95,
        return_token_ids=True,
        **extra,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert len(sampler.model.calls) == 7


@pytest.mark.parametrize("nonuniform", [False, True])
def test_real_gibbs_sampler_runs_at_fixed_nfe_and_preserves_original_framing(
    nonuniform,
):
    inputs = torch.tensor([[1, 4, 4, 8, 2, 3], [1, 8, 8, 8, 2, 3]])
    sampler = _sampler(_ContextualToyModel(), _real_process(nonuniform), "udlm")
    torch.manual_seed(119)
    first = sampler.generate(
        inputs, num_steps=4, gibbs_corrector=True, return_token_ids=True
    )
    assert len(sampler.model.calls) == 4
    assert torch.equal(first[inputs != 4], inputs[inputs != 4])
    torch.manual_seed(119)
    repeated = sampler.generate(
        inputs, num_steps=4, gibbs_corrector=True, return_token_ids=True
    )
    assert torch.equal(first, repeated)


@pytest.mark.parametrize("value", [None, 0, 1, "true", torch.tensor(True)])
def test_corrector_requires_strict_boolean_before_sampling(value):
    model, process = _UDLMModel(), _UDLMProcess()
    sampler = _sampler(model, process, "udlm")
    with pytest.raises(ValueError, match="gibbs_corrector must be a boolean"):
        sampler.generate(torch.tensor([[1, 4, 2]]), gibbs_corrector=value)
    assert model.calls == process.steps == []


def test_corrector_rejects_mdlm_before_model_evaluation():
    model, process = _MDLMModel(), _MDLMProcess()
    with pytest.raises(ValueError, match="only for UDLM"):
        _sampler(model, process, "mdlm").generate(
            torch.tensor([[1, 4, 2]]), gibbs_corrector=True, num_steps=4
        )
    assert model.calls == process.steps == []


@pytest.mark.parametrize("budget", [0, 1, 3, -2, 4.0, True, "4"])
def test_corrector_rejects_invalid_total_nfe_before_prior_sampling(budget):
    model, process = _UDLMModel(), _UDLMProcess()

    def forbidden_prior(*args, **kwargs):
        raise AssertionError("invalid budgets must fail before consuming RNG")

    process.sample_prior = forbidden_prior
    with pytest.raises(ValueError, match="even integer num_steps >= 2"):
        _sampler(model, process, "udlm").generate(
            torch.tensor([[1, 4, 2]]), gibbs_corrector=True, num_steps=budget
        )
    assert model.calls == process.steps == []


def test_corrector_validates_odd_budget_from_config():
    model, process = _UDLMModel(), _UDLMProcess()
    with pytest.raises(ValueError, match="even integer num_steps >= 2"):
        _sampler(model, process, "udlm").generate(
            torch.tensor([[1, 4, 2]]), gibbs_corrector=True
        )
