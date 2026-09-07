"""Scientific mask equivalence for the prospective V8 CT/CE comparison."""

import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import genmol.model as model_module
from test_udlm_model import _Tokenizer, _config


@pytest.fixture(autouse=True)
def tokenizer(monkeypatch):
    monkeypatch.setattr(model_module, "get_tokenizer", lambda: _Tokenizer())


def config(parameterization="raw_loo", *, mask_policy=None, explicit_policy=False):
    value = _config(
        diffusion="udlm",
        prior_variant="schedule_uniform",
        exclude_special=False,
        conditioning_variant="film_adaln",
        zero_init_conditioning=False,
    )
    value.training.udlm.parameterization = parameterization
    if explicit_policy:
        value.training.udlm.mask_all_special_tokens = mask_policy
    return value


@pytest.mark.parametrize("prior_variant", ["release_uniform", "schedule_uniform"])
def test_missing_ct_mask_policy_retains_legacy_targets_state_and_rng(prior_variant):
    old_config = config()
    old_config.training.udlm.prior_variant = prior_variant
    del old_config.training.udlm.parameterization
    snapshot = OmegaConf.to_container(old_config, resolve=False)
    explicit_config = copy.deepcopy(old_config)
    explicit_config.training.udlm.mask_all_special_tokens = False
    torch.manual_seed(211)
    old = model_module.GenMol(old_config)
    old_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(211)
    explicit = model_module.GenMol(explicit_config)
    assert old.udlm_mask_all_special_tokens is False
    assert explicit.udlm_mask_all_special_tokens is False
    assert torch.equal(old_rng, torch.random.get_rng_state())
    assert OmegaConf.to_container(old_config, resolve=False) == snapshot
    assert old.state_dict().keys() == explicit.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    ids = torch.tensor([[1, 5, 0, 6, 4, 2, 3, 7]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    expected = torch.tensor([[False, True, True, True, True, False, False, False]])
    assert torch.equal(old.diffusion_token_mask(ids, attention), expected)
    assert torch.equal(explicit.diffusion_token_mask(ids, attention), expected)


def test_missing_ce_mask_policy_is_identical_to_explicit_true():
    implicit_config = config("x0_denoiser")
    snapshot = OmegaConf.to_container(implicit_config, resolve=False)
    explicit_config = copy.deepcopy(implicit_config)
    explicit_config.training.udlm.mask_all_special_tokens = True
    torch.manual_seed(219)
    implicit = model_module.GenMol(implicit_config)
    implicit_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(219)
    explicit = model_module.GenMol(explicit_config)
    assert implicit.udlm_mask_all_special_tokens is True
    assert torch.equal(implicit_rng, torch.random.get_rng_state())
    assert OmegaConf.to_container(implicit_config, resolve=False) == snapshot
    for key, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])


@pytest.mark.parametrize("attention_dtype", [torch.bool, torch.int64])
def test_common_ct_ce_mask_fixes_unk_mask_and_preserves_attention(attention_dtype):
    ct = model_module.GenMol(config(mask_policy=True, explicit_policy=True))
    ce = model_module.GenMol(
        config("x0_denoiser", mask_policy=True, explicit_policy=True)
    )
    ids = torch.tensor([[1, 5, 0, 6, 4, 7, 2, 3, 8]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=attention_dtype)
    original_attention, original_ids = attention.clone(), ids.clone()
    expected = torch.tensor(
        [[False, True, False, True, False, True, False, False, False]]
    )
    masks = [model.diffusion_token_mask(ids, attention) for model in (ct, ce)]
    assert all(torch.equal(mask, expected) for mask in masks)
    assert all(mask.dtype == torch.bool for mask in masks)
    assert torch.equal(attention, original_attention)
    assert torch.equal(ids, original_ids)
    assert all(mask.data_ptr() != attention.data_ptr() for mask in masks)
    # Target masking is distinct from excluding IDs from the corruption support.
    for model in (ct, ce):
        assert torch.equal(model.mdlm.diffusion_token_ids, torch.arange(11))
        assert model.udlm_prior_metadata.excluded_token_ids == ()
    t = torch.tensor([0.8])
    noisy = [
        model.mdlm.forward_process(
            ids, t, mutable_mask=mask, generator=torch.Generator().manual_seed(61)
        )
        for model, mask in zip((ct, ce), masks, strict=True)
    ]
    assert torch.equal(noisy[0], noisy[1])
    assert all(torch.equal(sample[~expected], ids[~expected]) for sample in noisy)


def test_common_policy_uses_all_tokenizer_special_ids_including_extra_controls(
    monkeypatch,
):
    class TokenizerWithExtraControl(_Tokenizer):
        all_special_ids = [*_Tokenizer.all_special_ids, 10]

    monkeypatch.setattr(
        model_module, "get_tokenizer", lambda: TokenizerWithExtraControl()
    )
    ids = torch.tensor([[1, 5, 10, 6, 2]])
    attention = torch.ones_like(ids, dtype=torch.bool)
    expected = torch.tensor([[False, True, False, True, False]])
    for parameterization in ("raw_loo", "x0_denoiser"):
        model = model_module.GenMol(
            config(parameterization, mask_policy=True, explicit_policy=True)
        )
        assert torch.equal(model.diffusion_token_mask(ids, attention), expected)
        assert attention.all()


@pytest.mark.parametrize("parameterization", ["raw_loo", "x0_denoiser"])
@pytest.mark.parametrize("invalid_policy", [0, 1, "true", "false", None, [], {}])
def test_common_mask_selector_rejects_non_boolean_values(
    parameterization, invalid_policy
):
    with pytest.raises(ValueError, match="mask_all_special_tokens must be boolean"):
        model_module.GenMol(
            config(parameterization, mask_policy=invalid_policy, explicit_policy=True)
        )


def test_ce_cannot_select_the_legacy_target_mask():
    with pytest.raises(ValueError, match="requires mask_all_special_tokens=true"):
        model_module.GenMol(
            config("x0_denoiser", mask_policy=False, explicit_policy=True)
        )


def test_common_mask_ct_ce_have_equal_constructor_rng_and_a1_backbone():
    ct_config = config(mask_policy=True, explicit_policy=True)
    ce_config = copy.deepcopy(ct_config)
    ce_config.training.udlm.parameterization = "x0_denoiser"
    torch.manual_seed(1500)
    ct = model_module.GenMol(ct_config)
    ct_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(1500)
    ce = model_module.GenMol(ce_config)
    assert torch.equal(ct_rng, torch.random.get_rng_state())
    assert ct.udlm_conditioning_metadata == ce.udlm_conditioning_metadata
    assert ct.udlm_prior_metadata == ce.udlm_prior_metadata
    assert ce.state_dict().keys() - ct.state_dict().keys() == {
        model_module.UDLM_DENOISER_STATE_KEY
    }
    assert not ct.state_dict().keys() - ce.state_dict().keys()
    for key, value in ct.state_dict().items():
        assert torch.equal(value, ce.state_dict()[key])


def test_v8_configs_differ_only_in_parameterization_and_share_budget():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        ct = OmegaConf.to_container(
            compose(config_name="udlm_e_objective_ct"), resolve=False
        )
        ce = OmegaConf.to_container(
            compose(config_name="udlm_e_objective_ce"), resolve=False
        )
    assert ct["training"]["udlm"]["parameterization"] == "raw_loo"
    assert ce["training"]["udlm"]["parameterization"] == "x0_denoiser"
    ce["training"]["udlm"]["parameterization"] = "raw_loo"
    assert ct == ce
    udlm = ct["training"]["udlm"]
    assert udlm["mask_all_special_tokens"] is True
    assert udlm["exclude_special_tokens"] is False
    assert udlm["prior_variant"] == "empirical_frequency"
    assert udlm["empirical_uniform_mix"] == 0.0002
    assert udlm["conditioning_variant"] == "film_adaln"
    assert udlm["zero_init_conditioning"] is False
    assert ct["seed"] == 1500
    assert ct["trainer"]["max_steps"] == 1000
    assert ct["loader"]["global_batch_size"] == 128
    assert (
        ct["trainer"]["devices"]
        * ct["trainer"]["accumulate_grad_batches"]
        * ct["loader"]["batch_size"]
        == 128
    )
    assert ct["optim"]["lr"] == 0.0003
    assert ct["optim"]["scheduler"] == {
        "name": "half_cosine_with_linear_warmup_and_floor",
        "warmup_updates": 50,
        "horizon_updates": 1000,
        "decay_floor_lr": 0.000003,
    }
