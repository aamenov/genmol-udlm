"""CPU-only contract and actual tiny-checkpoint tests; no oracle/model jobs."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf

import genmol.sampler  # noqa: F401  -- runtime source-provenance checks
from scripts.exps.denovo import benchmark
from scripts.exps.pmo import udlm_sampling as sampling
from test_udlm_denoiser_benchmark import _checkpoint


def write_contract(tmp_path, **overrides):
    checkpoint = _checkpoint()
    path = tmp_path / "tiny.ckpt"
    torch.save(checkpoint, path)
    values = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "diffusion_type": "udlm",
        "parameterization": "x0_denoiser",
        "softmax_temp": 0.5,
        "randomness": 0,
        "min_add_len": 18,
        "num_steps": 3,
        "inference_eps": 1e-5,
        "exclude_special_tokens": False,
        "prior_variant": "schedule_uniform",
        "prior_metadata_sha256": benchmark._canonical_json_sha256(
            checkpoint["udlm_prior_metadata"]
        ),
        "raw_loo_top_p": 1.0,
        "temperature_space": "x0_denoiser",
        **overrides,
    }
    config = tmp_path / "sampling.yaml"
    config.write_text(yaml.safe_dump(values))
    return config, path, checkpoint


class TinyTokenizer:
    vocab_size = 5
    pad_token_id, bos_token_id, eos_token_id, mask_token_id = 3, 1, 2, 4

    def get_vocab(self):
        return {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "[PAD]": 3, "[MASK]": 4}

    def get_added_vocab(self):
        return {}

    def __len__(self):
        return self.vocab_size


def tiny_sampler_class(checkpoint, *, alteration=None):
    class TinySampler:
        def __init__(self, path, **kwargs):
            self.constructor = {"path": path, **kwargs}
            self.diffusion_type = "udlm"
            self.model = torch.nn.Module()
            self.model.backbone = torch.nn.Identity()
            self.model.device = torch.device("cpu")
            self.model.config = OmegaConf.create(
                checkpoint["hyper_parameters"]["config"]
            )
            self.model.udlm_parameterization = "x0_denoiser"
            self.model.udlm_prior_metadata = copy.deepcopy(
                checkpoint["udlm_prior_metadata"]
            )
            self.model.tokenizer = TinyTokenizer()
            self.mdlm = SimpleNamespace(to_device=lambda device: None)
            self.inference_weights = {
                "source": "ema",
                "ema_applied": True,
                "ema": {"num_updates": 20},
            }
            if alteration:
                alteration(self)

        def generate(self, _smiles, **kwargs):
            self.generation_kwargs = kwargs
            for _ in range(kwargs["num_steps"]):
                self.model.backbone(torch.tensor([1.0]))
            return "CCO"

        def mask_modification(self, smiles, **kwargs):
            return self.generate(smiles, **kwargs)

    return TinySampler


def prepared(tmp_path, *, alteration=None):
    config, path, checkpoint = write_contract(tmp_path)
    contract = sampling.read_contract(config, gamma=0, variant="released")
    return sampling.prepare(
        contract,
        model_path=path,
        device="cpu",
        gamma=0,
        guidance_scale=2,
        sampler_class=tiny_sampler_class(checkpoint, alteration=alteration),
    )


def test_actual_tiny_checkpoint_and_sources_bind_inference(tmp_path):
    adapter = prepared(tmp_path)
    receipt = adapter.receipt
    assert receipt["checkpoint"]["global_step"] == 20
    assert (
        receipt["checkpoint"]["udlm_denoiser_metadata"]
        == benchmark.UDLM_DENOISER_METADATA
    )
    assert receipt["checkpoint"]["byte_identity_verified_before_and_after_load"] is True
    assert receipt["inference_weights"]["ema_applied"] is True
    assert receipt["temperature_space"] == "x0_denoiser"
    assert receipt["oracle_call_protocol"] == sampling.ORACLE_CALL_PROTOCOL
    assert receipt["reverse_bridge_temperature"] == 1.0
    assert receipt["tokenizer"]["effective_size"] == 5
    assert {"pmo_runner", "pmo_sampling_adapter", "denoiser_source"} <= set(
        receipt["implementation_inputs"]
    )
    for item in receipt["implementation_inputs"].values():
        assert sampling._fingerprint(item["path"])["sha256"] == item["sha256"]
    assert adapter.sampler.constructor["require_ema"] is True
    assert (
        adapter.sampler.constructor["expected_checkpoint_sha256"]
        == receipt["checkpoint"]["sha256"]
    )
    assert adapter.sampler.constructor["length_distribution"]
    assert adapter.modify("CC") == "CCO"
    assert adapter.last_call == {
        "generation_calls": 1,
        "backbone_evaluations": 3,
        "pre_generation_fallbacks": 0,
    }
    assert adapter.sampler.generation_kwargs == {
        "num_steps": 3,
        "softmax_temp": 0.5,
        "randomness": 0.0,
        "gamma": 0,
        "w": 2,
        "raw_loo_top_p": 1.0,
        "temperature_space": "x0_denoiser",
    }
    assert "generate" not in vars(adapter.sampler)
    assert not adapter.sampler.model.backbone._forward_pre_hooks


@pytest.mark.parametrize(
    "overrides",
    [
        {"diffusion_type": "udlm", "unexpected": 1},
        {"min_add_len": 40},
        {"checkpoint_sha256": "f" * 63},
        {"num_steps": 0},
        {"temperature_space": "bogus"},
        {"raw_loo_top_p": 0.9},
        {"gibbs_corrector": True},
        {"parameterization": "raw_loo"},
    ],
)
def test_invalid_contracts_fail_without_sampler_or_oracle(tmp_path, overrides):
    config, _, _ = write_contract(tmp_path, **overrides)
    with pytest.raises((ValueError, RuntimeError)):
        sampling.read_contract(config, gamma=0, variant="released")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.1, "variant": "released"},
        {"gamma": 0, "variant": "running_mean"},
        {"gamma": 0, "variant": "released", "resume": True},
    ],
)
def test_unsupported_policy_guidance_resume_fail_early(tmp_path, kwargs):
    config, _, _ = write_contract(tmp_path)
    with pytest.raises(ValueError):
        sampling.read_contract(config, **kwargs)


def test_raw_loo_explicit_default_has_same_canonical_config(tmp_path):
    config, _, _ = write_contract(tmp_path, temperature_space="raw_loo")
    explicit = sampling.read_contract(config, gamma=0, variant="released")
    source = yaml.safe_load(config.read_text())
    del source["temperature_space"]
    config.write_text(yaml.safe_dump(source))
    implicit = sampling.read_contract(config, gamma=0, variant="released")
    assert explicit["configuration"] == implicit["configuration"]
    assert explicit["configuration_sha256"] == implicit["configuration_sha256"]
    assert "temperature_space" not in sampling.modification_kwargs(
        implicit, gamma=0, guidance_scale=2
    )


@pytest.mark.parametrize(
    "alteration",
    [
        lambda obj: setattr(obj, "diffusion_type", "mdlm"),
        lambda obj: setattr(obj.model, "udlm_parameterization", "raw_loo"),
        lambda obj: obj.model.config.training.udlm.update({"inference_eps": 0.01}),
        lambda obj: obj.model.config.training.udlm.update(
            {"exclude_special_tokens": True}
        ),
        lambda obj: obj.model.udlm_prior_metadata.update(
            {"variant": "release_uniform"}
        ),
        lambda obj: setattr(
            obj, "inference_weights", {"source": "raw", "ema_applied": False}
        ),
    ],
)
def test_actual_loaded_sampler_must_match_checkpoint_contract(tmp_path, alteration):
    with pytest.raises((ValueError, RuntimeError)):
        prepared(tmp_path, alteration=alteration)


def test_checkpoint_hash_and_ce_marker_rejected_before_sampler(tmp_path):
    config, path, checkpoint = write_contract(tmp_path)
    contract = sampling.read_contract(config, gamma=0, variant="released")
    del checkpoint["state_dict"]["_udlm_denoiser_ce_version"]
    torch.save(checkpoint, path)

    def forbidden(*_args, **_kwargs):
        pytest.fail("checkpoint validation must precede sampler construction")

    with pytest.raises((ValueError, RuntimeError)):
        sampling.prepare(
            contract,
            model_path=path,
            device="cpu",
            gamma=0,
            guidance_scale=2,
            sampler_class=forbidden,
        )
    source = yaml.safe_load(config.read_text())
    source["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config.write_text(yaml.safe_dump(source))
    contract = sampling.read_contract(config, gamma=0, variant="released")
    with pytest.raises((ValueError, RuntimeError)):
        sampling.prepare(
            contract,
            model_path=path,
            device="cpu",
            gamma=0,
            guidance_scale=2,
            sampler_class=forbidden,
        )


def instrumented(*, behavior="generate", failure=None, steps=3):
    checkpoint = _checkpoint()
    sampler = tiny_sampler_class(checkpoint)("unused")
    if failure:

        def generate(*_args, **_kwargs):
            raise failure

        sampler.generate = generate

    def modify(smiles, **kwargs):
        if behavior == "fallback":
            return smiles
        if behavior == "unexplained_mutation":
            return "COC"
        try:
            result = sampler.generate(smiles, **kwargs)
            if behavior == "twice":
                result = sampler.generate(smiles, **kwargs)
            return result
        except BaseException:
            return smiles  # Reproduce released addmask's bare except.

    sampler.mask_modification = modify
    return sampling.SamplingAdapter(
        sampler,
        {
            "sampler_kwargs": {"num_steps": steps},
            "configured_nfe_per_generation": 3,
        },
    )


def test_released_pre_generation_fallback_keeps_parent_without_claimed_nfe():
    adapter = instrumented(behavior="fallback")
    assert adapter.modify("CC") == "CC"
    assert adapter.last_call == {
        "generation_calls": 0,
        "backbone_evaluations": 0,
        "pre_generation_fallbacks": 1,
    }
    assert adapter.statistics["modification_attempts"] == 1
    assert adapter.statistics["pre_generation_fallbacks"] == 1


@pytest.mark.parametrize(
    "failure", [ValueError("generation failed"), KeyboardInterrupt(), SystemExit(9)]
)
def test_swallowed_generation_failures_and_interrupts_propagate(failure):
    adapter = instrumented(failure=failure)
    original = adapter.sampler.generate
    with pytest.raises(type(failure)) as caught:
        adapter.modify("CC")
    assert caught.value is failure
    assert adapter.statistics["generation_calls"] == 1
    assert adapter.statistics["pre_generation_fallbacks"] == 0
    assert adapter.sampler.generate is original
    assert not adapter.sampler.model.backbone._forward_pre_hooks


@pytest.mark.parametrize(
    "kwargs",
    [{"steps": 2}, {"behavior": "twice"}, {"behavior": "unexplained_mutation"}],
)
def test_ignored_nfe_or_multiple_generations_cannot_pass(kwargs):
    adapter = instrumented(**kwargs)
    with pytest.raises(RuntimeError):
        adapter.modify("CC")
    assert not adapter.sampler.model.backbone._forward_pre_hooks


def test_source_changes_reject_terminal_acceptance(tmp_path):
    adapter = prepared(tmp_path)
    source = adapter.receipt["contract"]["source"]["path"]
    from pathlib import Path

    Path(source).write_text(Path(source).read_text() + "\n# changed\n")
    with pytest.raises(RuntimeError, match="input changed"):
        adapter.validate_unchanged()


@pytest.mark.parametrize(
    "error", [ValueError("SAFE conversion"), KeyboardInterrupt(), SystemExit(8)]
)
def test_completion_preserves_ordinary_fallback_but_not_interruption(error):
    adapter = instrumented()

    def completion(*_args, **_kwargs):
        raise error

    def addmask(smiles, **kwargs):
        try:
            return adapter.sampler.fragment_completion(smiles, **kwargs)
        except BaseException:
            return smiles

    adapter.sampler.fragment_completion = completion
    adapter.sampler.mask_modification = addmask
    if isinstance(error, Exception):
        assert adapter.modify("CC") == "CC"
        assert adapter.statistics["pre_generation_fallbacks"] == 1
    else:
        with pytest.raises(type(error)) as caught:
            adapter.modify("CC")
        assert caught.value is error
        assert adapter.statistics["pre_generation_fallbacks"] == 0
    assert adapter.sampler.fragment_completion is completion
    assert not adapter.sampler.model.backbone._forward_pre_hooks


def test_explicit_mdlm_control_remains_supported(tmp_path):
    path = tmp_path / "mdlm.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "checkpoint_sha256": "b" * 64,
                "diffusion_type": "mdlm",
                "softmax_temp": 1.2,
                "randomness": 2.0,
                "min_add_len": 18,
            }
        )
    )
    contract = sampling.read_contract(path, gamma=0.3, variant="released")
    assert contract["configuration"]["num_steps"] is None
    assert sampling.modification_kwargs(contract, gamma=0.3, guidance_scale=2) == {
        "gamma": 0.3,
        "w": 2,
        "softmax_temp": 1.2,
        "randomness": 2.0,
    }
