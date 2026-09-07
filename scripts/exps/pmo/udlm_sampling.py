"""Opt-in, checkpoint-bound sampling for the released PMO population policy.

This module performs no optimization or oracle scoring. It leaves the historical
runner untouched when --sampling-config is absent. A prospective PMO budget/seed
panel and GPU controller remain separate from this adapter.
"""

from __future__ import annotations

import copy
import hashlib
import math
import numbers
from pathlib import Path

import yaml

from scripts.exps.denovo import benchmark


ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = {
    "pmo_sampling_adapter": Path(__file__),
    "pmo_runner": ROOT / "scripts/exps/pmo/run_ablation.py",
    "pmo_population": ROOT / "scripts/exps/pmo/main/genmol/fragment_population.py",
    "pmo_artifact_io": ROOT / "scripts/exps/pmo/main/genmol/experiment_io.py",
    "sampling_validator": Path(benchmark.__file__),
}
SAMPLING_KEYS = {
    "checkpoint_sha256",
    "diffusion_type",
    "parameterization",
    "softmax_temp",
    "randomness",
    "min_add_len",
    "num_steps",
    "inference_eps",
    "exclude_special_tokens",
    "prior_variant",
    "prior_metadata_sha256",
    "raw_loo_top_p",
    "gibbs_corrector",
    "temperature_space",
}
ORACLE_CALL_PROTOCOL = {
    "input": "singleton list containing the CachedOracle canonical SMILES",
    "output": "exactly one finite real scalar; booleans rejected",
    "exception_policy": "propagate ordinary non-docking TDC list-path evaluator failures",
    "budget": "CachedOracle charges only after a finite successful return",
}


def singleton_list_oracle(evaluator):
    """Use TDC's error-propagating list dispatch without changing valid scores.

    The installed scalar dispatch silently converts evaluator exceptions to zero.
    CachedOracle has already canonicalized/validated this one molecule and owns
    the unique-call budget. A failed evaluator never enters that cache.
    """

    def score(canonical_smiles):
        values = evaluator([canonical_smiles])
        if type(values) is not list or len(values) != 1:
            raise ValueError("PMO singleton-list oracle must return exactly one score")
        value = values[0]
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        ):
            raise ValueError("PMO oracle score must be a finite real scalar")
        return float(value)

    return score


def _fingerprint(path):
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def read_contract(path, *, gamma, variant, resume=False):
    """Read one explicit sampling law; YAML controls override legacy CLI defaults."""
    if variant != "released" or resume:
        raise ValueError(
            "sampling-config initially requires released policy and no resume"
        )
    path = Path(path).expanduser().resolve(strict=True)
    payload = path.read_bytes()
    source = yaml.safe_load(payload)
    if not isinstance(source, dict) or set(source) - SAMPLING_KEYS:
        raise ValueError(
            "sampling-config must contain only the declared sampling fields"
        )
    if "diffusion_type" not in source:
        raise ValueError("sampling-config requires an explicit diffusion_type")
    expected = source.get("checkpoint_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("sampling-config requires a lowercase checkpoint_sha256")
    sampling = benchmark.validate_sampling_config(source)
    if sampling["min_add_len"] != 18:
        raise ValueError("PMO preserves the released effective min_add_len=18")
    if sampling.get("gibbs_corrector", False):
        raise ValueError(
            "PMO sampling-config initially supports predictor-only sampling"
        )
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("PMO gamma must be finite and in [0,1]")
    if sampling["diffusion_type"] == "udlm" and gamma != 0:
        raise ValueError("UDLM PMO requires gamma=0 before any oracle scoring")
    return {
        "schema_version": 1,
        "source": {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        },
        "checkpoint_sha256": expected,
        "configuration": sampling,
        "configuration_sha256": benchmark._canonical_json_sha256(sampling),
        "control_precedence": "sampling YAML overrides CLI softmax_temp and randomness",
        "fragment_length_policy": {
            "completion_effective_min_add_len": 18,
            "remask_chunk_length": "released uniform integer 5..15, subject to capacity",
            "mask_modification_min_len": 30,
            "completion_mask_len_override": "ignored by released completion path",
        },
    }


def modification_kwargs(contract, *, gamma, guidance_scale):
    sampling = contract["configuration"]
    if sampling["diffusion_type"] == "udlm" and gamma != 0:
        raise ValueError("UDLM PMO requires gamma=0 before any oracle scoring")
    result = {
        "gamma": gamma,
        "w": guidance_scale,
        "softmax_temp": sampling["softmax_temp"],
        "randomness": sampling["randomness"],
    }
    if sampling["diffusion_type"] == "udlm":
        result.update(
            num_steps=sampling["num_steps"], raw_loo_top_p=sampling["raw_loo_top_p"]
        )
        if "temperature_space" in sampling:
            result["temperature_space"] = sampling["temperature_space"]
    # Deliberately do not forward min_add_len/mask_len: retain effective length18.
    return result


def _validate_loaded(sampler, checkpoint, sampling):
    if sampler.diffusion_type != sampling["diffusion_type"]:
        raise ValueError("loaded sampler diffusion_type differs from sampling-config")
    benchmark.validate_denoiser_sampling_identity(checkpoint, sampling)
    if sampler.diffusion_type == "udlm":
        config = sampler.model.config.training.get("udlm", {})
        for key, wanted in (
            ("inference_eps", sampling["inference_eps"]),
            ("exclude_special_tokens", sampling["exclude_special_tokens"]),
        ):
            if config.get(key, 1e-5 if key == "inference_eps" else False) != wanted:
                raise ValueError(f"loaded UDLM {key} differs from sampling-config")
        if getattr(sampler.model, "udlm_parameterization", "raw_loo") != sampling.get(
            "parameterization", "raw_loo"
        ):
            raise ValueError(
                "loaded UDLM parameterization differs from sampling-config"
            )
        benchmark._validate_loaded_udlm_prior_identity(
            sampler,
            prior_variant=sampling["prior_variant"],
            prior_metadata_sha256=sampling["prior_metadata_sha256"],
        )
    weights = sampler.inference_weights
    if (
        weights.get("source") != "ema"
        or weights.get("ema_applied") is not True
        or not weights.get("ema")
    ):
        raise ValueError("opt-in PMO requires verified EMA inference weights")
    return weights


def prepare(contract, *, model_path, device, gamma, guidance_scale, sampler_class):
    """Validate actual checkpoint and source before constructing any PMO oracle."""
    if (
        read_contract(contract["source"]["path"], gamma=gamma, variant="released")
        != contract
    ):
        raise ValueError("sampling-config changed after resolution")
    benchmark.assert_local_genmol_import()
    sampling = contract["configuration"]
    checkpoint = benchmark.checkpoint_metadata(
        Path(model_path), expected_sha256=contract["checkpoint_sha256"]
    )
    if checkpoint["diffusion_type"] != sampling["diffusion_type"]:
        raise ValueError("checkpoint diffusion_type differs from sampling-config")
    benchmark.validate_denoiser_sampling_identity(checkpoint, sampling)
    # Capture source only after reusable config/checkpoint validators have run.
    inputs = benchmark.load_implementation_input_snapshot(
        **benchmark.sampling_implementation_options(sampling)
    )
    sources = dict(inputs.provenance)
    sources.update({key: _fingerprint(path) for key, path in SOURCE_PATHS.items()})
    sampler = sampler_class(
        str(model_path),
        expected_checkpoint_sha256=contract["checkpoint_sha256"],
        require_ema=True,
        length_distribution=inputs.length_distribution,
    )
    sampler.model.to(device)
    sampler.mdlm.to_device(sampler.model.device)
    weights = _validate_loaded(sampler, checkpoint, sampling)
    benchmark.assert_runtime_module_provenance(inputs.provenance)
    receipt = {
        "schema_version": 1,
        "contract": copy.deepcopy(contract),
        "checkpoint": checkpoint,
        "inference_weights": weights,
        "implementation_inputs": sources,
        "tokenizer": benchmark.tokenizer_provenance(sampler.model.tokenizer),
        "sampler_kwargs": modification_kwargs(
            contract, gamma=gamma, guidance_scale=guidance_scale
        ),
        "configured_nfe_per_generation": sampling["num_steps"],
        "nfe_definition": "observed backbone forward calls; no forward call in attaching-only warmup",
        "temperature_space": (
            sampling.get("temperature_space", "raw_loo")
            if sampling["diffusion_type"] == "udlm"
            else "mdlm_clean_logits"
        ),
        "randomness_usage": (
            "MDLM randomness control ignored by UDLM; categorical prior and transition draws remain stochastic"
            if sampling["diffusion_type"] == "udlm"
            else "MDLM confidence-sampling Gumbel scale"
        ),
        "failure_policy": "propagate generation failures even if released addmask swallows them",
        "fallback_policy": "retain released pre-generation parent fallback, record zero generation calls/NFE",
        "oracle_call_protocol": dict(ORACLE_CALL_PROTOCOL),
    }
    if sampling.get("temperature_space") == "x0_denoiser":
        receipt.update(benchmark.DENOISER_TEMPERATURE_PROTOCOL)
    adapter = SamplingAdapter(sampler, receipt)
    adapter.validate_unchanged()
    return adapter


class SamplingAdapter:
    """Instrument PMO's one-generation mutation without changing fragment selection."""

    def __init__(self, sampler, receipt):
        self.sampler = sampler
        self.receipt = copy.deepcopy(receipt)
        self.statistics = {
            "modification_attempts": 0,
            "generation_calls": 0,
            "backbone_evaluations": 0,
            "pre_generation_fallbacks": 0,
        }
        self.last_call = None

    def validate_unchanged(self):
        records = [
            self.receipt["contract"]["source"],
            *self.receipt["implementation_inputs"].values(),
        ]
        for recorded in records:
            observed = _fingerprint(recorded["path"])
            if any(observed[key] != recorded[key] for key in ("sha256", "size_bytes")):
                raise RuntimeError(f"PMO sampling input changed: {recorded['path']}")

    def modify(self, smiles):
        self.statistics["modification_attempts"] += 1
        call = {
            "generation_calls": 0,
            "backbone_evaluations": 0,
            "pre_generation_fallbacks": 0,
        }
        errors = []
        original_generate = self.sampler.generate
        had_instance_generate = "generate" in vars(self.sampler)
        original_completion = getattr(self.sampler, "fragment_completion", None)
        had_instance_completion = "fragment_completion" in vars(self.sampler)

        def count_forward(_module, _args):
            call["backbone_evaluations"] += 1

        def checked_generate(*args, **kwargs):
            call["generation_calls"] += 1
            try:
                return original_generate(*args, **kwargs)
            except BaseException as error:
                errors.append(error)
                raise

        def checked_completion(*args, **kwargs):
            try:
                return original_completion(*args, **kwargs)
            except BaseException as error:
                # Ordinary SAFE conversion failures retain released fallback;
                # an interruption must survive addmask's bare except.
                if not isinstance(error, Exception):
                    errors.append(error)
                raise

        handle = self.sampler.model.backbone.register_forward_pre_hook(count_forward)
        self.sampler.generate = checked_generate
        if original_completion is not None:
            self.sampler.fragment_completion = checked_completion
        try:
            result = self.sampler.mask_modification(
                smiles, **self.receipt["sampler_kwargs"]
            )
            if errors:
                # Preserve interruptions as well as errors swallowed by bare addmask.
                raise errors[0]
            if call["generation_calls"] == 0:
                if result != smiles or call["backbone_evaluations"]:
                    raise RuntimeError("unexpected mutation without a generation call")
                call["pre_generation_fallbacks"] = 1
                return result
            if call["generation_calls"] != 1:
                raise RuntimeError(
                    "PMO modification executed multiple generation calls"
                )
            wanted = self.receipt["configured_nfe_per_generation"]
            if wanted is not None and call["backbone_evaluations"] != wanted:
                raise RuntimeError("observed UDLM NFE differs from sampling-config")
            return result
        finally:
            if had_instance_generate:
                self.sampler.generate = original_generate
            else:
                del self.sampler.generate
            if original_completion is not None:
                if had_instance_completion:
                    self.sampler.fragment_completion = original_completion
                else:
                    del self.sampler.fragment_completion
            handle.remove()
            self.last_call = dict(call)
            for key, value in call.items():
                self.statistics[key] += value
