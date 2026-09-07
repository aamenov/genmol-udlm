"""Opt-in posterior-space context guidance; sampler-only, no benchmark wiring.

The separate API returns raw IDs and an experimental receipt. It never changes
the existing GenMol sampler's default or PMO guidance acceptance paths.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import numbers
from pathlib import Path
import time

import torch

from genmol.diffusion import ContinuousCategoricalDiffusion


CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "method",
        "gamma",
        "scale",
        "context_seed",
        "predictor_steps",
        "execution",
    }
)
METHOD = "posterior_context"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "src/genmol/context_guidance.py",
    "src/genmol/sampler.py",
    "src/genmol/diffusion.py",
    "src/genmol/denoiser.py",
    "src/genmol/model.py",
    "src/genmol/backbone.py",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return _sha(
        _json_bytes({"dtype": str(value.dtype), "shape": list(value.shape)})
        + value.numpy().tobytes()
    )


def validate_config(config):
    _require(
        isinstance(config, Mapping) and set(config) == CONFIG_KEYS,
        "context guidance requires exactly the declared config fields",
    )
    _require(
        type(config["schema_version"]) is int
        and config["schema_version"] == 1
        and config["method"] == METHOD,
        "unsupported context guidance identity",
    )
    result = dict(config)
    for key, minimum, maximum in (("gamma", 0.0, 1.0), ("scale", 1.0, float("inf"))):
        value = config[key]
        _require(
            not isinstance(value, bool)
            and isinstance(value, numbers.Real)
            and math.isfinite(float(value))
            and minimum <= value <= maximum,
            f"invalid context guidance {key}",
        )
        result[key] = float(value)
    for key, minimum, maximum in (
        ("context_seed", 0, 2**63 - 1),
        ("predictor_steps", 1, 2**31 - 1),
    ):
        _require(
            type(config[key]) is int and minimum <= config[key] <= maximum,
            f"invalid context guidance {key}",
        )
    _require(
        isinstance(config["execution"], str)
        and config["execution"] in {"serial", "packed"},
        "context execution must be serial or packed",
    )
    return result


def _sources():
    return {name: _sha((ROOT / name).read_bytes()) for name in SOURCE_PATHS}


def _sampling_rng_state(device):
    if device.type == "cpu":
        state = torch.get_rng_state()
    elif device.type == "cuda":
        state = torch.cuda.get_rng_state(device)
    else:
        raise ValueError("experimental guidance supports CPU or CUDA sampling only")
    return _tensor_sha(state)


def _runtime_identity(sampler):
    model, process = sampler.model, sampler.mdlm
    _require(
        sampler.diffusion_type == "udlm"
        and type(process) is ContinuousCategoricalDiffusion
        and model.mdlm is process,
        "context guidance requires schedule-consistent categorical UDLM (not legacy R or MDLM)",
    )
    _require(
        isinstance(model, torch.nn.Module)
        and all(not module.training for module in model.modules()),
        "context guidance requires model.eval() with all modules in evaluation mode",
    )
    _require(
        model.udlm_parameterization in {"raw_loo", "x0_denoiser"},
        "unsupported denoiser parameterization",
    )
    model._validate_runtime_udlm_prior_identity()
    model._validate_udlm_parameterization_state_dict(model.state_dict())
    weights = sampler.inference_weights
    _require(
        weights["source"] == "ema" and weights["ema_applied"] is True,
        "context guidance requires recorded EMA inference weights",
    )
    ema = weights.get("ema")
    _require(
        isinstance(ema, Mapping)
        and set(ema) == {"shadow_parameter_count", "decay", "num_updates"},
        "context guidance requires complete load-time EMA metadata",
    )
    _require(
        type(ema["shadow_parameter_count"]) is int
        and ema["shadow_parameter_count"] > 0,
        "invalid EMA shadow count",
    )
    _require(
        not isinstance(ema["decay"], bool)
        and isinstance(ema["decay"], numbers.Real)
        and math.isfinite(float(ema["decay"]))
        and 0 <= ema["decay"] <= 1,
        "invalid EMA decay",
    )
    _require(
        ema["num_updates"] is None
        or (type(ema["num_updates"]) is int and ema["num_updates"] >= 0),
        "invalid EMA update count",
    )
    _require(
        torch.isfinite(process.stationary_probs).all()
        and torch.all(process.stationary_probs > 0),
        "context prior must have finite full support",
    )
    prior = model.udlm_prior_metadata.to_dict()
    identity = dict(
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        parameterization=model.udlm_parameterization,
        prior_metadata=prior,
        prior_metadata_sha256=_sha(_json_bytes(prior)),
        stationary_probability_tensor_sha256=_tensor_sha(process.stationary_probs),
        active_token_ids_tensor_sha256=_tensor_sha(process.diffusion_token_ids),
        inference_weights=weights,
        checkpoint_bytes_revalidated=False,
        checkpoint_scope="runtime prior/parameterization checks and load-time EMA receipt only; caller must bind checkpoint bytes separately",
    )
    if model.udlm_parameterization == "x0_denoiser":
        from genmol.model import UDLM_DENOISER_METADATA

        identity["denoiser_metadata"] = dict(UDLM_DENOISER_METADATA)
    return identity


def _initial_state(sampler, token_ids, config):
    _require(
        isinstance(token_ids, torch.Tensor)
        and token_ids.ndim == 2
        and token_ids.dtype == torch.int64
        and min(token_ids.shape) > 0,
        "token_ids must be nonempty int64 [B,L]",
    )
    x = token_ids.to(sampler.model.device).clone()
    sampler.mdlm._compact_indices(x)  # validates full-vocabulary token range
    editable = x == sampler.model.mask_index
    controls = tuple(sorted(set(sampler.model.tokenizer.all_special_ids)))
    required = {
        sampler.model.bos_index,
        sampler.model.eos_index,
        sampler.pad_index,
        sampler.model.mask_index,
    }
    _require(
        all(type(value) is int for value in controls) and required <= set(controls),
        "tokenizer must declare BOS/EOS/PAD/MASK among all_special_ids",
    )
    eligible = ~editable
    for token in controls:
        eligible &= x != token
    counts = [int(math.floor(config["gamma"] * int(row.sum()))) for row in eligible]
    eps = float(
        sampler.model.config.training.get("udlm", {}).get("inference_eps", 1e-5)
    )
    _require(math.isfinite(eps) and 0 < eps < 1, "invalid context-guidance endpoint")
    grid = torch.linspace(
        1.0, eps, config["predictor_steps"] + 1, dtype=torch.float32, device=x.device
    )
    _require(
        torch.all(grid[1:] > 0) and torch.all(grid[:-1] > grid[1:]),
        "context guidance needs distinct positive-time grid points",
    )
    return x, editable, eligible, counts, controls, grid


def posterior_law(process, logits, original_xt, t, s, editable, parameterization):
    """One branch, original editable observation, float64 active probabilities."""
    _require(
        logits.shape == (*original_xt.shape, process.num_classes)
        and logits.is_floating_point(),
        "predictor logits must be floating [B,L,K]",
    )
    active_logits = logits.index_select(-1, process.diffusion_token_ids)
    _require(
        torch.isfinite(active_logits[editable]).all(),
        "nonfinite active predictor logits at editable coordinates",
    )
    # Predictions at clamped coordinates are irrelevant, including framing IDs
    # outside the active alphabet. They cannot contaminate probability sampling.
    raw = torch.where(
        editable[..., None],
        logits.double(),
        torch.zeros_like(logits, dtype=torch.float64),
    )
    if parameterization == "x0_denoiser":
        from genmol.denoiser import denoiser_to_loo_logits

        raw = denoiser_to_loo_logits(
            process, raw, original_xt, t, mutable_mask=editable
        )
    else:
        _require(parameterization == "raw_loo", "unsupported denoiser parameterization")
    _, active = process._compact_indices(original_xt)
    _require(
        not torch.any(editable & ~active),
        "editable noisy states must belong to the active alphabet",
    )
    bridge_xt = torch.where(active, original_xt, process.diffusion_token_ids[0])
    probabilities = process.posterior_probs(
        raw, bridge_xt, t, s, temperature=1.0, raw_loo_top_p=1.0
    )
    # Sentinel laws at immutable coordinates are not sampled model predictions.
    # Replace only unused laws with a uniform dummy: a rare sentinel can have
    # underflowed entries even when every editable law is representable.
    probabilities = torch.where(
        editable[..., None],
        probabilities,
        torch.full_like(probabilities, 1.0 / process.diffusion_vocab_size),
    )
    _require(
        torch.isfinite(probabilities).all() and torch.all(probabilities > 0),
        "reverse law lost finite positive support; no floors are applied",
    )
    _require(
        torch.allclose(
            probabilities.sum(-1),
            torch.ones_like(probabilities[..., 0]),
            atol=1e-12,
            rtol=1e-12,
        ),
        "reverse law is not normalized",
    )
    return probabilities


def blend_reverse_laws(conditional, poor, scale):
    """Float64 geometric extrapolation, rejecting zeros/overflow/underflow."""
    _require(
        conditional.shape == poor.shape
        and conditional.dtype == poor.dtype == torch.float64,
        "reverse laws must be matching float64 tensors",
    )
    _require(
        math.isfinite(scale) and scale >= 1, "guidance scale must be finite and >=1"
    )
    for value in (conditional, poor):
        _require(
            torch.isfinite(value).all() and torch.all(value > 0),
            "reverse laws require finite positive support",
        )
        _require(
            torch.allclose(
                value.sum(-1), torch.ones_like(value[..., 0]), atol=1e-12, rtol=1e-12
            ),
            "reverse laws must be normalized",
        )
    scores = scale * conditional.log() + (1.0 - scale) * poor.log()
    _require(torch.isfinite(scores).all(), "guided log scores overflowed")
    result = (scores - scores.logsumexp(-1, keepdim=True)).exp()
    _require(
        torch.isfinite(result).all() and torch.all(result > 0),
        "guided law lost finite positive support; no floors are applied",
    )
    _require(
        torch.allclose(
            result.sum(-1), torch.ones_like(result[..., 0]), atol=1e-12, rtol=1e-12
        ),
        "guided law is not normalized",
    )
    return result


@torch.no_grad()
def generate_context_guided(sampler, token_ids, *, config):
    started = time.perf_counter()
    config = validate_config(config)  # precedes every state/RNG operation
    identity, sources = _runtime_identity(sampler), _sources()
    original, editable, eligible, counts, controls, grid = _initial_state(
        sampler, token_ids, config
    )
    batch, length = original.shape
    rng_before = _sampling_rng_state(original.device)
    noop = (
        "scale_one"
        if config["scale"] == 1
        else "gamma_zero"
        if config["gamma"] == 0
        else "no_selected_context"
        if not any(counts)
        else "no_editable_tokens"
        if not editable.any()
        else None
    )
    observed_batches, masks = [], hashlib.sha256()
    context_draws = 0

    def record_forward(_module, args, kwargs):
        inputs = args[0] if args else kwargs.get("input_ids")
        _require(
            isinstance(inputs, torch.Tensor) and inputs.ndim == 2,
            "cannot identify observed backbone batch",
        )
        observed_batches.append(int(inputs.shape[0]))

    handle = sampler.model.backbone.register_forward_pre_hook(
        record_forward, with_kwargs=True
    )
    try:
        if noop is not None:
            result = sampler.generate(
                token_ids,
                softmax_temp=1.0,
                randomness=0,
                gamma=0,
                num_steps=config["predictor_steps"],
                raw_loo_top_p=1.0,
                gibbs_corrector=False,
                temperature_space="raw_loo",
                return_token_ids=True,
            )
        else:
            context_rng = torch.Generator(device="cpu")
            context_rng.manual_seed(config["context_seed"])
            eligible_ids = [row.nonzero(as_tuple=True)[0].cpu() for row in eligible]
            x = torch.where(
                editable,
                sampler.mdlm.sample_prior(original.shape, device=original.device),
                original,
            )
            attention = original != sampler.pad_index
            for step in range(config["predictor_steps"]):
                selected = torch.zeros_like(editable)
                for row, (indices, count) in enumerate(zip(eligible_ids, counts)):
                    if count:
                        chosen = indices[
                            torch.randperm(len(indices), generator=context_rng)[:count]
                        ]
                        selected[row, chosen.to(x.device)] = True
                        context_draws += 1
                _require(
                    not torch.any(selected & editable),
                    "context selection crossed the fixed editable boundary",
                )
                masks.update(selected.cpu().numpy().tobytes())
                poor_x = torch.where(selected, sampler.model.mask_index, x)
                _require(
                    torch.equal(poor_x[editable], x[editable]),
                    "poor view changed editable noisy tokens",
                )
                t, s = grid[step].expand(batch), grid[step + 1].expand(batch)
                if config["execution"] == "packed":
                    packed = sampler.model(
                        torch.cat((x, poor_x)),
                        torch.cat((attention, attention)),
                        t=torch.cat((t, t)),
                    )
                    _require(
                        packed.shape[0] == 2 * batch, "packed prediction batch differs"
                    )
                    conditional_logits, poor_logits = packed.split(batch)
                else:
                    conditional_logits = sampler.model(x, attention, t=t)
                    poor_logits = sampler.model(poor_x, attention, t=t)
                p_c = posterior_law(
                    sampler.mdlm,
                    conditional_logits,
                    x,
                    t,
                    s,
                    editable,
                    identity["parameterization"],
                )
                p_u = posterior_law(
                    sampler.mdlm,
                    poor_logits,
                    x,
                    t,
                    s,
                    editable,
                    identity["parameterization"],
                )
                law = blend_reverse_laws(p_c, p_u, config["scale"])
                # Rows without selected context in a mixed batch still incurred
                # both model predictions; only their law is conditional here.
                active_rows = torch.tensor(
                    [count > 0 for count in counts], device=x.device
                )[:, None, None]
                law = torch.where(active_rows, law, p_c)
                sampled = torch.multinomial(
                    law.reshape(-1, sampler.mdlm.diffusion_vocab_size), 1
                ).reshape(batch, length)
                x = torch.where(
                    editable, sampler.mdlm.diffusion_token_ids[sampled], original
                )
                _require(
                    torch.equal(x[~editable], original[~editable]),
                    "immutable context changed",
                )
            result = x
    finally:
        handle.remove()
    _require(
        torch.equal(result[~editable], original[~editable]), "immutable context changed"
    )
    _require(
        _runtime_identity(sampler) == identity and _sources() == sources,
        "runtime identity or implementation sources changed during guidance",
    )
    expected_batches = (
        [batch] * config["predictor_steps"]
        if noop is not None
        else (
            [2 * batch] * config["predictor_steps"]
            if config["execution"] == "packed"
            else [batch] * (2 * config["predictor_steps"])
        )
    )
    _require(
        observed_batches == expected_batches,
        "observed backbone work differs from the declared execution",
    )
    receipt = dict(
        schema_version=1,
        status="completed",
        scientific_status="experimental sampler-only hypothesis; no benchmark acceptance or efficacy claim",
        config=config,
        config_sha256=_sha(_json_bytes(config)),
        identity=identity,
        source_sha256=sources,
        precision="float64 reverse probabilities and log-space blend; reject nonfinite or zero support"
        if noop is None
        else "exact historical predictor path",
        application_order="per-branch CE-to-LOO if applicable; same original editable state/time/prior; reverse posterior; log blend; categorical draw; clamp original context",
        fixed_controls=dict(temperature=1.0, raw_loo_top_p=1.0, gibbs_corrector=False),
        context_policy=dict(
            eligibility="original immutable per-row non-control tokens",
            control_ids=list(controls),
            resampling="fresh subset per row per predictor step; dedicated CPU torch generator",
            context_seed=config["context_seed"],
            eligible_counts=[int(row.sum()) for row in eligible],
            requested_subset_sizes=counts,
            selected_counts=counts if noop is None else [0] * batch,
            subset_draws=context_draws,
            selection_sha256=masks.hexdigest(),
            initial_ids_sha256=_tensor_sha(original),
            editable_mask_sha256=_tensor_sha(editable),
        ),
        observations=dict(
            batch_size=batch,
            predictor_steps=config["predictor_steps"],
            noop_reason=noop,
            backbone_forward_calls=len(observed_batches),
            backbone_batch_sizes=observed_batches,
            candidate_equivalent_evaluations=sum(observed_batches),
            evaluations_per_candidate=sum(observed_batches) / batch,
            output_token_ids_sha256=_tensor_sha(result),
            immutable_context_preserved=True,
            sampling_rng_state_before_sha256=rng_before,
            sampling_rng_state_after_sha256=_sampling_rng_state(original.device),
            elapsed_seconds=time.perf_counter() - started,
        ),
        sampling_rng="caller-owned global Torch RNG; context RNG is separate; no packed/serial bitwise-equivalence claim",
        immutable_probability_policy="uniform active-alphabet dummy at immutable coordinates only; sampled values discarded by original-context clamp",
    )
    # Detach every metadata container from both caller config and model records.
    return {"token_ids": result, "receipt": json.loads(_json_bytes(receipt))}
