"""Opt-in single-coordinate Gibbs correction for rank-one UDLM processes.

This module is not wired into the sampler or any registered benchmark. With an
exact, untempered leave-one-out (LOO) predictor and a fixed editable set,
uniform random-scan Gibbs preserves the noisy-data marginal conditioned on the
immutable context. A learned predictor can depend on its own noisy token;
temperature and nucleus filtering also change its conditional. Those settings
are approximate sampling hypotheses, not an exact-stationarity guarantee.

Reference: Uniform Diffusion Models Revisited, Appendix E / Algorithm 5,
https://arxiv.org/abs/2605.22765. The nonuniform stationary-prior specialization
here uses the project's existing rank-one categorical process.
"""

from __future__ import annotations

import torch

from genmol.diffusion import (
    ContinuousCategoricalDiffusion,
    ContinuousUniformDiffusion,
)


def gibbs_conditional_probs(
    diffusion: ContinuousUniformDiffusion,
    logits: torch.Tensor,
    s: torch.Tensor,
    *,
    temperature: float = 1.0,
    raw_loo_top_p: float = 1.0,
) -> torch.Tensor:
    """Return the inferred Gibbs conditional on the active token alphabet.

    ``logits`` is floating ``[B,L,K]`` in full model-vocabulary coordinates;
    ``s`` is floating ``[B]`` with ``0 <= s <= 1``. The returned probabilities
    are float64 ``[B,L,A]``, ordered by ``diffusion.diffusion_token_ids``, where
    ``A`` is the active vocabulary size. Float64 preserves the categorical
    process's small positive stationary masses when lower-precision logits
    are supplied. Inputs and process buffers are never mutated.

    If ``r`` is the raw LOO distribution after optional temperature/top-p,
    this computes ``alpha(s) * r + (1-alpha(s)) * pi``. It deliberately omits
    the observed-token likelihood used in a reverse bridge: a Gibbs update
    conditions on the *other* coordinates. At ``s=0`` the conditional is r.
    At ``s=1`` the release schedule still has its configured residual signal.
    """

    if not isinstance(diffusion, ContinuousUniformDiffusion):
        raise TypeError("corrector requires a uniform or categorical UDLM process")
    if logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("logits must be a floating tensor with shape [B,L,K]")
    if logits.shape[-1] != diffusion.num_classes:
        raise ValueError("logits vocabulary does not match the diffusion process")
    if s.shape != (logits.shape[0],) or not s.is_floating_point():
        raise ValueError("s must be a floating tensor with shape [B]")
    if s.device != logits.device:
        raise ValueError("s and logits must be on the same device")
    if not torch.isfinite(s).all() or torch.any((s < 0) | (s > 1)):
        raise ValueError("corrector time requires finite 0 <= s <= 1")
    if diffusion.diffusion_token_ids.device != logits.device:
        raise ValueError("diffusion and logits must be on the same device")

    log_r = diffusion.clean_log_probs(
        logits,
        temperature=temperature,
        raw_loo_top_p=raw_loo_top_p,
    ).to(dtype=torch.float64)
    typed_s = s.to(dtype=torch.float64)
    alpha = diffusion.alpha(typed_s)[:, None, None]
    # Compute the small refresh mass directly: 1-alpha can round to zero
    # for tiny positive s even though a positive stationary mass is intended.
    refresh = -torch.expm1(-diffusion.sigma(typed_s))[:, None, None]
    if isinstance(diffusion, ContinuousCategoricalDiffusion):
        pi = diffusion.stationary_probs
    else:
        pi = torch.full(
            (diffusion.diffusion_vocab_size,),
            1.0 / diffusion.diffusion_vocab_size,
            dtype=torch.float64,
            device=logits.device,
        )
    probabilities = alpha * log_r.exp() + refresh * pi
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError(
            "LOO predictions must produce finite nonnegative probabilities"
        )
    totals = probabilities.sum(dim=-1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError("every Gibbs conditional must have positive total mass")
    # Remove accepted softmax/schedule floating-point summation drift only.
    return probabilities / totals


@torch.no_grad()
def random_scan_gibbs_step(
    diffusion: ContinuousUniformDiffusion,
    logits: torch.Tensor,
    xt: torch.Tensor,
    s: torch.Tensor,
    *,
    mutable_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    raw_loo_top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Resample one uniformly selected editable coordinate per nonempty row.

    ``xt`` contains int32/int64 model token IDs ``[B,L]``. ``mutable_mask`` is
    Boolean ``[B,L]`` (default: all positions). Excluded control tokens are
    legal only at immutable positions. Rows with no editable coordinate are
    unchanged; a wholly immutable batch consumes no random draws. A sampled
    token may equal the previous token, so the number of *changed* positions
    per row is zero or one. Output shape/dtype/device match xt; inputs are not
    modified. The optional generator controls both coordinate and token draws
    and must be compatible with the tensors' device.

    Use logits freshly evaluated at ``xt,s``. Keep the editable set independent
    of the current token values for the exact Gibbs interpretation. In
    particular, deriving this mask from low model confidence or choosing a
    low-margin coordinate is a different, approximate kernel. Repeat calls
    require fresh model predictions after the preceding sampled update.
    """

    if xt.ndim != 2 or xt.dtype not in (torch.int32, torch.int64):
        raise ValueError("xt must be an int32/int64 tensor with shape [B,L]")
    if logits.shape[:-1] != xt.shape or logits.device != xt.device:
        raise ValueError("logits and xt must have matching [B,L] shape and device")
    if not isinstance(diffusion, ContinuousUniformDiffusion):
        raise TypeError("corrector requires a uniform or categorical UDLM process")
    if torch.any((xt < 0) | (xt >= diffusion.num_classes)):
        raise ValueError("xt contains token IDs outside the model vocabulary")
    if mutable_mask is None:
        mutable_mask = torch.ones_like(xt, dtype=torch.bool)
    elif (
        mutable_mask.shape != xt.shape
        or mutable_mask.dtype != torch.bool
        or mutable_mask.device != xt.device
    ):
        raise ValueError("mutable_mask must be Boolean [B,L] on xt's device")
    if diffusion.token_to_diffusion_index.device != xt.device:
        raise ValueError("diffusion and xt must be on the same device")
    compact_ids = diffusion.token_to_diffusion_index[xt.long()]
    if torch.any(mutable_mask & (compact_ids < 0)):
        raise ValueError("editable positions must contain active-alphabet token IDs")

    conditionals = gibbs_conditional_probs(
        diffusion,
        logits,
        s,
        temperature=temperature,
        raw_loo_top_p=raw_loo_top_p,
    )
    result = xt.clone()
    active_rows = mutable_mask.any(dim=-1).nonzero(as_tuple=True)[0]
    if active_rows.numel() == 0:
        return result
    positions = torch.multinomial(
        mutable_mask[active_rows].to(dtype=torch.float64),
        num_samples=1,
        generator=generator,
    ).squeeze(-1)
    sampled_compact = torch.multinomial(
        conditionals[active_rows, positions],
        num_samples=1,
        generator=generator,
    ).squeeze(-1)
    result[active_rows, positions] = diffusion.diffusion_token_ids[sampled_compact].to(
        dtype=xt.dtype
    )
    return result
