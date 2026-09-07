"""Optional clean-token CE predictions and their conversion to raw LOO logits.

This is a proposed objective/parameterization change, not the released UDLM
objective. Exact denoiser probabilities convert to exact coordinate LOO
conditionals; learned calibration errors and factorized reverse sampling remain
approximations. See docs/udlm_denoiser_ce_hypothesis.md for the derivation.
"""

from __future__ import annotations

import torch

from genmol.diffusion import ContinuousCategoricalDiffusion


def _validate_inputs(diffusion, logits, token_ids, mask):
    if not isinstance(diffusion, ContinuousCategoricalDiffusion):
        raise TypeError("clean denoiser requires schedule-consistent categorical UDLM")
    if (
        logits.ndim != 3
        or not logits.is_floating_point()
        or logits.shape[-1] != diffusion.num_classes
        or token_ids.shape != logits.shape[:-1]
        or token_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("expected floating logits [B,L,K] and integer token IDs [B,L]")
    if (
        token_ids.device != logits.device
        or diffusion.diffusion_token_ids.device != logits.device
    ):
        raise ValueError("inputs and diffusion must be on the same device")
    if mask is None:
        mask = torch.ones_like(token_ids, dtype=torch.bool)
    if (
        mask.shape != token_ids.shape
        or mask.dtype != torch.bool
        or mask.device != logits.device
    ):
        raise ValueError("mask must be Boolean [B,L] on the logits device")
    compact, allowed = diffusion._compact_indices(token_ids.long())
    if torch.any(mask & ~allowed):
        raise ValueError("selected positions must contain active-alphabet tokens")
    return compact, mask


def denoiser_to_loo_logits(
    diffusion: ContinuousCategoricalDiffusion,
    logits: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    *,
    mutable_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Subtract local forward log likelihood before temperature or top-p.

    Input/output logits are full-vocabulary [B,L,K]; xt is [B,L], and t is
    floating [B] with 0<t<=1. For active candidate j and observed token k,
    subtract log(alpha(t)*1[j=k]+(1-alpha(t))*pi[k]). The result's active
    softmax is proportional to D[j]/L[j]. No probability division is used.
    Excluded vocabulary columns and immutable positions retain their logits.
    Float64 stays float64; lower precision is promoted to float32 as in UDLM.
    """
    compact, mask = _validate_inputs(diffusion, logits, xt, mutable_mask)
    if (
        t.shape != (logits.shape[0],)
        or not t.is_floating_point()
        or t.device != logits.device
    ):
        raise ValueError("t must be floating [B] on the logits device")
    if not torch.isfinite(t).all() or torch.any((t <= 0) | (t > 1)):
        raise ValueError("denoiser conversion requires finite 0 < t <= 1")
    result = logits.clone() if logits.dtype == torch.float64 else logits.float().clone()
    typed_t = t.to(dtype=result.dtype)
    refresh = (1.0 - diffusion.noise_eps) * typed_t
    log_alpha = torch.log1p(-refresh)[:, None, None]
    log_refresh = refresh.log()[:, None, None]
    # Log the float64 prior before narrowing, preserving very small masses.
    log_pi = diffusion.stationary_probs.log().to(dtype=result.dtype)
    log_likelihood = (
        (log_refresh + log_pi[compact].unsqueeze(-1))
        .expand(*xt.shape, diffusion.diffusion_vocab_size)
        .clone()
    )
    current_log_likelihood = torch.logaddexp(
        log_alpha, log_refresh + log_pi[compact].unsqueeze(-1)
    )
    log_likelihood.scatter_(-1, compact.unsqueeze(-1), current_log_likelihood)
    active_logits = result.index_select(-1, diffusion.diffusion_token_ids)
    converted = active_logits - torch.where(
        mask.unsqueeze(-1), log_likelihood, torch.zeros_like(log_likelihood)
    )
    result.index_copy_(-1, diffusion.diffusion_token_ids, converted)
    return result


def clean_denoiser_loss(
    diffusion: ContinuousCategoricalDiffusion,
    logits: torch.Tensor,
    x0: torch.Tensor,
    *,
    mask: torch.Tensor,
    global_mean: bool = False,
) -> torch.Tensor:
    """CE over all selected clean tokens, including unchanged noisy inputs.

    Active-alphabet normalization gives excluded logits zero gradient. Immutable
    tokens have zero loss/gradient. Match the existing CT reduction: return [B]
    per-row token means, or a scalar global token mean when requested.
    """
    compact, mask = _validate_inputs(diffusion, logits, x0, mask)
    if not torch.any(mask):
        raise ValueError("clean denoiser loss mask selects no active tokens")
    log_d = diffusion.clean_log_probs(logits)
    nll = -log_d.gather(-1, compact.unsqueeze(-1)).squeeze(-1)
    nll = torch.where(mask, nll, torch.zeros_like(nll))
    if global_mean:
        return nll.sum() / mask.sum()
    return nll.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
