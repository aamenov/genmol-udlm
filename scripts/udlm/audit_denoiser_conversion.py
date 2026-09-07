"""CPU-only exact toy proof and gradient audit; no training or artifact writes.

Run with the project Python. The accompanying derivation and limitations are in
docs/udlm_denoiser_ce_hypothesis.md. Assertions are the executable checks; JSON
on stdout records their results. Do not run Python with assertion disabling -O.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from fractions import Fraction as F
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATES = tuple(product(range(3), repeat=2))
WEIGHTS = ((11, 1, 2), (1, 9, 1), (2, 1, 7))
DATA = {x: F(WEIGHTS[x[0]][x[1]], 35) for x in STATES}
PRIORS = ((F(1, 3),) * 3, (F(72, 100), F(23, 100), F(5, 100)))
ALPHAS = ((F(4, 5), F(1, 3)), (F(1), F(9, 10)), (F(1, 5), F(1, 100)))


def normalize(values):
    total = sum(values)
    return [v / total for v in values]


def kernel(alpha, earlier, later, pi):
    return alpha * (earlier == later) + (1 - alpha) * pi[later]


def conditional_clean(xt, alpha_t, pi):
    return normalize(
        [
            DATA[x]
            * math.prod(
                kernel(alpha_t, x[position], xt[position], pi) for position in range(2)
            )
            for x in STATES
        ]
    )


def marginals(probabilities):
    return [
        [
            sum(p for x, p in zip(STATES, probabilities) if x[position] == j)
            for j in range(3)
        ]
        for position in range(2)
    ]


def convert(denoiser, current, alpha_t, pi):
    return normalize([denoiser[j] / kernel(alpha_t, j, current, pi) for j in range(3)])


def bridge(raw_loo, current, alpha_s, alpha_t, pi):
    return normalize(
        [
            kernel(alpha_t / alpha_s, i, current, pi)
            * (alpha_s * raw_loo[i] + (1 - alpha_s) * pi[i])
            for i in range(3)
        ]
    )


def exact_audit():
    """Enumerate all x0,zs,zt trajectories, independently of production code."""
    checks = 0
    max_direct_error = F(0)
    max_joint_tv = F(0)
    cases = []
    for pi, (alpha_s, alpha_t), xt in product(PRIORS, ALPHAS, STATES):
        clean = conditional_clean(xt, alpha_t, pi)
        denoiser = marginals(clean)
        reverse_joint = normalize(
            [
                sum(
                    DATA[x]
                    * math.prod(
                        kernel(alpha_s, x[position], zs[position], pi)
                        * kernel(alpha_t / alpha_s, zs[position], xt[position], pi)
                        for position in range(2)
                    )
                    for x in STATES
                )
                for zs in STATES
            ]
        )
        exact_reverse = marginals(reverse_joint)
        raw_loo = []
        for position in range(2):
            current = xt[position]
            converted = convert(denoiser[position], current, alpha_t, pi)
            direct_loo = normalize(
                [
                    sum(
                        DATA[x] * kernel(alpha_t, x[1 - position], xt[1 - position], pi)
                        for x in STATES
                        if x[position] == j
                    )
                    for j in range(3)
                ]
            )
            posterior_mixture = [
                sum(
                    denoiser[position][j]
                    * kernel(alpha_t / alpha_s, i, current, pi)
                    * kernel(alpha_s, j, i, pi)
                    / kernel(alpha_t, j, current, pi)
                    for j in range(3)
                )
                for i in range(3)
            ]
            converted_bridge = bridge(converted, current, alpha_s, alpha_t, pi)
            assert converted == direct_loo
            assert converted_bridge == posterior_mixture == exact_reverse[position]
            checks += 3
            direct = bridge(denoiser[position], current, alpha_s, alpha_t, pi)
            max_direct_error = max(
                max_direct_error,
                *(abs(a - b) for a, b in zip(direct, exact_reverse[position])),
            )
            raw_loo.append(converted)
        factorized = [
            math.prod(exact_reverse[position][x[position]] for position in range(2))
            for x in STATES
        ]
        tv = sum(abs(a - b) for a, b in zip(reverse_joint, factorized)) / 2
        max_joint_tv = max(max_joint_tv, tv)
        cases.append((pi, alpha_s, alpha_t, xt, denoiser, raw_loo, exact_reverse))
    assert checks == 324 and max_direct_error > F(3, 10) and max_joint_tv > F(1, 5)
    return cases, {
        "posterior_scalar_equalities": checks,
        "leave_one_out_scalar_equalities": checks,
        "trajectory_count": len(PRIORS) * len(ALPHAS) * len(STATES) ** 3,
        "uncorrected_denoiser_max_abs_error": float(max_direct_error),
        "factorized_vs_true_joint_max_tv": float(max_joint_tv),
        "arithmetic": "fractions.Fraction; all asserted identities exact",
    }


def torch_audit(cases):
    # Hide GPUs before importing Torch; all tensors/modules explicitly stay on CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import torch.nn.functional as nnf

    from genmol.diffusion import (
        ContinuousCategoricalDiffusion,
        ContinuousUniformDiffusion,
    )

    torch.set_num_threads(1)

    def tensor(values):
        return torch.tensor(values, dtype=torch.float64, device="cpu")

    max_error = 0.0
    calls = 0
    max_conversion_error = 0.0
    for pi, alpha_s, alpha_t, xt, denoiser, raw_loo, exact in cases:
        diffusion = ContinuousCategoricalDiffusion(3, [float(p) for p in pi])
        models = [diffusion]
        if pi == PRIORS[0]:
            models.append(ContinuousUniformDiffusion(3))
        log_d = tensor([[list(map(float, row)) for row in denoiser]]).log()
        log_likelihood = tensor(
            [
                [
                    [float(kernel(alpha_t, j, xt[position], pi)) for j in range(3)]
                    for position in range(2)
                ]
            ]
        ).log()
        logits = log_d - log_likelihood
        expected_loo = tensor([[list(map(float, row)) for row in raw_loo]])
        max_conversion_error = max(
            max_conversion_error, (logits.softmax(-1) - expected_loo).abs().max().item()
        )
        expected = tensor([[list(map(float, row)) for row in exact]])
        current = torch.tensor([xt], dtype=torch.long, device="cpu")
        for model in models:
            t = tensor([(1 - float(alpha_t)) / (1 - model.noise_eps)])
            s = tensor([(1 - float(alpha_s)) / (1 - model.noise_eps)])
            result = model.posterior_probs(logits, current, t, s)
            max_error = max(max_error, (result - expected).abs().max().item())
            calls += 1
    assert calls == 81 and max_error < 1e-12 and max_conversion_error < 1e-12

    pi = PRIORS[1]
    model = ContinuousCategoricalDiffusion(3, [float(p) for p in pi])
    current_tuple = (1, 2)
    clean_states = torch.tensor(STATES, dtype=torch.long, device="cpu")
    current_states = torch.tensor([current_tuple] * 9, dtype=torch.long, device="cpu")
    expected_gradients = []
    for time in (F(1, 100), F(1, 2), F(99, 100)):
        alpha_t = 1 - (1 - F(1, 1000)) * time
        conditional = conditional_clean(current_tuple, alpha_t, pi)
        denoiser = marginals(conditional)
        raw_loo = [
            convert(denoiser[position], current_tuple[position], alpha_t, pi)
            for position in range(2)
        ]
        weights = tensor(list(map(float, conditional)))
        times = tensor([float(time)] * 9)
        row = {"t": float(time)}
        for name, prediction in (("D", denoiser), ("R", raw_loo)):
            logits = tensor([list(map(float, p)) for p in prediction]).log()
            logits.requires_grad_(True)
            expanded = logits.unsqueeze(0).expand(9, -1, -1)
            losses = {
                "ct": model.loss(expanded, clean_states, current_states, times),
                "ce": nnf.cross_entropy(
                    expanded.reshape(-1, 3), clean_states.reshape(-1), reduction="none"
                )
                .reshape(9, 2)
                .mean(-1),
            }
            for objective, loss in losses.items():
                grad = torch.autograd.grad(
                    (weights * loss).sum(), logits, retain_graph=True
                )[0]
                row[f"{objective}_gradient_norm_at_{name}"] = grad.norm().item()
        assert row["ct_gradient_norm_at_R"] < 1e-10
        assert row["ce_gradient_norm_at_D"] < 1e-12
        assert row["ct_gradient_norm_at_D"] > 1e-3
        assert row["ce_gradient_norm_at_R"] > 1e-3
        expected_gradients.append(row)

    single_gradients = []
    clean = torch.tensor([[0]], dtype=torch.long, device="cpu")
    current = torch.tensor([[1]], dtype=torch.long, device="cpu")
    for time in (0.001, 0.01, 0.5, 0.99, 0.9999):
        logits = tensor([[[0.0, 8.0, 0.0]]]).requires_grad_(True)
        losses = {
            "ct": model.loss(logits, clean, current, tensor([time])).mean(),
            "ce": nnf.cross_entropy(logits.reshape(1, 3), clean.reshape(1)),
        }
        row = {"t": time}
        for name, loss in losses.items():
            grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            row[f"{name}_loss"] = loss.item()
            row[f"{name}_gradient_norm"] = grad.norm().item()
        single_gradients.append(row)
    return {
        "device": "cpu",
        "torch_version": torch.__version__,
        "production_posterior_calls": calls,
        "production_posterior_max_abs_error": max_error,
        "log_space_conversion_max_abs_error": max_conversion_error,
        "conditional_expected_gradients": expected_gradients,
        "single_example_gradients": single_gradients,
    }


def main():
    if not __debug__:
        raise RuntimeError("Run without -O: this audit requires assertions")
    cases, exact = exact_audit()
    result = {
        "status": "passed",
        "claim": "algebra_and_toy_gradients_only_no_empirical_improvement",
        "exact": exact,
        "production": torch_audit(cases),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / "src/genmol/diffusion.py")
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
