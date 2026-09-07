"""CPU toy audit of temperature before versus after CE-to-LOO conversion.

No model checkpoint, molecular data, sampling seeds or GPU are used. This
script composes existing production primitives; it does not modify sampling.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from fractions import Fraction as F
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PI = (
    (F(1, 3),) * 3,
    (F(72, 100), F(23, 100), F(5, 100)),
    (F(9, 10), F(999, 10000), F(1, 10000)),
)
DENOISERS = (
    (F(1, 3),) * 3,
    (F(1, 1000), F(9, 1000), F(99, 100)),
    (F(1, 5), F(3, 10), F(1, 2)),
)
TIMES = ((F(1, 2), F(2, 5)), (F(1), F(0)), (F(1, 100000), F(0)))
NOISE_EPS = F(1, 1000)


def normalize(values):
    total = sum(values)
    return tuple(value / total for value in values)


def kernel(alpha, earlier, later, pi):
    return alpha * (earlier == later) + (1 - alpha) * pi[later]


def mixed_bridge(denoiser, current, alpha_t, alpha_s, pi):
    """Independently mix exact forward bridges for specified clean tokens."""
    return tuple(
        sum(
            denoiser[clean]
            * kernel(alpha_s, clean, earlier, pi)
            * kernel(alpha_t / alpha_s, earlier, current, pi)
            / kernel(alpha_t, clean, current, pi)
            for clean in range(3)
        )
        for earlier in range(3)
    )


def exact_cases():
    cases = []
    for pi, denoiser, (t, s), current, power in product(
        PI, DENOISERS, TIMES, range(3), (1, 2, 3)
    ):
        alpha_t = 1 - (1 - NOISE_EPS) * t
        alpha_s = 1 - (1 - NOISE_EPS) * s
        likelihood = tuple(kernel(alpha_t, j, current, pi) for j in range(3))
        raw = normalize([denoiser[j] / likelihood[j] for j in range(3)])
        old_raw = normalize([value**power for value in raw])
        implied_old = normalize([old_raw[j] * likelihood[j] for j in range(3)])
        direct_old = normalize(
            [denoiser[j] ** power * likelihood[j] ** (1 - power) for j in range(3)]
        )
        new_denoiser = normalize([value**power for value in denoiser])
        new_raw = normalize([new_denoiser[j] / likelihood[j] for j in range(3)])
        recovered_new = normalize([new_raw[j] * likelihood[j] for j in range(3)])
        if implied_old != direct_old or recovered_new != new_denoiser:
            raise AssertionError("temperature/conversion identity failed")
        if power == 1 and (old_raw != new_raw or implied_old != new_denoiser):
            raise AssertionError("temperature-one identity failed")
        old_bridge = mixed_bridge(implied_old, current, alpha_t, alpha_s, pi)
        new_bridge = mixed_bridge(new_denoiser, current, alpha_t, alpha_s, pi)
        if sum(old_bridge) != 1 or sum(new_bridge) != 1:
            raise AssertionError("forward bridge mixture is not normalized")
        cases.append(
            dict(
                pi=pi,
                denoiser=denoiser,
                t=t,
                s=s,
                current=current,
                power=power,
                implied_old=implied_old,
                new_denoiser=new_denoiser,
                old_bridge=old_bridge,
                new_bridge=new_bridge,
            )
        )
    return cases


def production_audit(cases):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from genmol.denoiser import denoiser_to_loo_logits
    from genmol.diffusion import ContinuousCategoricalDiffusion

    torch.set_num_threads(1)
    maximum = 0.0
    for case in cases:
        process = ContinuousCategoricalDiffusion(
            3, [float(x) for x in case["pi"]], noise_eps=float(NOISE_EPS)
        )
        logits = torch.tensor(
            [[[float(x) for x in case["denoiser"]]]], dtype=torch.float64
        ).log()
        current = torch.tensor([[case["current"]]])
        t = torch.tensor([float(case["t"])], dtype=torch.float64)
        s = torch.tensor([float(case["s"])], dtype=torch.float64)
        old_logits = denoiser_to_loo_logits(process, logits, current, t)
        new_logits = denoiser_to_loo_logits(process, logits * case["power"], current, t)
        for actual, expected in (
            (
                process.posterior_probs(
                    old_logits, current, t, s, temperature=1 / case["power"]
                ),
                case["old_bridge"],
            ),
            (
                process.posterior_probs(new_logits, current, t, s, temperature=1.0),
                case["new_bridge"],
            ),
        ):
            reference = torch.tensor(
                [[[float(x) for x in expected]]], dtype=torch.float64
            )
            maximum = max(maximum, (actual - reference).abs().max().item())
            torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    return maximum


def main():
    files = [
        Path(__file__),
        ROOT / "src/genmol/denoiser.py",
        ROOT / "src/genmol/diffusion.py",
    ]
    before = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    cases = exact_cases()
    maximum = production_audit(cases)
    after = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    if before != after:
        raise RuntimeError("audit source changed during execution")
    example = next(
        case
        for case in cases
        if case["pi"] == PI[2]
        and case["denoiser"] == DENOISERS[1]
        and case["t"] == F(1, 2)
        and case["s"] == F(2, 5)
        and case["current"] == 2
        and case["power"] == 2
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "schema_version": 1,
                "scope": "CPU algebra and existing production primitives; no molecular claim",
                "molecular_samples": 0,
                "checkpoint_reads": 0,
                "rational_cases": len(cases),
                "production_scalar_checks": 6 * len(cases),
                "maximum_production_absolute_error": maximum,
                "temperature_one_cases": sum(case["power"] == 1 for case in cases),
                "example": {
                    key: (
                        [float(x) for x in value]
                        if isinstance(value, tuple)
                        else float(value) if isinstance(value, F) else value
                    )
                    for key, value in example.items()
                },
                "source_sha256": before,
                "limitations": [
                    "chosen toy probabilities",
                    "coordinate rather than joint tempering",
                    "no learned-model or benchmark measurement",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
