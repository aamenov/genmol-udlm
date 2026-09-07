"""Enumerate a tiny oracle reverse chain on CPU; no molecular/model claim.

Run with the project virtual environment. Assertions check exact-joint recovery,
factorization for independent data, and production coordinate-bridge agreement.
No weights, GPU, randomness, or output files are used. JSON goes to stdout.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np

STATES = np.array(tuple(product(range(3), repeat=2)), dtype=np.int64)
DATA = np.array([[11, 1, 2], [1, 9, 1], [2, 1, 7]], dtype=np.float64).ravel() / 35
PRIORS = {"uniform": np.full(3, 1 / 3), "skewed": np.array([0.72, 0.23, 0.05])}
NOISE_EPS = 1e-3
INFERENCE_EPS = 1e-5
NFES = (1, 8, 32, 64, 128, 256, 512)


def forward_matrix(alpha, prior):
    """Rows are earlier states; columns are later states in the nine-state space."""
    local = alpha * np.eye(3) + (1 - alpha) * prior[None, :]
    return (
        local[STATES[:, None, 0], STATES[None, :, 0]]
        * local[STATES[:, None, 1], STATES[None, :, 1]]
    )


def tv(first, second):
    return float(np.abs(first - second).sum() / 2)


def reverse_matrices(data, prior, alpha_s, alpha_t):
    """Bayes-enumerate the joint reverse row, then multiply its two marginals."""
    ps = data @ forward_matrix(alpha_s, prior)
    joint = (ps[:, None] * forward_matrix(alpha_t / alpha_s, prior)).T
    joint /= joint.sum(axis=1, keepdims=True)
    marginals = np.stack(
        [
            np.stack([joint[:, STATES[:, position] == j].sum(1) for j in range(3)], 1)
            for position in range(2)
        ],
        1,
    )
    factorized = marginals[:, 0, STATES[:, 0]] * marginals[:, 1, STATES[:, 1]]
    assert np.max(np.abs(joint.sum(1) - 1)) < 1e-14
    assert np.max(np.abs(factorized.sum(1) - 1)) < 1e-14
    return joint, factorized, marginals


def chain(data, prior, nfe):
    times = np.linspace(1.0, INFERENCE_EPS, nfe + 1)
    alphas = 1 - (1 - NOISE_EPS) * times
    start = data @ forward_matrix(alphas[0], prior)
    prior_joint = prior[STATES[:, 0]] * prior[STATES[:, 1]]
    exact = start.copy()
    factorized = start.copy()
    initialized_from_prior = prior_joint.copy()
    for alpha_t, alpha_s in zip(alphas[:-1], alphas[1:]):
        joint, independent, _ = reverse_matrices(data, prior, alpha_s, alpha_t)
        exact = exact @ joint
        factorized = factorized @ independent
        initialized_from_prior = initialized_from_prior @ independent
    endpoint = data @ forward_matrix(alphas[-1], prior)
    exact_error = tv(exact, endpoint)
    assert exact_error < 1e-12
    return {
        "nfe": nfe,
        "last_model_time": float(times[-2]),
        "exact_joint_tv_to_endpoint": exact_error,
        "factorized_tv_to_endpoint_exact_start": tv(factorized, endpoint),
        "factorized_tv_to_endpoint_iid_prior_start": tv(
            initialized_from_prior, endpoint
        ),
        "factorized_tv_to_clean_data_iid_prior_start": tv(initialized_from_prior, data),
        "endpoint_tv_to_clean_data": tv(endpoint, data),
        "iid_prior_tv_to_exact_start": tv(prior_joint, start),
    }


def production_audit():
    """Check both final grid steps and a large jump against production float64."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import torch

    from genmol.diffusion import ContinuousCategoricalDiffusion

    torch.set_num_threads(1)
    max_error = 0.0
    cases = 0
    for prior in PRIORS.values():
        model = ContinuousCategoricalDiffusion(3, prior, noise_eps=NOISE_EPS)
        for t, s in (
            (1.0, INFERENCE_EPS),
            (INFERENCE_EPS + (1 - INFERENCE_EPS) / 128, INFERENCE_EPS),
            (INFERENCE_EPS + (1 - INFERENCE_EPS) / 512, INFERENCE_EPS),
        ):
            alpha_t = 1 - (1 - NOISE_EPS) * t
            alpha_s = 1 - (1 - NOISE_EPS) * s
            _, _, expected = reverse_matrices(DATA, prior, alpha_s, alpha_t)
            local = alpha_t * np.eye(3) + (1 - alpha_t) * prior[None, :]
            raw_loo = np.empty((9, 2, 3), dtype=np.float64)
            for row, current in enumerate(STATES):
                for position in range(2):
                    other = 1 - position
                    weights = DATA * local[STATES[:, other], current[other]]
                    raw_loo[row, position] = [
                        weights[STATES[:, position] == j].sum() for j in range(3)
                    ]
                    raw_loo[row, position] /= raw_loo[row, position].sum()
            actual = model.posterior_probs(
                torch.from_numpy(raw_loo).log(),
                torch.from_numpy(STATES),
                torch.full((9,), t, dtype=torch.float64),
                torch.full((9,), s, dtype=torch.float64),
            ).numpy()
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            cases += 1
    assert max_error < 1e-11
    return {"nine_state_cases": cases, "max_abs_error": max_error, "device": "cpu"}


def main():
    rows = []
    independent = np.outer([0.2, 0.3, 0.5], [0.4, 0.35, 0.25]).ravel()
    max_independent_error = 0.0
    for name, prior in PRIORS.items():
        for nfe in NFES:
            row = chain(DATA, prior, nfe)
            rows.append({"prior": name, **row})
            independent_result = chain(independent, prior, nfe)
            max_independent_error = max(
                max_independent_error,
                independent_result["factorized_tv_to_endpoint_exact_start"],
            )
    assert max_independent_error < 1e-12
    print(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "CPU oracle finite-step factorization diagnostic, not molecular evidence",
                "data_probability_matrix": DATA.reshape(3, 3).tolist(),
                "priors": {name: value.tolist() for name, value in PRIORS.items()},
                "noise_eps": NOISE_EPS,
                "inference_eps": INFERENCE_EPS,
                "temperature": 1.0,
                "arithmetic": "numpy/torch float64 enumeration; no Monte Carlo",
                "production_coordinate_bridge": production_audit(),
                "independent_data_max_factorization_tv": max_independent_error,
                "rows": rows,
                "limitations": [
                    "Two positions and three categories with oracle conditionals are not learned molecular sequences.",
                    "Finer steps need not improve a learned or temperature-controlled model.",
                    "512 predictor evaluations cost four times the 128-evaluation budget.",
                    "An iid stationary start differs from the residual-clean terminal forward distribution.",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
