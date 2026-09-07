"""Exact, model-free reverse-limit audit; stdout only, standard library only.

This is a mathematical hypothesis diagnostic, not molecular evidence. Fractions
allow finite-prior Bayes mixtures and raw-LOO bridges to be compared exactly.
The three categories are clean A, clean B, and MASK. No production imports,
checkpoint reads, sampling, network access, or experiment selection occur here.
"""

from fractions import Fraction as F
import json


MASK = 2
ALPHA_T = F(2, 5)
ALPHA_S = F(7, 10)
BASE = (F(1, 5), F(1, 2), F(3, 10))
WEIGHTS = (F(3, 5), F(2, 5), F(0))


def normalize(values):
    values = tuple(F(value) for value in values)
    if not values or min(values) < 0 or sum(values) <= 0:
        raise ValueError("nonnegative weights with positive total are required")
    return tuple(value / sum(values) for value in values)


def prior(mask_weight, base=BASE):
    """pi_lambda = lambda delta_MASK + (1-lambda) base, with lambda < 1."""
    mask_weight = F(mask_weight)
    base = tuple(F(value) for value in base)
    if len(base) != 3 or min(base) <= 0 or sum(base) != 1:
        raise ValueError("base must be a positive three-category distribution")
    if not 0 <= mask_weight < 1:
        raise ValueError("finite prior requires 0 <= lambda < 1")
    return tuple(
        (1 - mask_weight) * value + mask_weight * (index == MASK)
        for index, value in enumerate(base)
    )


def _validate(pi, observed, alpha_t, alpha_s):
    if len(pi) != 3 or min(pi) <= 0 or sum(pi) != 1:
        raise ValueError("finite bridge requires positive normalized pi")
    if type(observed) is not int or observed not in range(3):
        raise ValueError("observed category must be 0, 1, or 2")
    if not 0 < alpha_t < alpha_s < 1:
        raise ValueError("require fixed interior 0 < alpha_t < alpha_s < 1")


def local_likelihood(pi, observed, alpha_t):
    return tuple(
        alpha_t * (clean == observed) + (1 - alpha_t) * pi[observed]
        for clean in range(3)
    )


def bridge(raw_loo, pi, observed, alpha_t=ALPHA_T, alpha_s=ALPHA_S):
    """Normalize q(zt=k|zs=i) [alpha_s R_i + (1-alpha_s) pi_i]."""
    _validate(pi, observed, alpha_t, alpha_s)
    raw_loo = normalize(raw_loo)
    if len(raw_loo) != 3:
        raise ValueError("three model weights are required")
    ratio = alpha_t / alpha_s
    weights = tuple(
        (ratio * (earlier == observed) + (1 - ratio) * pi[observed])
        * (alpha_s * raw_loo[earlier] + (1 - alpha_s) * pi[earlier])
        for earlier in range(3)
    )
    assert sum(weights) == alpha_t * raw_loo[observed] + (1 - alpha_t) * pi[observed]
    return normalize(weights)


def posterior_mixture(denoiser, pi, observed, alpha_t=ALPHA_T, alpha_s=ALPHA_S):
    """Independently sum D_j q(zs=i|zt=k,x0=j), including rare j != k."""
    _validate(pi, observed, alpha_t, alpha_s)
    denoiser = normalize(denoiser)
    if len(denoiser) != 3:
        raise ValueError("three model weights are required")
    ratio = alpha_t / alpha_s
    result = [F(0)] * 3
    for clean in range(3):
        evidence = alpha_t * (clean == observed) + (1 - alpha_t) * pi[observed]
        conditional = []
        for earlier in range(3):
            before = alpha_s * (earlier == clean) + (1 - alpha_s) * pi[earlier]
            transition = ratio * (observed == earlier) + (1 - ratio) * pi[observed]
            conditional.append(before * transition / evidence)
        assert sum(conditional) == 1
        result = [
            value + denoiser[clean] * conditional[i] for i, value in enumerate(result)
        ]
    assert sum(result) == 1
    return tuple(result)


def denoiser_to_loo(denoiser, pi, observed, alpha_t=ALPHA_T):
    denoiser = normalize(denoiser)
    return normalize(
        value / likelihood
        for value, likelihood in zip(denoiser, local_likelihood(pi, observed, alpha_t))
    )


def temper(weights, inverse_temperature):
    """Integer inverse temperatures keep this audit entirely rational."""
    if type(inverse_temperature) is not int or inverse_temperature < 1:
        raise ValueError("exact audit requires positive integer inverse temperature")
    return normalize(value**inverse_temperature for value in normalize(weights))


def effective_denoiser(denoiser, pi, observed, inverse_temperature):
    """D_eff proportional to D**h L**(1-h), h = 1/T after D/L conversion."""
    if type(inverse_temperature) is not int or inverse_temperature < 1:
        raise ValueError("exact audit requires positive integer inverse temperature")
    return normalize(
        value**inverse_temperature * likelihood ** (1 - inverse_temperature)
        for value, likelihood in zip(
            normalize(denoiser), local_likelihood(pi, observed, ALPHA_T)
        )
    )


def limiting_clean_observation(denoiser, observed, alpha_t=ALPHA_T, alpha_s=ALPHA_S):
    """Fixed D, D_MASK=0; the lambda->1 limit of the CE posterior mixture."""
    denoiser = normalize(denoiser)
    if observed not in (0, 1) or len(denoiser) != 3 or denoiser[MASK] != 0:
        raise ValueError("requires a clean observation and zero clean MASK mass")
    if not 0 < alpha_t < alpha_s < 1:
        raise ValueError("require interior alphas")
    carry = alpha_t * (1 - alpha_s) / (alpha_s * (1 - alpha_t))
    reveal = (alpha_s - alpha_t) / (1 - alpha_t)
    remask = (alpha_s - alpha_t) * (1 - alpha_s) / (alpha_s * (1 - alpha_t))
    result = [reveal * value for value in denoiser]
    result[observed] = denoiser[observed] + (1 - denoiser[observed]) * carry
    result[MASK] = (1 - denoiser[observed]) * remask
    assert sum(result) == 1
    return tuple(result)


def limiting_mask_observation(weights, alpha_t=ALPHA_T, alpha_s=ALPHA_S):
    weights = normalize(weights)
    if len(weights) != 3 or weights[MASK] != 0:
        raise ValueError("clean MASK weight must be zero")
    if not 0 < alpha_t < alpha_s < 1:
        raise ValueError("require interior alphas")
    return (
        (alpha_s - alpha_t) * weights[0] / (1 - alpha_t),
        (alpha_s - alpha_t) * weights[1] / (1 - alpha_t),
        (1 - alpha_s) / (1 - alpha_t),
    )


def tv(first, second):
    return sum(abs(a - b) for a, b in zip(first, second)) / 2


def laws(pi, observed):
    converted = denoiser_to_loo(WEIGHTS, pi, observed)
    return {
        "raw_loo_T1": bridge(WEIGHTS, pi, observed),
        "ce_T1": posterior_mixture(WEIGHTS, pi, observed),
        "ce_then_raw_T_half": bridge(temper(converted, 2), pi, observed),
        "clean_T_half_then_ce": posterior_mixture(temper(WEIGHTS, 2), pi, observed),
        "raw_loo_T_half": bridge(temper(WEIGHTS, 2), pi, observed),
    }


def run_audit():
    equalities = 0
    for base in (BASE, (F(1, 3),) * 3, (F(1, 1000), F(499, 1000), F(1, 2))):
        for power in (1, 2, 3, 4, 6, 12):
            pi = prior(1 - F(1, 10**power), base)
            for denoiser in (WEIGHTS, (F(1, 4), F(3, 4), F(0)), (F(1), F(0), F(0))):
                for observed in range(3):
                    converted = denoiser_to_loo(denoiser, pi, observed)
                    assert bridge(converted, pi, observed) == posterior_mixture(
                        denoiser, pi, observed
                    )
                    equalities += 1
                    for h in (1, 2, 3):
                        assert bridge(
                            temper(converted, h), pi, observed
                        ) == posterior_mixture(
                            effective_denoiser(denoiser, pi, observed, h), pi, observed
                        )
                        equalities += 1
    limits = {
        "observed_A": {
            "raw_loo_T1": (F(1), F(0), F(0)),
            "ce_T1": limiting_clean_observation(WEIGHTS, 0),
            "ce_then_raw_T_half": limiting_clean_observation((F(0), F(1), F(0)), 0),
            "clean_T_half_then_ce": limiting_clean_observation(temper(WEIGHTS, 2), 0),
            "raw_loo_T_half": (F(1), F(0), F(0)),
        },
        "observed_MASK": {
            "raw_loo_T1": limiting_mask_observation(WEIGHTS),
            "ce_T1": limiting_mask_observation(WEIGHTS),
            "ce_then_raw_T_half": limiting_mask_observation(temper(WEIGHTS, 2)),
            "clean_T_half_then_ce": limiting_mask_observation(temper(WEIGHTS, 2)),
            "raw_loo_T_half": limiting_mask_observation(temper(WEIGHTS, 2)),
        },
    }
    rows = []
    for power in (1, 2, 3, 4, 6, 12):
        pi = prior(1 - F(1, 10**power))
        for label, observed in (("observed_A", 0), ("observed_MASK", MASK)):
            observed_laws = laws(pi, observed)
            for method, law in observed_laws.items():
                error = tv(law, limits[label][method])
                if power == 12:
                    assert error < F(1, 10**10)
                rows.append(
                    {
                        "lambda": str(1 - F(1, 10**power)),
                        "observation": label,
                        "method": method,
                        "probabilities_exact": [str(value) for value in law],
                        "probabilities_decimal": [float(value) for value in law],
                        "tv_to_limit_exact": str(error),
                    }
                )
    return {
        "schema_version": 1,
        "status": "exact_arithmetic_checks_passed",
        "scientific_status": "one-coordinate hypothetical weights; no molecular efficacy or model calibration claim",
        "production_imports_or_checkpoint_reads": False,
        "category_order": ["A", "B", "MASK"],
        "alpha_t": str(ALPHA_T),
        "alpha_s": str(ALPHA_S),
        "base_prior": [str(value) for value in BASE],
        "fixed_R_and_D": [str(value) for value in WEIGHTS],
        "exact_bridge_mixture_equalities": equalities,
        "limits": {
            label: {key: [str(p) for p in law] for key, law in methods.items()}
            for label, methods in limits.items()
        },
        "finite_prior_rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, sort_keys=True, allow_nan=False))
