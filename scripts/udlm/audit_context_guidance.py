"""Exact three-category oracle for a proposed posterior-space context heuristic.

Standard library only: no models, molecular data, scoring or production imports.
"""

from fractions import Fraction as F
from itertools import product
import json
import math


def normalize(values):
    values = tuple(F(value) for value in values)
    if not values or any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("Need nonnegative weights with positive total")
    total = sum(values)
    return tuple(value / total for value in values)


def reverse_mixture(denoiser, prior, observed, alpha_s, alpha_t):
    """Sum exact q(i at s | k at t, j at 0) against clean weights D_j."""
    d, pi = normalize(denoiser), normalize(prior)
    if len(d) != len(pi) or any(p <= 0 for p in pi):
        raise ValueError("Need a shared full-support alphabet")
    if not 0 < alpha_t < alpha_s < 1 or not 0 <= observed < len(pi):
        raise ValueError("Toy uses distinct interior times and a valid observed ID")
    ratio = alpha_t / alpha_s
    likelihood = [
        alpha_t * int(j == observed) + (1 - alpha_t) * pi[observed]
        for j in range(len(pi))
    ]
    return tuple(
        sum(
            d[j]
            * (ratio * int(i == observed) + (1 - ratio) * pi[observed])
            * (alpha_s * int(i == j) + (1 - alpha_s) * pi[i])
            / likelihood[j]
            for j in range(len(pi))
        )
        for i in range(len(pi))
    )


def reverse_from_loo(denoiser, prior, observed, alpha_s, alpha_t):
    """Independent rank-one bridge expression after R = normalize(D / L)."""
    d, pi = normalize(denoiser), normalize(prior)
    r = normalize(
        d[j] / (alpha_t * int(j == observed) + (1 - alpha_t) * pi[observed])
        for j in range(len(pi))
    )
    ratio = alpha_t / alpha_s
    return normalize(
        (ratio * int(i == observed) + (1 - ratio) * pi[observed])
        * (alpha_s * r[i] + (1 - alpha_s) * pi[i])
        for i in range(len(pi))
    )


def guide_exact(conditional, poor, weight):
    """Integer guidance weights make the entire power product rational."""
    if type(weight) is not int or weight < 0:
        raise ValueError("Exact toy accepts nonnegative integer guidance weights")
    conditional, poor = normalize(conditional), normalize(poor)
    if len(conditional) != len(poor) or any(p <= 0 for p in conditional + poor):
        raise ValueError("Guidance requires identical strictly positive support")
    return normalize(c**weight * u ** (1 - weight) for c, u in zip(conditional, poor))


def stable_log_guidance(log_conditional, log_poor, weight):
    """Return normalized log probabilities; never floor or clip tiny categories.

    Float exp may underflow even with finite output logs. A future categorical
    implementation must address that explicitly rather than claim exact positivity.
    """
    if isinstance(weight, bool) or not math.isfinite(weight) or weight < 0:
        raise ValueError("Need a finite nonnegative guidance weight")
    if not log_conditional or len(log_conditional) != len(log_poor):
        raise ValueError("Need matching nonempty supports")
    if not all(math.isfinite(value) for value in (*log_conditional, *log_poor)):
        raise ValueError("Need finite log probabilities on the common support")
    # Separate additive constants are irrelevant and can safely be removed.
    cm, um = max(log_conditional), max(log_poor)
    scores = [
        weight * (c - cm) + (1 - weight) * (u - um)
        for c, u in zip(log_conditional, log_poor)
    ]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Guidance arithmetic overflowed; no implicit clipping")
    maximum = max(scores)
    centered = [value - maximum for value in scores]
    if not all(math.isfinite(value) for value in centered):
        raise ValueError("Guidance normalization overflowed")
    log_total = math.log(math.fsum(math.exp(value) for value in centered))
    return tuple(value - log_total for value in centered)


def degraded_view(
    current, initial, editable, selected_context, *, mask_id, control_ids
):
    """Mask only selected ORIGINAL immutable non-control coordinates, no RNG."""
    if len(current) != len(initial) or len(current) != len(editable):
        raise ValueError("Coordinate shapes differ")
    if any(type(value) is not bool for value in editable):
        raise ValueError("Editable flags must be bool")
    if any(current[i] != initial[i] for i in range(len(current)) if not editable[i]):
        raise ValueError("Immutable context was already modified")
    eligible = {
        i
        for i in range(len(current))
        if not editable[i] and initial[i] not in control_ids
    }
    selected = tuple(selected_context)
    if len(set(selected)) != len(selected) or not set(selected) <= eligible:
        raise ValueError("Only original immutable non-control context is eligible")
    return tuple(mask_id if i in selected else value for i, value in enumerate(current))


def clamped_product(laws, original, editable):
    """Enumerate a small independent-coordinate proposal and clamp fixed context."""
    if not len(laws) == len(original) == len(editable) or any(
        type(e) is not bool for e in editable
    ):
        raise ValueError("Coordinate shapes/flags differ")
    laws = tuple(normalize(law) for law in laws)
    result = {}
    for sequence in product(*(range(len(law)) for law in laws)):
        probability = F(1)
        for i, token in enumerate(sequence):
            probability *= laws[i][token] if editable[i] else int(token == original[i])
        result[sequence] = probability
    return result


def audit():
    priors = [
        (F(1, 3),) * 3,
        (F(1, 5), F(1, 2), F(3, 10)),
        (F(9, 10), F(1, 20), F(1, 20)),
    ]
    denoisers = [
        (F(3, 5), F(3, 10), F(1, 10)),
        (F(1, 5), F(1, 5), F(3, 5)),
        (F(1, 3),) * 3,
    ]
    times = [(F(4, 5), F(1, 5)), (F(2, 3), F(1, 3)), (F(9, 10), F(4, 5))]
    cases, bridge_checks, guidance_checks, joint_checks = 0, 0, 0, 0
    maximum_error = 0.0
    for pi, (a_s, a_t), k, d_c, d_u in product(
        priors, times, range(3), denoisers, denoisers
    ):
        p_c = reverse_mixture(d_c, pi, k, a_s, a_t)
        p_u = reverse_mixture(d_u, pi, k, a_s, a_t)
        for d, p in ((d_c, p_c), (d_u, p_u)):
            assert p == reverse_from_loo(d, pi, k, a_s, a_t)
            assert sum(p) == 1 and min(p) > 0
            bridge_checks += 1
        assert guide_exact(p_c, p_u, 1) == p_c
        assert guide_exact(p_c, p_u, 0) == p_u
        for w in (0, 1, 2, 3):
            guided = guide_exact(p_c, p_u, w)
            assert guide_exact(p_c, p_c, w) == p_c
            assert sum(guided) == 1 and min(guided) > 0
            logs = stable_log_guidance(
                tuple(math.log(float(p)) for p in p_c),
                tuple(math.log(float(p)) for p in p_u),
                w,
            )
            maximum_error = max(
                maximum_error,
                max(abs(math.exp(log_p) - float(p)) for log_p, p in zip(logs, guided)),
            )
            joint = clamped_product((p_u, guided), (1, k), (False, True))
            assert sum(joint.values()) == 1
            assert all(
                probability == 0
                for sequence, probability in joint.items()
                if sequence[0] != 1
            )
            assert (
                tuple(
                    sum(value for sequence, value in joint.items() if sequence[1] == i)
                    for i in range(3)
                )
                == guided
            )
            guidance_checks += 1
            joint_checks += len(joint)
        cases += 1
    pi, d_c, d_u = priors[1], denoisers[0], denoisers[1]
    a_s, a_t, k, w = F(2, 3), F(1, 3), 0, 2
    p_c, p_u = (reverse_mixture(d, pi, k, a_s, a_t) for d in (d_c, d_u))
    posterior_guided = guide_exact(p_c, p_u, w)
    clean_mix_then_bridge = reverse_mixture(guide_exact(d_c, d_u, w), pi, k, a_s, a_t)
    posterior_temperature_only = normalize(p**w for p in p_c)
    assert posterior_guided != clean_mix_then_bridge
    assert posterior_guided != posterior_temperature_only
    original, current, editable = (1, 0), (1, 2), (False, True)
    assert degraded_view(
        current, original, editable, (0,), mask_id=0, control_ids={0}
    ) == (0, 2)
    assert (
        degraded_view(current, original, editable, (), mask_id=0, control_ids={0})
        == current
    )
    return dict(
        schema_version=1,
        claim="three-category algebra only; no molecular evidence",
        oracle_cases=cases,
        exact_bridge_equalities=bridge_checks,
        exact_guided_laws=guidance_checks,
        enumerated_joint_states=joint_checks,
        max_float_probability_error=maximum_error,
        example=dict(
            prior=[str(v) for v in pi],
            denoiser_conditional=[str(v) for v in d_c],
            denoiser_poor=[str(v) for v in d_u],
            alpha_s=str(a_s),
            alpha_t=str(a_t),
            observed=k,
            weight=w,
            conditional_reverse=[str(v) for v in p_c],
            poor_reverse=[str(v) for v in p_u],
            posterior_guided=[str(v) for v in posterior_guided],
            clean_mix_then_bridge=[str(v) for v in clean_mix_then_bridge],
            posterior_temperature_only=[str(v) for v in posterior_temperature_only],
        ),
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
