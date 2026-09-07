"""Exact rational tests: no production, torch, model, or oracle dependencies."""

from fractions import Fraction as F
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.udlm.audit_near_absorbing_limits import (
    ALPHA_T,
    BASE,
    MASK,
    WEIGHTS,
    bridge,
    denoiser_to_loo,
    laws,
    limiting_clean_observation,
    local_likelihood,
    normalize,
    posterior_mixture,
    prior,
    run_audit,
    temper,
    tv,
)


def test_exact_bayes_mixture_and_converted_bridge_panel():
    report = run_audit()
    assert report["exact_bridge_mixture_equalities"] == 648
    assert len(report["finite_prior_rows"]) == 60
    assert report["production_imports_or_checkpoint_reads"] is False


@pytest.mark.parametrize(
    "base", [BASE, (F(1, 3),) * 3, (F(1, 1000), F(499, 1000), F(1, 2))]
)
@pytest.mark.parametrize("observed", [0, 1])
def test_fixed_positive_raw_observed_weight_copies_in_limit(base, observed):
    expected = tuple(F(i == observed) for i in range(3))
    errors = [
        tv(bridge(WEIGHTS, prior(1 - F(1, 10**n), base), observed), expected)
        for n in (2, 4, 8, 12)
    ]
    assert all(first > second > 0 for first, second in zip(errors, errors[1:]))
    assert errors[-1] < F(1, 10**10)


def test_explicit_counterexample_limits_and_finite_full_support():
    pi = prior(1 - F(1, 10**12))
    assert min(pi) > 0 and sum(pi) == 1
    expected = {
        "raw_loo_T1": (F(1), F(0), F(0)),
        "raw_loo_T_half": (F(1), F(0), F(0)),
        "ce_T1": (F(5, 7), F(1, 5), F(3, 35)),
        "ce_then_raw_T_half": (F(2, 7), F(1, 2), F(3, 14)),
        "clean_T_half_then_ce": (F(71, 91), F(2, 13), F(6, 91)),
    }
    for key, law in laws(pi, 0).items():
        assert tv(law, expected[key]) < F(1, 10**10)
    assert limiting_clean_observation(WEIGHTS, 0) == expected["ce_T1"]


def test_mask_observation_equalities_hold_at_every_finite_prior():
    for n in (1, 2, 4, 12):
        pi = prior(1 - F(1, 10**n))
        values = laws(pi, MASK)
        assert values["raw_loo_T1"] == values["ce_T1"]
        assert (
            values["raw_loo_T_half"]
            == values["ce_then_raw_T_half"]
            == values["clean_T_half_then_ce"]
        )
    assert tv(values["ce_T1"], (F(3, 10), F(1, 5), F(1, 2))) < F(1, 10**10)


def test_zero_raw_current_mass_breaks_visible_copy_assumption():
    pi = prior(1 - F(1, 10**12))
    value = bridge((F(0), F(1), F(0)), pi, 0)
    assert tv(value, (F(2, 7), F(1, 2), F(3, 14))) < F(1, 10**10)
    assert value[0] < F(1, 3)


def test_temperature_above_one_limit_with_exact_square_root_construction():
    # T=2, h=1/2: choose L_A/L_B=n^2 so the square root stays rational.
    # This checks the memo's T>1 case without approximating irrational powers.
    denoiser = (F(9, 25), F(16, 25), F(0))
    for n in (100, 10_000, 1_000_000):
        pi_a = F(2, 3 * (n * n - 1))
        pi = prior(1 - pi_a / BASE[0])
        likelihood = local_likelihood(pi, 0, ALPHA_T)
        assert likelihood[0] / likelihood[1] == n * n
        half_power = normalize((F(3), F(4 * n), F(0)))
        assert temper(half_power, 2) == denoiser_to_loo(denoiser, pi, 0)
        value = bridge(half_power, pi, 0)
    assert tv(value, (F(1), F(0), F(0))) < F(1, 100_000)


def test_bayes_consistent_denoiser_recovers_raw_law_exactly():
    for n in (1, 4, 12):
        pi = prior(1 - F(1, 10**n))
        likelihood = local_likelihood(pi, 0, ALPHA_T)
        bayes_d = normalize(r * likelihood[j] for j, r in enumerate(WEIGHTS))
        assert denoiser_to_loo(bayes_d, pi, 0) == WEIGHTS
        assert posterior_mixture(bayes_d, pi, 0) == bridge(WEIGHTS, pi, 0)
    assert bayes_d[0] > 1 - F(1, 10**10)


def test_noncommuting_model_error_and_noise_limits_for_raw_temperature_half():
    epsilon = F(1, 10**30)
    pi = prior(1 - epsilon)
    # Error small relative to one but much larger than sqrt(epsilon).
    eta = F(1, 10**6)
    d = (1 - eta, eta, F(0))
    value = bridge(temper(denoiser_to_loo(d, pi, 0), 2), pi, 0)
    assert tv(value, (F(2, 7), F(1, 2), F(3, 14))) < F(1, 10**10)
    # Making model error negligible before the prior vanishes restores copying.
    eta = F(1, 10**30)
    d = (1 - eta, eta, F(0))
    value = bridge(temper(denoiser_to_loo(d, pi, 0), 2), pi, 0)
    assert tv(value, (F(1), F(0), F(0))) < F(1, 10**10)


def test_perfect_clean_denoiser_is_not_exact_copy_at_finite_positive_prior():
    delta = (F(1), F(0), F(0))
    pi = prior(F(9, 10))
    converted = denoiser_to_loo(delta, pi, 0)
    assert converted == delta and temper(converted, 2) == delta
    law = bridge(delta, pi, 0)
    assert law == posterior_mixture(delta, pi, 0)
    assert 0 < law[0] < 1 and law[1] > 0 and law[MASK] > 0


@pytest.mark.parametrize(
    "alpha_t,alpha_s", [(F(1, 5), F(3, 5)), (F(2, 5), F(7, 10)), (F(9, 10), F(19, 20))]
)
def test_visible_ce_limit_is_not_specific_to_selected_alphas(alpha_t, alpha_s):
    pi = prior(1 - F(1, 10**12))
    expected = limiting_clean_observation(WEIGHTS, 1, alpha_t, alpha_s)
    actual = posterior_mixture(WEIGHTS, pi, 1, alpha_t, alpha_s)
    assert tv(actual, expected) < F(1, 10**10)
    assert expected[1] < 1 and expected[MASK] > 0


@pytest.mark.parametrize("value", [F(1), F(-1, 10), F(11, 10)])
def test_no_absorbing_or_invalid_prior_is_silently_accepted(value):
    with pytest.raises(ValueError, match="lambda"):
        prior(value)


def test_rejects_singular_bridge_and_noninteger_temperature_audit():
    with pytest.raises(ValueError, match="positive normalized"):
        bridge(WEIGHTS, (F(0), F(0), F(1)), 0)
    with pytest.raises(ValueError, match="interior"):
        bridge(WEIGHTS, prior(F(9, 10)), 0, ALPHA_T, ALPHA_T)
    with pytest.raises(ValueError, match="integer"):
        temper(WEIGHTS, F(1, 2))


def test_cli_is_standalone_stdlib_and_byte_reproducible():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/udlm/audit_near_absorbing_limits.py"
    )
    command = [sys.executable, "-I", str(script)]
    first = subprocess.check_output(command)
    assert first == subprocess.check_output(command)
    assert json.loads(first) == run_audit()
