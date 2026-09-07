"""CPU categorical mathematics; no model, chemistry or production imports."""

import ast
from fractions import Fraction as F
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.udlm import audit_context_guidance as toy


def test_exhaustive_rational_bridge_guidance_and_clamping():
    result = toy.audit()
    assert result["oracle_cases"] == 243
    assert result["exact_bridge_equalities"] == 486
    assert result["exact_guided_laws"] == 972
    assert result["enumerated_joint_states"] == 8748
    assert result["max_float_probability_error"] < 1e-14
    assert result["claim"] == "three-category algebra only; no molecular evidence"


def test_explicit_counterexample_mixing_before_bridge_is_different():
    pi, d_c, d_u = map(toy.normalize, [(2, 5, 3), (6, 3, 1), (2, 2, 6)])
    args = pi, 0, F(2, 3), F(1, 3)
    p_c, p_u = toy.reverse_mixture(d_c, *args), toy.reverse_mixture(d_u, *args)
    assert p_c == (F(24, 35), F(31, 140), F(13, 140))
    assert p_u == (F(3, 7), F(29, 140), F(51, 140))
    guided = toy.guide_exact(p_c, p_u, 2)
    assert guided == (F(141984, 175679), F(245055, 1405432), F(24505, 1405432))
    clean_first = toy.reverse_mixture(toy.guide_exact(d_c, d_u, 2), *args)
    assert clean_first == (F(1929, 2380), F(73, 476), F(43, 1190))
    assert guided != clean_first
    assert guided != toy.normalize(p**2 for p in p_c)


@pytest.mark.parametrize("weight", [0, 1, 2, 3, 12])
def test_gamma_zero_identical_laws_and_weight_one_identity(weight):
    conditional, poor = toy.normalize((7, 2, 1)), toy.normalize((2, 3, 5))
    assert toy.guide_exact(conditional, conditional, weight) == conditional
    assert toy.guide_exact(conditional, poor, 1) == conditional
    guided = toy.guide_exact(conditional, poor, weight)
    assert min(guided) > 0 and sum(guided) == 1
    assert guided[0] / guided[1] == (conditional[0] / conditional[1]) ** weight * (
        poor[0] / poor[1]
    ) ** (1 - weight)


def test_context_degradation_keeps_original_editable_state_and_excludes_controls():
    initial, current, editable = (0, 1, 0), (0, 1, 2), (False, False, True)
    poor = toy.degraded_view(
        current, initial, editable, (1,), mask_id=0, control_ids={0}
    )
    assert poor == (0, 0, 2)
    assert poor[2] == current[2]  # both analytic bridges must use the same k
    assert (
        toy.degraded_view(current, initial, editable, (), mask_id=0, control_ids={0})
        == current
    )
    for forbidden in ((0,), (2,), (1, 1)):
        with pytest.raises(ValueError, match="original immutable non-control"):
            toy.degraded_view(
                current, initial, editable, forbidden, mask_id=0, control_ids={0}
            )


def test_context_eligibility_is_per_row_and_not_borrowed_from_row_zero():
    first = toy.degraded_view(
        (1, 2), (1, 0), (False, True), (0,), mask_id=0, control_ids={0}
    )
    second = toy.degraded_view(
        (2, 1), (0, 1), (True, False), (1,), mask_id=0, control_ids={0}
    )
    assert first == (0, 2) and second == (2, 0)
    with pytest.raises(ValueError, match="original immutable"):
        toy.degraded_view(
            (2, 1), (0, 1), (True, False), (0,), mask_id=0, control_ids={0}
        )


def test_original_immutable_context_is_clamped_even_if_both_predictors_disagree():
    proposed = toy.clamped_product(
        ((F(1, 100), F(99, 100)), (F(2, 3), F(1, 3))), (0, 1), (False, True)
    )
    assert proposed == {(0, 0): F(2, 3), (0, 1): F(1, 3), (1, 0): 0, (1, 1): 0}
    assert sum(proposed.values()) == 1
    with pytest.raises(ValueError, match="already modified"):
        toy.degraded_view((1, 0), (0, 0), (False, True), (), mask_id=0, control_ids={0})


@pytest.mark.parametrize(
    "alpha_s,alpha_t",
    [(F(1), F(1, 2)), (F(1, 2), F(1, 2)), (F(1, 3), F(2, 3)), (F(1, 2), F(0))],
)
def test_strict_support_proof_does_not_claim_boundary_time_support(alpha_s, alpha_t):
    with pytest.raises(ValueError, match="interior times"):
        toy.reverse_mixture((1, 2, 3), (1, 1, 1), 0, alpha_s, alpha_t)


def test_zero_clean_weight_still_gives_positive_interior_bridge():
    p = toy.reverse_mixture((0, 1, 0), (1, 2, 3), 0, F(2, 3), F(1, 3))
    assert sum(p) == 1 and min(p) > 0
    assert p == toy.reverse_from_loo((0, 1, 0), (1, 2, 3), 0, F(2, 3), F(1, 3))


def test_zero_reverse_support_is_rejected_before_negative_power():
    with pytest.raises(ValueError, match="strictly positive support"):
        toy.guide_exact((1, 2, 3), (0, 1, 2), 2)
    with pytest.raises(ValueError, match="full-support alphabet"):
        toy.reverse_mixture((1, 2, 3), (0, 1, 2), 0, F(2, 3), F(1, 3))


def test_stable_log_guidance_handles_large_offsets_and_real_weight():
    lc, lu = (-10000.0, -10001.0, -10010.0), (-20003.0, -20004.0, -20001.0)
    result = toy.stable_log_guidance(lc, lu, 1.75)
    shifted = toy.stable_log_guidance(
        tuple(v + 20000 for v in lc), tuple(v + 30000 for v in lu), 1.75
    )
    assert result == shifted and all(math.isfinite(p) for p in result)
    assert math.fsum(math.exp(p) for p in result) == pytest.approx(1.0)


def test_stable_logs_do_not_promise_float_exponential_full_support():
    result = toy.stable_log_guidance(
        (0.0, -2000.0, -3000.0), (0.0, -1000.0, -1500.0), 2
    )
    assert all(math.isfinite(p) for p in result)
    assert result == (0.0, -3000.0, -4500.0)
    assert math.exp(result[1]) == 0.0  # disclosed finite-precision limitation


@pytest.mark.parametrize(
    "weight,lc,lu",
    [
        (float("inf"), (0.0, -1.0), (0.0, -2.0)),
        (2, (0.0, float("-inf")), (0.0, -1.0)),
        (2, (0.0, float("nan")), (0.0, -1.0)),
        (1e308, (0.0, -10.0), (0.0, -1.0)),
        (-1, (0.0, -1.0), (0.0, -2.0)),
    ],
)
def test_invalid_or_overflowed_log_guidance_is_not_silently_clipped(weight, lc, lu):
    with pytest.raises(ValueError):
        toy.stable_log_guidance(lc, lu, weight)


def test_standalone_oracle_is_reproducible_and_has_no_external_imports():
    path = Path(toy.__file__)
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom)
    }
    imports |= {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports <= {"fractions", "itertools", "json", "math"}
    command = [sys.executable, str(path)]
    first = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    assert first == second
    assert json.loads(first) == toy.audit()
