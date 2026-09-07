"""Installed TDC dispatch with injected synthetic scores; no real oracle factory."""

import math

import numpy as np
import pytest
from tdc import Oracle

from scripts.exps.pmo.run_ablation import CachedOracle
from scripts.exps.pmo.udlm_sampling import singleton_list_oracle


def synthetic_tdc(function, normalize=lambda value: value):
    # Deliberately bypass __init__: never load a property model, descriptor oracle,
    # downloaded artifact, or real fexofenadine evaluator.
    evaluator = Oracle.__new__(Oracle)
    evaluator.name = "synthetic_non_docking_fixture"
    evaluator.evaluator_func = function
    evaluator.normalize = normalize
    evaluator.default_property = 0.0
    evaluator.num_called = 0
    evaluator.num_max_call = None
    return evaluator


def test_singleton_preserves_real_dispatch_normalization_and_exact_argument():
    seen = []

    def score(smiles):
        seen.append(smiles)
        return 0.8

    scalar = synthetic_tdc(score, normalize=lambda value: value / 2)
    listed = synthetic_tdc(score, normalize=lambda value: value / 2)
    assert scalar("CCO") == singleton_list_oracle(listed)("CCO") == 0.4
    assert seen == ["CCO", "CCO"]
    assert scalar.num_called == listed.num_called == 1
    arguments = []
    wrapper = singleton_list_oracle(lambda values: arguments.append(values) or [0.2])
    assert wrapper("CCO") == 0.2
    assert arguments == [["CCO"]]


@pytest.mark.parametrize("value", [0, 0.0, np.float64(0.0), np.float32(0.25)])
def test_finite_real_zero_is_a_valid_charged_score_and_duplicates_are_cached(value):
    seen = []
    evaluator = synthetic_tdc(lambda smiles: seen.append(smiles) or value)
    cached = CachedOracle(singleton_list_oracle(evaluator), budget=2)
    first = cached.score("C(C)O")
    second = cached.score("CCO")
    assert first.valid and first.charged and first.score == float(value)
    assert second.valid and not second.charged and second.score == first.score
    assert cached.calls == 1 and seen == ["CCO"]
    assert first.call_index == second.call_index == 1


def test_scalar_tdc_hides_synthetic_failure_while_wrapper_propagates():
    failure = ValueError("synthetic scorer failure")

    def broken(smiles):
        raise failure

    assert synthetic_tdc(broken)("CCO") == 0.0
    cached = CachedOracle(singleton_list_oracle(synthetic_tdc(broken)), budget=2)
    with pytest.raises(ValueError) as caught:
        cached.score("CCO")
    assert caught.value is failure
    assert cached.calls == 0 and cached.buffer == {}


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_interruptions_propagate_without_any_cache_charge(error_type):
    failure = error_type("synthetic interruption")

    def interrupted(smiles):
        raise failure

    cached = CachedOracle(singleton_list_oracle(synthetic_tdc(interrupted)), budget=2)
    with pytest.raises(error_type) as caught:
        cached.score("CCO")
    assert caught.value is failure
    assert cached.calls == 0 and cached.buffer == {}


@pytest.mark.parametrize(
    "result",
    [
        0.2,
        (0.2,),
        np.array([0.2]),
        [],
        [0.2, 0.3],
        [True],
        [np.bool_(False)],
        ["0.2"],
        [None],
        [complex(0.2, 0)],
        [[0.2]],
        [math.nan],
        [math.inf],
        [-math.inf],
    ],
)
def test_malformed_or_nonfinite_results_never_charge(result):
    calls = []
    cached = CachedOracle(
        singleton_list_oracle(lambda arg: calls.append(arg) or result), budget=2
    )
    with pytest.raises(ValueError):
        cached.score("CCO")
    assert calls == [["CCO"]]
    assert cached.calls == 0 and cached.buffer == {}
