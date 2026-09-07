from __future__ import annotations

from fractions import Fraction
from itertools import product
from unittest.mock import patch

import pytest
import torch

from genmol.corrector import gibbs_conditional_probs, random_scan_gibbs_step
from genmol.diffusion import (
    ContinuousCategoricalDiffusion,
    ContinuousUniformDiffusion,
)


def _exact_binary_fixture(prior):
    """Enumerate the forward joint and LOO target without corrector formulas."""
    states = list(product(range(2), repeat=2))
    clean = [Fraction(4, 10), Fraction(1, 10), Fraction(2, 10), Fraction(3, 10)]
    alpha = Fraction(3, 5)

    def forward(clean_token, noisy_token):
        return alpha * (clean_token == noisy_token) + (1 - alpha) * prior[noisy_token]

    noisy_joint = []
    loo = []
    for noisy in states:
        noisy_joint.append(
            sum(
                probability * forward(x[0], noisy[0]) * forward(x[1], noisy[1])
                for x, probability in zip(states, clean)
            )
        )
        row = []
        for coordinate in range(2):
            other = 1 - coordinate
            denominator = sum(
                probability * forward(x[other], noisy[other])
                for x, probability in zip(states, clean)
            )
            row.append(
                [
                    sum(
                        probability
                        * (x[coordinate] == token)
                        * forward(x[other], noisy[other])
                        for x, probability in zip(states, clean)
                    )
                    / denominator
                    for token in range(2)
                ]
            )
        loo.append(row)
    return (
        states,
        torch.tensor([float(value) for value in noisy_joint], dtype=torch.float64),
        torch.tensor(
            [[[float(value) for value in pair] for pair in row] for row in loo],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("nonuniform", [False, True])
def test_actual_random_scan_kernel_preserves_explicit_correlated_joint(nonuniform):
    prior = (
        [Fraction(3, 4), Fraction(1, 4)]
        if nonuniform
        else [Fraction(1, 2), Fraction(1, 2)]
    )
    process = (
        ContinuousCategoricalDiffusion(2, [float(value) for value in prior])
        if nonuniform
        else ContinuousUniformDiffusion(2)
    )
    states, target, loo = _exact_binary_fixture(prior)
    expected_target = (
        [54 / 125, 21 / 125, 57 / 250, 43 / 250]
        if nonuniform
        else [79 / 250, 23 / 125, 61 / 250, 32 / 125]
    )
    torch.testing.assert_close(
        target, torch.tensor(expected_target, dtype=torch.float64), atol=1e-15, rtol=0
    )
    time = torch.tensor([0.4 / (1 - process.noise_eps)], dtype=torch.float64)
    transition = torch.zeros((4, 4), dtype=torch.float64)

    # Enumerate the two actual RNG calls to reconstruct the implementation's
    # exact finite-state transition, including coordinate choice and output
    # mutation. This checks the update itself, not only its probability helper.
    for source_index, source_state in enumerate(states):
        for coordinate, token in product(range(2), repeat=2):
            captured_weights = []
            forced_choices = iter((coordinate, token))

            def forced_multinomial(weights, num_samples, *, generator=None):
                assert weights.shape == (1, 2)
                assert num_samples == 1
                captured_weights.append(weights[0] / weights[0].sum())
                return torch.tensor([[next(forced_choices)]], dtype=torch.long)

            with patch("genmol.corrector.torch.multinomial", forced_multinomial):
                result = random_scan_gibbs_step(
                    process,
                    loo[source_index : source_index + 1].log(),
                    torch.tensor([source_state]),
                    time,
                )
            assert len(captured_weights) == 2
            destination = states.index(tuple(result[0].tolist()))
            transition[source_index, destination] += (
                captured_weights[0][coordinate] * captured_weights[1][token]
            )

    torch.testing.assert_close(transition.sum(-1), torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(target @ transition, target, atol=1e-14, rtol=0)
    flux = target[:, None] * transition
    torch.testing.assert_close(flux, flux.T, atol=1e-14, rtol=0)

    # This correlated fixture distinguishes Gibbs from updating both tokens
    # independently using stale conditionals: that tempting alternative fails.
    simultaneous = torch.zeros_like(transition)
    for source_index in range(4):
        conditional = gibbs_conditional_probs(
            process, loo[source_index : source_index + 1].log(), time
        )[0]
        for destination, state in enumerate(states):
            simultaneous[source_index, destination] = (
                conditional[0, state[0]] * conditional[1, state[1]]
            )
    assert (target @ simultaneous - target).abs().max() > 0.03


@pytest.mark.parametrize("nonuniform", [False, True])
def test_active_mapping_framing_and_single_coordinate_invariants(nonuniform):
    process = (
        ContinuousCategoricalDiffusion(6, [0.2, 0.3, 0.5], excluded_token_ids=[0, 2, 4])
        if nonuniform
        else ContinuousUniformDiffusion(6, excluded_token_ids=[0, 2, 4])
    )
    xt = torch.tensor([[0, 1, 3, 5, 2, 4], [0, 3, 1, 3, 2, 4]], dtype=torch.int32)
    mutable = torch.tensor([[False, True, True, True, False, False], [False] * 6])
    logits = torch.zeros(2, 6, 6)
    logits[..., [0, 2, 4]] = 1000  # Excluded IDs must never win the softmax.
    time = torch.tensor([1.0, 0.25])
    original_ids, original_logits, original_mask = (
        xt.clone(),
        logits.clone(),
        mutable.clone(),
    )
    conditional = gibbs_conditional_probs(process, logits, time)
    expected_pi = torch.tensor(
        [0.2, 0.3, 0.5] if nonuniform else [1 / 3] * 3, dtype=torch.float64
    )
    expected = process.noise_eps / 3 + (1 - process.noise_eps) * expected_pi
    torch.testing.assert_close(conditional[0, 0], expected)
    assert conditional.shape == (2, 6, 3)
    assert conditional.dtype == torch.float64
    for seed in range(20):
        result = random_scan_gibbs_step(
            process,
            logits,
            xt,
            time,
            mutable_mask=mutable,
            generator=torch.Generator().manual_seed(seed),
        )
        assert result.shape == xt.shape and result.dtype == xt.dtype
        assert torch.equal(result[~mutable], xt[~mutable])
        assert torch.all((result != xt).sum(-1) <= 1)
        assert set(result[mutable].tolist()) <= {1, 3, 5}
    assert torch.equal(xt, original_ids)
    assert torch.equal(logits, original_logits)
    assert torch.equal(mutable, original_mask)


def test_uniform_coordinate_choice_and_caller_generator_reproducibility():
    # At s=0 this conditional always chooses token 1, revealing precisely which
    # coordinate was selected. Repeated independent rows check uniform scan.
    process = ContinuousUniformDiffusion(2)
    xt = torch.zeros((4096, 5), dtype=torch.long)
    mutable = torch.tensor([False, True, False, True, True]).expand_as(xt)
    logits = torch.tensor([-torch.inf, 0.0]).expand(4096, 5, 2)
    time = torch.zeros(4096)
    first = torch.Generator().manual_seed(73)
    second = torch.Generator().manual_seed(73)
    global_rng_before = torch.random.get_rng_state().clone()
    actual = random_scan_gibbs_step(
        process, logits, xt, time, mutable_mask=mutable, generator=first
    )
    repeated = random_scan_gibbs_step(
        process, logits, xt, time, mutable_mask=mutable, generator=second
    )
    assert torch.equal(actual, repeated)
    assert torch.equal(first.get_state(), second.get_state())
    assert torch.equal(torch.random.get_rng_state(), global_rng_before)
    assert torch.all(actual.sum(-1) == 1)
    frequencies = actual[:, [1, 3, 4]].double().mean(0)
    assert torch.all((frequencies - 1 / 3).abs() < 0.03)


def test_wholly_immutable_rows_do_not_consume_rng_or_change_ids():
    process = ContinuousUniformDiffusion(3)
    xt = torch.tensor([[0, 1, 2]])
    generator = torch.Generator().manual_seed(41)
    before = generator.get_state().clone()
    result = random_scan_gibbs_step(
        process,
        torch.zeros(1, 3, 3),
        xt,
        torch.tensor([0.5]),
        mutable_mask=torch.zeros_like(xt, dtype=torch.bool),
        generator=generator,
    )
    assert torch.equal(result, xt)
    assert result.data_ptr() != xt.data_ptr()
    assert torch.equal(generator.get_state(), before)


def test_top_p_truncation_does_not_remove_refresh_support_or_narrow_tiny_prior():
    process = ContinuousCategoricalDiffusion(3, [0.7, 0.3, 1e-50])
    logits = torch.tensor([[[0.0, -3.0, -1000.0]]], dtype=torch.float32)
    conditionals = gibbs_conditional_probs(
        process, logits, torch.tensor([0.5]), raw_loo_top_p=0.5
    )
    refresh = (1 - process.noise_eps) * 0.5
    assert conditionals[0, 0, 2] > 0
    assert conditionals[0, 0, 2].item() == pytest.approx(
        refresh * 1e-50, rel=1e-12, abs=0
    )
    assert conditionals[0, 0, 1].item() == pytest.approx(refresh * 0.3)
    zero_time = gibbs_conditional_probs(
        process, logits, torch.tensor([0.0]), raw_loo_top_p=0.5
    )
    torch.testing.assert_close(
        zero_time, torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    )


@pytest.mark.parametrize("time", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_times_fail_before_sampling(time):
    with pytest.raises(ValueError, match="finite 0 <= s <= 1"):
        random_scan_gibbs_step(
            ContinuousUniformDiffusion(2),
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, dtype=torch.long),
            torch.tensor([time]),
        )


def test_excluded_editable_tokens_are_rejected_but_clamped_tokens_are_legal():
    process = ContinuousUniformDiffusion(3, excluded_token_ids=[0])
    with pytest.raises(ValueError, match="editable positions"):
        random_scan_gibbs_step(
            process, torch.zeros(1, 2, 3), torch.tensor([[0, 1]]), torch.tensor([0.5])
        )
    with pytest.raises(ValueError, match="Boolean"):
        random_scan_gibbs_step(
            process,
            torch.zeros(1, 2, 3),
            torch.tensor([[0, 1]]),
            torch.tensor([0.5]),
            mutable_mask=torch.tensor([[0, 1]]),
        )


def test_nonfinite_active_predictions_are_rejected():
    with pytest.raises(ValueError, match="finite nonnegative"):
        gibbs_conditional_probs(
            ContinuousUniformDiffusion(2),
            torch.full((1, 1, 2), -torch.inf),
            torch.tensor([0.25]),
        )
