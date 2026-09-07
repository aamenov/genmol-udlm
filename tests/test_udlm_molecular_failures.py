"""Lexical audit guards: SMILES ring reuse and bracket numbers are distinct."""

import pytest

from scripts.udlm.audit_molecular_failures import lexical_flags


@pytest.mark.parametrize(
    "text",
    ["C1CC1", "C1CC1C1CC1", "[13CH3:7]C", "[NH4+]1.C1", "C%12.C%12", "C%(123).C%(123)"],
)
def test_balanced_labels_and_bracket_numbers_are_not_syntax_failures(text):
    assert lexical_flags(text) == {"odd_ring": False, "bad_parentheses": False}


@pytest.mark.parametrize("text", ["C1CC", "C1CC1.C1", "C%12.C%13", "C%(123)"])
def test_unmatched_ring_label_is_detected(text):
    assert lexical_flags(text) == {"odd_ring": True, "bad_parentheses": False}


@pytest.mark.parametrize("text", ["C(C", "C)C(", "C((C)"])
def test_parenthesis_prefix_and_final_balance_are_checked(text):
    assert lexical_flags(text) == {"odd_ring": False, "bad_parentheses": True}


def test_ring_and_parenthesis_failures_can_overlap():
    assert lexical_flags("C1(C") == {"odd_ring": True, "bad_parentheses": True}
