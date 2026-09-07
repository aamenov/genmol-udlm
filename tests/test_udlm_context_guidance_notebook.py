"""Stage29 exact laws, independent execution and historical-cell preservation."""

import ast
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess

import pytest

from scripts.udlm import update_notebook as updater

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "genmol_from_scratch.ipynb"
PREVIOUS = "ccdbe64e2147f3d85f232a165ffd6400167c394a"


def execute_pairs():
    cells = updater._posterior_context_guidance_cells()
    assert len(cells) == 8
    namespaces = []
    for markdown, code in zip(cells[::2], cells[1::2]):
        assert markdown["cell_type"] == "markdown"
        assert code["cell_type"] == "code"
        for heading in (
            "Paper correspondence",
            "Intuition and motivation",
            "Mathematics, with every symbol defined",
            "Small concrete example",
            "Code below, shapes, and invariants",
            "Differences from released implementations",
            "Comprehension checkpoint",
            "Expected reasoning",
        ):
            assert heading in markdown["source"]
        tree = ast.parse(code["source"])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module in {"math", "fractions"}
            assert not isinstance(node, ast.Import)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"open", "eval", "exec", "__import__"}
        namespace = {}
        exec(compile(tree, code["id"], "exec"), namespace)
        namespaces.append(namespace)
    return cells, namespaces


def test_four_pairs_execute_from_independent_clean_namespaces():
    cells, ns = execute_pairs()
    text = "\n".join(
        cell["source"] for cell in cells if cell["cell_type"] == "markdown"
    )
    for phrase in (
        "per-row",
        "current row-zero",
        "original editable",
        "raw-LOO",
        "null-label",
        "context RNG draws",
        "candidate-equivalent",
        "underflow",
        "globally",
        "no molecular efficacy claim",
        "invokes no production API",
    ):
        assert phrase.lower() in text.lower()
    assert ns[0]["stage29_hidden"] == ((1,), (2,))
    assert ns[0]["stage29_clamped"] == ((1, 5, 6, 7, 2, 3), (1, 8, 6, 8, 2, 3))
    assert ns[3]["stage29_float_probabilities"] == (1.0, 0.0)


@pytest.mark.parametrize("observed", [0, 1, 2])
@pytest.mark.parametrize("prior", [(F(1, 3),) * 3, (F(9, 10), F(1, 20), F(1, 20))])
def test_ce_bridge_matches_explicit_forward_matrix_conditionals(prior, observed):
    _, ns = execute_pairs()
    denoiser = (F(3, 5), F(3, 10), F(1, 10))
    alpha_s, alpha_t = F(7, 10), F(2, 5)
    _, result = ns[1]["stage29_reverse"](denoiser, prior, observed, alpha_s, alpha_t)
    q_s = [
        [alpha_s * (i == j) + (1 - alpha_s) * prior[i] for i in range(3)]
        for j in range(3)
    ]
    transition = [
        [
            alpha_t / alpha_s * (i == k) + (1 - alpha_t / alpha_s) * prior[k]
            for k in range(3)
        ]
        for i in range(3)
    ]
    expected = []
    for earlier in range(3):
        value = F(0)
        for clean in range(3):
            normalizer = sum(q_s[clean][i] * transition[i][observed] for i in range(3))
            value += (
                denoiser[clean]
                * q_s[clean][earlier]
                * transition[earlier][observed]
                / normalizer
            )
        expected.append(value)
    assert result == tuple(expected) and sum(result) == 1 and min(result) > 0


def test_exact_noncommuting_counterexample_and_guided_odds():
    _, namespaces = execute_pairs()
    ns = namespaces[2]
    assert ns["stage29_guided"] == (
        F(141984, 175679),
        F(245055, 1405432),
        F(24505, 1405432),
    )
    assert ns["stage29_clean_first"] == (F(1929, 2380), F(73, 476), F(43, 1190))
    c, u, guided = (
        ns["stage29_conditional"],
        ns["stage29_degraded"],
        ns["stage29_guided"],
    )
    assert guided[0] / guided[1] == (c[0] / c[1]) ** 2 / (u[0] / u[1])
    assert (
        ns["stage29_guide"](c, u, 0) == u
    )  # Algebra only; proposed API range is w>=1.


def test_batch_weighted_work_and_identity_rng_accounting():
    _, namespaces = execute_pairs()
    fn = namespaces[3]["stage29_work"]
    for batch, steps in ((1, 1), (3, 5), (4, 128)):
        for execution in ("serial", "packed"):
            active = fn(
                batch,
                steps,
                execution,
                gamma=F(1, 2),
                weight=2,
                eligible_per_row=(4,) * batch,
            )
            assert active["candidate_evaluations"] == 2 * batch * steps
            assert active["forward_invocations"] == steps * (
                2 if execution == "serial" else 1
            )
            for gamma, weight, count in ((0, 2, 4), (F(1, 2), 1, 4), (F(1, 2), 2, 1)):
                identity = fn(
                    batch,
                    steps,
                    execution,
                    gamma=gamma,
                    weight=weight,
                    eligible_per_row=(count,) * batch,
                )
                assert identity["candidate_evaluations"] == batch * steps
                assert identity["context_subset_draws"] == 0


def test_all_160_previous_cells_exact_and_new_stage_before_final_report(tmp_path):
    previous = json.loads(
        subprocess.check_output(
            ["git", "show", PREVIOUS + ":genmol_from_scratch.ipynb"], cwd=ROOT
        )
    )
    assert len(previous["cells"]) == 160
    assert sum(cell["cell_type"] == "code" for cell in previous["cells"]) == 76
    first, second = tmp_path / "first.ipynb", tmp_path / "second.ipynb"
    updater.update_notebook(NOTEBOOK, first)
    updater.update_notebook(first, second)
    assert first.read_bytes() == second.read_bytes() == NOTEBOOK.read_bytes()
    current = json.loads(first.read_text())
    new = updater._posterior_context_guidance_cells()
    ids = {cell["id"] for cell in new}
    assert [cell for cell in current["cells"] if cell["id"] not in ids] == previous[
        "cells"
    ]
    assert current["cells"][-3:] == previous["cells"][-3:]
    assert current["cells"][-11:-3] == new
    assert (
        len(current["cells"]) == len({cell["id"] for cell in current["cells"]}) == 168
    )
    for cell in current["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=cell["id"])
            assert cell["execution_count"] is None and cell["outputs"] == []
