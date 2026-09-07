"""Independent CPU arithmetic and preservation checks for Stage 28."""

import ast
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

import pytest
import yaml

from scripts.udlm import update_notebook as updater
from scripts.exps.pmo.main.genmol import experiment_io


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "genmol_from_scratch.ipynb"
FROZEN_PREVIOUS = "48473d4febbd06d9bc07986ca96ceb93926c91ec"


def execute_pairs():
    cells = updater._pmo_optimization_cells()
    assert len(cells) == 8
    namespaces = []
    for markdown, code in zip(cells[::2], cells[1::2]):
        assert markdown["cell_type"] == "markdown" and code["cell_type"] == "code"
        for requirement in (
            "Paper correspondence",
            "Intuition and motivation",
            "Mathematics, with every symbol defined",
            "Small concrete example",
            "Code below, shapes, and invariants",
            "Differences from released implementations",
            "Comprehension checkpoint",
            "Expected reasoning",
        ):
            assert requirement in markdown["source"]
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


def test_all_four_pairs_execute_independently_without_external_state():
    cells, namespaces = execute_pairs()
    text = "\n".join(
        cell["source"] for cell in cells if cell["cell_type"] == "markdown"
    )
    for phrase in (
        "does **not**",
        "graph-level fragment preservation",
        "not decoded text inspection",
        "singleton-list",
        "legitimate zero",
        "at most 1001 new calls",
        "rejected\nproposals is insufficient",
        "training amounts",
        "No outcome table",
        "Final de novo",
    ):
        assert phrase in text
    assert namespaces[0]["stage28_final_mask_count"] == 1
    assert namespaces[0]["stage28_editable_count"] == 2
    assert namespaces[2]["stage28_remask_enabled"] == [False, False, True]


@pytest.mark.parametrize(
    "similarity,tpsa,logp,expected",
    [
        (0.1, 90, 4, 0.5),
        (0.8, 80, 5, math.exp(-1 / 3)),
        (0.8, 100, 3, 1.0),
        (0, 90, 4, 0.0),
        (0.8, 70, 4, math.exp(-2 / 3)),
        (0.8, 90, 6, math.exp(-2 / 3)),
    ],
)
def test_fex_components_match_independent_hand_cases(similarity, tpsa, logp, expected):
    _, ns = execute_pairs()
    components, score = ns[1]["stage28_descriptor_score"](similarity, tpsa, logp)
    assert all(0 <= value <= 1 for value in components)
    assert score == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("spacing", [1, 2, 3, 9])
def test_fraction_auc_matches_released_metric_for_synthetic_scores(spacing):
    _, ns = execute_pairs()
    budget = ns[2]
    scores = budget["stage28_charged"]
    curve, auc = budget["stage28_exact_auc"](scores, spacing)
    assert isinstance(auc, Fraction)
    assert curve[0] == (0, Fraction(0)) and curve[-1][0] == 4
    actual = experiment_io.top_k_auc(
        [float(score) for score in scores],
        k=10,
        reporting_frequency=spacing,
        budget=4,
        pad_to_budget=False,
    )
    assert actual == pytest.approx(float(auc), abs=1e-14)
    assert budget["stage28_top10"]([Fraction(i, 11) for i in range(12)]) == Fraction(
        13, 22
    )


def test_cache_order_and_prospective_panel_match_frozen_inputs():
    _, ns = execute_pairs()
    budget, panel = ns[2], ns[3]
    assert budget["stage28_reasons"].count("charged") == 4
    assert len(budget["stage28_reasons"]) == 7
    assert budget["stage28_auc"] == Fraction(2, 5)
    assert budget["stage28_exact_auc"](budget["stage28_early"], 2)[1] == Fraction(
        23, 40
    )
    protocol = json.loads(
        (ROOT / "experiments/udlm/protocols/engineering_v14_pmo.json").read_text()
    )
    assert set(panel["stage28_planned_seeds"]) == {
        entry["seed"] for entry in protocol["entries"]
    }
    assert panel["stage28_requested_runs"] == len(protocol["entries"]) == 6
    assert panel["stage28_requested_call_ceiling"] == sum(
        entry["max_oracle_calls"] for entry in protocol["entries"]
    )
    names = {"MDLM": "mdlm", "S CT": "s_ct", "MASK CE": "mask_ce"}
    for arm in panel["stage28_planned_arms"]:
        config = yaml.safe_load(
            (
                ROOT
                / f"experiments/udlm/protocols/engineering_v14_pmo_configs/{names[arm['name']]}.yaml"
            ).read_text()
        )
        assert arm["checkpoint_sha256"] == config["checkpoint_sha256"]
        assert arm["temperature"] == config["softmax_temp"]
        assert arm["randomness"] == config["randomness"]
        assert arm["nfe"] == config.get("num_steps")
        if arm["name"] != "MDLM":
            assert arm["prior"] == config["prior_variant"]


def test_previous_152_cells_are_exactly_preserved_and_regeneration_is_idempotent(
    tmp_path,
):
    previous = json.loads(
        subprocess.check_output(
            ["git", "show", FROZEN_PREVIOUS + ":genmol_from_scratch.ipynb"], cwd=ROOT
        )
    )
    assert len(previous["cells"]) == 152
    assert sum(cell["cell_type"] == "code" for cell in previous["cells"]) == 72
    first, second = tmp_path / "first.ipynb", tmp_path / "second.ipynb"
    updater.update_notebook(NOTEBOOK, first)
    updater.update_notebook(first, second)
    assert first.read_bytes() == second.read_bytes() == NOTEBOOK.read_bytes()
    current = json.loads(first.read_text())
    added = updater._pmo_optimization_cells()
    new_ids = {cell["id"] for cell in added}
    previous_ids = {cell["id"] for cell in previous["cells"]}
    assert [cell for cell in current["cells"] if cell["id"] in previous_ids] == previous[
        "cells"
    ]
    assert [cell for cell in current["cells"] if cell["id"] in new_ids] == added
    assert len(current["cells"]) >= 160  # Later teaching stages may append before the report.
    assert len({cell["id"] for cell in current["cells"]}) == len(current["cells"])
    for index, cell in enumerate(current["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=cell["id"])
            if cell["id"] in new_ids:
                assert current["cells"][index - 1]["cell_type"] == "markdown"
