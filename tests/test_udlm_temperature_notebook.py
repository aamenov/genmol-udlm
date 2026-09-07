"""Exact independent temperature teaching, with every earlier notebook cell retained."""

import ast
import json
from fractions import Fraction
from pathlib import Path

import pytest

from scripts.udlm import update_notebook as updater


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "genmol_from_scratch.ipynb"


def execute_stage():
    markdown, cell = updater._denoiser_temperature_cells()
    tree = ast.parse(cell["source"])
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert len(imports) == 1
    assert isinstance(imports[0], ast.ImportFrom) and imports[0].module == "fractions"
    namespace = {}
    exec(compile(tree, cell["id"], "exec"), namespace)
    return markdown["source"], namespace


def test_stage27_defines_the_control_and_teaching_boundaries():
    markdown, _ = execute_stage()
    for fragment in (
        "Paper correspondence",
        "Intuition and motivation",
        "Mathematics, with every symbol defined",
        "Small concrete example",
        "Code below, shapes, and invariants",
        "Differences from released implementations",
        "Comprehension checkpoint",
        "Expected reasoning",
        "not a bug fix",
        "temperature_space: x0_denoiser",
        "top-p one and no Gibbs",
        "V12 remains\nunchanged",
        "final seeds 0/1/2 remain reserved",
        "coordinate reverse marginal",
        "original trained prior",
    ):
        assert fragment in markdown


def test_chosen_clean_weights_and_reverse_steps_are_distinct_exact_distributions():
    _, ns = execute_stage()
    old, new = ns["stage27_old"], ns["stage27_new"]
    assert old[2] == Fraction(489559950, 900010909)
    assert new[2] == Fraction(490050, 490091)
    assert old[2] == pytest.approx(0.5439489067348627)
    assert new[2] == pytest.approx(0.9999163420670855)
    assert ns["stage27_old_step"][2] == pytest.approx(0.8480830919723489)
    assert ns["stage27_new_step"][2] == pytest.approx(0.9999588529490707)
    for values in (old, new, ns["stage27_old_step"], ns["stage27_new_step"]):
        assert len(values) == 3 and sum(values) == 1
        assert all(isinstance(value, Fraction) and value > 0 for value in values)
    assert old != ns["stage27_old_step"] and new != ns["stage27_new_step"]


@pytest.mark.parametrize("current", [0, 1, 2])
def test_temperature_one_agrees_and_zero_time_bridge_recovers_clean_weights(current):
    _, ns = execute_stage()
    d, pi, alpha = ns["stage27_denoiser"], ns["stage27_prior"], ns["stage27_alpha_t"]
    likelihood = tuple(
        alpha * (j == current) + (1 - alpha) * pi[current] for j in range(3)
    )
    old, new = ns["stage27_rules"](d, likelihood, 1)
    assert old == new == d
    # At s=0, the earlier token is clean. This catches a swapped kernel index.
    assert ns["stage27_bridge"](d, pi, current, alpha, Fraction(1)) == d


def test_mask_observation_preserves_nonmask_clean_odds_but_not_mask_odds():
    _, ns = execute_stage()
    d, pi, alpha = ns["stage27_denoiser"], ns["stage27_prior"], ns["stage27_alpha_t"]
    likelihood = tuple(alpha * (j == 0) + (1 - alpha) * pi[0] for j in range(3))
    old, new = ns["stage27_rules"](d, likelihood, 2)
    assert old[1] / old[2] == new[1] / new[2] == (d[1] / d[2]) ** 2
    assert old[0] / old[2] != new[0] / new[2]


def test_double_temperature_does_not_implement_the_declared_new_control():
    _, ns = execute_stage()
    old_again, _ = ns["stage27_rules"](ns["stage27_new"], ns["stage27_likelihood"], 2)
    assert old_again != ns["stage27_new"]


def test_only_two_cells_are_added_and_regeneration_is_byte_idempotent(tmp_path):
    current = json.loads(NOTEBOOK.read_text())
    stage = updater._denoiser_temperature_cells()
    stage_ids = {cell["id"] for cell in stage}
    historical = [cell for cell in current["cells"] if cell["id"] not in stage_ids]
    source = dict(current, cells=historical)
    previous = tmp_path / "previous.ipynb"
    first, second = tmp_path / "first.ipynb", tmp_path / "second.ipynb"
    previous.write_text(json.dumps(source))
    updater.update_notebook(previous, first)
    updater.update_notebook(first, second)
    assert first.read_bytes() == second.read_bytes() == NOTEBOOK.read_bytes()
    regenerated = json.loads(first.read_text())["cells"]
    assert [cell for cell in regenerated if cell["id"] not in stage_ids] == historical
    assert [cell for cell in regenerated if cell["id"] in stage_ids] == stage
    assert len({cell["id"] for cell in regenerated}) == len(regenerated)
    indices = [i for i, cell in enumerate(regenerated) if cell["id"] in stage_ids]
    assert indices[1] == indices[0] + 1
    assert regenerated[indices[0] - 1]["id"] == "stage-26-mask-rich-prior-code"
    for cell in regenerated:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=cell["id"])
