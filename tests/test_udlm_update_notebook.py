from __future__ import annotations

import ast
import copy
import csv
import io
import json
import subprocess
from pathlib import Path

import pytest

from scripts.udlm import update_notebook as updater


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPOSITORY_ROOT / "genmol_from_scratch.ipynb"


def _notebook() -> dict:
    return json.loads(SOURCE_NOTEBOOK.read_text())


def _write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def _cells_by_id(notebook: dict) -> dict[str, dict]:
    return {cell["id"]: cell for cell in notebook["cells"]}


def test_update_replaces_generated_cells_and_preserves_stages_20_6_through_20_8(
    tmp_path: Path,
) -> None:
    notebook = _notebook()
    before = _cells_by_id(notebook)
    preserved_before = {
        cell_id: copy.deepcopy(before[cell_id])
        for cell_id in updater.PRESERVED_STAGE20_TAGS_BY_ID
    }
    generated = updater.stage_cells()
    late_generated = updater._selection_bound_scale_up_cells()  # noqa: SLF001
    before[generated[0]["id"]]["source"] = "corrupt generated source\n"
    before[late_generated[0]["id"]]["source"] = "corrupt late source\n"

    source = tmp_path / "source.ipynb"
    destination = tmp_path / "updated.ipynb"
    _write_notebook(source, notebook)
    updater.update_notebook(source, destination)

    updated = json.loads(destination.read_text())
    updated_by_id = _cells_by_id(updated)
    assert updated_by_id[generated[0]["id"]] == generated[0]
    assert updated_by_id[late_generated[0]["id"]] == late_generated[0]
    assert {
        cell_id: updated_by_id[cell_id]
        for cell_id in updater.PRESERVED_STAGE20_TAGS_BY_ID
    } == preserved_before

    stage20_ids = [
        cell["id"]
        for cell in updated["cells"]
        if cell["id"].startswith(updater.STAGE_TAG_PREFIX)
    ]
    assert stage20_ids == [
        *(cell["id"] for cell in generated),
        *updater.PRESERVED_STAGE20_TAGS_BY_ID,
        *(cell["id"] for cell in late_generated),
    ]


def test_update_rejects_unknown_stage20_tag(tmp_path: Path) -> None:
    notebook = _notebook()
    notebook["cells"].append(
        {
            "cell_type": "markdown",
            "id": "retired-stage-20-cell",
            "metadata": {"tags": ["stage-20-udlm-retired"]},
            "source": "retired content\n",
        }
    )
    source = tmp_path / "unknown.ipynb"
    _write_notebook(source, notebook)

    with pytest.raises(ValueError, match="unknown .*prefixed cell"):
        updater.update_notebook(source, tmp_path / "must-not-exist.ipynb")


def test_update_rejects_duplicate_cell_ids(tmp_path: Path) -> None:
    notebook = _notebook()
    notebook["cells"].append(copy.deepcopy(notebook["cells"][0]))
    source = tmp_path / "duplicate.ipynb"
    _write_notebook(source, notebook)

    with pytest.raises(ValueError, match="duplicate cell id"):
        updater.update_notebook(source, tmp_path / "must-not-exist.ipynb")


def test_update_is_byte_idempotent(tmp_path: Path) -> None:
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"

    updater.update_notebook(SOURCE_NOTEBOOK, first)
    updater.update_notebook(first, second)

    assert first.read_bytes() == SOURCE_NOTEBOOK.read_bytes()
    assert second.read_bytes() == first.read_bytes()


def test_generated_stage0_uses_utilization_based_shared_gpu_policy(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "updated.ipynb"
    updater.update_notebook(SOURCE_NOTEBOOK, destination)
    cells = _cells_by_id(json.loads(destination.read_text()))
    code = "".join(cells["stage0-setup"]["source"])
    count_code = "".join(cells["stage0-gpu-count"]["source"])
    markdown = " ".join("".join(cells["stage0-setup-note"]["source"]).split())

    for fragment in (
        "MIN_IDLE_FREE_MEMORY_MIB = 30000",
        "MAX_IDLE_UTILIZATION_PERCENT = 10",
        "memory.free,utilization.gpu,compute_mode",
        "gpu['compute_processes'] = process_rows",
        "if utilization_percent >= MAX_IDLE_UTILIZATION_PERCENT",
        "-gpu['free_memory_mib']",
        "if final['uuid'] != initial['uuid']",
    ):
        assert fragment in code
    assert "if process_rows:" not in code
    assert "a nonempty inventory is allowed" in markdown
    assert "card at exactly 10% is rejected" in markdown
    assert "never interrupts or kills" in markdown
    assert "1 <= NUM_GPUS <= 4" in count_code
    assert "hard ceiling: 4" in count_code
    compile(code, "stage0-setup", "exec")


def test_later_notebook_gpu_rechecks_reuse_stage0_policy_and_allow_processes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "updated.ipynb"
    updater.update_notebook(SOURCE_NOTEBOOK, destination)
    cells = _cells_by_id(json.loads(destination.read_text()))

    stage5_note = " ".join(
        "".join(cells["stage5-forward-note"]["source"]).split()
    )
    stage5_code = "".join(cells["f4963b0c"]["source"])
    stage6_note = " ".join(
        "".join(cells["stage6-bert-update-note"]["source"]).split()
    )
    stage6_code = "".join(cells["stage6-bert-update"]["source"])

    assert "foreign_compute_processes" not in stage5_code
    assert "probe_physical_gpu(STAGE5_PHYSICAL_GPU_ID)" in stage5_code
    assert 'stage5_gpu_reprobe.get("uuid") != SELECTED_GPU_UUIDS[0]' in stage5_code
    assert 'stage5_gpu_reprobe["compute_processes"]' in stage5_code
    assert "allowing and recording active process rows" in stage5_note

    assert "foreign_compute_processes" not in stage6_code
    assert "probe_physical_gpu(stage6_physical_candidate)" in stage6_code
    assert "SELECTED_GPU_UUIDS[stage6_logical_candidate]" in stage6_code
    assert 'stage6_reprobe["utilization_percent"]' in stage6_code
    assert 'stage6_reprobe["compute_processes"]' in stage6_code
    assert "utilization strictly below 10 percent" in stage6_note
    assert "active process rows are recorded but do not disqualify" in stage6_note

    compile(stage5_code, "stage5-forward", "exec")
    compile(stage6_code, "stage6-bert-update", "exec")


@pytest.mark.parametrize(
    ("free_memory_mib", "utilization", "compute_mode", "eligible", "reason"),
    [
        (30_000, 9, "Default", True, None),
        (30_000, 10, "Default", False, "not below 10%"),
        (29_999, 9, "Default", False, "below 30000 MiB"),
        (30_000, 9, "Prohibited", False, "compute mode is prohibited"),
    ],
)
def test_generated_stage0_gpu_boundaries_allow_recorded_processes(
    free_memory_mib: int,
    utilization: int,
    compute_mode: str,
    eligible: bool,
    reason: str | None,
) -> None:
    parsed = ast.parse(updater.STAGE0_SETUP_CODE)
    names = {"MIN_IDLE_FREE_MEMORY_MIB", "MAX_IDLE_UTILIZATION_PERCENT"}
    selected_nodes = [
        node
        for node in parsed.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            )
        )
        or isinstance(node, ast.FunctionDef)
        and node.name == "probe_physical_gpu"
    ]
    namespace = {"csv": csv, "io": io}
    exec(
        compile(
            ast.Module(body=selected_nodes, type_ignores=[]), "stage0-policy", "exec"
        ),
        namespace,
    )
    responses = iter(
        (
            subprocess.CompletedProcess(
                [],
                0,
                stdout=(
                    "2, GPU-stage0-test, Synthetic GPU, "
                    f"{free_memory_mib}, {utilization}, {compute_mode}\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess([], 0, stdout="4321, 512 MiB\n", stderr=""),
        )
    )
    namespace["_run_nvidia_smi"] = lambda *_args: next(responses)

    state = namespace["probe_physical_gpu"](2)

    assert state["eligible"] is eligible
    assert state["compute_processes"] == [{"pid": 4321, "used_memory": "512 MiB"}]
    assert state["compute_process_count"] == 1
    if reason is not None:
        assert any(reason in item for item in state["reasons"])


def test_prior_floor_teaching_binds_retrospective_artifact_without_rewriting_history():
    notebook = _notebook()
    cells = _cells_by_id(notebook)
    markdown = "".join(cells["stage-20-udlm-prior-geometry"]["source"])
    code = "".join(cells["stage-20-udlm-prior-geometry-code"]["source"])

    for fragment in (
        "retrospective replication, not a preregistered confirmation",
        "historical/manual categorical configuration",
        "reviewed pilot launches only",
        "Does the lower unigram NLL mean",
    ):
        assert fragment in markdown
    for fragment in (
        "floor_selection_train_rows_10001_30000.json",
        "02908dafaf589ca9a49e560aa1eab470a18d6bfe616b781164784c489f54a9f1",
        "6424b323084358ea050ba22d7e13ef8d45962496",
        '"historical_manual_weight": 0.01',
        '"reviewed_pilot_weight": 0.0002',
    ):
        assert fragment in code
    compile(code, "stage-20-udlm-prior-geometry-code", "exec")


def test_generated_schedule_teaching_distinguishes_official_recipe(tmp_path: Path):
    destination = tmp_path / "updated.ipynb"
    updater.update_notebook(SOURCE_NOTEBOOK, destination)
    cells = _cells_by_id(json.loads(destination.read_text()))
    markdown = "".join(cells["stage-20-udlm-evidence"]["source"])

    assert "QM9 recipe uses 25,000 optimizer steps" in markdown
    assert "not the released UDLM QM9 schedule" in markdown
    assert "pilot hypothesis" in markdown
    assert "not an exact replay of the official recipe" in markdown


def test_generated_math_teaches_plugin_output_as_loo_predictor(tmp_path: Path):
    destination = tmp_path / "updated.ipynb"
    updater.update_notebook(SOURCE_NOTEBOOK, destination)
    cells = _cells_by_id(json.loads(destination.read_text()))
    markdown = " ".join(
        "".join(cells["stage-20-udlm-math"]["source"]).split()
    )

    for fragment in (
        "https://arxiv.org/abs/2605.22765",
        "simplex-valued plug-in bridge parameter",
        "leave-one-out (LOO) posterior",
        "not the ordinary denoising posterior",
        "temperature/top-p should act on the raw LOO logits",
        "does not enforce exact invariance",
    ):
        assert fragment in markdown


def test_generated_health_teaching_binds_exact_gate_and_later_diagnostic(
    tmp_path: Path,
):
    destination = tmp_path / "updated.ipynb"
    updater.update_notebook(SOURCE_NOTEBOOK, destination)
    cells = _cells_by_id(json.loads(destination.read_text()))
    markdown = "".join(cells["stage-20-udlm-evidence"]["source"])
    compact_markdown = " ".join(markdown.split())
    code = "".join(cells["stage-20-udlm-evidence-code"]["source"])
    all_markdown = "\n".join(
        "".join(cell["source"])
        for cell in cells.values()
        if cell["cell_type"] == "markdown"
    )

    for fragment in (
        "scripts/udlm/launch_health_panel.py",
        "scripts/udlm/validate_health_panel.py",
        "health-w{W}-e-{H}",
        "Completed health-to-selection firewall",
        "R3=`b49e900`",
        "permits only screen authorization",
        "$B_{\\mathrm{eff}}=Wma_W=16$",
        "Only after each 1,000-update training receipt validates",
        "not the 10-update health-terminal receipt",
    ):
        assert fragment in compact_markdown
    for fragment in (
        '"launcher_variant_order": ["udlm", "schedule_uniform", "udlm_categorical"]',
        '"supported_world_sizes": [1, 2]',
        '"num_nodes": 1',
        '"optimizer_updates_each": 10',
        '"training_seed": 1',
        '"loader_workers": 1',
        '"micro_batch_size_per_process": 2',
        '"gradient_accumulation_by_world_size": {1: 8, 2: 4}',
        '"effective_global_batch_size": 16',
        '"vocabulary_size": 1880',
        '"exclude_special_tokens": False',
        '"scratch_mode": False',
        '"empirical_uniform_mix": 0.0002',
        '"checkpoint_project_relative_path"',
        '"checkpoint_size_bytes": 1396998679',
        "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6",
        '"max_utilization_percent": 10',
        '"active_compute_processes_allowed": True',
        '"min_free_memory_mib": 30000',
        '"normalized_evidence_schema_version": 1',
        '"successful_exit_receipt_schema_version": 5',
        '"terminal_e_receipt_required_before"',
        '"screen_config_materialization"',
        '"screen_registry_freeze"',
        '"candidate_lock": False',
        '"post_training_decode_diagnostic"',
        '"after_optimizer_updates_each": 1000',
        '"seed": 1100',
        '"requested": 32',
    ):
        assert fragment in code
    assert '"health_generation"' not in code
    compact_all_markdown = " ".join(all_markdown.split())
    assert "at exactly 10% is rejected" in compact_all_markdown
    assert "never interrupts or kills" in compact_all_markdown
    assert "deterministic `health-w{W}-{r,s,e}-{H}` names" in compact_all_markdown
    assert "Only after all three training receipts pass may seed 1100 x 32 requests" in (
        compact_all_markdown
    )
    compile(code, "stage-20-udlm-evidence-code", "exec")


def test_stage20_9_teaches_and_verifies_selection_bound_scale_up(tmp_path: Path):
    destination = tmp_path / "updated.ipynb"
    updater.update_notebook(SOURCE_NOTEBOOK, destination)
    notebook = json.loads(destination.read_text())
    cells = _cells_by_id(notebook)
    markdown = " ".join(
        "".join(cells["stage-20-udlm-selection-bound-scale-up"]["source"]).split()
    )
    code = "".join(
        cells["stage-20-udlm-selection-bound-scale-up-code"]["source"]
    )

    for fragment in (
        "https://arxiv.org/abs/2412.10193",
        "https://arxiv.org/abs/2501.06158",
        "R versus S changes only",
        "S versus E",
        "Concrete example and intuition",
        "Code and tensor invariants",
        "Difference from released implementations",
        "Comprehension checkpoint",
        "conditional on E-tuned optimization and conditioning",
        "adds 14,962,176 conditioning parameters",
        "first scale-up rung will use one GPU",
        "no molecular benchmark or superiority result",
        "[B,L,1880]",
        "[B,1,H]",
        "utilization strictly below 10%",
        "up to four",
    ):
        assert fragment in markdown
    for fragment in (
        "scheduler_evidence.json",
        "scheduler_selection.json",
        "conditioning_evidence.json",
        "conditioning_selection.json",
        "76a4ec9771dd0e875bf4532beb217da0ed54426427d39d92dbb218c50673c705",
        "e93d1face65bab573b32898da1a0a7a209a95a1a21fde58c509bcb3db8bac272",
        "e63010c52f97e788bb25ea8f46b4f5ba9f2ae1e95bca484ee139a871ae57d612",
        "ea1473ccfbff5b55e6f0e27dea4ea1c6ecca20de1a69935857da782b929193d0",
        '"selected_arm_id"] == "E-L1"',
        '"selected_arm_id"] == "E-A1"',
        '"generation_metrics_included"] is False',
        '"final_generation_seeds_used"] == []',
        '"screen_selection_artifacts_alone_authorize_scale_up": False',
        '"supported_gpu_counts": [1, 2, 3, 4]',
        '"planned_first_registered_gpu_count": 1',
        '"reseed_after_model_initialization_each": True',
    ):
        assert fragment in code
    compile(code, "stage-20-udlm-selection-bound-scale-up-code", "exec")

    ordered_ids = [cell["id"] for cell in notebook["cells"]]
    assert ordered_ids.index("stage-20-udlm-selection-bound-scale-up") == (
        ordered_ids.index("stage-20-udlm-stream-sharding-code") + 1
    )
    assert ordered_ids.index("stage-20-udlm-selection-bound-scale-up-code") == (
        ordered_ids.index("stage-20-udlm-selection-bound-scale-up") + 1
    )
    assert ordered_ids.index("stage19-report-note") == (
        ordered_ids.index("stage-20-udlm-selection-bound-scale-up-code") + 1
    )
