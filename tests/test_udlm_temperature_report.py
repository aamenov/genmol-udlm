"""Synthetic temperature-space reports; these are not molecular observations."""

import copy
import csv
import hashlib
import io
import json

import pytest
from pypdf import PdfReader

from scripts.udlm import report_exploration as report
from test_udlm_paired_report import synthetic_screen


def temperature_screen(root):
    original = synthetic_screen(root, prior=True)
    protocol = original["protocol"]["configuration"]
    declarations = protocol["design"].pop("prior_comparison")
    protocol["design"]["temperature_space_comparison"] = declarations
    entries = {entry["config_id"]: entry for entry in protocol["entries"]}
    for index, declaration in enumerate(declarations.values()):
        for role in ("control", "treatment"):
            entry = entries[declaration[f"{role}_config"]]
            entry.update(
                temperature_space="raw_loo" if role == "control" else "x0_denoiser",
                prior_variant=(
                    "empirical_frequency" if index == 0 else "mask_rich_empirical"
                ),
                prior_metadata_sha256=("e" if index == 0 else "f") * 64,
                checkpoint_sha256=("a" if index == 0 else "b") * 64,
            )
    for run in original["runs"]:
        entry = entries[run["config_id"]]
        sampling = run["config"]["sampling"]
        sampling.update(
            {
                field: entry[field]
                for field in ("prior_variant", "prior_metadata_sha256")
            }
        )
        sampling.update(num_steps=128, raw_loo_top_p=1.0)
        if entry["temperature_space"] == "x0_denoiser":
            sampling["temperature_space"] = "x0_denoiser"
        run["checkpoint"]["sha256"] = entry["checkpoint_sha256"]
        summary_path = root / run["run_directory"] / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["config"] = run["config"]
        summary["checkpoint"] = run["checkpoint"]
        summary_path.write_text(json.dumps(summary))
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    output = root / "output/engineering"
    for entry in protocol["entries"]:
        receipt_path = output / "controller_receipts" / f"{entry['attempt_id']}.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["identity"]["entry"] = entry
        receipt["identity"]["protocol_sha256"] = hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest()
        receipt["artifacts"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (output / entry["attempt_id"]).glob("seed_*/*")
        }
        receipt_path.write_text(json.dumps(receipt))

    def rescore(path, *, root):
        run = next(
            run
            for run in original["runs"]
            if root / run["run_directory"] == path.parent
        )
        result = copy.deepcopy(run["independent_rescore"])
        result["summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    result = report.build_report(protocol_path, output, root=root, rescore=rescore)
    result["caveats"].insert(0, "SYNTHETIC COUNTS ONLY; NOT V13 MOLECULAR RESULTS.")
    return result


@pytest.fixture
def paired(tmp_path):
    return temperature_screen(tmp_path)


def recompute(paired):
    return report._paired_contrasts(paired["protocol"]["configuration"], paired["runs"])


def test_full_pipeline_keeps_both_signed_contrasts(paired):
    assert paired["status"] == "complete"
    assert paired["accounting"]["independently_rescored_requests"] == 80
    primary, secondary = paired["paired_contrasts"]
    assert {x["direction"] for x in (primary, secondary)} == {
        "clean_temperature_minus_raw_loo_temperature"
    }
    values = primary["metrics"]["released_comparable"]["quality"]
    assert [x["difference"] for x in values["per_seed"]] == [0.1, -0.1]
    assert values["mean_difference"] == 0
    assert primary["metrics"]["strict"]["quality"]["mean_difference"] == -0.15
    assert (
        secondary["metrics"]["released_comparable"]["quality"]["mean_difference"]
        == 0.25
    )
    paired["runs"].reverse()
    assert recompute(paired) == paired["paired_contrasts"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature_space", "raw_loo"),
        ("checkpoint_sha256", "f" * 64),
        ("prior_metadata_sha256", "a" * 64),
        ("prior_variant", "mask_rich_empirical"),
        ("parameterization", "raw_loo"),
    ],
)
def test_declared_treatment_relabeling_rejected(paired, field, value):
    protocol = paired["protocol"]["configuration"]
    treatment = protocol["design"]["temperature_space_comparison"]["primary"][
        "treatment_config"
    ]
    next(x for x in protocol["entries"] if x["config_id"] == treatment)[field] = value
    with pytest.raises(ValueError):
        recompute(paired)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature_space", "raw_loo"),
        ("num_steps", 512),
        ("prior_metadata_sha256", "a" * 64),
        ("prior_variant", "mask_rich_empirical"),
        ("raw_loo_top_p", 0.9),
        ("gibbs_corrector", True),
    ],
)
def test_observed_control_or_mode_drift_rejected(paired, field, value):
    treatment = next(
        x
        for x in paired["runs"]
        if x["config"]["sampling"].get("temperature_space") == "x0_denoiser"
    )
    treatment["config"]["sampling"][field] = value
    with pytest.raises(ValueError):
        recompute(paired)


@pytest.mark.parametrize("status", ["missing", "failed", "invalid", "undefined"])
def test_unavailable_pair_withholds_only_affected_metric(paired, status):
    treatment = next(
        x
        for x in paired["runs"]
        if x["config"]["sampling"].get("temperature_space") == "x0_denoiser"
    )
    if status == "missing":
        paired["runs"].remove(treatment)
    elif status == "undefined":
        treatment["metrics"]["released_comparable"]["quality"] = None
    else:
        treatment.update(status=status, metrics=None)
    primary, secondary = recompute(paired)
    values = primary["metrics"]["released_comparable"]["quality"]
    assert values["mean_difference"] is None and values["sample_sd"] is None
    assert values["defined_seed_count"] == 1
    assert values["per_seed"][1]["difference"] == -0.1
    assert secondary["metrics"]["released_comparable"]["quality"]["complete"]


def test_empty_or_mixed_design_rejected(paired):
    protocol = paired["protocol"]["configuration"]
    protocol["design"]["objective_comparison"] = {}
    with pytest.raises(ValueError, match="declare either"):
        recompute(paired)
    protocol["design"].pop("objective_comparison")
    protocol["design"]["temperature_space_comparison"] = {}
    with pytest.raises(ValueError, match="at least one"):
        recompute(paired)


def test_pdf_and_csv_use_temperature_labels(paired):
    rows = list(csv.DictReader(io.StringIO(report._paired_csv_bytes(paired).decode())))
    assert len(rows) == 48
    assert {row["direction"] for row in rows} == {
        "clean_temperature_minus_raw_loo_temperature"
    }
    text = "\n".join(
        page.extract_text()
        for page in PdfReader(io.BytesIO(report._pdf_bytes(paired))).pages
    )
    assert "Temperature-space comparison" in text
    assert "Clean minus LOO" in text
    assert "Predeclared paired CE minus CT" not in text
    assert "Predeclared paired MASK minus empirical" not in text
    assert "same frozen CE checkpoint" in " ".join(paired["caveats"])
