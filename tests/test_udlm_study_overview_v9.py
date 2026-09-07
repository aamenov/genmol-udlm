import copy
import csv
import io
import json
import statistics

import pytest

from scripts.udlm import generate_study_overview_v9 as overview


def paired_fixture():
    protocol = {"seeds": [1600, 1601], "design": {"objective_comparison": {}}}
    report = {"runs": [], "paired_contrasts": []}
    for contrast_id, temperature, code in (
        ("primary", 1.0, "100"),
        ("secondary", 0.5, "050"),
    ):
        control_config, treatment_config = f"ct_t{code}", f"ce_t{code}"
        declared = dict(
            control_config=control_config,
            treatment_config=treatment_config,
            temperature=temperature,
        )
        protocol["design"]["objective_comparison"][contrast_id] = declared
        contrast = dict(
            declared,
            contrast_id=contrast_id,
            direction="CE_minus_CT",
            expected_seeds=[1600, 1601],
            summary_policy="all_declared_pairs_required_for_each_metric",
            metrics={},
        )
        for arm, config_id in (("CT", control_config), ("CE", treatment_config)):
            for index, seed in enumerate(protocol["seeds"]):
                value = (0.30 if arm == "CT" else 0.35) + index * (
                    0.02 if arm == "CT" else 0.05
                )
                report["runs"].append(
                    {
                        "config_id": config_id,
                        "seed": seed,
                        "metrics": {
                            branch: dict.fromkeys(overview.METRICS, value)
                            for branch in overview.BRANCHES
                        },
                    }
                )
        for branch in overview.BRANCHES:
            contrast["metrics"][branch] = {}
            for metric in overview.METRICS:
                rows = []
                for index, seed in enumerate(protocol["seeds"]):
                    control, treatment = 0.30 + index * 0.02, 0.35 + index * 0.05
                    row = dict(
                        seed=seed,
                        requested_samples=100,
                        control_status="completed",
                        treatment_status="completed",
                        status="defined",
                        control_value=control,
                        treatment_value=treatment,
                        difference=treatment - control,
                    )
                    if metric == "quality":
                        row.update(
                            control_quality_count=round(control * 100),
                            treatment_quality_count=round(treatment * 100),
                        )
                    rows.append(row)
                differences = [row["difference"] for row in rows]
                contrast["metrics"][branch][metric] = dict(
                    complete=True,
                    defined_seed_count=2,
                    expected_seed_count=2,
                    per_seed=rows,
                    mean_difference=statistics.mean(differences),
                    sample_sd=statistics.stdev(differences),
                )
        report["paired_contrasts"].append(contrast)
    return report, protocol


def test_valid_paired_arithmetic():
    report, protocol = paired_fixture()
    overview.validate_paired_contrasts(report, protocol)


@pytest.mark.parametrize(
    "mutation", ["delta", "mean", "count", "seed", "direction", "duplicate_run"]
)
def test_rejects_changed_paired_evidence(mutation):
    report, protocol = paired_fixture()
    contrast = report["paired_contrasts"][0]
    quality = contrast["metrics"]["strict"]["quality"]
    if mutation == "delta":
        quality["per_seed"][0]["difference"] *= -1
    elif mutation == "mean":
        quality["mean_difference"] = 0.0
    elif mutation == "count":
        quality["per_seed"][0]["treatment_quality_count"] += 1
    elif mutation == "seed":
        quality["per_seed"][1]["seed"] = 0
    elif mutation == "direction":
        contrast["direction"] = "CT_minus_CE"
    else:
        report["runs"].append(copy.deepcopy(report["runs"][0]))
    with pytest.raises(ValueError):
        overview.validate_paired_contrasts(report, protocol)


def test_undefined_metric_withholds_all_pair_summary():
    report, protocol = paired_fixture()
    for run in report["runs"]:
        if run["config_id"] == "ce_t100" and run["seed"] == 1601:
            run["metrics"]["strict"]["diversity"] = None
    value = report["paired_contrasts"][0]["metrics"]["strict"]["diversity"]
    value["per_seed"][1].update(
        treatment_value=None, difference=None, status="undefined_metric"
    )
    value.update(
        complete=False, defined_seed_count=1, mean_difference=None, sample_sd=None
    )
    overview.validate_paired_contrasts(report, protocol)
    value["mean_difference"] = value["per_seed"][0]["difference"]
    with pytest.raises(ValueError, match="summary arithmetic"):
        overview.validate_paired_contrasts(report, protocol)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "run_seed",
        "rescore_seed",
        "entry",
        "controller_seeds",
        "controller_count",
        "controller_source",
    ],
)
def test_run_slot_is_bound_to_saved_seed_and_controller(mutation):
    protocol = {"seeds": [1600, 1601], "num_samples": 100}
    entry = dict(
        attempt_id="v9-ct-t100", candidate_id="ct", arm_id="CT", config_id="ct_t100"
    )
    run = dict(
        entry, seed=1600, requested_samples=100, independent_rescore={"seed": 1600}
    )
    summary = {"seed": 1600, "num_samples": 100, "git": {"commit": "source"}}
    receipt = {
        "identity": {
            "entry": copy.deepcopy(entry),
            "seeds": [1600, 1601],
            "num_samples": 100,
            "source": {"head": "source", "upstream": "source"},
        }
    }
    if mutation == "run_seed":
        run["seed"] = 1601
    elif mutation == "rescore_seed":
        run["independent_rescore"]["seed"] = 1601
    elif mutation == "entry":
        receipt["identity"]["entry"]["attempt_id"] = "another-attempt"
    elif mutation == "controller_seeds":
        receipt["identity"]["seeds"] = [0, 1]
    elif mutation == "controller_count":
        receipt["identity"]["num_samples"] = 64
    elif mutation == "controller_source":
        receipt["identity"]["source"]["upstream"] = "other-source"
    if mutation is None:
        overview.validate_run_slot(run, summary, receipt, entry, protocol)
    else:
        with pytest.raises(ValueError):
            overview.validate_run_slot(run, summary, receipt, entry, protocol)


def test_seed_aggregate_withholds_partly_undefined_diversity():
    runs = [
        {
            "metrics": {
                branch: dict.fromkeys(overview.METRICS, value)
                for branch in overview.BRANCHES
            }
        }
        for value in (0.4, 0.6)
    ]
    aggregate = {
        "metrics": {
            branch: {
                metric: {
                    "mean": 0.5,
                    "sample_sd": statistics.stdev([0.4, 0.6]),
                    "defined_seed_count": 2,
                }
                for metric in overview.METRICS
            }
            for branch in overview.BRANCHES
        }
    }
    runs[1]["metrics"]["strict"]["diversity"] = None
    aggregate["metrics"]["strict"]["diversity"] = {
        "mean": None,
        "sample_sd": None,
        "defined_seed_count": 1,
    }
    overview.validate_seed_aggregate(aggregate, runs)
    aggregate["metrics"]["strict"]["diversity"]["mean"] = 0.4
    with pytest.raises(ValueError, match="aggregation mismatch"):
        overview.validate_seed_aggregate(aggregate, runs)


def test_incomplete_v9_cannot_create_an_overview(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"status": "incomplete"}))
    with pytest.raises(ValueError, match="not a complete"):
        overview.load_v9(overview.Inputs(tmp_path), tmp_path, tmp_path)


def test_input_hashes_and_final_mutation_check(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_bytes(b"original\n")
    inputs = overview.Inputs(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        inputs.read(path, "receipt", "0" * 64)
    inputs.read(path, "receipt", overview.digest(b"original\n"))
    path.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="changed before publication"):
        inputs.manifest()


def test_existing_output_is_preserved(tmp_path):
    destination = tmp_path / "original_report"
    destination.mkdir()
    sentinel = destination / "study_overview.pdf"
    sentinel.write_bytes(b"historical PDF bytes")
    with pytest.raises(ValueError, match="must be fresh"):
        overview.publish_bundle(
            destination, {"study_overview.pdf": b"replacement"}, workspace=tmp_path
        )
    assert sentinel.read_bytes() == b"historical PDF bytes"
    assert list(destination.iterdir()) == [sentinel]


def test_configuration_csv_uses_distinct_study_rows_and_lf():
    records = [
        dict(
            study=study,
            arm_id="E",
            config_id="one",
            temperature=1.0,
            kernel="128P",
            samples=samples,
            seeds=seeds,
            metrics={
                branch: {
                    metric: {"mean": 0.5, "sample_sd": 0.1}
                    for metric in overview.METRICS
                }
                for branch in overview.BRANCHES
            },
        )
        for study, samples, seeds in (
            ("V5", 128, [1200, 1201]),
            ("V9", 200, [1600, 1601]),
        )
    ]
    payload = overview.configuration_csv(records)
    assert b"\r" not in payload
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    assert len(rows) == 4
    assert [(row["study"], row["samples"], row["seeds"]) for row in rows[::2]] == [
        ("V5", "128", "1200/1201"),
        ("V9", "200", "1600/1601"),
    ]
