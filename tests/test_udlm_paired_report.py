"""Synthetic report fixtures; these counts are not molecular experiment results."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import pytest

from scripts.udlm import report_exploration as report


def synthetic_screen(root: Path):
    """Exercise the real report pipeline with explicitly synthetic rescore values."""
    protocol_path = root / "protocol.json"
    output = root / "output/engineering"
    protocol = {
        "study_id": "SYNTHETIC paired report preview - NOT V9 results",
        "entries": [],
        "seeds": [1600, 1601],
        "num_samples": 10,
        "design": {
            "objective_comparison": {
                "primary": {
                    "control_config": "ct_t100",
                    "treatment_config": "ce_t100",
                    "temperature": 1.0,
                },
                "secondary": {
                    "control_config": "ct_t050",
                    "treatment_config": "ce_t050",
                    "temperature": 0.5,
                },
            }
        },
    }
    quality_counts = {
        "ct_t100": ([5, 7], [4, 6]),
        "ce_t100": ([6, 6], [2, 5]),
        "ct_t050": ([6, 5], [5, 4]),
        "ce_t050": ([8, 8], [6, 5]),
    }

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    for index, config_id in enumerate(quality_counts):
        ce = config_id.startswith("ce")
        entry = {
            "attempt_id": f"synthetic-{config_id}",
            "candidate_id": config_id,
            "arm_id": "CE" if ce else "CT",
            "config_id": config_id,
            "checkpoint_sha256": ("a" if ce else "b") * 64,
            "config_sha256": str(index) * 64,
            "parameterization": "x0_denoiser" if ce else "raw_loo",
        }
        protocol["entries"].append(entry)
        for seed in protocol["seeds"]:
            directory = output / entry["attempt_id"] / f"seed_{seed}"
            summary = {
                "seed": seed,
                "num_samples": 10,
                "checkpoint": {"sha256": entry["checkpoint_sha256"]},
                "config": {
                    "sha256": entry["config_sha256"],
                    "sampling": {
                        "softmax_temp": 1.0 if config_id.endswith("100") else 0.5,
                        **({"parameterization": "x0_denoiser"} if ce else {}),
                    },
                },
                "run": {
                    "final_protocol_eligible": False,
                    "generation_protocol": {
                        "diffusion_type": "udlm",
                        "nfe": 128,
                    },
                },
                "git": {"commit": "c" * 40},
                "environment": {},
                "runtime_seconds": {"generation": 1.0},
                "metrics": {"producer_quality": "deliberately not metric truth"},
            }
            write(directory / "summary.json", summary)
            (directory / "raw_samples.csv").write_text(
                "sample_index,raw_model_text\n"
                + "".join(f"{i},SYNTHETIC\n" for i in range(10))
            )
    write(protocol_path, protocol)
    for entry in protocol["entries"]:
        artifacts = {
            path.relative_to(root)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in (output / entry["attempt_id"]).glob("seed_*/*")
        }
        write(
            output / "controller_receipts" / f"{entry['attempt_id']}.json",
            {
                "identity": {
                    "entry": entry,
                    "seeds": protocol["seeds"],
                    "num_samples": 10,
                    "protocol_sha256": hashlib.sha256(
                        protocol_path.read_bytes()
                    ).hexdigest(),
                    "source": {"head": "c" * 40, "upstream": "c" * 40},
                },
                "artifacts": artifacts,
                "return_code": 0,
            },
        )

    def rescore(summary_path, *, root):
        summary = json.loads(summary_path.read_text())
        config_id = summary_path.parent.parent.name.removeprefix("synthetic-")
        seed_index = protocol["seeds"].index(summary["seed"])
        metrics = {}
        for branch, counts in zip(report.BRANCHES, quality_counts[config_id]):
            count = counts[seed_index]
            metrics[branch] = {
                "validity": 1.0,
                "uniqueness": 1.0,
                "quality": count / 10,
                "quality_count": count,
                "quality_denominator": 10,
                "diversity": 0.7 + 0.1 * seed_index,
            }
        return {
            "status": "exact_match",
            "seed": summary["seed"],
            "metrics": metrics,
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "raw_samples_sha256": hashlib.sha256(
                summary_path.with_name("raw_samples.csv").read_bytes()
            ).hexdigest(),
            "identity": {"generation": {"inference_weights": {"source": "ema"}}},
        }

    result = report.build_report(protocol_path, output, root=root, rescore=rescore)
    result["caveats"].insert(
        0,
        "SYNTHETIC PREVIEW ONLY: mock rescore counts; these are not V9 molecular results.",
    )
    return result


@pytest.fixture
def paired(tmp_path):
    return synthetic_screen(tmp_path)


def _recompute(result):
    return report._paired_contrasts(result["protocol"]["configuration"], result["runs"])


def test_real_report_pipeline_includes_both_signed_quality_contrasts(paired):
    assert paired["status"] == "complete"
    assert paired["accounting"]["independently_rescored_requests"] == 80
    primary, secondary = paired["paired_contrasts"]
    assert [primary["contrast_id"], secondary["contrast_id"]] == [
        "primary",
        "secondary",
    ]
    repaired = primary["metrics"]["released_comparable"]["quality"]
    assert [row["difference"] for row in repaired["per_seed"]] == [0.1, -0.1]
    assert repaired["mean_difference"] == 0.0
    assert repaired["sample_sd"] == pytest.approx(math.sqrt(0.02))
    assert [row["control_quality_count"] for row in repaired["per_seed"]] == [5, 7]
    strict = primary["metrics"]["strict"]["quality"]
    assert [row["difference"] for row in strict["per_seed"]] == [-0.2, -0.1]
    assert strict["mean_difference"] == -0.15
    assert strict["sample_sd"] == pytest.approx(math.sqrt(0.005))
    assert (
        secondary["metrics"]["released_comparable"]["quality"]["mean_difference"]
        == 0.25
    )
    assert secondary["metrics"]["strict"]["quality"]["sample_sd"] == 0.0
    assert paired["superiority_established"] is False


def test_pairs_match_seed_identity_and_do_not_depend_on_run_order(paired):
    before = copy.deepcopy(paired["paired_contrasts"])
    paired["runs"].reverse()
    assert _recompute(paired) == before


@pytest.mark.parametrize("status", ["pending", "failed", "invalid"])
def test_missing_pair_withholds_mean_without_discarding_observed_negative_delta(
    paired, status
):
    run = next(
        run
        for run in paired["runs"]
        if run["config_id"] == "ce_t100" and run["seed"] == 1600
    )
    run.update(status=status, metrics=None)
    primary, secondary = _recompute(paired)
    summary = primary["metrics"]["released_comparable"]["quality"]
    assert summary["complete"] is False
    assert summary["defined_seed_count"] == 1
    assert summary["mean_difference"] is summary["sample_sd"] is None
    assert summary["per_seed"][0]["treatment_status"] == status
    assert summary["per_seed"][0]["difference"] is None
    assert summary["per_seed"][1]["difference"] == -0.1
    assert secondary["metrics"]["strict"]["quality"]["complete"] is True


def test_undefined_diversity_is_not_zero_and_does_not_hide_quality(paired):
    run = next(
        run
        for run in paired["runs"]
        if run["config_id"] == "ce_t100" and run["seed"] == 1601
    )
    run["metrics"]["strict"]["diversity"] = None
    primary = _recompute(paired)[0]
    diversity = primary["metrics"]["strict"]["diversity"]
    assert diversity["per_seed"][1]["status"] == "undefined_metric"
    assert diversity["mean_difference"] is diversity["sample_sd"] is None
    assert primary["metrics"]["strict"]["quality"]["mean_difference"] == -0.15


@pytest.mark.parametrize(
    "field,value",
    [("quality_count", 100), ("quality_denominator", 11), ("quality", 0.8)],
)
def test_inconsistent_certified_quality_counts_fail_closed(paired, field, value):
    paired["runs"][0]["metrics"]["strict"][field] = value
    with pytest.raises(ValueError, match="quality count/denominator"):
        _recompute(paired)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_config",
        "duplicate_config",
        "wrong_direction",
        "wrong_temperature",
        "duplicate_seed",
    ],
)
def test_ambiguous_or_mismatched_contrasts_are_rejected(paired, mutation):
    protocol = paired["protocol"]["configuration"]
    if mutation == "unknown_config":
        protocol["design"]["objective_comparison"]["primary"][
            "control_config"
        ] = "absent"
    elif mutation == "duplicate_config":
        protocol["entries"].append(copy.deepcopy(protocol["entries"][0]))
    elif mutation == "wrong_direction":
        protocol["entries"][0]["parameterization"] = "x0_denoiser"
    elif mutation == "wrong_temperature":
        paired["runs"][0]["config"]["sampling"]["softmax_temp"] = 0.5
    else:
        paired["runs"].append(copy.deepcopy(paired["runs"][0]))
    with pytest.raises(ValueError, match="paired"):
        _recompute(paired)


def test_paired_bundle_contains_all_metrics_and_preserves_previous_snapshot(
    paired, tmp_path
):
    from pypdf import PdfReader

    before = copy.deepcopy(paired)
    paths = report.write_report(paired, tmp_path / "reports/synthetic", root=tmp_path)
    document = json.loads(Path(paths["json"]).read_text())
    csv_payload = Path(paths["paired_contrasts_csv"]).read_bytes()
    assert (
        document["report_artifacts"]["paired_contrasts_csv_sha256"]
        == hashlib.sha256(csv_payload).hexdigest()
    )
    assert document["paired_contrasts"] == paired["paired_contrasts"]
    rows = list(csv.DictReader(io.StringIO(csv_payload.decode())))
    assert len(rows) == 2 * 2 * 4 * 3
    assert {row["contrast_id"] for row in rows} == {"primary", "secondary"}
    assert {row["metric"] for row in rows} == set(report.METRICS)
    strict_summary = next(
        row
        for row in rows
        if row["contrast_id"] == "primary"
        and row["branch"] == "strict"
        and row["metric"] == "quality"
        and row["row_type"] == "summary"
    )
    assert strict_summary["mean_difference"] == "-0.15"
    text = " ".join(page.extract_text() for page in PdfReader(paths["pdf"]).pages)
    for expected in [
        "Predeclared paired CE minus CT contrasts",
        "primary | temperature 1.0",
        "secondary | temperature 0.5",
        "-0.150000",
        "-0.100000",
        "0.141421",
        "SYNTHETIC",
        "not a confidence interval",
    ]:
        assert expected in text
    snapshots = {key: Path(path).read_bytes() for key, path in paths.items()}
    with pytest.raises((FileExistsError, ValueError)):
        report.write_report(paired, tmp_path / "reports/synthetic", root=tmp_path)
    assert snapshots == {key: Path(path).read_bytes() for key, path in paths.items()}
    assert paired == before


def test_legacy_objective_metadata_does_not_invent_contrasts(paired):
    protocol = paired["protocol"]["configuration"]
    protocol["design"]["objective_comparison"] = {"primary_temperature": 1.0}
    assert _recompute(paired) == []
