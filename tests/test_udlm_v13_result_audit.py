"""Synthetic checks only; prospective V13 outcomes are not read by these tests."""

import copy
import json

import pytest

from scripts.exps.denovo import benchmark
from scripts.udlm import audit_v13_results as audit
from scripts.udlm import report_exploration as reporter


def synthetic_report():
    protocol = json.loads((audit.ROOT / audit.PROTOCOL).read_text())
    runs = []
    for entry in protocol["entries"]:
        sampling = benchmark.validate_sampling_config(
            benchmark.load_yaml_config(audit.ROOT / entry["config"])
        )
        for seed in audit.SEEDS:
            clean = entry["temperature_space"] == "x0_denoiser"
            if entry["arm_id"] == "E_CE":
                count = (
                    (40 if seed == 2100 else 60)
                    if clean
                    else (25 if seed == 2100 else 70)
                )
            else:
                count = (
                    (20 if seed == 2100 else 65)
                    if clean
                    else (15 if seed == 2100 else 90)
                )
            metrics = {
                branch: {
                    **dict.fromkeys(audit.evidence.METRICS, count / 100),
                    "quality_count": count,
                    "quality_denominator": 100,
                }
                for branch in audit.evidence.BRANCHES
            }
            receipt = dict(audit.TEMPERATURE_RECEIPT) if clean else {}
            runs.append(
                {
                    **entry,
                    "status": "completed",
                    "seed": seed,
                    "requested_samples": 100,
                    "metrics": metrics,
                    "config": {
                        "sampling": dict(sampling),
                        "sha256": entry["config_sha256"],
                    },
                    "checkpoint": {"sha256": entry["checkpoint_sha256"]},
                    "generation_protocol": {"nfe": 128, **receipt},
                    "independent_rescore": {"identity": {"generation": receipt}},
                }
            )
    return protocol, {
        "runs": runs,
        "paired_contrasts": reporter._paired_contrasts(protocol, runs),
    }


def test_declared_modes_and_both_signed_contrasts():
    protocol, report = synthetic_report()
    entries = {row["config_id"]: row for row in protocol["entries"]}
    for run in report["runs"]:
        audit.validate_mode(run, entries[run["config_id"]])
    audit.check_contrasts(report, protocol)
    primary, secondary = report["paired_contrasts"]
    assert primary["metrics"]["strict"]["quality"]["mean_difference"] == 0.025
    assert primary["metrics"]["strict"]["quality"]["sample_sd"] == pytest.approx(
        0.1767766952966369
    )
    assert (
        secondary["metrics"]["released_comparable"]["quality"]["mean_difference"]
        == -0.1
    )
    assert secondary["metrics"]["released_comparable"]["quality"][
        "sample_sd"
    ] == pytest.approx(0.21213203435596426)
    assert primary["direction"] == secondary["direction"] == audit.DIRECTION


@pytest.mark.parametrize(
    "mutation",
    ["mode", "order", "bridge", "boolean_bridge", "nfe", "gibbs", "checkpoint"],
)
def test_denoiser_mode_receipts_and_frozen_controls_reject_drift(mutation):
    protocol, report = synthetic_report()
    run = report["runs"][2]
    entry = next(
        entry for entry in protocol["entries"] if entry["config_id"] == run["config_id"]
    )
    if mutation == "mode":
        run["config"]["sampling"]["temperature_space"] = "raw_loo"
    elif mutation == "order":
        run["generation_protocol"]["temperature_application"] = "after_conversion"
    elif mutation in ("bridge", "boolean_bridge"):
        run["independent_rescore"]["identity"]["generation"][
            "reverse_bridge_temperature"
        ] = 0.5 if mutation == "bridge" else True
    elif mutation == "nfe":
        run["config"]["sampling"]["num_steps"] = 512
    elif mutation == "gibbs":
        run["config"]["sampling"]["gibbs_corrector"] = True
    else:
        run["checkpoint"]["sha256"] = "a" * 64
    with pytest.raises(ValueError):
        audit.validate_mode(run, entry)


@pytest.mark.parametrize("mutation", ["receipt", "canonical"])
def test_raw_control_retains_historical_omission(mutation):
    protocol, report = synthetic_report()
    run, entry = report["runs"][0], protocol["entries"][0]
    if mutation == "receipt":
        run["generation_protocol"].update(audit.TEMPERATURE_RECEIPT)
    else:
        run["config"]["sampling"]["temperature_space"] = "raw_loo"
    with pytest.raises(ValueError):
        audit.validate_mode(run, entry)


@pytest.mark.parametrize(
    "mutation",
    ["direction", "checkpoint", "prior", "temperature", "difference", "quality_count"],
)
def test_pair_identity_and_signed_arithmetic_are_bound(mutation):
    protocol, report = synthetic_report()
    contrast = report["paired_contrasts"][0]
    if mutation == "direction":
        contrast["direction"] = "MASK_minus_empirical"
    elif mutation == "checkpoint":
        protocol["entries"][1]["checkpoint_sha256"] = "a" * 64
    elif mutation == "prior":
        protocol["entries"][1]["prior_metadata_sha256"] = "a" * 64
    elif mutation == "temperature":
        report["runs"][2]["config"]["sampling"]["softmax_temp"] = 0.7
    elif mutation == "difference":
        contrast["metrics"]["strict"]["quality"]["mean_difference"] *= -1
    else:
        contrast["metrics"]["strict"]["quality"]["per_seed"][0][
            "control_quality_count"
        ] += 1
    with pytest.raises(ValueError):
        audit.check_contrasts(report, protocol)


def test_undefined_metric_withholds_both_paired_statistics():
    protocol, report = synthetic_report()
    report["runs"][0]["metrics"]["strict"]["diversity"] = None
    report["paired_contrasts"] = reporter._paired_contrasts(protocol, report["runs"])
    audit.check_contrasts(report, protocol)
    value = report["paired_contrasts"][0]["metrics"]["strict"]["diversity"]
    assert value["mean_difference"] is value["sample_sd"] is None
    altered = copy.deepcopy(report)
    altered["paired_contrasts"][0]["metrics"]["strict"]["diversity"][
        "mean_difference"
    ] = -0.1
    with pytest.raises(ValueError):
        audit.check_contrasts(altered, protocol)


def test_pending_report_rejects_before_reading_any_run(monkeypatch, tmp_path):
    class Inputs:
        def read_json(self, path, role, expected):
            if str(path).endswith(audit.PROTOCOL):
                return {"report_root": "output/never_read"}
            return {"status": "pending"}

    monkeypatch.setattr(audit.evidence, "Inputs", Inputs)
    monkeypatch.setattr(
        audit.reporter,
        "build_report",
        lambda *a, **kw: pytest.fail("must not inspect unfinished runs"),
    )
    with pytest.raises(ValueError, match="complete before"):
        audit.audit(tmp_path, "a" * 64)
