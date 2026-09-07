import copy

import pytest

from scripts.udlm.report_followthrough_resolution import paired_contrasts


def fixture_report():
    runs = []
    for arm in ("CT", "CE"):
        for nfe in (128, 512):
            for seed in (17300, 17301):
                quality = 0.4 if nfe == 128 else (0.3 if seed == 17300 else 0.6)
                runs.append({"arm_id": arm, "seed": seed, "status": "completed",
                             "generation_protocol": {"nfe": nfe},
                             "metrics": {branch: {metric: quality for metric in
                                          ("quality", "validity", "uniqueness", "diversity")}
                                         for branch in ("strict", "released_comparable")}})
    return {"status": "complete", "unexpected_run_directories": [], "runs": runs}


def test_paired_differences_keep_both_signs():
    result = paired_contrasts(fixture_report())
    assert len(result) == 16
    assert result[0]["seed_17300_difference"] == pytest.approx(-0.1)
    assert result[0]["seed_17301_difference"] == pytest.approx(0.2)
    assert result[0]["mean_difference"] == pytest.approx(0.05)


@pytest.mark.parametrize("problem", ["missing", "failed", "duplicate"])
def test_incomplete_or_duplicated_evidence_cannot_form_contrast(problem):
    report = fixture_report()
    if problem == "missing":
        report["runs"].pop()
    elif problem == "failed":
        report["runs"][0]["status"] = "failed"
    else:
        report["runs"].append(copy.deepcopy(report["runs"][0]))
    with pytest.raises(ValueError):
        paired_contrasts(report)


def test_undefined_metric_is_not_imputed_or_silently_dropped():
    report = fixture_report()
    report["runs"][0]["metrics"]["strict"]["diversity"] = None
    row = next(row for row in paired_contrasts(report)
               if (row["arm"], row["decoding"], row["metric"]) == ("CT", "strict", "diversity"))
    assert row["mean_difference"] is None
    assert row["sample_sd"] is None


def test_unscheduled_run_blocks_conclusion_even_with_complete_planned_pairs():
    report = fixture_report()
    report["unexpected_run_directories"] = ["extra/seed_17300"]
    with pytest.raises(ValueError):
        paired_contrasts(report)


def test_resolution_configs_change_only_number_of_steps():
    import hashlib
    import json
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "experiments/udlm/protocols/followthrough_resolution.json").read_text())
    for arm in ("CT", "CE"):
        entries = [entry for entry in protocol["entries"] if entry["arm_id"] == arm]
        configs = []
        for entry in entries:
            payload = (root / entry["config"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry["config_sha256"]
            config = yaml.safe_load(payload)
            assert config.pop("num_steps") == entry["nfe"]
            assert config["softmax_temp"] == 1.0
            configs.append(config)
        assert entries[0]["checkpoint_sha256"] == entries[1]["checkpoint_sha256"]
        assert configs[0] == configs[1]
