import statistics

import pytest

from scripts.udlm import analyze_v10_resolution as analysis


def test_exact_resolution_deltas_and_ratio_of_runs():
    difference = analysis.paired_summary(
        {1700: (0.25, 0.4), 1701: (0.7, 0.6)}, [1700, 1701]
    )
    assert difference["operation"] == "512_minus_128"
    assert difference["mean"] == pytest.approx(0.025)
    assert difference["sample_sd"] == pytest.approx(statistics.stdev([0.15, -0.1]))
    timing = analysis.paired_summary(
        {1700: (10, 30), 1701: (20, 100)}, [1700, 1701], ratio=True
    )
    assert timing["operation"] == "512_over_128"
    assert timing["mean"] == 4.0
    assert timing["mean"] != (30 + 100) / (10 + 20)


@pytest.mark.parametrize(
    "pairs", [{1700: (1.0, 2.0)}, {1700: (1.0, 2.0), 1701: (None, 2.0)}]
)
def test_missing_or_undefined_seed_withholds_summary(pairs):
    value = analysis.paired_summary(pairs, [1700, 1701])
    assert value["mean"] is value["sample_sd"] is None
    assert value["complete"] is False
    assert value["defined_seed_count"] == 1
    assert value["expected_seed_count"] == 2
    assert value["per_seed"][1]["status"] == "missing_or_undefined_pair"


def test_runtime_denominator_and_components_are_validated():
    with pytest.raises(ValueError, match="denominator"):
        analysis.paired_summary({1700: (0.0, 2.0)}, [1700], ratio=True)
    with pytest.raises(ValueError, match="finite"):
        analysis.paired_summary({1700: (1.0, float("nan"))}, [1700])
    analysis.validate_runtime(
        {
            "generation": 12.0,
            "model_sampling_and_tokenizer": 10.0,
            "released_postprocessing": 2.0,
        }
    )
    with pytest.raises(ValueError, match="do not add up"):
        analysis.validate_runtime(
            {
                "generation": 13.0,
                "model_sampling_and_tokenizer": 10.0,
                "released_postprocessing": 2.0,
            }
        )


def contrast_fixture():
    protocol = {
        "seeds": [1700, 1701],
        "design": {
            "resolution_comparison": {
                "contrasts": [
                    {
                        "contrast_id": "ct_512_minus_128",
                        "method": "CT",
                        "control_config": "ct128",
                        "treatment_config": "ct512",
                    },
                    {
                        "contrast_id": "ce_512_minus_128",
                        "method": "CE",
                        "control_config": "ce128",
                        "treatment_config": "ce512",
                    },
                ]
            }
        },
    }
    report, counts = {"runs": []}, {}
    for method in ("CT", "CE"):
        for nfe in (128, 512):
            config = f"{method.lower()}{nfe}"
            for seed in protocol["seeds"]:
                report["runs"].append(
                    {
                        "config_id": config,
                        "seed": seed,
                        "arm_id": method,
                        "checkpoint": {"sha256": method},
                        "config": {"sampling": {"num_steps": nfe, "softmax_temp": 0.5}},
                        "metrics": {
                            branch: dict.fromkeys(
                                analysis.METRICS, 0.2 if nfe == 128 else 0.3
                            )
                            for branch in analysis.BRANCHES
                        },
                        "runtime_seconds": {"generation": nfe / 16},
                    }
                )
                counts[(config, seed)] = dict.fromkeys(
                    analysis.LEXICAL_COUNTS, 10 if nfe == 128 else 5
                )
    return report, protocol, counts


def test_within_method_contrasts_include_all_metrics_and_lexical_counts():
    report, protocol, counts = contrast_fixture()
    contrasts = analysis.analyze(report, protocol, counts)
    assert [row["contrast_id"] for row in contrasts] == [
        "ct_512_minus_128",
        "ce_512_minus_128",
    ]
    for contrast in contrasts:
        assert contrast["metrics"]["strict"]["quality"]["mean"] == pytest.approx(0.1)
        assert contrast["generation_runtime_ratio"]["mean"] == 4
        assert contrast["lexical_counts"]["odd_ring"]["mean"] == -5


@pytest.mark.parametrize("mutation", ["checkpoint", "method", "temperature"])
def test_rejects_pairs_that_change_more_than_resolution(mutation):
    report, protocol, counts = contrast_fixture()
    row = next(row for row in report["runs"] if row["config_id"] == "ct512")
    if mutation == "checkpoint":
        row["checkpoint"]["sha256"] = "another"
    elif mutation == "method":
        row["arm_id"] = "CE"
    else:
        row["config"]["sampling"]["softmax_temp"] = 1.0
    with pytest.raises(ValueError):
        analysis.analyze(report, protocol, counts)


def test_missing_resolution_run_withholds_all_metrics():
    report, protocol, counts = contrast_fixture()
    report["runs"] = [
        row
        for row in report["runs"]
        if (row["config_id"], row["seed"]) != ("ct512", 1701)
    ]
    counts.pop(("ct512", 1701))
    contrast = analysis.analyze(report, protocol, counts)[0]
    assert contrast["metrics"]["strict"]["quality"]["mean"] is None
    assert contrast["generation_runtime_ratio"]["mean"] is None
    assert contrast["lexical_counts"]["odd_ring"]["mean"] is None
