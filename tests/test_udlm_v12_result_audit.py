import base64
import copy
import hashlib
import json
import struct

import pytest

from scripts.udlm import audit_v12_results as audit


def encoded(raw):
    return {
        "encoding": "rfc4648_base64",
        "compression": "none",
        "decoded_byte_count": len(raw),
        "decoded_sha256": hashlib.sha256(raw).hexdigest(),
        "data_base64": base64.b64encode(raw).decode(),
    }


def token_fixture():
    value = {
        "rows": 2,
        "columns": 5,
        "control_token_ids": {"unk": 0, "bos": 1, "eos": 2, "pad": 3, "mask": 4},
        "model_vocab_size": 1880,
        "control_token_counts": {
            "final_sampled_editable_positions": {
                "pad": 0,
                "bos": 0,
                "eos": 0,
                "unk": 0,
                "mask": 1,
            }
        },
    }
    for key, ids in (
        ("sampler_input_ids", [1, 4, 4, 2, 3, 1, 4, 2, 3, 3]),
        ("final_sampled_ids", [1, 4, 5, 2, 3, 1, 6, 2, 3, 3]),
    ):
        value[key] = {
            **encoded(struct.pack("<10H", *ids)),
            "dtype": "uint16",
            "byte_order": "little",
            "array_order": "C",
            "element_count": 10,
        }
    value["editable_mask"] = {
        **encoded(bytes([0b01100010, 0])),
        "packing": "one_bit_per_position",
        "bit_order": "msb0",
        "array_order": "C",
        "logical_bit_count": 10,
        "unused_tail_bit_count": 6,
    }
    return value


def test_editable_bit_denominator_excludes_framing_padding():
    result = audit.token_counts(token_fixture())
    assert result["editable_positions"] == 3
    assert result["final_editable_mask"] == 1
    assert result["mask_fraction_of_editable_positions"] == 1 / 3
    assert result["rows_with_editable_mask"] == 1
    assert result["row_fraction_with_editable_mask"] == 0.5


@pytest.mark.parametrize(
    "mutation", ["hash", "tail", "denominator", "count", "immutable"]
)
def test_rejects_token_audit_corruption(mutation):
    value = token_fixture()
    if mutation == "hash":
        value["final_sampled_ids"]["decoded_sha256"] = "0" * 64
    elif mutation == "tail":
        value["editable_mask"].update(encoded(bytes([0b01100010, 1])))
    elif mutation == "denominator":
        value["editable_mask"].update(encoded(bytes([0b11100010, 0])))
    elif mutation == "count":
        value["control_token_counts"]["final_sampled_editable_positions"]["mask"] = 0
    else:
        value["final_sampled_ids"].update(
            encoded(struct.pack("<10H", 1, 4, 5, 2, 3, 1, 6, 2, 5, 3))
        )
    with pytest.raises(ValueError):
        audit.token_counts(value)


def test_metric_denominators_and_deduplication_are_branch_specific():
    rows = []
    for smiles, unique, quality, qed in (
        ("A", True, True, 0.7),
        ("A", False, False, 0.7),
        ("B", True, False, 0.2),
        ("", False, False, ""),
    ):
        rows.append(
            {
                "released_smiles": smiles,
                "released_is_first_unique": str(unique),
                "released_quality_counted": str(quality),
                "released_qed": qed,
                "released_sa": 3,
            }
        )
    metrics = {
        "valid_count": 3,
        "unique_count": 2,
        "quality_count": 1,
        "validity_denominator": 4,
        "uniqueness_denominator": 3,
        "quality_denominator": 4,
        "diversity_input_count": 2,
        "validity": 0.75,
        "uniqueness": 2 / 3,
        "quality": 0.25,
        "diversity": 0.6,
        "quality_thresholds": {"qed_min_inclusive": 0.6, "sa_max_inclusive": 4},
    }
    result = audit.metric_counts(rows, metrics, "released_comparable")
    assert result["quality_denominator"] == 4
    assert result["uniqueness_denominator"] == 3
    metrics["uniqueness_denominator"] = 4
    with pytest.raises(ValueError, match="denominator"):
        audit.metric_counts(rows, metrics, "released_comparable")


def synthetic_report():
    from scripts.udlm import report_exploration as report

    protocol = json.loads((audit.FEATURE_ROOT / audit.PROTOCOL).read_text())
    runs = []
    for entry in protocol["entries"]:
        for seed in protocol["seeds"]:
            mask = entry["prior_variant"] == "mask_rich_empirical"
            amount = (
                (25 if seed == 2000 else 70)
                if not mask
                else (40 if seed == 2000 else 60)
            )
            metrics = {
                branch: {
                    **dict.fromkeys(audit.evidence.METRICS, amount / 100),
                    "quality_count": amount,
                    "quality_denominator": 100,
                }
                for branch in audit.evidence.BRANCHES
            }
            runs.append(
                {
                    **entry,
                    "seed": seed,
                    "requested_samples": 100,
                    "status": "completed",
                    "metrics": metrics,
                    "config": {
                        "sampling": {
                            **{
                                k: entry[k]
                                for k in (
                                    "parameterization",
                                    "prior_variant",
                                    "prior_metadata_sha256",
                                )
                            },
                            "softmax_temp": 1.0
                            if entry["config_id"].endswith("t100")
                            else 0.5,
                        }
                    },
                }
            )
    return protocol, {
        "runs": runs,
        "paired_contrasts": report._paired_contrasts(protocol, runs),
    }


def test_both_signed_prior_contrasts_all_metrics():
    protocol, report = synthetic_report()
    audit.check_contrasts(report, protocol)
    for contrast in report["paired_contrasts"]:
        result = contrast["metrics"]["strict"]["quality"]
        assert result["mean_difference"] == 0.025
        assert result["sample_sd"] == pytest.approx(0.1767766952966369)
    report["paired_contrasts"][0]["metrics"]["strict"]["quality"][
        "mean_difference"
    ] *= -1
    with pytest.raises(ValueError, match="sample SD"):
        audit.check_contrasts(report, protocol)


def test_undefined_pair_withholds_mean_and_sample_sd():
    from scripts.udlm import report_exploration as reporter

    protocol, report = synthetic_report()
    report["runs"][0]["metrics"]["strict"]["diversity"] = None
    report["paired_contrasts"] = reporter._paired_contrasts(protocol, report["runs"])
    audit.check_contrasts(report, protocol)
    value = report["paired_contrasts"][0]["metrics"]["strict"]["diversity"]
    assert value["mean_difference"] is value["sample_sd"] is None
    changed = copy.deepcopy(report)
    changed["paired_contrasts"][0]["metrics"]["strict"]["diversity"][
        "mean_difference"
    ] = -0.1
    with pytest.raises(ValueError, match="sample SD"):
        audit.check_contrasts(changed, protocol)


@pytest.mark.parametrize("mutation", ["attempt", "temperature", "quality_count"])
def test_paired_labels_and_counts_bind_original_runs(mutation):
    protocol, report = synthetic_report()
    if mutation == "attempt":
        report["paired_contrasts"][0]["control_attempt_id"] = "another-attempt"
    elif mutation == "temperature":
        report["runs"][0]["config"]["sampling"]["softmax_temp"] = 0.7
    else:
        report["paired_contrasts"][0]["metrics"]["strict"]["quality"]["per_seed"][0][
            "control_quality_count"
        ] += 1
    with pytest.raises(ValueError):
        audit.check_contrasts(report, protocol)


def execution_fixture():
    probe = {
        "uuid": "GPU-example",
        "index": 6,
        "name": "CPU fixture",
        "memory_total_mib": 49000,
        "memory_used_mib": 9000,
        "utilization_percent": 9,
        "compute_processes": [{"pid": 123}],
    }
    policy = {
        "requested_gpu_count": 2,
        "max_utilization_percent": 10,
        "utilization_comparison": "strictly_less_than",
        "min_free_memory_mib": 30000,
        "active_compute_processes_allowed": True,
    }
    logical = {"logical_index": 0}
    snapshot = {"physical_gpu_at_final_uuid_probe": probe, "policy": policy}
    launch = {
        "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT": json.dumps(snapshot),
        "CUDA_VISIBLE_DEVICES": "GPU-example",
        "GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX": "6",
    }
    summary = {
        "environment": {"launch_environment": launch, "cuda_device": logical},
        "runtime_seconds": {
            "generation": 12.0,
            "model_sampling_and_tokenizer": 10.0,
            "released_postprocessing": 2.0,
        },
    }
    run = {
        "gpu_mapping": {
            "cuda_visible_devices": "GPU-example",
            "physical_index": "6",
            "logical_device": logical,
            "final_probe": probe,
            "launch_policy": policy,
        }
    }
    return run, summary


def test_runtime_definition_and_dynamic_mapping_are_explicit():
    run, summary = execution_fixture()
    record = audit.execution_record(run, summary)
    assert record["runtime_seconds"]["generation"] == 12
    assert record["gpu_mapping"]["physical_index"] == 6
    assert record["gpu_mapping"]["logical_index"] == 0
    assert record["gpu_mapping"]["final_probe_free_memory_mib"] == 40000
    assert record["gpu_mapping"]["existing_compute_process_count"] == 1


@pytest.mark.parametrize("mutation", ["runtime", "mapping", "threshold"])
def test_runtime_mapping_and_launch_boundary_reject_drift(mutation):
    run, summary = execution_fixture()
    if mutation == "runtime":
        summary["runtime_seconds"]["generation"] = 13
    elif mutation == "mapping":
        run["gpu_mapping"]["physical_index"] = "0"
    else:
        snapshot = json.loads(
            summary["environment"]["launch_environment"][
                "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"
            ]
        )
        snapshot["physical_gpu_at_final_uuid_probe"]["utilization_percent"] = 10
        summary["environment"]["launch_environment"][
            "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"
        ] = json.dumps(snapshot)
        run["gpu_mapping"]["final_probe"]["utilization_percent"] = 10
    with pytest.raises(ValueError):
        audit.execution_record(run, summary)
