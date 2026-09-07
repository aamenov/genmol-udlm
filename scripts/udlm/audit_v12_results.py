"""Local V12 evidence cross-check. Requires a completed, explicitly pinned report.

No model loads, chemistry re-evaluation, sampling, or canonical writes. Output is
JSON to stdout; this supplements the already independent chemistry rescorer.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import sys

FEATURE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FEATURE_ROOT))
from scripts.udlm import generate_study_overview_v9 as evidence  # noqa: E402

PROTOCOL = "experiments/udlm/protocols/engineering_v12_mask_prior.json"
PROTOCOL_SHA = "f4ea756fabb9e5133c72f302e8398bcbb8d0eabaa81f183b017af8c19b02cadd"
require = evidence.require
same = evidence.same_number


def decoded(record, size):
    require(record["encoding"] == "rfc4648_base64", "Unexpected token encoding")
    require(record["compression"] == "none", "Unexpected token compression")
    raw = base64.b64decode(record["data_base64"], validate=True)
    require(
        base64.b64encode(raw).decode() == record["data_base64"], "Noncanonical base64"
    )
    require(len(raw) == record["decoded_byte_count"] == size, "Token byte size differs")
    require(
        hashlib.sha256(raw).hexdigest() == record["decoded_sha256"],
        "Token hash differs",
    )
    return raw


def token_counts(audit):
    rows, columns = audit["rows"], audit["columns"]
    require(
        type(rows) is int
        and type(columns) is int
        and 0 < rows <= 1000
        and 0 < columns <= 256,
        "Invalid token shape",
    )
    n = rows * columns
    arrays = []
    for name in ("sampler_input_ids", "final_sampled_ids"):
        record = audit[name]
        require(
            record["dtype"] == "uint16"
            and record["byte_order"] == "little"
            and record["array_order"] == "C"
            and record["element_count"] == n,
            "Unexpected ID array layout",
        )
        arrays.append(struct.unpack(f"<{n}H", decoded(record, 2 * n)))
    before, after = arrays
    record = audit["editable_mask"]
    require(
        record["logical_bit_count"] == n
        and record["packing"] == "one_bit_per_position"
        and record["bit_order"] == "msb0"
        and record["array_order"] == "C",
        "Unexpected editable bitmask layout",
    )
    packed = decoded(record, (n + 7) // 8)
    tail = len(packed) * 8 - n
    require(
        record["unused_tail_bit_count"] == tail
        and (not tail or packed[-1] & ((1 << tail) - 1) == 0),
        "Nonzero editable tail padding",
    )
    editable = [bool(packed[i // 8] & (128 >> (i % 8))) for i in range(n)]
    ids = audit["control_token_ids"]
    require(
        ids == {"unk": 0, "bos": 1, "eos": 2, "pad": 3, "mask": 4},
        "Unexpected control token IDs",
    )
    require(
        audit["model_vocab_size"] == 1880
        and all(0 <= x < 1880 for values in arrays for x in values),
        "Token ID outside model alphabet",
    )
    require(
        editable == [x == ids["mask"] for x in before],
        "Editable denominator is not sampler-input MASK positions",
    )
    require(
        all(x == y for x, y, edit in zip(before, after, editable) if not edit),
        "Immutable token changed",
    )
    for row in range(rows):
        start = row * columns
        values = list(before[start : start + columns])
        end = values.index(ids["eos"])
        require(
            end > 1
            and values == [1] + [4] * (end - 1) + [2] + [3] * (columns - end - 1),
            "Sampler input framing differs",
        )
    editable_counts = {
        name: sum(edit and value == token for value, edit in zip(after, editable))
        for name, token in ids.items()
    }
    require(
        editable_counts
        == audit["control_token_counts"]["final_sampled_editable_positions"],
        "Recorded editable control counts differ from exact arrays",
    )
    denominator = sum(editable)
    mask_rows = sum(
        any(
            editable[i] and after[i] == 4
            for i in range(row * columns, (row + 1) * columns)
        )
        for row in range(rows)
    )
    return {
        "requests": rows,
        "editable_positions": denominator,
        "final_editable_mask": editable_counts["mask"],
        "mask_fraction_of_editable_positions": editable_counts["mask"] / denominator,
        "rows_with_editable_mask": mask_rows,
        "row_fraction_with_editable_mask": mask_rows / rows,
        "all_editable_control_counts": editable_counts,
    }


def metric_counts(rows, metrics, branch):
    prefix = "released" if branch == "released_comparable" else "strict"
    valid = unique = quality = 0
    seen = set()
    for row in rows:
        smiles = row[f"{prefix}_smiles"]
        is_unique = bool(smiles) and smiles not in seen
        if smiles:
            valid += 1
            seen.add(smiles)
        counted = (
            is_unique
            and float(row[f"{prefix}_qed"]) >= 0.6
            and float(row[f"{prefix}_sa"]) <= 4
        )
        require(
            row[f"{prefix}_is_first_unique"] == str(is_unique),
            "Saved uniqueness flag differs from per-seed identity",
        )
        require(
            row[f"{prefix}_quality_counted"] == str(counted),
            "Saved quality flag differs from thresholds",
        )
        unique += is_unique
        quality += counted
    expected = {
        "valid_count": valid,
        "unique_count": unique,
        "quality_count": quality,
        "validity_denominator": len(rows),
        "uniqueness_denominator": valid,
        "quality_denominator": len(rows),
        "diversity_input_count": unique,
    }
    require(
        all(metrics[key] == value for key, value in expected.items()),
        "Metric numerator/denominator differs from raw rows",
    )
    require(
        same(metrics["validity"], valid / len(rows))
        and same(metrics["quality"], quality / len(rows))
        and same(metrics["uniqueness"], unique / valid if valid else None),
        "Metric fraction differs",
    )
    require(
        metrics["quality_thresholds"]
        == {"qed_min_inclusive": 0.6, "sa_max_inclusive": 4.0},
        "Quality thresholds differ",
    )
    require(
        (metrics["diversity"] is None) == (unique == 0), "Diversity definedness differs"
    )
    return expected


def check_contrasts(report, protocol):
    runs = {(run["config_id"], run["seed"]): run for run in report["runs"]}
    entries = {entry["config_id"]: entry for entry in protocol["entries"]}
    require(
        [c["contrast_id"] for c in report["paired_contrasts"]]
        == ["primary", "secondary"],
        "Missing a declared prior contrast",
    )
    for contrast in report["paired_contrasts"]:
        declared = protocol["design"]["prior_comparison"][contrast["contrast_id"]]
        require(
            contrast["direction"] == "MASK_minus_empirical"
            and all(contrast[k] == declared[k] for k in declared)
            and contrast["expected_seeds"] == [2000, 2001]
            and contrast["summary_policy"]
            == "all_declared_pairs_required_for_each_metric"
            and all(
                contrast[role + "_attempt_id"]
                == entries[contrast[role + "_config"]]["attempt_id"]
                for role in ("control", "treatment")
            ),
            "Prior contrast identity differs",
        )
        require(
            all(
                runs[(contrast[role + "_config"], seed)]["config"]["sampling"][
                    "softmax_temp"
                ]
                == declared["temperature"]
                for role in ("control", "treatment")
                for seed in (2000, 2001)
            ),
            "Observed paired temperature differs",
        )
        for branch in evidence.BRANCHES:
            for metric in evidence.METRICS:
                result = contrast["metrics"][branch][metric]
                require(
                    [row["seed"] for row in result["per_seed"]] == [2000, 2001],
                    "Paired seed slot differs",
                )
                differences = []
                for row in result["per_seed"]:
                    values = [
                        runs[(contrast[key], row["seed"])]["metrics"][branch][metric]
                        for key in ("control_config", "treatment_config")
                    ]
                    difference = (
                        values[1] - values[0]
                        if all(value is not None for value in values)
                        else None
                    )
                    require(
                        row["requested_samples"] == 100
                        and row["control_status"]
                        == row["treatment_status"]
                        == "completed",
                        "Paired acceptance differs",
                    )
                    require(
                        same(row["control_value"], values[0])
                        and same(row["treatment_value"], values[1])
                        and same(row["difference"], difference),
                        "MASK-minus-empirical arithmetic differs",
                    )
                    require(
                        row["status"]
                        == (
                            "completed"
                            if difference is not None
                            else "undefined_metric"
                        ),
                        "Paired metric status differs",
                    )
                    if metric == "quality":
                        require(
                            all(
                                row[role + "_quality_count"]
                                == runs[(contrast[role + "_config"], row["seed"])][
                                    "metrics"
                                ][branch]["quality_count"]
                                for role in ("control", "treatment")
                            ),
                            "Paired quality count differs",
                        )
                    if difference is not None:
                        differences.append(difference)
                complete = len(differences) == 2
                require(
                    result["complete"] is complete
                    and result["defined_seed_count"] == len(differences)
                    and result["expected_seed_count"] == 2,
                    "Paired completeness differs",
                )
                require(
                    same(
                        result["mean_difference"],
                        statistics.mean(differences) if complete else None,
                    )
                    and same(
                        result["sample_sd"],
                        statistics.stdev(differences) if complete else None,
                    ),
                    "Paired mean/sample SD differs",
                )


def execution_record(run, summary):
    runtime = summary["runtime_seconds"]
    require(
        all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in runtime.values()
        ),
        "Runtime must be finite and nonnegative",
    )
    require(
        same(
            runtime["generation"],
            runtime["model_sampling_and_tokenizer"]
            + runtime["released_postprocessing"],
        ),
        "Generation runtime components differ",
    )
    environment = summary["environment"]
    launch = environment["launch_environment"]
    snapshot = json.loads(launch["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"])
    probe, policy = snapshot["physical_gpu_at_final_uuid_probe"], snapshot["policy"]
    expected_mapping = {
        "cuda_visible_devices": launch["CUDA_VISIBLE_DEVICES"],
        "physical_index": launch["GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX"],
        "logical_device": environment["cuda_device"],
        "final_probe": probe,
        "launch_policy": policy,
    }
    require(
        run["gpu_mapping"] == expected_mapping,
        "Report GPU mapping differs from bound summary",
    )
    free = probe["memory_total_mib"] - probe["memory_used_mib"]
    require(
        probe["uuid"] == launch["CUDA_VISIBLE_DEVICES"]
        and str(probe["index"]) == launch["GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX"]
        and environment["cuda_device"]["logical_index"] == 0,
        "Physical/logical GPU mapping differs",
    )
    require(
        policy["requested_gpu_count"] == 2
        and policy["max_utilization_percent"] == 10
        and policy["utilization_comparison"] == "strictly_less_than"
        and policy["min_free_memory_mib"] == 30000
        and policy["active_compute_processes_allowed"] is True
        and 0 <= probe["utilization_percent"] < 10
        and free >= 30000,
        "Final launch resource policy differs",
    )
    return {
        "runtime_seconds": runtime,
        "gpu_mapping": {
            "uuid": probe["uuid"],
            "physical_index": probe["index"],
            "logical_index": 0,
            "name": probe["name"],
            "final_probe_utilization_percent": probe["utilization_percent"],
            "final_probe_free_memory_mib": free,
            "existing_compute_process_count": len(probe["compute_processes"]),
            "requested_gpu_count": policy["requested_gpu_count"],
        },
    }


def audit(root, report_sha, source):
    inputs = evidence.Inputs()
    protocol = inputs.read_json(root / PROTOCOL, "Frozen V12 protocol", PROTOCOL_SHA)
    report_dir = root / protocol["report_root"]
    report = inputs.read_json(
        report_dir / "report.json", "Completed V12 independent report", report_sha
    )
    require(
        report["status"] == "complete"
        and not report["unexpected_run_directories"]
        and report["superiority_established"] is False,
        "V12 is not a completed engineering report",
    )
    require(
        report["accounting"]
        == {
            "scheduled_runs": 8,
            "scheduled_requests": 800,
            "independently_rescored_requests": 800,
            "status_counts": {"completed": 8},
        },
        "V12 acceptance accounting differs",
    )
    require(
        report["protocol"]["sha256"] == PROTOCOL_SHA
        and report["protocol"]["configuration"] == protocol,
        "Frozen protocol differs",
    )
    for name, key in (
        ("report.csv", "csv_sha256"),
        ("report.pdf", "pdf_sha256"),
        ("paired_contrasts.csv", "paired_contrasts_csv_sha256"),
    ):
        inputs.read(
            report_dir / name,
            "Original V12 companion artifact",
            report["report_artifacts"][key],
        )
    entries = {entry["config_id"]: entry for entry in protocol["entries"]}
    require(
        len(report["runs"]) == 8
        and {(run["config_id"], run["seed"]) for run in report["runs"]}
        == {(key, seed) for key in entries for seed in (2000, 2001)},
        "Unexpected/duplicate V12 run slot",
    )
    for entry in entries.values():
        inputs.read(
            root / entry["config"], "Frozen inference YAML", entry["config_sha256"]
        )
    rows_out = []
    for run in report["runs"]:
        entry = entries[run["config_id"]]
        require(
            run["status"] == "completed"
            and run["requested_samples"] == run["raw_row_count"] == 100
            and set(run["artifacts"]) == {"summary.json", "raw_samples.csv"},
            "Incomplete V12 run",
        )
        summary = inputs.reference(
            root, run["artifacts"]["summary.json"], "Original seed summary"
        )
        raw = inputs.read(
            root / run["artifacts"]["raw_samples.csv"]["relative_path"],
            "Original raw rows",
            run["artifacts"]["raw_samples.csv"]["sha256"],
        )
        controller = inputs.reference(
            root, run["controller"]["artifact"], "Attempt controller receipt"
        )
        require(
            controller == run["controller"]["receipt"]
            and controller["return_code"] == 0
            and run["controller"]["status"] == "completed"
            and controller["identity"]["protocol_sha256"] == PROTOCOL_SHA,
            "Controller not successful/protocol-bound",
        )
        evidence.validate_run_slot(run, summary, controller, entry, protocol)
        require(
            summary["git"]["commit"] == source
            and summary["run"]["final_protocol_eligible"] is False,
            "Generation source/final eligibility differs",
        )
        for key, summary_key in (
            ("config", "config"),
            ("checkpoint", "checkpoint"),
            ("source", "git"),
            ("runtime_seconds", "runtime_seconds"),
        ):
            require(
                run[key] == summary[summary_key], "Report changed original run identity"
            )
        require(
            run["checkpoint"]["sha256"] == entry["checkpoint_sha256"]
            and run["config"]["sha256"] == entry["config_sha256"],
            "Checkpoint/configuration binding differs",
        )
        sampling = run["config"]["sampling"]
        require(
            all(
                sampling[k] == entry[k]
                for k in ("parameterization", "prior_variant", "prior_metadata_sha256")
            )
            and sampling["num_steps"] == 128
            and not sampling.get("gibbs_corrector", False),
            "CE/prior/NFE identity differs",
        )
        require(
            run["generation_protocol"] == summary["run"]["generation_protocol"]
            and run["generation_protocol"]["nfe"] == 128,
            "Generation NFE protocol differs",
        )
        acceptance = run["independent_rescore"]
        require(
            acceptance["status"] == "exact_match"
            and acceptance["summary_sha256"]
            == run["artifacts"]["summary.json"]["sha256"]
            and acceptance["raw_samples_sha256"] == evidence.digest(raw),
            "Independent acceptance binds different bytes",
        )
        require(
            run["metrics"] == acceptance["metrics"],
            "Metrics differ from independent acceptance",
        )
        run_dir = root / run["run_directory"]
        require(
            not (run_dir / "failure_receipt.json").exists(),
            "Unexpected failure receipt",
        )
        for artifact in run["artifacts"].values():
            require(
                controller["artifacts"][artifact["relative_path"]]
                == artifact["sha256"],
                "Controller binds different seed output",
            )
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        require(len(rows) == 100, "Raw row count differs")
        denominators = {
            branch: metric_counts(rows, run["metrics"][branch], branch)
            for branch in evidence.BRANCHES
        }
        tokens = token_counts(summary["sampled_token_control_audit"])
        require(tokens["requests"] == 100, "Token audit request count differs")
        control_audit = acceptance["identity"]["sampled_token_control_audit"]
        require(
            control_audit["exact_batch_decode_match"] is True
            and control_audit["control_token_counts"][
                "final_sampled_editable_positions"
            ]
            == tokens["all_editable_control_counts"],
            "Independent token acceptance differs",
        )
        for key in ("sampler_input_ids", "final_sampled_ids", "editable_mask"):
            require(
                control_audit[key + "_sha256"]
                == summary["sampled_token_control_audit"][key]["decoded_sha256"],
                "Independent token hash differs",
            )
        rows_out.append(
            {
                "config_id": run["config_id"],
                "seed": run["seed"],
                "denominators": denominators,
                "tokens": tokens,
                "metrics": run["metrics"],
                **execution_record(run, summary),
            }
        )
    require(
        len(report["aggregates"]) == 4
        and {a["config_id"] for a in report["aggregates"]} == set(entries),
        "Aggregate coverage differs",
    )
    for aggregate in report["aggregates"]:
        matching = [
            run for run in report["runs"] if run["config_id"] == aggregate["config_id"]
        ]
        evidence.validate_seed_aggregate(aggregate, matching)
    check_contrasts(report, protocol)
    total_control_counts = {
        key: sum(run["tokens"]["all_editable_control_counts"][key] for run in rows_out)
        for key in ("unk", "bos", "eos", "pad", "mask")
    }
    return {
        "status": "independent_evidence_checks_passed",
        "protocol_sha256": PROTOCOL_SHA,
        "report_sha256": report_sha,
        "generation_source": source,
        "accounting": report["accounting"],
        "runs": rows_out,
        "aggregates": report["aggregates"],
        "paired_contrasts": report["paired_contrasts"],
        "total_editable_positions": sum(
            row["tokens"]["editable_positions"] for row in rows_out
        ),
        "total_editable_control_counts": total_control_counts,
        "inputs": inputs.manifest(),
        "limits": "No chemistry or model rerun; chemistry metrics inherit exact independent per-seed rescoring. MASK counts are reconstructed from final IDs and editable bits, not decoded text. Paired sample SD across two engineering seeds is not a confidence interval or superiority test. Pooled token counts describe this heterogeneous sample only. Runtime is descriptive: generation includes model sampling/tokenizer and released postprocessing, excludes model load, independent scoring and token audit; shared GPUs and small samples preclude a speed claim.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--generation-source", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.root.resolve(strict=True),
                args.report_sha256,
                args.generation_source,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
