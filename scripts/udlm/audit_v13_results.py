"""CPU evidence audit of the fixed V13 clean-versus-LOO temperature panel.

Requires a complete explicitly pinned report. Replays report receipt validation
using the saved independent chemistry results, then separately checks arithmetic,
raw-row denominators and exact token arrays. No model, chemistry or GPU work.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.udlm import audit_v12_results as previous  # noqa: E402
from scripts.udlm import report_exploration as reporter  # noqa: E402

evidence = previous.evidence
require, same = evidence.require, evidence.same_number
PROTOCOL = "experiments/udlm/protocols/engineering_v13_temperature_space.json"
PROTOCOL_SHA = "daf8c8108772e0ebcbdd39f15dab07478fe8945ba6b33f4b68858a8536ad7319"
SOURCE = "0fdc54d91baade2f84b48de9223ceb19696fadb7"
SEEDS = [2100, 2101]
DIRECTION = "clean_temperature_minus_raw_loo_temperature"
TEMPERATURE_RECEIPT = {
    "temperature_space": "x0_denoiser",
    "temperature_application": "clean_denoiser_before_loo_conversion",
    "reverse_bridge_temperature": 1.0,
}


def validate_mode(run, entry):
    sampling = run["config"]["sampling"]
    require(
        sampling.get("temperature_space", "raw_loo") == entry["temperature_space"],
        "Observed temperature space differs",
    )
    require(
        all(
            sampling[key] == entry[key]
            for key in ("parameterization", "prior_variant", "prior_metadata_sha256")
        ),
        "CE/prior identity differs",
    )
    fixed = {
        "diffusion_type": "udlm",
        "parameterization": "x0_denoiser",
        "softmax_temp": 0.5,
        "raw_loo_top_p": 1.0,
        "randomness": 0.0,
        "min_add_len": 40,
        "num_steps": 128,
        "inference_eps": 1e-5,
        "exclude_special_tokens": False,
    }
    require(
        all(sampling[key] == value for key, value in fixed.items())
        and not sampling.get("gibbs_corrector", False),
        "A fixed V13 sampling control differs",
    )
    require(
        run["checkpoint"]["sha256"] == entry["checkpoint_sha256"]
        and run["config"]["sha256"] == entry["config_sha256"],
        "Frozen checkpoint/configuration differs",
    )
    require(run["generation_protocol"]["nfe"] == 128, "NFE differs")
    for receipt in (
        run["generation_protocol"],
        run["independent_rescore"]["identity"]["generation"],
    ):
        if entry["temperature_space"] == "x0_denoiser":
            require(
                all(
                    receipt.get(key) == value and type(receipt.get(key)) is type(value)
                    for key, value in TEMPERATURE_RECEIPT.items()
                ),
                "Clean-temperature order/unit bridge differs",
            )
        else:
            require(
                not set(TEMPERATURE_RECEIPT).intersection(receipt),
                "Raw control carries an unconfigured temperature receipt",
            )
            require(
                "temperature_space" not in sampling,
                "Raw temperature space must retain its historical canonical omission",
            )


def check_contrasts(report, protocol):
    """Recompute signed means/SD independently of the report's Fraction helper."""
    runs = {(run["config_id"], run["seed"]): run for run in report["runs"]}
    entries = {entry["config_id"]: entry for entry in protocol["entries"]}
    require(
        [c["contrast_id"] for c in report["paired_contrasts"]]
        == ["primary", "secondary"],
        "Both temperature contrasts must remain visible",
    )
    for contrast in report["paired_contrasts"]:
        declared = protocol["design"]["temperature_space_comparison"][
            contrast["contrast_id"]
        ]
        require(
            contrast["direction"] == DIRECTION
            and contrast["expected_seeds"] == SEEDS
            and contrast["summary_policy"]
            == "all_declared_pairs_required_for_each_metric"
            and all(contrast[key] == value for key, value in declared.items()),
            "Temperature contrast declaration differs",
        )
        selected = [
            entries[contrast[role + "_config"]] for role in ("control", "treatment")
        ]
        require(
            [entry["temperature_space"] for entry in selected]
            == ["raw_loo", "x0_denoiser"]
            and all(
                selected[0][key] == selected[1][key]
                for key in (
                    "checkpoint_sha256",
                    "prior_variant",
                    "prior_metadata_sha256",
                )
            ),
            "Temperature contrast changes checkpoint/prior or reverses direction",
        )
        common_sampling = []
        for role, entry in zip(("control", "treatment"), selected):
            require(
                contrast[role + "_attempt_id"] == entry["attempt_id"],
                "Paired attempt identity differs",
            )
            for seed in SEEDS:
                sampling = dict(runs[(entry["config_id"], seed)]["config"]["sampling"])
                require(
                    sampling.get("temperature_space", "raw_loo")
                    == entry["temperature_space"],
                    "Paired observed mode differs",
                )
                sampling.pop("temperature_space", None)
                common_sampling.append(sampling)
        require(
            all(value == common_sampling[0] for value in common_sampling),
            "Pair changes more than temperature space",
        )
        for branch in evidence.BRANCHES:
            for metric in evidence.METRICS:
                result = contrast["metrics"][branch][metric]
                require(
                    [row["seed"] for row in result["per_seed"]] == SEEDS,
                    "Paired seed slots differ",
                )
                differences = []
                for row in result["per_seed"]:
                    metrics = [
                        runs[(entry["config_id"], row["seed"])]["metrics"][branch]
                        for entry in selected
                    ]
                    values = [value[metric] for value in metrics]
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
                        row["status"]
                        == (
                            "completed"
                            if difference is not None
                            else "undefined_metric"
                        )
                        and same(row["control_value"], values[0])
                        and same(row["treatment_value"], values[1])
                        and same(row["difference"], difference),
                        "Clean-minus-raw arithmetic/status differs",
                    )
                    if metric == "quality":
                        require(
                            all(
                                row[role + "_quality_count"] == value["quality_count"]
                                for role, value in zip(
                                    ("control", "treatment"), metrics
                                )
                            ),
                            "Paired quality numerator differs",
                        )
                    if difference is not None:
                        differences.append(difference)
                complete = len(differences) == 2
                require(
                    result["complete"] is complete
                    and result["defined_seed_count"] == len(differences)
                    and result["expected_seed_count"] == 2
                    and same(
                        result["mean_difference"],
                        statistics.mean(differences) if complete else None,
                    )
                    and same(
                        result["sample_sd"],
                        statistics.stdev(differences) if complete else None,
                    ),
                    "Paired all-seed mean/sample SD differs",
                )


def audit(root, report_sha):
    inputs = evidence.Inputs()
    protocol_path = root / PROTOCOL
    protocol = inputs.read_json(protocol_path, "Frozen V13 protocol", PROTOCOL_SHA)
    directory = root / protocol["report_root"]
    report = inputs.read_json(
        directory / "report.json", "Completed V13 report", report_sha
    )
    require(
        report["status"] == "complete"
        and report["superiority_established"] is False
        and not report["unexpected_run_directories"],
        "V13 must be complete before outcome auditing",
    )
    require(
        report["accounting"]
        == {
            "scheduled_runs": 8,
            "scheduled_requests": 800,
            "independently_rescored_requests": 800,
            "status_counts": {"completed": 8},
        },
        "V13 acceptance accounting differs",
    )
    require(
        report["protocol"]["configuration"] == protocol
        and report["protocol"]["sha256"] == PROTOCOL_SHA,
        "Protocol binding differs",
    )
    entries = {entry["config_id"]: entry for entry in protocol["entries"]}
    require(
        protocol["seeds"] == SEEDS
        and protocol["num_samples"] == 100
        and protocol["nfe"] == 128,
        "Fixed engineering budget differs",
    )
    require(
        len(entries) == 4
        and len(report["runs"]) == 8
        and {(run["config_id"], run["seed"]) for run in report["runs"]}
        == {(key, seed) for key in entries for seed in SEEDS},
        "Missing or repeated run slot",
    )
    for name, key in (
        ("report.csv", "csv_sha256"),
        ("report.pdf", "pdf_sha256"),
        ("paired_contrasts.csv", "paired_contrasts_csv_sha256"),
    ):
        inputs.read(
            directory / name, "Original V13 companion", report["report_artifacts"][key]
        )
    for entry in entries.values():
        inputs.read(
            root / entry["config"], "Frozen inference YAML", entry["config_sha256"]
        )
    accepted = {}
    rows_out = []
    for run in report["runs"]:
        require(
            run["status"] == "completed"
            and run["independent_rescore"]["status"] == "exact_match",
            "Run lacks independent acceptance",
        )
        summary_path = root / run["artifacts"]["summary.json"]["relative_path"]
        summary = inputs.reference(
            root, run["artifacts"]["summary.json"], "Original seed summary"
        )
        raw = inputs.read(
            root / run["artifacts"]["raw_samples.csv"]["relative_path"],
            "Original raw rows",
            run["artifacts"]["raw_samples.csv"]["sha256"],
        )
        inputs.reference(
            root, run["controller"]["artifact"], "Original attempt controller"
        )
        require(summary["git"]["commit"] == SOURCE, "Frozen generation source differs")
        accepted[summary_path.resolve()] = run["independent_rescore"]
        validate_mode(run, entries[run["config_id"]])
        require(
            run["independent_rescore"]["identity"]["source"]["denoiser_source_sha256"]
            == summary["implementation_inputs"]["denoiser_source"]["sha256"],
            "Clean-denoiser source acceptance differs",
        )
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        require(len(rows) == 100, "Raw request count differs")
        denominators = {
            branch: previous.metric_counts(rows, run["metrics"][branch], branch)
            for branch in evidence.BRANCHES
        }
        tokens = previous.token_counts(summary["sampled_token_control_audit"])
        require(tokens["requests"] == 100, "Token row count differs")
        certified_tokens = run["independent_rescore"]["identity"][
            "sampled_token_control_audit"
        ]
        require(
            certified_tokens["exact_batch_decode_match"] is True
            and certified_tokens["control_token_counts"][
                "final_sampled_editable_positions"
            ]
            == tokens["all_editable_control_counts"],
            "Token acceptance differs",
        )
        for key in ("sampler_input_ids", "final_sampled_ids", "editable_mask"):
            require(
                certified_tokens[key + "_sha256"]
                == summary["sampled_token_control_audit"][key]["decoded_sha256"],
                "Accepted token hash differs",
            )
        repair = {
            "request_denominator": len(rows),
            "recovered_requests": sum(
                row["released_was_recovered"] == "True" for row in rows
            ),
            "largest_component_selected_requests": sum(
                row["released_largest_component_applied"] == "True" for row in rows
            ),
        }
        rows_out.append(
            {
                "config_id": run["config_id"],
                "seed": run["seed"],
                "metrics": run["metrics"],
                "denominators": denominators,
                "tokens": tokens,
                "repair_counts": repair,
                **previous.execution_record(run, summary),
            }
        )
    # This replays receipt/hash/seed validation only: each callback returns the
    # already certified chemistry result pinned by the original report hash.
    replayed = reporter.build_report(
        protocol_path,
        root / protocol["output_root"],
        root=root,
        rescore=lambda path, **kwargs: accepted[path.resolve()],
    )
    for key in (
        "status",
        "accounting",
        "protocol",
        "runs",
        "aggregates",
        "paired_contrasts",
        "unexpected_run_directories",
    ):
        require(
            replayed[key] == report[key],
            "Saved report differs from bound receipt replay: " + key,
        )
    for aggregate in report["aggregates"]:
        evidence.validate_seed_aggregate(
            aggregate,
            [
                run
                for run in report["runs"]
                if run["config_id"] == aggregate["config_id"]
            ],
        )
    check_contrasts(report, protocol)
    return {
        "status": "independent_evidence_checks_passed",
        "protocol_sha256": PROTOCOL_SHA,
        "report_sha256": report_sha,
        "generation_source": SOURCE,
        "accounting": report["accounting"],
        "runs": rows_out,
        "aggregates": report["aggregates"],
        "paired_contrasts": report["paired_contrasts"],
        "total_editable_positions": sum(
            row["tokens"]["editable_positions"] for row in rows_out
        ),
        "total_editable_control_counts": {
            key: sum(
                row["tokens"]["all_editable_control_counts"][key] for row in rows_out
            )
            for key in ("unk", "bos", "eos", "pad", "mask")
        },
        "inputs": inputs.manifest(),
        "limits": "No new chemistry evaluation, model load or generation. Original exact independent chemistry acceptances are retained and replayed; metric fractions, seed arithmetic and token counts are checked separately. Two engineering seeds do not establish superiority. Runtime is descriptive on shared GPUs. Pooled controls are a heterogeneous census. Recovery means the saved released_was_recovered flag; component selection means released_largest_component_applied. Counts may overlap and each divides by all requests.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.root.resolve(strict=True), args.report_sha256),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
