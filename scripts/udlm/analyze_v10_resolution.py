"""CPU-only V10 appendix: within-method 512-minus-128 metrics and syntax counts.

Requires the completed independent report and its explicit SHA-256. Original
reports remain untouched. No sampling, model loading or chemistry rescoring.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "scripts/udlm/audit_molecular_failures.py").is_file()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.udlm import audit_molecular_failures as lexical  # noqa: E402
from scripts.udlm import generate_study_overview_v9 as evidence  # noqa: E402

PROTOCOL = Path("experiments/udlm/protocols/engineering_v10_resolution.json")
PROTOCOL_SHA = "cc6e3ced657f6ec66f611398e65b8b0e79598b3e76acbef77c72cfc4a7d0c4fd"
METRICS = evidence.METRICS
BRANCHES = evidence.BRANCHES
LEXICAL_COUNTS = (
    "odd_ring",
    "bad_parentheses",
    "both_flags",
    "syntax_flag",
    "flagged_strict_valid",
)
REPAIR_COUNTS = ("recovered", "largest_component")
require = evidence.require


def paired_summary(pairs, seeds, *, ratio=False):
    """Withhold summaries unless every declared seed has two defined values."""
    rows, values = [], []
    for seed in seeds:
        control, treatment = pairs.get(seed, (None, None))
        defined = control is not None and treatment is not None
        if defined:
            require(
                math.isfinite(control) and math.isfinite(treatment),
                "Paired values must be finite",
            )
            require(
                not ratio or control > 0, "Runtime-ratio denominator must be positive"
            )
        value = (
            (treatment / control if ratio else treatment - control) if defined else None
        )
        rows.append(
            {
                "seed": seed,
                "control_value": control,
                "treatment_value": treatment,
                "value": value,
                "status": "defined" if defined else "missing_or_undefined_pair",
            }
        )
        if value is not None:
            values.append(value)
    complete = len(values) == len(seeds) and bool(seeds)
    return {
        "operation": "512_over_128" if ratio else "512_minus_128",
        "complete": complete,
        "expected_seed_count": len(seeds),
        "defined_seed_count": len(values),
        "per_seed": rows,
        "mean": statistics.mean(values) if complete else None,
        "sample_sd": statistics.stdev(values) if complete and len(values) > 1 else None,
    }


def validate_runtime(runtime):
    for key in (
        "generation",
        "model_sampling_and_tokenizer",
        "released_postprocessing",
    ):
        require(
            isinstance(runtime[key], (int, float))
            and math.isfinite(runtime[key])
            and runtime[key] >= 0,
            "Runtime values must be finite and nonnegative",
        )
    require(
        evidence.same_number(
            runtime["generation"],
            runtime["model_sampling_and_tokenizer"]
            + runtime["released_postprocessing"],
        ),
        "Generation runtime components do not add up",
    )


def load_report(inputs, root, directory, expected_report_sha):
    report = inputs.read_json(
        directory / "report.json",
        "Completed independent V10 report",
        expected_report_sha,
    )
    require(
        report["status"] == "complete"
        and report["superiority_established"] is False
        and not report["unexpected_run_directories"],
        "V10 requires a complete engineering report",
    )
    require(
        report["accounting"]
        == {
            "scheduled_runs": 8,
            "scheduled_requests": 800,
            "independently_rescored_requests": 800,
            "status_counts": {"completed": 8},
        },
        "Unexpected V10 run/request accounting",
    )
    protocol = inputs.read_json(
        root / PROTOCOL, "Prospective V10 protocol", PROTOCOL_SHA
    )
    require(
        report["protocol"]["sha256"] == PROTOCOL_SHA
        and report["protocol"]["configuration"] == protocol,
        "Report does not bind the frozen V10 protocol",
    )
    for extension in ("csv", "pdf"):
        inputs.read(
            directory / f"report.{extension}",
            "Unmodified V10 original report",
            report["report_artifacts"][f"{extension}_sha256"],
        )
    for reference in [
        protocol["prospective_design"],
        *[v for v in protocol["prior_evidence"].values() if isinstance(v, dict)],
    ]:
        inputs.read(
            root / reference["relative_path"],
            "Prospective design/prior evidence",
            reference["sha256"],
        )
    entries = {entry["config_id"]: entry for entry in protocol["entries"]}
    require(
        len(report["runs"]) == 8
        and len(report["aggregates"]) == 4
        and {row["config_id"] for row in report["aggregates"]} == set(entries),
        "Missing V10 configurations or runs",
    )
    for entry in entries.values():
        inputs.read(
            root / entry["config"],
            "Pinned inference configuration",
            entry["config_sha256"],
        )
    counts = {}
    for run in report["runs"]:
        entry = entries[run["config_id"]]
        require(
            run["status"] == "completed"
            and run["requested_samples"] == run["raw_row_count"] == 100
            and run["independent_rescore"]["status"] == "exact_match"
            and set(run["artifacts"]) == {"summary.json", "raw_samples.csv"},
            "Run lacks complete independently rescored artifacts",
        )
        summary = inputs.reference(
            root, run["artifacts"]["summary.json"], "Original per-seed summary"
        )
        raw = inputs.read(
            root / run["artifacts"]["raw_samples.csv"]["relative_path"],
            "Original raw molecular rows",
            run["artifacts"]["raw_samples.csv"]["sha256"],
        )
        controller = inputs.reference(
            root, run["controller"]["artifact"], "Original controller receipt"
        )
        require(
            controller == run["controller"]["receipt"]
            and run["controller"]["status"] == "completed"
            and controller["return_code"] == 0
            and controller["identity"]["protocol_sha256"] == PROTOCOL_SHA,
            "Controller receipt differs from the declared experiment",
        )
        evidence.validate_run_slot(run, summary, controller, entry, protocol)
        require(
            summary["config"] == run["config"]
            and summary["checkpoint"] == run["checkpoint"]
            and summary["git"] == run["source"]
            and summary["runtime_seconds"] == run["runtime_seconds"]
            and summary["run"]["generation_protocol"] == run["generation_protocol"],
            "Report changed original generation identity",
        )
        require(
            run["independent_rescore"]["summary_sha256"]
            == run["artifacts"]["summary.json"]["sha256"]
            and run["independent_rescore"]["raw_samples_sha256"]
            == evidence.digest(raw),
            "Independent rescore binds different input bytes",
        )
        require(
            run["config"]["sha256"] == entry["config_sha256"]
            and run["checkpoint"]["sha256"] == entry["checkpoint_sha256"],
            "Checkpoint/configuration bytes differ from protocol",
        )
        sampling = run["config"]["sampling"]
        nfe = protocol["design"]["nfe_by_configuration"][run["config_id"]]
        require(
            run["generation_protocol"]["nfe"] == sampling["num_steps"] == nfe
            and not sampling.get("gibbs_corrector", False)
            and sampling.get("parameterization", "raw_loo")
            == entry["parameterization"],
            "Unexpected inference objective or NFE",
        )
        validate_runtime(run["runtime_seconds"])
        rows = lexical.parse_rows(raw, 100)
        counts[(run["config_id"], run["seed"])] = lexical.tally(
            [lexical.row_diagnostics(row) for row in rows]
        )
        for branch in BRANCHES:
            for metric in METRICS:
                require(
                    evidence.same_number(
                        run["metrics"][branch][metric],
                        run["independent_rescore"]["metrics"][branch][metric],
                    ),
                    "Run metric differs from independent rescore",
                )
    require(len(counts) == 8, "Duplicated V10 configuration/seed")
    for aggregate in report["aggregates"]:
        runs = sorted(
            [
                run
                for run in report["runs"]
                if run["config_id"] == aggregate["config_id"]
            ],
            key=lambda row: row["seed"],
        )
        require(
            aggregate["complete"]
            and [run["seed"] for run in runs] == protocol["seeds"],
            "Missing a declared V10 seed",
        )
        evidence.validate_seed_aggregate(aggregate, runs)
    return report, protocol, counts


def analyze(report, protocol, counts):
    runs = {(row["config_id"], row["seed"]): row for row in report["runs"]}
    contrasts = []
    for declared in protocol["design"]["resolution_comparison"]["contrasts"]:
        control_id, treatment_id = (
            declared["control_config"],
            declared["treatment_config"],
        )
        contrast = dict(
            declared, operation="512_minus_128", metrics={}, lexical_counts={}
        )
        for seed in protocol["seeds"]:
            control, treatment = (
                runs.get((control_id, seed)),
                runs.get((treatment_id, seed)),
            )
            if control is None or treatment is None:
                continue
            require(
                control["arm_id"] == treatment["arm_id"] == declared["method"]
                and control["checkpoint"]["sha256"]
                == treatment["checkpoint"]["sha256"],
                "Resolution contrast changes method or checkpoint",
            )
            configs = [dict(row["config"]["sampling"]) for row in (control, treatment)]
            require(
                [config.pop("num_steps") for config in configs] == [128, 512]
                and configs[0] == configs[1],
                "Resolution pair changes more than predictor NFE",
            )
        for branch in BRANCHES:
            contrast["metrics"][branch] = {}
            for metric in METRICS:
                pairs = {
                    seed: tuple(
                        runs.get((config, seed), {})
                        .get("metrics", {})
                        .get(branch, {})
                        .get(metric)
                        for config in (control_id, treatment_id)
                    )
                    for seed in protocol["seeds"]
                }
                contrast["metrics"][branch][metric] = paired_summary(
                    pairs, protocol["seeds"]
                )
        runtime_pairs = {
            seed: tuple(
                runs.get((config, seed), {})
                .get("runtime_seconds", {})
                .get("generation")
                for config in (control_id, treatment_id)
            )
            for seed in protocol["seeds"]
        }
        contrast["generation_runtime_ratio"] = paired_summary(
            runtime_pairs, protocol["seeds"], ratio=True
        )
        for field in LEXICAL_COUNTS:
            pairs = {
                seed: tuple(
                    counts.get((config, seed), {}).get(field)
                    for config in (control_id, treatment_id)
                )
                for seed in protocol["seeds"]
            }
            contrast["lexical_counts"][field] = paired_summary(pairs, protocol["seeds"])
        contrasts.append(contrast)
    return contrasts


def csv_bytes(contrasts, run_counts):
    stream = io.StringIO()
    fields = [
        "record_type",
        "contrast_id",
        "method",
        "config_id",
        "seed",
        "branch",
        "measure",
        "operation",
        "control_value",
        "treatment_value",
        "value",
        "mean",
        "sample_sd",
        "defined_seed_count",
        "expected_seed_count",
        "status",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for contrast in contrasts:
        groups = [
            (branch, metric, value)
            for branch in BRANCHES
            for metric, value in contrast["metrics"][branch].items()
        ]
        groups += [
            ("runtime", "generation_seconds", contrast["generation_runtime_ratio"])
        ]
        groups += [
            ("raw_lexical", field, value)
            for field, value in contrast["lexical_counts"].items()
        ]
        for branch, measure, value in groups:
            base = {
                "contrast_id": contrast["contrast_id"],
                "method": contrast["method"],
                "branch": branch,
                "measure": measure,
                "operation": value["operation"],
            }
            for row in value["per_seed"]:
                writer.writerow(dict(base, record_type="paired_seed", **row))
            writer.writerow(
                dict(
                    base,
                    record_type="paired_summary",
                    mean=value["mean"],
                    sample_sd=value["sample_sd"],
                    defined_seed_count=value["defined_seed_count"],
                    expected_seed_count=value["expected_seed_count"],
                    status="complete" if value["complete"] else "withheld",
                )
            )
    for row in run_counts:
        for field in ("n", *LEXICAL_COUNTS, *REPAIR_COUNTS):
            writer.writerow(
                dict(
                    record_type="released_repair_count"
                    if field in REPAIR_COUNTS
                    else "raw_lexical_count",
                    config_id=row["config_id"],
                    method=row["method"],
                    seed=row["seed"],
                    branch="released_comparable"
                    if field in REPAIR_COUNTS
                    else "raw_lexical",
                    measure=field,
                    value=row["counts"][field],
                    status="observed",
                )
            )
    return stream.getvalue().encode()


def pdf_bytes(report, contrasts, run_counts):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallV10", fontSize=8, leading=11, spaceAfter=7))
    styles.add(ParagraphStyle(name="TinyV10", fontSize=7, leading=9))
    story = []
    width = landscape(A4)[0] - 64

    def add(text, style="SmallV10"):
        story.append(Paragraph(text, styles[style]))

    def table(rows, widths):
        obj = Table(
            [
                [Paragraph(escape(str(cell)), styles["TinyV10"]) for cell in row]
                for row in rows
            ],
            colWidths=widths,
            repeatRows=1,
        )
        obj.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce7f1")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f0f4f7")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(obj)

    def number(value, metric):
        if value is None:
            return "undefined"
        return f"{value:.5f}" if metric == "diversity" else f"{100 * value:.3f}"

    def mean_sd(value, metric):
        return (
            number(value["mean"], metric) + " +/- " + number(value["sample_sd"], metric)
        )

    add("V10: sampling resolution within CT and CE", "Title")
    add(
        "512 predictor evaluations minus 128, separately within each method; temperature 0.5, seeds 1700/1701, "
        "100 requests per seed. Four configurations / eight runs / 800 requests, all independently rescored. "
        "This is a resolution contrast, not CE minus CT and not equal compute: 512 uses four times the model evaluations."
    )
    add(
        "<b>No superiority established.</b> V5/V6 and complete V9 outcomes informed this design. The separately audited "
        "V8 CT and completed V8b CE checkpoints are reused without retraining. The original V8 controller/campaign remains failed. "
        "Final UDLM seeds 0/1/2 remain reserved. Local MDLM quality 85.8% and paper GenMol V1 84.6% are contextual, differently budgeted comparisons."
    )
    rows = [
        [
            "Configuration",
            "Branch",
            "Validity %",
            "Uniqueness %",
            "Quality %",
            "Diversity",
        ]
    ]
    for aggregate in report["aggregates"]:
        for branch in BRANCHES:
            rows.append(
                [
                    aggregate["config_id"],
                    "repaired" if branch == "released_comparable" else "strict",
                    *[
                        mean_sd(aggregate["metrics"][branch][metric], metric)
                        for metric in METRICS
                    ],
                ]
            )
    table(rows, [120, 70] + [(width - 190) / 4] * 4)
    add(
        "Per-configuration means +/- sample SD weight seeds equally; no pooling across configurations. Validity and "
        "quality divide by requests; uniqueness divides by valid molecules. Quality counts unique valid molecules with "
        "QED &gt;= 0.6 and SA &lt;= 4. Diversity is the released PyTDC metric on unique valid molecules. Repaired decoding "
        "uses SAFE fix=True and largest-component selection by SMILES length; strict uses fix=False plus RDKit validation."
    )
    story.append(PageBreak())
    add("Paired molecular changes: 512 minus 128", "Heading1")
    rows = [
        [
            "Method / seed",
            "Branch",
            "Delta validity pp",
            "Delta uniqueness pp",
            "Delta quality pp",
            "Delta diversity",
        ]
    ]
    for contrast in contrasts:
        for branch in BRANCHES:
            for index, seed in enumerate((1700, 1701)):
                rows.append(
                    [
                        f'{contrast["method"]} / {seed}',
                        "repaired" if branch == "released_comparable" else "strict",
                        *[
                            number(
                                contrast["metrics"][branch][metric]["per_seed"][index][
                                    "value"
                                ],
                                metric,
                            )
                            for metric in METRICS
                        ],
                    ]
                )
            rows.append(
                [
                    f'{contrast["method"]} mean +/- SD',
                    "repaired" if branch == "released_comparable" else "strict",
                    *[
                        mean_sd(contrast["metrics"][branch][metric], metric)
                        for metric in METRICS
                    ],
                ]
            )
    table(rows, [120, 70] + [(width - 190) / 4] * 4)
    add(
        "Pairing is by seed/run labels, not molecule-level trajectories: different step counts change random draws and "
        "reverse paths. All declared pairs must define a metric before its mean/SD is shown; undefined is never zero. "
        "Both methods and negative changes remain disclosed. Two-seed SD is descriptive, not a confidence interval or significance test."
    )
    add("Generation wall time: 512 / 128", "Heading2")
    rows = [
        [
            "Method / seed",
            "128-step seconds",
            "512-step seconds",
            "Ratio / mean +/- sample SD",
        ]
    ]
    for contrast in contrasts:
        timing = contrast["generation_runtime_ratio"]
        for row in timing["per_seed"]:
            rows.append(
                [
                    f'{contrast["method"]} / {row["seed"]}',
                    f'{row["control_value"]:.3f}',
                    f'{row["treatment_value"]:.3f}',
                    f'{row["value"]:.3f}',
                ]
            )
        rows.append(
            [
                f'{contrast["method"]} mean +/- SD',
                "--",
                "--",
                f'{timing["mean"]:.3f} +/- {timing["sample_sd"]:.3f}',
            ]
        )
    table(rows, [120, 170, 170, width - 460])
    add(
        "Generation seconds include model sampling/tokenizer decoding plus released SAFE postprocessing; they exclude "
        "checkpoint loading and metric scoring. The mean is the equal-seed mean of paired ratios, not a ratio of means. "
        "Sequential GPU scheduling, device mapping and external activity can change wall time; four times NFE need not mean four times wall time."
    )
    story.append(PageBreak())
    add("Raw SAFE ring and parenthesis diagnostics", "Heading1")
    rows = [
        [
            "Configuration / seed",
            "Requests",
            "Odd ring label",
            "Bad parentheses",
            "Both flags",
            "Either flag",
            "Flagged strict-valid",
        ]
    ]
    for row in run_counts:
        rows.append(
            [
                f'{row["config_id"]} / {row["seed"]}',
                row["counts"]["n"],
                *[row["counts"][field] for field in LEXICAL_COUNTS],
            ]
        )
    table(rows, [160, 65] + [(width - 225) / 5] * 5)
    add(
        "Odd ring label: remove bracket atoms, recognize single-digit, %NN and %(integer) labels, normalize integer labels "
        "and flag odd counts. Even counts allow legitimate label reuse and do not establish chemical validity. Parentheses: "
        "remove bracket atoms and %(integer) labels, then flag a negative prefix depth or nonzero final depth. Counts overlap."
    )
    rows = [
        [
            "Method: mean 512-minus-128 count +/- SD",
            "Odd ring",
            "Parentheses",
            "Both",
            "Either",
            "Flagged strict-valid",
        ]
    ]
    for contrast in contrasts:
        rows.append(
            [
                contrast["method"],
                *[
                    f'{contrast["lexical_counts"][field]["mean"]:.3f} +/- {contrast["lexical_counts"][field]["sample_sd"]:.3f}'
                    for field in LEXICAL_COUNTS
                ],
            ]
        )
    table(rows, [225] + [(width - 225) / 5] * 5)
    add("Released repair behavior: counts across the two 100-request seeds", "Heading2")
    rows = [
        [
            "Configuration",
            "Requests",
            "Recovered: count / requests (%)",
            "Largest-component selection: count / requests (%)",
        ]
    ]
    for aggregate in report["aggregates"]:
        selected = [
            row["counts"]
            for row in run_counts
            if row["config_id"] == aggregate["config_id"]
        ]
        total = sum(row["n"] for row in selected)
        values = [sum(row[field] for row in selected) for field in REPAIR_COUNTS]
        rows.append(
            [
                aggregate["config_id"],
                total,
                *[
                    f"{value} / {total} ({100 * value / total:.1f}%)"
                    for value in values
                ],
            ]
        )
    table(rows, [160, 65, (width - 225) / 2, (width - 225) / 2])
    add(
        "Recovered means released decoding produced a valid selected molecule where strict decoding did not. "
        "Largest-component selection is the saved released-selection flag. Both fractions divide by all requests, "
        "not only valid molecules, and can overlap. CSV/JSON preserve the per-seed counts; these descriptive counts "
        "do not isolate a causal effect of repair."
    )
    add(
        "Lexical flags are limited diagnostics, not a complete SAFE/SMILES parser or a causal explanation of quality. "
        "Each count uses all 100 requested rows, including decode failures and repeats; no deduplication across seeds or "
        "configurations. Fewer flags need not improve QED, SA, validity or quality. The finite-state oracle motivated "
        "resolution testing but does not prove gains for learned, temperature-0.5 molecular models."
    )
    add(
        "The original complete V10 report remains unchanged and contains all configurations, generation identities, device "
        "mappings and per-run metrics. This appendix hashes that report, its PDF/CSV, raw rows, summaries, controllers, "
        "prospective/prior evidence and the exact reused lexical/helper source. JSON/CSV retain exact values and counts. "
        "No new molecular sampling, chemistry rescoring, checkpoint load or automatic promotion occurred here."
    )
    output = io.BytesIO()
    SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        leftMargin=32,
        rightMargin=32,
        topMargin=25,
        bottomMargin=25,
        title="V10 within-method predictor resolution analysis",
        invariant=1,
    ).build(story)
    return output.getvalue()


def build_bundle(root, directory, expected_report_sha):
    root, directory = (
        Path(root).resolve(strict=True),
        Path(directory).resolve(strict=True),
    )
    inputs = evidence.Inputs()
    generator = inputs.read(Path(__file__), "V10 appendix generator")
    for module, role in (
        (lexical, "Tested lexical diagnostics implementation"),
        (evidence, "Hash and identity helper implementation"),
    ):
        inputs.read(Path(module.__file__), role)
    report, protocol, counts = load_report(inputs, root, directory, expected_report_sha)
    contrasts = analyze(report, protocol, counts)
    run_counts = [
        {
            "config_id": run["config_id"],
            "method": run["arm_id"],
            "seed": run["seed"],
            "nfe": run["generation_protocol"]["nfe"],
            "counts": counts[(run["config_id"], run["seed"])],
            "runtime_seconds": run["runtime_seconds"],
            "gpu_mapping": run["gpu_mapping"],
            "source": run["source"],
            "checkpoint": run["checkpoint"],
        }
        for run in report["runs"]
    ]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "study_id": protocol["study_id"],
        "superiority_established": False,
        "source_report_sha256": expected_report_sha,
        "protocol_sha256": PROTOCOL_SHA,
        "accounting": report["accounting"],
        "configuration_metrics": report["aggregates"],
        "contrasts": contrasts,
        "runs": run_counts,
        "definitions": {
            key: lexical.DEFINITIONS[key]
            for key in ("ring_parity", "parentheses", "strict", "repaired", "quality")
        },
        "limitations": protocol["limitations"],
    }
    outputs = {
        "analysis.json": evidence.json_bytes(summary),
        "analysis.csv": csv_bytes(contrasts, run_counts),
        "analysis.pdf": pdf_bytes(report, contrasts, run_counts),
        "analyze_v10_resolution.py": generator,
    }
    outputs["input_hash_manifest.json"] = evidence.json_bytes(
        {
            "schema_version": 1,
            "inputs": inputs.manifest(),
            "generator_sha256": evidence.digest(generator),
            "outputs": {
                name: {"sha256": evidence.digest(payload), "size_bytes": len(payload)}
                for name, payload in outputs.items()
            },
        }
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(
        not args.output_directory.resolve().exists(), "Output directory must be fresh"
    )
    outputs = build_bundle(
        args.input_root, args.report_directory, args.expected_report_sha256
    )
    evidence.publish_bundle(args.output_directory, outputs)
    print(
        json.dumps(
            {
                "output_directory": str(args.output_directory.resolve()),
                "pdf_sha256": evidence.digest(outputs["analysis.pdf"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
