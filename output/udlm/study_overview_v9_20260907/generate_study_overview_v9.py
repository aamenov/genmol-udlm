"""Combine completed V5/V6, training evidence and the paired V9 report on CPU.

Use a fresh output directory. Original reports are read, hash-checked and
appended unchanged; this overview neither samples nor re-runs chemistry or
checkpoint validation. It requires the completed paired V9 supplement.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

import pypdf
import reportlab
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

WORKSPACE = Path("/home/aidar.alimbayev/Documents/genmolv2")
HISTORICAL = Path("output/udlm/study_overview_20260907")
HISTORICAL_PDF_SHA = "96934c4eeabc89d61f3d507eb6a455e157adb51ac7fa48cb76a82344ab3b3963"
V9_PROTOCOL = Path("experiments/udlm/protocols/engineering_v9_objectives.json")
V9_PROTOCOL_SHA = "718893198b0e08a2405e33177c64989499914207ff39b6880edbc5eff2b35c7d"
V7_TERMINAL = Path(
    "output/udlm/engineering_v7/ct_e_throughput20_b128_w1/terminal_manifest.json"
)
METRICS = ("validity", "uniqueness", "quality", "diversity")
BRANCHES = ("released_comparable", "strict")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def canonical_digest(value):
    return digest(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


class Inputs:
    """Retain verified small evidence bytes; checkpoint identities are inherited."""

    def __init__(self, workspace=WORKSPACE):
        self.workspace = workspace.resolve(strict=True)
        self.payloads = {}
        self.roles = {}

    def read(self, path, role, expected=None):
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(self.workspace), "Input is outside workspace")
        payload = path.read_bytes()
        require(
            expected is None or digest(payload) == expected,
            f"Input hash mismatch: {path}",
        )
        require(
            path not in self.payloads or self.payloads[path] == payload,
            f"Input changed: {path}",
        )
        self.payloads[path] = payload
        self.roles.setdefault(path, set()).add(role)
        return payload

    def read_json(self, path, role, expected=None):
        return json.loads(self.read(path, role, expected))

    def reference(self, root, reference, role):
        payload = self.read(
            root / reference["relative_path"], role, reference["sha256"]
        )
        require(
            "size_bytes" not in reference or len(payload) == reference["size_bytes"],
            "Input size mismatch",
        )
        return json.loads(payload)

    def manifest(self):
        result = []
        for path, payload in sorted(self.payloads.items()):
            require(
                path.read_bytes() == payload,
                f"Input changed before publication: {path}",
            )
            result.append(
                {
                    "workspace_relative_path": str(path.relative_to(self.workspace)),
                    "sha256": digest(payload),
                    "size_bytes": len(payload),
                    "roles": sorted(self.roles[path]),
                }
            )
        return result


def load_historical(inputs, root):
    directory = root / HISTORICAL
    manifest = inputs.read_json(
        directory / "input_hash_manifest.json", "Historical overview manifest"
    )
    require(manifest["status"] == "complete", "Historical overview is incomplete")
    inputs.read(
        directory / "generate_study_overview.py",
        "Preserved historical generator",
        manifest["generator_sha256"],
    )
    for name, record in manifest["outputs"].items():
        require(Path(name).name == name, "Unexpected historical output path")
        inputs.read(directory / name, "Preserved historical output", record["sha256"])
    for record in manifest["inputs"]:
        inputs.read(
            inputs.workspace / record["workspace_relative_path"],
            "Historical upstream evidence",
            record["sha256"],
        )
    overview = inputs.read_json(
        directory / "overview.json", "Historical configuration means"
    )
    require(
        overview["status"] == "complete"
        and overview["configuration_count"] == 18
        and overview["scheduled_runs"] == 36
        and overview["independently_rescored_requests"] == 2304
        and overview["superiority_established"] is False,
        "Unexpected historical accounting",
    )
    pdf = inputs.read(
        directory / "study_overview.pdf",
        "Unmodified V5/V6/baseline appendix",
        HISTORICAL_PDF_SHA,
    )
    return overview, pdf


def load_training_terminal(inputs, root, reference, expected_status):
    terminal = inputs.reference(root, reference, "Training terminal")
    require(terminal["status"] == expected_status, "Training controller status changed")
    directory = Path(reference["relative_path"]).parent
    for name, field in (
        ("request_manifest.json", "request_sha256"),
        ("launch_manifest.json", "launch_sha256"),
    ):
        receipt = inputs.read_json(
            root / directory / name, "Training request/launch", terminal[field]
        )
        require(
            receipt["source"] == terminal["source"]
            and receipt["plan"] == terminal["plan"],
            "Training receipt identity mismatch",
        )
    plan = terminal["plan"]
    require(
        canonical_digest(plan["config"]) == plan["config_sha256"],
        "Training resolved configuration mismatch",
    )
    for path, expected in terminal["artifact_hashes"].items():
        if Path(path).suffix != ".ckpt":
            inputs.read(root / path, "Training log or telemetry", expected)
    if expected_status == "completed":
        require(
            terminal["training_return_code"] == 0
            and terminal["leases_release_authorized"] is True,
            "Completed training lacks accepted exit/lease evidence",
        )
        require(
            terminal["completed_example_exposures"] == plan["example_exposures"],
            "Training exposure mismatch",
        )
        require(
            same_number(
                terminal["end_to_end_training_examples_per_second"],
                plan["example_exposures"] / terminal["training_subprocess_seconds"],
            ),
            "Training throughput mismatch",
        )
        require(
            terminal["artifact_hashes"][terminal["checkpoint"]["relative_path"]]
            == terminal["checkpoint"]["sha256"],
            "Training checkpoint identity mismatch",
        )
    else:
        require(
            terminal["checkpoint"] is None
            and terminal["completed_example_exposures"] is None
            and terminal["end_to_end_training_examples_per_second"] is None,
            "Failed CT receipt was converted into a successful receipt",
        )
    return terminal


def load_training(inputs, root, protocol):
    training = protocol["training"]
    v7_payload = inputs.read(root / V7_TERMINAL, "V7 throughput pilot terminal")
    v7 = load_training_terminal(
        inputs,
        root,
        {"relative_path": str(V7_TERMINAL), "sha256": digest(v7_payload)},
        "completed",
    )
    ct = load_training_terminal(
        inputs, root, training["arms"]["CT"]["terminal_receipt"], "failed"
    )
    ce = load_training_terminal(
        inputs, root, training["arms"]["CE"]["terminal_receipt"], "completed"
    )
    audit = inputs.reference(
        root,
        training["arms"]["CT"]["post_exit_audit"],
        "Separate CT post-exit CPU audit",
    )
    campaign = inputs.reference(
        root, training["original_failed_campaign"], "Original failed V8 campaign"
    )
    require(
        campaign["status"] == audit["original_campaign_status"] == "failed",
        "Original V8 failure must remain disclosed",
    )
    require(
        audit["original_terminal_sha256"]
        == training["arms"]["CT"]["terminal_receipt"]["sha256"]
        and audit["original_campaign_terminal_sha256"]
        == training["original_failed_campaign"]["sha256"],
        "CT audit receipt mismatch",
    )
    require(
        audit["checkpoint_status"] == "separately_validated_after_controller_failure"
        and audit["leases_released"] is True
        and audit["checkpoint"]["global_step"] == 1000,
        "CT separate audit is not accepted",
    )
    require(
        audit["checkpoint"]["sha256"] == training["arms"]["CT"]["checkpoint_sha256"]
        and ce["checkpoint"]["sha256"] == training["arms"]["CE"]["checkpoint_sha256"],
        "Objective checkpoint mismatch",
    )
    require(
        canonical_digest(audit["training_implementation_sha256"])
        == training["training_implementation_hashes_sha256"],
        "Training implementation map mismatch",
    )
    inputs.reference(
        root, training["original_v8_protocol"], "Original V8 objective protocol"
    )
    inputs.reference(
        root, training["arms"]["CE"]["protocol"], "Separate V8b CE protocol"
    )
    for arm, terminal in (("CT", ct), ("CE", ce)):
        require(
            terminal["source"]["head"] == training["arms"][arm]["source_revision"]
            and terminal["plan"]["config_sha256"]
            == training["arms"][arm]["resolved_config_sha256"],
            "Training source/config binding mismatch",
        )
    common = []
    for terminal in (ct, ce):
        config = copy.deepcopy(terminal["plan"]["config"])
        config["callback"]["dirpath"] = "<arm-output>/checkpoints"
        config["training"]["udlm"].pop("parameterization", None)
        common.append(config)
    require(common[0] == common[1], "CT/CE common training settings differ")
    require(
        canonical_digest(common[0]) == training["matched_common_config_sha256"],
        "Common training settings differ from the prospective hash",
    )
    require(
        v7["checkpoint"]["global_step"] == 20
        and v7["completed_example_exposures"] == 2560,
        "Unexpected V7 throughput scope",
    )
    require(
        ce["checkpoint"]["global_step"] == 1000
        and ce["completed_example_exposures"] == 128000
        and audit["configured_example_exposures_supported_by_step_and_batch"] == 128000,
        "Unexpected objective-training scope",
    )
    return {
        "V7": v7,
        "V8_CT": ct,
        "V8_CT_post_exit_audit": audit,
        "V8b_CE": ce,
        "original_V8_campaign": campaign,
    }


def load_v9(inputs, root, directory):
    report = inputs.read_json(directory / "report.json", "Completed V9 paired report")
    require(
        report["status"] == "complete"
        and report["superiority_established"] is False
        and not report["unexpected_run_directories"],
        "V9 is not a complete engineering report",
    )
    require(
        report["accounting"]
        == {
            "scheduled_runs": 8,
            "scheduled_requests": 800,
            "independently_rescored_requests": 800,
            "status_counts": {"completed": 8},
        },
        "V9 accounting differs from the fixed eight-run design",
    )
    protocol = inputs.read_json(
        root / V9_PROTOCOL, "Frozen V9 protocol", V9_PROTOCOL_SHA
    )
    require(
        report["protocol"]["sha256"] == V9_PROTOCOL_SHA
        and report["protocol"]["configuration"] == protocol,
        "V9 protocol identity mismatch",
    )
    inputs.read(
        root / protocol["prospective_design"]["relative_path"],
        "Prospective V9 design",
        protocol["prospective_design"]["sha256"],
    )
    for extension in ("csv", "pdf"):
        inputs.read(
            directory / f"report.{extension}",
            f"V9 paired report {extension}",
            report["report_artifacts"][f"{extension}_sha256"],
        )
    inputs.read(
        directory / "paired_contrasts.csv",
        "Paired CE minus CT CSV",
        report["report_artifacts"]["paired_contrasts_csv_sha256"],
    )
    require("paired_contrasts" in report, "The completed paired supplement is required")
    entries = {entry["config_id"]: entry for entry in protocol["entries"]}
    for entry in entries.values():
        inputs.read(
            root / entry["config"], "V9 inference configuration", entry["config_sha256"]
        )
    require(
        len(report["runs"]) == 8 and len(report["aggregates"]) == 4,
        "Missing V9 runs/configurations",
    )
    require(
        {row["config_id"] for row in report["aggregates"]} == set(entries),
        "V9 aggregate configurations differ from the declared entries",
    )
    for run in report["runs"]:
        entry = entries[run["config_id"]]
        require(
            set(run["artifacts"]) == {"summary.json", "raw_samples.csv"},
            "V9 completed run must retain its summary and raw rows without a failure receipt",
        )
        require(
            run["status"] == "completed"
            and run["requested_samples"] == run["raw_row_count"] == 100
            and run["independent_rescore"]["status"] == "exact_match",
            "V9 run lacks exact independent rescoring",
        )
        require(
            run["checkpoint"]["sha256"] == entry["checkpoint_sha256"],
            "V9 run checkpoint mismatch",
        )
        require(
            run["config"]["sha256"] == entry["config_sha256"]
            and run["config"]["sampling"].get("parameterization", "raw_loo")
            == entry["parameterization"],
            "V9 run configuration/objective mismatch",
        )
        controller = run["controller"]
        receipt = inputs.reference(
            root, controller["artifact"], "V9 controller receipt"
        )
        require(
            controller["status"] == "completed"
            and receipt == controller["receipt"]
            and receipt["return_code"] == 0
            and receipt["identity"]["protocol_sha256"] == V9_PROTOCOL_SHA,
            "V9 controller receipt mismatch",
        )
        gp = run["generation_protocol"]
        require(
            gp["nfe"] == 128 and not gp.get("gibbs_corrector", False),
            "Unexpected V9 sampling budget",
        )
        for name, artifact in run["artifacts"].items():
            payload = inputs.read(
                root / artifact["relative_path"],
                "V9 original run artifact",
                artifact["sha256"],
            )
            if name == "raw_samples.csv":
                require(
                    len(list(csv.DictReader(io.StringIO(payload.decode())))) == 100,
                    "V9 raw row count mismatch",
                )
                require(
                    digest(payload) == run["independent_rescore"]["raw_samples_sha256"],
                    "V9 independent rescore bound different raw rows",
                )
            if name == "summary.json":
                summary = json.loads(payload)
                require(
                    digest(payload) == run["independent_rescore"]["summary_sha256"]
                    and summary["config"] == run["config"]
                    and summary["checkpoint"] == run["checkpoint"]
                    and summary["git"] == run["source"]
                    and summary["run"]["generation_protocol"] == gp,
                    "V9 saved summary differs from report identity",
                )
                validate_run_slot(run, summary, receipt, entry, protocol)
        for branch in BRANCHES:
            for metric in METRICS:
                require(
                    same_number(
                        run["metrics"][branch][metric],
                        run["independent_rescore"]["metrics"][branch][metric],
                    ),
                    "V9 rescore metric mismatch",
                )
    records = []
    for aggregate in report["aggregates"]:
        runs = sorted(
            (
                run
                for run in report["runs"]
                if run["config_id"] == aggregate["config_id"]
            ),
            key=lambda run: run["seed"],
        )
        require(
            aggregate["complete"] and [run["seed"] for run in runs] == [1600, 1601],
            "V9 paired seeds changed",
        )
        validate_seed_aggregate(aggregate, runs)
        records.append(
            dict(
                aggregate,
                study="V9",
                temperature=runs[0]["config"]["effective"]["softmax_temp"],
                kernel="128P",
                gibbs_corrector=False,
                samples=200,
                seeds=[1600, 1601],
            )
        )
    validate_paired_contrasts(report, protocol)
    return report, protocol, records


def validate_run_slot(run, summary, receipt, entry, protocol):
    require(
        summary["seed"] == run["seed"] == run["independent_rescore"]["seed"]
        and run["seed"] in protocol["seeds"]
        and summary["num_samples"]
        == run["requested_samples"]
        == protocol["num_samples"],
        "V9 seed/sample slot differs from its saved summary or independent rescore",
    )
    require(
        all(
            run[key] == entry[key]
            for key in ("attempt_id", "candidate_id", "arm_id", "config_id")
        )
        and receipt["identity"]["entry"] == entry
        and receipt["identity"]["seeds"] == protocol["seeds"]
        and receipt["identity"]["num_samples"] == protocol["num_samples"]
        and receipt["identity"]["source"]
        == {"head": summary["git"]["commit"], "upstream": summary["git"]["commit"]},
        "V9 run/controller identity differs from its prospective slot",
    )


def validate_seed_aggregate(aggregate, runs):
    for branch in BRANCHES:
        for metric in METRICS:
            values = [
                run["metrics"][branch][metric]
                for run in runs
                if run["metrics"][branch][metric] is not None
            ]
            all_defined = len(values) == len(runs)
            value = aggregate["metrics"][branch][metric]
            require(
                value["defined_seed_count"] == len(values)
                and same_number(
                    value["mean"],
                    statistics.mean(values) if values and all_defined else None,
                )
                and same_number(
                    value["sample_sd"],
                    statistics.stdev(values)
                    if len(values) > 1 and all_defined
                    else None,
                ),
                "V9 seed aggregation mismatch",
            )


def validate_paired_contrasts(report, protocol):
    """Check the supplement's arithmetic against the certified per-run values."""
    contrasts = report["paired_contrasts"]
    require(
        [row["contrast_id"] for row in contrasts] == ["primary", "secondary"],
        "Both declared objective contrasts are required",
    )
    runs = {(run["config_id"], run["seed"]): run for run in report["runs"]}
    require(len(runs) == len(report["runs"]), "Duplicate V9 configuration/seed")
    for contrast in contrasts:
        declared = protocol["design"]["objective_comparison"][contrast["contrast_id"]]
        require(
            contrast["direction"] == "CE_minus_CT"
            and contrast["expected_seeds"] == protocol["seeds"]
            and contrast["summary_policy"]
            == "all_declared_pairs_required_for_each_metric"
            and all(
                contrast[key] == declared[key]
                for key in ("control_config", "treatment_config", "temperature")
            ),
            "Objective contrast does not match the prospective design",
        )
        for branch in BRANCHES:
            for metric in METRICS:
                value = contrast["metrics"][branch][metric]
                require(
                    [row["seed"] for row in value["per_seed"]] == protocol["seeds"],
                    "Contrast seed accounting changed",
                )
                differences = []
                for row in value["per_seed"]:
                    control = runs[(contrast["control_config"], row["seed"])][
                        "metrics"
                    ][branch][metric]
                    treatment = runs[(contrast["treatment_config"], row["seed"])][
                        "metrics"
                    ][branch][metric]
                    difference = (
                        treatment - control
                        if control is not None and treatment is not None
                        else None
                    )
                    require(
                        same_number(row["control_value"], control)
                        and same_number(row["treatment_value"], treatment)
                        and same_number(row["difference"], difference),
                        "Paired CE-minus-CT arithmetic mismatch",
                    )
                    require(
                        row["requested_samples"] == 100
                        and row["control_status"]
                        == row["treatment_status"]
                        == "completed",
                        "Contrast does not describe the complete original runs",
                    )
                    if difference is not None:
                        differences.append(difference)
                    if metric == "quality":
                        require(
                            row["control_quality_count"] == round(control * 100)
                            and row["treatment_quality_count"]
                            == round(treatment * 100),
                            "Paired quality numerator mismatch",
                        )
                complete = len(differences) == len(protocol["seeds"])
                require(
                    value["complete"] is complete
                    and value["expected_seed_count"] == len(protocol["seeds"])
                    and value["defined_seed_count"] == len(differences)
                    and same_number(
                        value["mean_difference"],
                        statistics.mean(differences) if complete else None,
                    )
                    and same_number(
                        value["sample_sd"],
                        statistics.stdev(differences) if complete else None,
                    ),
                    "Paired objective summary arithmetic mismatch",
                )


def configuration_csv(records):
    stream = io.StringIO()
    columns = [
        "study",
        "arm_id",
        "config_id",
        "temperature",
        "kernel",
        "samples",
        "seeds",
        "branch",
    ]
    columns += [
        f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sample_sd")
    ]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in records:
        for branch in BRANCHES:
            row = {name: record[name] for name in columns[:6]}
            row.update(seeds="/".join(map(str, record["seeds"])), branch=branch)
            row.update(
                {
                    f"{metric}_{suffix}": record["metrics"][branch][metric][suffix]
                    for metric in METRICS
                    for suffix in ("mean", "sample_sd")
                }
            )
            writer.writerow(row)
    return stream.getvalue().encode()


def make_cover(records, training, protocol, contrasts):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="StudySmall",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(name="StudyTiny", fontName="Helvetica", fontSize=7, leading=9)
    )
    story = []
    width, _ = landscape(A4)
    usable = width - 64

    def add(text, style="StudySmall"):
        story.append(Paragraph(text, styles[style]))

    def table(rows, widths):
        item = Table(
            [
                [Paragraph(escape(str(cell)), styles["StudyTiny"]) for cell in row]
                for row in rows
            ],
            colWidths=widths,
            repeatRows=1,
        )
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde8f2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f2f5f8")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(item)
        story.append(Spacer(1, 8))

    def metric(value, scale=100):
        if value["mean"] is None:
            return "undefined"
        sd = (
            "undefined"
            if value["sample_sd"] is None
            else f'{value["sample_sd"] * scale:.3f}'
        )
        return f'{value["mean"] * scale:.3f} +/- {sd}'

    best = max(
        records,
        key=lambda row: row["metrics"]["released_comparable"]["quality"]["mean"],
    )
    quality = best["metrics"]["released_comparable"]["quality"]["mean"]
    add("GenMol / UDLM study through the paired objective evaluation", "Title")
    add(
        "Completed V5/V6 screens, V7 resource pilot, V8 incident audit, separate V8b CE training and V9 molecular results"
    )
    add(
        f"<b>No superiority established.</b> The highest observed repaired quality among the 22 disclosed engineering configurations is "
        f"{quality * 100:.3f}% ({escape(best['study'])}, {escape(best['config_id'])}); contextual local MDLM quality is 85.8% "
        "and published GenMol V1 quality is 84.6%. This maximum is selected from small exploratory screens. "
        "A higher pilot mean, if present, does not establish superiority or authorize a final candidate promotion."
    )
    quality_deltas = "; ".join(
        f'{row["contrast_id"]} at T={row["temperature"]:.1f}: '
        f'{100 * row["metrics"]["released_comparable"]["quality"]["mean_difference"]:+.3f} percentage points'
        for row in contrasts
    )
    add(
        f"<b>V9 clean CE minus CT repaired quality.</b> {escape(quality_deltas)}. "
        "These are descriptive paired-seed differences; all strict/repaired metrics and every configuration remain disclosed below."
    )
    table(
        [
            ["Study", "Question and complete scope", "Interpretation"],
            [
                "V5",
                "R/S/E x four temperatures; 24 runs / 1,536 requests",
                "Historical 1,000-update batch-16 CT models; temperature exploration",
            ],
            [
                "V6",
                "R/S/E x predictor/Gibbs; 12 runs / 768 requests",
                "128 predictors versus 64 predictors + 64 Gibbs evaluations; mixed gains",
            ],
            [
                "V7",
                "20-update batch-128 CT-E throughput pilot; 2,560 exposures",
                "Resource feasibility only; no molecular evaluation or warm-start reuse",
            ],
            [
                "V8 / V8b",
                "Separate 1,000-update CT and CE adaptations; 128,000 configured exposures each",
                "Original CT controller/campaign failed; CT accepted separately; new CE controller completed",
            ],
            [
                "V9",
                "CT/CE x temperatures 1.0 / 0.5; 8 runs / 800 requests",
                "Primary CE minus CT at 1.0; informed secondary contrast at 0.5",
            ],
            [
                "Local MDLM",
                "50,000 updates; 3 seeds / 3,000 requests",
                "Contextual baseline; training/inference budgets and sample counts differ",
            ],
        ],
        [85, 340, usable - 425],
    )
    add(
        "<b>New evidence and historical context.</b> All 3,104 V5/V6/V9 molecular requests were independently rescored "
        "within their original runs; they are not pooled into a single model estimate. The historical 27-page report "
        "is appended without modifying its original bytes. Its CE-untrained sentence describes the earlier snapshot and "
        "is superseded by the completed V8b/V9 evidence here. Archived failure states remain failures."
    )
    add(
        "<b>Metrics.</b> Validity = valid/requested; uniqueness = unique/valid; quality = unique valid molecules with "
        "QED &gt;= 0.6 and SA &lt;= 4, divided by requests; diversity is the released PyTDC fingerprint metric on unique "
        "valid molecules. Repaired decoding uses SAFE fix=True and largest-component selection by SMILES length; "
        "strict decoding uses fix=False and RDKit validation. Means and sample SD weight seeds equally; two-seed SD "
        "is not a confidence interval. Diversity and configurations are not pooled."
    )
    story.append(PageBreak())
    add("Training, controller outcomes and checkpoint semantics", "Heading1")
    rows = [
        [
            "Run",
            "Updates / global batch / seed / GPUs",
            "Training subprocess seconds",
            "Accepted exposures / examples per second",
            "Observed memory MiB",
        ]
    ]
    for name in ("V7", "V8_CT", "V8b_CE"):
        terminal = training[name]
        config = terminal["plan"]["config"]
        throughput = terminal["end_to_end_training_examples_per_second"]
        rows.append(
            [
                name,
                f'{config["trainer"]["max_steps"]} / {config["loader"]["global_batch_size"]} / {config["seed"]} / {terminal["plan"]["gpu_count"]}',
                f'{terminal["training_subprocess_seconds"]:.3f}',
                "not accepted by failed controller"
                if throughput is None
                else f'{terminal["completed_example_exposures"]:,} / {throughput:.3f}',
                ", ".join(
                    str(x)
                    for x in terminal["max_observed_aggregate_gpu_used_mib"].values()
                ),
            ]
        )
    table(rows, [75, 230, 120, 215, usable - 640])
    add(
        "<b>V8 CT incident.</b> CT reached step 1,000 and exited zero, but the controller immediately found a residual "
        "distributed process group and failed, retaining both leases. CE never started in that campaign. A separately "
        "hashed CPU audit later verified the saved CT checkpoint and absent processes, then released the exact retained "
        "leases. It supports 128,000 configured exposures; the original accepted-exposure and throughput fields remain null. "
        "The exact transient group members were not captured, so the teardown-race explanation remains an inference."
    )
    grace = training["V8b_CE"]["process_group_exit_grace"]
    add(
        f"<b>Separate V8b CE run.</b> Fresh MDLM EMA initialization preserved the original CE settings. A repaired controller "
        f"allowed up to {grace['grace_seconds']:g} seconds for teardown; its group exited after {grace['elapsed_seconds']:.3f} seconds "
        f"with {grace['probe_count']} probes. Its own completion receipt accepted step 1,000, optimizer state, EMA updates, "
        "finite saved tensors and the CE marker/metadata. This does not convert the failed V8 campaign into a success."
    )
    add(
        "<b>Matched training.</b> CT and CE each start fresh from MDLM 50k EMA, seed 1500, global batch 128 = 2 GPUs x "
        "microbatch 16 x accumulation 4; 1,000 updates = 128,000 configured exposures per arm. Both share the empirical "
        "prior with 0.0002 uniform mixture, FiLM, L1 schedule, full corruption alphabet and all-special-token clean-target "
        "mask. V7 and the earlier V5/V6 models used historical masks; V9 is not an isolated continuation of those screens."
    )
    add(
        "<b>Timing limits.</b> Training-subprocess time includes startup and checkpoint saving. Short V7 startup overhead, "
        "one versus two GPUs, sequential runs and external processes prevent a controlled speedup claim. Memory values "
        "are two-second external aggregate GPU observations including other processes, not allocator peaks. Exposures "
        "are configured accounting, not distinct molecules or an independent live row trace. Saved-tensor finiteness "
        "does not prove every intermediate update was finite. CT/CE loss magnitudes are not comparable quality scores."
    )
    table(
        [
            ["Checkpoint", "SHA-256"],
            [
                "V8 CT: separately audited",
                protocol["training"]["arms"]["CT"]["checkpoint_sha256"],
            ],
            [
                "V8b CE: completed separately",
                protocol["training"]["arms"]["CE"]["checkpoint_sha256"],
            ],
            [
                "Common warm-start MDLM 50k EMA",
                training["V8b_CE"]["plan"]["checkpoint_sha256"],
            ],
        ],
        [210, usable - 210],
    )
    story.append(PageBreak())
    add("V9: all four objective settings", "Heading1")
    add(
        "All V9 rows use 200 requests (two 100-request seeds, 1600/1601), EMA weights, 128 predictor evaluations, "
        "no Gibbs corrector, top-p 1.0, endpoint 1e-5, randomness 0 and min_add_len 40. CE clean logits are converted "
        "to LOO at the current noisy state/time before temperature and top-p; CT already predicts raw LOO. The "
        "conversion adds no backbone evaluation. Temperature 1.0 is primary; 0.5 was informed by V5/V6."
    )
    rows = [
        ["Config", "Branch", "Validity %", "Uniqueness %", "Quality %", "Diversity"]
    ]
    for row in records:
        if row["study"] == "V9":
            for branch in BRANCHES:
                rows.append(
                    [
                        row["config_id"],
                        "repaired" if branch == "released_comparable" else "strict",
                        *[
                            metric(
                                row["metrics"][branch][name],
                                1 if name == "diversity" else 100,
                            )
                            for name in METRICS
                        ],
                    ]
                )
    table(rows, [95, 70] + [(usable - 165) / 4] * 4)
    story.append(PageBreak())
    add("V9: paired CE minus CT effects", "Heading1")
    add(
        "Positive deltas favor CE for that metric. Validity/uniqueness/quality deltas are percentage points; "
        "diversity uses its original scale. Values are the equal-seed mean +/- sample SD. An undefined metric "
        "withholds its summary until both declared seeds define it. The appendix retains each per-seed difference."
    )
    rows = [
        [
            "Contrast / T",
            "Branch",
            "Delta validity",
            "Delta uniqueness",
            "Delta quality",
            "Delta diversity",
        ]
    ]
    for contrast in contrasts:
        for branch in BRANCHES:
            values = []
            for name in METRICS:
                value = contrast["metrics"][branch][name]
                values.append(
                    metric(
                        {
                            "mean": value["mean_difference"],
                            "sample_sd": value["sample_sd"],
                        },
                        1 if name == "diversity" else 100,
                    )
                )
            rows.append(
                [
                    f'{contrast["contrast_id"]} / {contrast["temperature"]:.1f}',
                    "repaired" if branch == "released_comparable" else "strict",
                    *values,
                ]
            )
    table(rows, [105, 70] + [(usable - 175) / 4] * 4)
    add(
        "Seed pairing is not molecule-level matching: objectives and sampling paths differ. Both contrasts remain "
        "disclosed even when negative. The fixed design predates these checkpoint molecular outputs; earlier engineering "
        "results were already known, so this is not independent confirmation. Higher diversity alone does not imply "
        "higher validity or molecular quality. The appendix reports every requested metric under both decoding paths."
    )
    add(
        "<b>Reading the appendices.</b> The first appendix is the complete paired V9 report with original per-run "
        "configuration, raw/rescore identities, CE helper provenance, generation counts and contrasts. The second is "
        "the preserved V5/V6 plus baseline overview, including all 18 historical settings and Gibbs ablations. "
        "The companion all_configurations.csv contains all 22 configuration means and SDs. The input manifest hashes "
        "the original reports, raw/run artifacts, training receipts/logs, CT audit and this generator. Large checkpoint "
        "identities are inherited from verified training/generation records; this overview does not reload the weights."
    )
    add(
        "Equal updates/examples and shared hyperparameters do not imply equal compute or separately optimized objectives. "
        "All adaptations inherit the large MDLM pretraining budget; a matched extra-update MDLM control remains absent. "
        "Final UDLM seeds 0/1/2 remain reserved. No automatic promotion, further training or superiority declaration follows this report."
    )
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        leftMargin=32,
        rightMargin=32,
        topMargin=25,
        bottomMargin=25,
        title="GenMol UDLM study through V9",
        invariant=1,
    )
    document.build(story)
    return output.getvalue()


def build_bundle(root, v9_directory, *, workspace=WORKSPACE):
    root = Path(root).resolve(strict=True)
    inputs = Inputs(workspace)
    generator = inputs.read(Path(__file__), "Overview generator source")
    historical, historical_pdf = load_historical(inputs, root)
    v9, protocol, records9 = load_v9(
        inputs, root, Path(v9_directory).resolve(strict=True)
    )
    training = load_training(inputs, root, protocol)
    records = historical["configurations"] + records9
    cover = make_cover(records, training, protocol, v9["paired_contrasts"])
    parts = [
        ("Updated study overview through V9", cover),
        (
            "V9 complete independent report and paired objective contrasts",
            inputs.read(Path(v9_directory) / "report.pdf", "V9 appendix"),
        ),
        ("Historical V5/V6 and local MDLM overview (preserved)", historical_pdf),
    ]
    writer, sections, cursor = PdfWriter(), [], 1
    for title, payload in parts:
        reader = PdfReader(io.BytesIO(payload))
        count = len(reader.pages)
        writer.append(reader, outline_item=title)
        sections.append(
            {
                "title": title,
                "first_page": cursor,
                "last_page": cursor + count - 1,
                "page_count": count,
                "input_pdf_sha256": digest(payload),
            }
        )
        cursor += count
    writer.add_metadata(
        {
            "/Title": "GenMol / UDLM: engineering study through V9",
            "/Subject": "Engineering comparisons; no superiority established",
        }
    )
    combined = io.BytesIO()
    writer.write(combined)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "superiority_established": False,
        "configuration_count": 22,
        "scheduled_runs": 44,
        "independently_rescored_requests": 3104,
        "baseline_requests": 3000,
        "reserved_udlm_final_seeds": [0, 1, 2],
        "formal_promotions": [],
        "snapshot_utc": v9["created_at_utc"],
        "configurations": records,
        "paired_objective_contrasts": v9["paired_contrasts"],
        "paired_gibbs_deltas": historical["paired_gibbs_deltas"],
        "training": training,
        "historical_context": "Original appendix retained unchanged; its CE-untrained status was true at its earlier snapshot.",
        "pdf_sections": sections,
    }
    outputs = {
        "study_overview.pdf": combined.getvalue(),
        "overview_cover.pdf": cover,
        "overview.json": json_bytes(summary),
        "all_configurations.csv": configuration_csv(records),
        "generate_study_overview_v9.py": generator,
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "inputs": inputs.manifest(),
        "generator_sha256": digest(generator),
        "pdf_sections": sections,
        "generation_mode": "CPU aggregation of independently rescored reports; no sampling, GPU probing or checkpoint reload",
        "software": {"pypdf": pypdf.__version__, "reportlab": reportlab.Version},
        "outputs": {
            name: {"sha256": digest(payload), "size_bytes": len(payload)}
            for name, payload in outputs.items()
        },
    }
    outputs["input_hash_manifest.json"] = json_bytes(manifest)
    return outputs, summary


def publish_bundle(destination, outputs, *, workspace=WORKSPACE):
    destination = Path(destination).resolve()
    require(
        destination.is_relative_to(Path(workspace).resolve(strict=True)),
        "Output is outside workspace",
    )
    require(
        not destination.exists(),
        "Output directory must be fresh; original reports cannot be overwritten",
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in outputs.items():
        require(Path(name).name == name, "Unexpected output path")
        with (destination / name).open("xb") as stream:
            stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--v9-report-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(
        not args.output_directory.resolve().exists(), "Output directory must be fresh"
    )
    outputs, summary = build_bundle(args.input_root, args.v9_report_directory)
    publish_bundle(args.output_directory, outputs)
    print(
        json.dumps(
            {
                "output_directory": str(args.output_directory.resolve()),
                "configurations": summary["configuration_count"],
                "independently_rescored_requests": summary[
                    "independently_rescored_requests"
                ],
                "pdf_sha256": digest(outputs["study_overview.pdf"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
