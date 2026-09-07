"""Rebuild a deterministic study overview from completed, hash-bound reports.

CPU only: no model/checkpoint loading, GPU probing, generation or chemistry
rescoring. The upstream reports contain the independent chemistry rescoring.
This script rechecks their artifacts and seed aggregation, then appends their
original PDF pages. Run with the project .venv and a fresh --output-directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import pypdf
import reportlab
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

WORKSPACE = Path("/home/aidar.alimbayev/Documents/genmolv2")
MAIN = WORKSPACE / "run_sources/udlm_genmol_worktree"
BASELINE_PDF_SHA = "c3acd529f909da4e223904608c1e33aaedaa1a3bd4c5766378963c757911818e"
BASELINE_CKPT_SHA = "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
METRICS = ("validity", "uniqueness", "quality", "diversity")
BRANCHES = ("released_comparable", "strict")


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Inputs:
    def __init__(self):
        self.payloads = {}
        self.roles = defaultdict(set)

    def read(self, path, role, expected=None):
        path = path.resolve()
        if not path.is_relative_to(WORKSPACE):
            raise ValueError(f"Input outside workspace: {path}")
        payload = path.read_bytes()
        if expected is not None and sha(payload) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        if path in self.payloads and payload != self.payloads[path]:
            raise ValueError(f"Input changed during generation: {path}")
        self.payloads[path] = payload
        self.roles[path].add(role)
        return payload

    def manifest(self):
        records = []
        for path, payload in sorted(self.payloads.items()):
            if path.read_bytes() != payload:
                raise ValueError(f"Input changed before publication: {path}")
            records.append(
                {
                    "workspace_relative_path": str(path.relative_to(WORKSPACE)),
                    "sha256": sha(payload),
                    "size_bytes": len(payload),
                    "roles": sorted(self.roles[path]),
                }
            )
        return records


def close(actual, expected):
    return (
        actual is expected
        if expected is None
        else actual is not None
        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    )


def load_screen(inputs, version, expected_runs, expected_requests, expected_configs):
    directory = MAIN / f"output/udlm/engineering_v{version}_reports/complete"
    report = json.loads(
        inputs.read(directory / "report.json", f"V{version} independent report")
    )
    if (
        report["status"] != "complete"
        or report["superiority_established"] is not False
        or report["unexpected_run_directories"]
        or len(report["aggregates"]) != expected_configs
        or report["accounting"]
        != {
            "scheduled_runs": expected_runs,
            "scheduled_requests": expected_requests,
            "independently_rescored_requests": expected_requests,
            "status_counts": {"completed": expected_runs},
        }
    ):
        raise ValueError(f"V{version} is not the expected complete engineering screen")
    pdf = inputs.read(
        directory / "report.pdf",
        f"V{version} original PDF",
        report["report_artifacts"]["pdf_sha256"],
    )
    inputs.read(
        directory / "report.csv",
        f"V{version} report CSV",
        report["report_artifacts"]["csv_sha256"],
    )
    protocol = report["protocol"]
    parsed_protocol = json.loads(
        inputs.read(
            MAIN / protocol["relative_path"], f"V{version} protocol", protocol["sha256"]
        )
    )
    if parsed_protocol != protocol["configuration"]:
        raise ValueError("Embedded protocol differs from original bytes")
    groups = defaultdict(list)
    for run in report["runs"]:
        if (
            run["status"] != "completed"
            or run["requested_samples"] != 64
            or run["raw_row_count"] != 64
            or run["independent_rescore"]["status"] != "exact_match"
            or run["generation_protocol"]["nfe"] != 128
        ):
            raise ValueError(
                "Run is incomplete, not independently matched, or has unexpected NFE"
            )
        for name, artifact in run["artifacts"].items():
            payload = inputs.read(
                MAIN / artifact["relative_path"],
                f"V{version} run {name}",
                artifact["sha256"],
            )
            if (
                name == "raw_samples.csv"
                and len(list(csv.DictReader(io.StringIO(payload.decode())))) != 64
            ):
                raise ValueError("Raw CSV count changed")
        for branch in BRANCHES:
            for metric in METRICS:
                if not close(
                    run["metrics"][branch][metric],
                    run["independent_rescore"]["metrics"][branch][metric],
                ):
                    raise ValueError("Run metrics disagree with independent rescoring")
        groups[run["config_id"]].append(run)
    records = []
    for aggregate in report["aggregates"]:
        runs = sorted(groups[aggregate["config_id"]], key=lambda run: run["seed"])
        expected_seeds = [1200, 1201] if version == 5 else [1300, 1301]
        if not aggregate["complete"] or [run["seed"] for run in runs] != expected_seeds:
            raise ValueError("Configuration does not contain both expected seeds")
        for branch in BRANCHES:
            for metric in METRICS:
                values = [
                    r["metrics"][branch][metric]
                    for r in runs
                    if r["metrics"][branch][metric] is not None
                ]
                m = aggregate["metrics"][branch][metric]
                mean = statistics.mean(values) if values else None
                sd = statistics.stdev(values) if len(values) > 1 else None
                if (
                    m["defined_seed_count"] != len(values)
                    or not close(m["mean"], mean)
                    or not close(m["sample_sd"], sd)
                ):
                    raise ValueError("Seed aggregation mismatch")
        effective = runs[0]["config"]["effective"]
        gibbs = effective.get("gibbs_corrector", False)
        if gibbs:
            for run in runs:
                gp = run["generation_protocol"]
                if (
                    gp.get("predictor_transitions_per_molecule") != 64
                    or gp.get("corrector_steps_per_molecule") != 64
                ):
                    raise ValueError(
                        "Gibbs allocation is not 64 predictor +64 corrector"
                    )
        records.append(
            dict(
                aggregate,
                study=f"V{version}",
                temperature=effective["softmax_temp"],
                gibbs_corrector=gibbs,
                kernel="64P+64G" if gibbs else "128P",
                samples=128,
                seeds=expected_seeds,
                runs=runs,
            )
        )
    for entry in parsed_protocol["entries"]:
        inputs.read(
            MAIN / entry["config"],
            f"V{version} sampling configuration",
            entry["config_sha256"],
        )
    return report, pdf, records


def paired_deltas(records):
    result = []
    for arm in ("R", "S", "E"):
        pair = {r["gibbs_corrector"]: r for r in records if r["arm_id"] == arm}
        if set(pair) != {False, True}:
            raise ValueError("Expected one same-arm predictor/Gibbs pair")
        control, treatment = pair[False], pair[True]
        for c, g in zip(control["runs"], treatment["runs"], strict=True):
            if (
                c["seed"] != g["seed"]
                or c["checkpoint"]["sha256"] != g["checkpoint"]["sha256"]
            ):
                raise ValueError("Paired comparison changes seed or checkpoint")
            c_config, g_config = dict(c["config"]["effective"]), dict(
                g["config"]["effective"]
            )
            c_config.pop("gibbs_corrector", None)
            g_config.pop("gibbs_corrector", None)
            if c_config != g_config:
                raise ValueError(
                    "Paired comparison changes configuration beyond corrector"
                )
        for branch in BRANCHES:
            metrics = {}
            for metric in METRICS:
                values = [
                    g["metrics"][branch][metric] - c["metrics"][branch][metric]
                    for c, g in zip(control["runs"], treatment["runs"], strict=True)
                    if g["metrics"][branch][metric] is not None
                    and c["metrics"][branch][metric] is not None
                ]
                metrics[metric] = {
                    "mean": statistics.mean(values) if values else None,
                    "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                    "paired_seed_count": len(values),
                    "values_by_seed": dict(zip(control["seeds"], values)),
                }
            result.append({"arm_id": arm, "branch": branch, "metrics": metrics})
    return result


def make_cover(records, deltas, baseline, training, snapshot):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallStudy",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(name="TinyStudy", fontName="Helvetica", fontSize=6.8, leading=9)
    )
    styles.add(
        ParagraphStyle(
            name="HeaderStudy", parent=styles["TinyStudy"], textColor=colors.white
        )
    )
    width, height = landscape(A4)
    usable = width - 64
    story = []

    def p(text, style="SmallStudy"):
        return Paragraph(text, styles[style])

    def add(text, style="SmallStudy"):
        story.append(p(text, style))

    def table(rows, widths=None):
        cooked = [
            [p(str(v), "HeaderStudy" if index == 0 else "TinyStudy") for v in row]
            for index, row in enumerate(rows)
        ]
        item = Table(cooked, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#eef3f7")],
                    ),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#16324f")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(item)

    def fmt(value, metric, signed=False):
        if value["mean"] is None:
            return "undefined"
        scale = 1 if metric == "diversity" else 100
        precision = 3 if metric == "diversity" else 2
        sign = "+" if signed else ""
        result = format(value["mean"] * scale, f"{sign}.{precision}f")
        sd = value["sample_sd"]
        return result + (
            " +/- " + format(sd * scale, f".{precision}f")
            if sd is not None
            else " (SD undefined)"
        )

    best = max(
        records, key=lambda r: r["metrics"]["released_comparable"]["quality"]["mean"]
    )
    best_q = best["metrics"]["released_comparable"]["quality"]["mean"]
    base_q = baseline["aggregate_metrics"]["released_comparable"]["quality"]["mean"]
    add("GenMol / UDLM engineering study", "Title")
    add(
        f"Complete V5 temperature screen and V6 Gibbs comparison | source snapshot {escape(snapshot)}"
    )
    add(
        "<b>No superiority established. No formal promotion or final candidate lock.</b> "
        f"The highest observed repaired quality across the 18 engineering configurations is {100*best_q:.2f}% "
        f'({best["study"]}, {escape(best["config_id"])}), versus {100*base_q:.2f}% for the frozen local MDLM baseline. '
        "This descriptive maximum is selected from a small screen; it is not a confirmatory estimate."
    )
    table(
        [
            [
                "Evidence",
                "Configurations / runs / requested samples",
                "Seed and compute budget",
            ],
            [
                "V5 temperature screen",
                "12 configurations; 24 runs; 1,536 samples, all independently rescored",
                "R/S/E x temperatures 0.50, 0.70, 0.85, 1.00; seeds 1200/1201; 128 predictor NFEs",
            ],
            [
                "V6 Gibbs comparison",
                "6 configurations; 12 runs; 768 samples, all independently rescored",
                "R/S/E x predictor/Gibbs; seeds 1300/1301; temperature 0.50; 128 total NFEs",
            ],
            [
                "Frozen local MDLM baseline",
                "1 configuration; 3 runs; 3,000 samples",
                "Seeds 0/1/2; released confidence sampling; inference and training compute not matched",
            ],
        ],
        [145, 310, usable - 455],
    )
    story.append(Spacer(1, 9))
    add(
        "<b>128 samples per engineering configuration versus 3,000 baseline samples.</b> "
        "Every engineering result averages two 64-sample seeds. All 2,304 V5+V6 requests remain disclosed; "
        "they are not pooled to estimate a single model configuration. V6 temperature 0.50 was chosen using partial V5 results, "
        "so it is an informed engineering choice, not a separately selected global optimum."
    )
    add(
        "<b>Training and checkpoints.</b> Each R/S/E model starts from the same 50,000-update MDLM EMA and adds "
        "1,000 updates at global batch 16 (16,000 example exposures; one GPU; microbatch 2; accumulation 8; seed 17). "
        "All three retain their original checkpoint bytes. The local MDLM training used batch 2,046 for 50,000 updates "
        "(102.3 million requested exposures); the paper used batch 2,048. A matched extra-update MDLM control is still absent. "
        "Scheduler and FiLM choices were selected on E loss, then shared across arms."
    )
    add(
        "<b>Arm definitions.</b> R uses released uniform UDLM with its known loss/forward schedule mismatch; "
        "S repairs that schedule using uniform categorical diffusion; E uses the same repaired process and empirical SAFE "
        "prior with a 0.0002 uniform mixture. None is a published reproduction of a fully optimized UDLM molecular model."
    )
    add(
        "<b>Metrics and uncertainty.</b> Validity = valid/requested; uniqueness = unique/valid; quality = distinct valid "
        "molecules with QED &gt;= 0.6 and SA &lt;= 4, divided by requested samples. QED summarizes drug-likeness; SA is the "
        "synthetic-accessibility score (lower is easier). Diversity is the released PyTDC fingerprint diversity on unique "
        "valid molecules. Means and sample SD average seeds equally; diversity is not pooled. Two-seed SD is fragile, "
        "not a confidence interval or significance test; configuration selection and molecular dependence add uncertainty."
    )
    add(
        "<b>Decoding and next work.</b> Repaired metrics use SAFE fix=True and select the largest component by SMILES "
        "string length. Strict metrics use fix=False plus RDKit validation, without repair or component selection. "
        "Final UDLM seeds 0/1/2 remain reserved (the historical baseline already used those seeds). CE denoising with "
        "checkpoint-bound CE-to-LOO conversion is implemented and CPU-tested, but remains untrained and has no molecular "
        "results in this report. The archived V4 failure remains failed and contributes no ranked configuration."
    )

    story.append(PageBreak())
    add("Every configuration: mean +/- sample SD across two seeds", "Heading1")
    add(
        "V/U/Q are percentages; D is diversity. P = predictor backbone evaluation; G = fresh Gibbs evaluation. "
        "All engineering rows use 128 samples/configuration and top-p=1.00. Baseline rows are context, not matched tests."
    )
    header = [
        "Study / arm",
        "T",
        "Kernel",
        "Repair V%",
        "Repair U%",
        "Repair Q%",
        "Repair D",
        "Strict V%",
        "Strict U%",
        "Strict Q%",
        "Strict D",
    ]
    rows = [header]
    ordered = sorted(
        records,
        key=lambda r: (
            r["study"],
            "RSE".index(r["arm_id"]),
            r["temperature"],
            r["gibbs_corrector"],
        ),
    )
    for record in ordered:
        rows.append(
            [
                f'{record["study"]} / {record["arm_id"]}',
                f'{record["temperature"]:.2f}',
                record["kernel"],
                *[
                    fmt(record["metrics"][branch][metric], metric)
                    for branch in BRANCHES
                    for metric in METRICS
                ],
            ]
        )
    rows.append(
        [
            "Local MDLM",
            "0.50",
            "released",
            *[
                fmt(baseline["aggregate_metrics"][b][m], m)
                for b in BRANCHES
                for m in METRICS
            ],
        ]
    )
    table(rows, [53, 28, 55] + [(usable - 136) / 8] * 8)
    add(
        "The local MDLM row averages three 1,000-sample seeds. Published GenMol V1 repaired reference: "
        "validity 100.0%, uniqueness 99.7%, quality 84.6%, diversity 0.818 (paper run means). Hardware, training, "
        "data provenance and sample counts differ; higher diversity alone does not establish better molecules."
    )

    story.append(PageBreak())
    add("Within-arm V6 Gibbs deltas at fixed total NFE", "Heading1")
    add(
        "Treatment minus predictor control, pairing seed 1300 with 1300 and 1301 with 1301 at the same checkpoint. "
        "Control: 128 predictor evaluations. Treatment: 64 predictor transitions + 64 fresh random-scan Gibbs updates. "
        "The paired RNG streams diverge because Gibbs changes draw counts. These are descriptive paired-seed deltas, "
        "not per-molecule paired effects or a confidence interval. Temperature is 0.50 in every pair."
    )
    rows = [
        [
            "Arm",
            "Decoding",
            "Delta validity (pp)",
            "Delta uniqueness (pp)",
            "Delta quality (pp)",
            "Delta diversity",
        ]
    ]
    for delta in deltas:
        rows.append(
            [
                delta["arm_id"],
                "repaired" if delta["branch"] == "released_comparable" else "strict",
                *[fmt(delta["metrics"][m], m, signed=True) for m in METRICS],
            ]
        )
    table(rows, [35, 90] + [(usable - 125) / 4] * 4)
    story.append(Spacer(1, 10))
    add(
        "The means show mixed effects: E improves repaired quality but loses strict quality; S improves both quality "
        "means while strict validity falls; R declines in both quality and validity branches. There is no universal "
        "Gibbs gain. V5 and V6 use different seeds, so their differences are not paired comparisons."
    )
    add(
        "A random-scan Gibbs step resamples one uniformly chosen editable coordinate per sequence and may leave "
        "its value unchanged. Exact stationarity requires true compatible LOO conditionals and untempered sampling; "
        "these learned predictions at temperature 0.50 form an approximate corrector. This comparison also trades "
        "away half the predictor transitions, so it evaluates that allocation, not adding free extra compute."
    )
    rows = [
        ["Checkpoint / budget", "SHA-256 (original source bytes; not reserialized)"]
    ]
    rows.append(["MDLM 50k EMA warm-start source", BASELINE_CKPT_SHA])
    for arm in ("R", "S", "E"):
        rows.append(
            [
                f"{arm}: +1,000 updates / +16,000 exposures",
                training[arm]["final_checkpoint"]["sha256"],
            ]
        )
    table(rows, [230, usable - 230])
    story.append(Spacer(1, 9))
    add(
        "<b>Artifact trail.</b> The input manifest hashes both report JSON/CSV/PDF files, all 36 run summaries and raw "
        "CSVs, both protocols and their configuration bytes, the three continuation training summaries, the local "
        "baseline aggregate and its verified PDF, and this generator. Checkpoint identities are inherited from the "
        "independently validated generation/training records; this overview does not reload multi-gigabyte checkpoints."
    )
    add(
        "<b>Unmodified appendices follow:</b> complete independently rescored V5 report; complete independently rescored "
        "V6 report; frozen local MDLM baseline report. They retain original page labels and historical context. "
        "The companion CSV/JSON contain all configuration means, SDs, paired deltas, hashes and original identifiers. "
        "No significance test, superiority declaration, formal promotion, or reserved-seed UDLM generation occurred here."
    )
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        leftMargin=32,
        rightMargin=32,
        topMargin=25,
        bottomMargin=25,
        title="GenMol UDLM engineering study overview",
        author="GenMol research workspace",
        invariant=1,
    )

    def footer(canvas, _doc):
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(width - 32, 13, f"Study overview | {_doc.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    destination = args.output_directory.resolve()
    if not destination.is_relative_to(WORKSPACE):
        raise ValueError("Output must remain inside the workspace")
    allowed = {"generate_study_overview.py"}
    if destination.exists() and set(p.name for p in destination.iterdir()) - allowed:
        raise ValueError(
            "Output directory must be fresh (the generator itself may exist)"
        )
    inputs = Inputs()
    generator = inputs.read(Path(__file__), "reproducible overview generator")
    v5, v5_pdf, records5 = load_screen(inputs, 5, 24, 1536, 12)
    v6, v6_pdf, records6 = load_screen(inputs, 6, 12, 768, 6)
    baseline_pdf = inputs.read(
        WORKSPACE / "output/pdf/genmol_denovo_50000_benchmark.pdf",
        "frozen local baseline PDF",
        BASELINE_PDF_SHA,
    )
    baseline = json.loads(
        inputs.read(
            WORKSPACE / "output/benchmarks/denovo_50000/report/aggregate.json",
            "frozen local baseline aggregate",
        )
    )
    if (
        baseline["status"] != "completed"
        or baseline["training_context"]["checkpoint"]["sha256"] != BASELINE_CKPT_SHA
    ):
        raise ValueError("Unexpected local baseline")
    training = {}
    for arm in ("R", "S", "E"):
        entry = next(
            e for e in v6["protocol"]["configuration"]["entries"] if e["arm_id"] == arm
        )
        summary_path = (
            MAIN / Path(entry["checkpoint"]).parent.parent / "training_summary.json"
        )
        summary = json.loads(
            inputs.read(summary_path, f"{arm} continuation training summary")
        )
        account = summary["training_accounting"]
        if (
            summary["status"] != "completed"
            or summary["final_checkpoint"]["sha256"] != entry["checkpoint_sha256"]
            or account["optimizer_updates"] != 1000
            or account["total_requested_example_exposures"] != 16000
            or account["effective_global_examples_per_optimizer_step"] != 16
            or summary["startup"]["verified_mdlm_warm_start_report"]["source_sha256"]
            != BASELINE_CKPT_SHA
        ):
            raise ValueError("Training budget or initialization mismatch")
        training[arm] = summary
    records = records5 + records6
    deltas = paired_deltas(records6)
    snapshot = max(v5["created_at_utc"], v6["created_at_utc"])
    cover_pdf = make_cover(records, deltas, baseline, training, snapshot)
    parts = [
        ("Study overview", cover_pdf),
        ("V5 full independent report", v5_pdf),
        ("V6 full independent report", v6_pdf),
        ("Local MDLM baseline report", baseline_pdf),
    ]
    writer, page_ranges, cursor = PdfWriter(), [], 1
    for title, data in parts:
        reader = PdfReader(io.BytesIO(data))
        count = len(reader.pages)
        writer.append(reader, outline_item=title)
        page_ranges.append(
            {
                "title": title,
                "first_page": cursor,
                "last_page": cursor + count - 1,
                "page_count": count,
                "input_pdf_sha256": sha(data),
            }
        )
        cursor += count
    writer.add_metadata(
        {
            "/Title": "GenMol / UDLM: completed engineering study V5 and V6",
            "/Author": "GenMol research workspace",
            "/Subject": "Engineering evidence; no superiority established",
        }
    )
    combined = io.BytesIO()
    writer.write(combined)
    public_records = [{k: v for k, v in row.items() if k != "runs"} for row in records]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "snapshot_utc": snapshot,
        "superiority_established": False,
        "formal_promotions": [],
        "ce_status": "implemented_untrained",
        "configuration_count": 18,
        "scheduled_runs": 36,
        "independently_rescored_requests": 2304,
        "samples_per_engineering_configuration": 128,
        "baseline_requests": 3000,
        "reserved_udlm_final_seeds": [0, 1, 2],
        "configurations": public_records,
        "paired_gibbs_deltas": deltas,
        "pdf_sections": page_ranges,
        "training": {
            a: {
                "checkpoint": s["final_checkpoint"]["sha256"],
                "accounting": s["training_accounting"],
            }
            for a, s in training.items()
        },
    }
    csv_output = io.StringIO()
    fields = [
        "study",
        "arm_id",
        "config_id",
        "temperature",
        "kernel",
        "samples",
        "seeds",
        "branch",
    ]
    fields += [
        f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sample_sd")
    ]
    csv_writer = csv.DictWriter(csv_output, fieldnames=fields)
    csv_writer.writeheader()
    for record in public_records:
        for branch in BRANCHES:
            row = {k: record[k] for k in fields[:6]}
            row.update(
                samples=record["samples"],
                seeds="/".join(map(str, record["seeds"])),
                branch=branch,
            )
            row.update(
                {
                    f"{metric}_{suffix}": record["metrics"][branch][metric][suffix]
                    for metric in METRICS
                    for suffix in ("mean", "sample_sd")
                }
            )
            csv_writer.writerow(row)
    outputs = {
        "study_overview.pdf": combined.getvalue(),
        "overview_cover.pdf": cover_pdf,
        "overview.json": (
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode(),
        "all_configurations.csv": csv_output.getvalue().encode(),
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "inputs": inputs.manifest(),
        "generator_sha256": sha(generator),
        "software": {"reportlab": reportlab.Version, "pypdf": pypdf.__version__},
        "generation_mode": "CPU deterministic aggregation and PDF concatenation; no new molecular sampling or rescoring",
        "pdf_sections": page_ranges,
        "outputs": {
            name: {"sha256": sha(data), "size_bytes": len(data)}
            for name, data in outputs.items()
        },
    }
    outputs["input_hash_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    destination.mkdir(parents=True, exist_ok=True)
    copied_generator = destination / "generate_study_overview.py"
    if not copied_generator.exists():
        with copied_generator.open("xb") as stream:
            stream.write(generator)
    for name, data in outputs.items():
        with (destination / name).open("xb") as stream:
            stream.write(data)
    print(
        json.dumps(
            {
                "output_directory": str(destination),
                "pages": cursor - 1,
                "configurations": 18,
                "independently_rescored_requests": 2304,
                "pdf_sha256": sha(outputs["study_overview.pdf"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
