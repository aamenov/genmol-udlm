"""Append a terminal V13 temperature panel to the pinned published V12 overview.

CPU saved-evidence aggregation only. Missing/pending inputs cannot publish;
terminal failures retain unavailable metrics. No model, checkpoint or GPU reads.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "scripts/udlm/generate_study_overview_v12.py").is_file()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.udlm import generate_study_overview_v12 as previous  # noqa: E402

old, reporter, require = previous.old, previous.reporter, previous.require
V12_DIRECTORY = Path("output/udlm/study_overview_v12_20260907")
V12_MANIFEST_SHA = "1f41402e69a04e996c2836ca7254b0bb98f44b87bf70b702b4dcf467dcd1f955"
V12_PDF_SHA = "b6be1949a3f163e90d4437e5e1f66c2a406839fe791fda6d97edc9f9af29e250"
V13_PROTOCOL_SHA = "daf8c8108772e0ebcbdd39f15dab07478fe8945ba6b33f4b68858a8536ad7319"
V13_SOURCE = "0fdc54d91baade2f84b48de9223ceb19696fadb7"
V13_PANEL = dict(
    study="V13",
    study_id="engineering-v13-ce-temperature-space",
    seeds=(2100, 2101),
    config_ids=(
        "e_ce_raw_t050",
        "e_ce_clean_t050",
        "mask_ce_raw_t050",
        "mask_ce_clean_t050",
    ),
    direction="clean_temperature_minus_raw_loo_temperature",
)
TEMPERATURE_PROTOCOL = {
    "temperature_space": "x0_denoiser",
    "temperature_application": "clean_denoiser_before_loo_conversion",
    "reverse_bridge_temperature": 1.0,
}
# The old manifest remains unchanged. Only these two source locations moved
# after publication; identical content survives in the frozen V12 source tree.
V12_SOURCE_ALIASES = {
    f"run_sources/udlm_genmol_worktree/scripts/udlm/{name}": f"run_sources/udlm_overview_v12_worktree/scripts/udlm/{name}"
    for name in ("generate_study_overview_v12.py", "report_exploration.py")
}


def load_v13(inputs, root, directory, expected_report_sha):
    require(
        isinstance(expected_report_sha, str)
        and len(expected_report_sha) == 64
        and all(char in "0123456789abcdef" for char in expected_report_sha),
        "An explicit terminal V13 report SHA-256 is required",
    )
    report, protocol, records, masks = previous.load_terminal_panel(
        inputs, root, directory, expected_report_sha, **V13_PANEL
    )
    require(
        report["protocol"]["sha256"] == V13_PROTOCOL_SHA,
        "V13 protocol is not the fixed published panel",
    )
    for run in report["runs"]:
        require(
            run["controller"]["receipt"]["identity"]["source"]
            == {"head": V13_SOURCE, "upstream": V13_SOURCE},
            "V13 frozen generation source differs",
        )
        if run["status"] != "completed":
            continue
        sampling, generation = run["config"]["sampling"], run["generation_protocol"]
        require(
            sampling["softmax_temp"] == 0.5 and sampling["raw_loo_top_p"] == 1.0,
            "V13 fixed temperature/top-p differs",
        )
        clean = sampling.get("temperature_space", "raw_loo") == "x0_denoiser"
        for identity in (
            generation,
            run["independent_rescore"]["identity"]["generation"],
        ):
            actual = {
                key: identity[key] for key in TEMPERATURE_PROTOCOL if key in identity
            }
            require(
                actual == (TEMPERATURE_PROTOCOL if clean else {}),
                "V13 temperature application identity differs",
            )
            if clean:
                require(
                    type(identity["reverse_bridge_temperature"]) is float,
                    "V13 bridge temperature type differs",
                )
    return report, protocol, records, masks


def make_cover(summary, *, synthetic=False):
    """A compact addition; all older tables/configurations remain in appendices."""
    stream = io.BytesIO()
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    story = []

    def text(value, style="BodyText"):
        story.extend([Paragraph(escape(value), styles[style]), Spacer(1, 6)])

    def table(rows, widths):
        wrapped = [
            [Paragraph(escape(str(cell)), styles["BodyText"]) for cell in row]
            for row in rows
        ]
        result = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef4")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([result, Spacer(1, 9)])

    prefix = "SYNTHETIC TEST DATA — " if synthetic else ""
    text(prefix + "GenMol / UDLM study through V13", "Title")
    counts = summary["accounting"]["total"]
    text(
        f"{counts['configurations']} configurations; {counts['scheduled_runs']} scheduled seed runs; {counts['scheduled_requests']:,} planned requests; {counts['accepted_requests']:,} independently rescored accepted requests. Status: {summary['status']}. No superiority or formal promotion is established."
    )
    text(
        "V13 tests temperature placement using the same frozen CE checkpoint within each empirical/MASK-rich prior pair. Both modes use T=0.5, 128 predictor network evaluations per molecule, top-p=1, no Gibbs, and seeds 2100/2101 with 100 requests each. Raw mode converts CE to LOO before tempering; clean mode tempers CE logits first, converts to LOO, and uses bridge temperature one."
    )
    text("Four V13 settings", "Heading2")
    table(
        [["Configuration / accepted", "Scoring", "Quality % ± SD", "Diversity ± SD"]]
        + [
            [
                f"{r['config_id']}\n{r['accepted_samples']}/{r['samples']}; {r['statuses']}",
                "repaired" if branch == "released_comparable" else "strict",
                previous.format_metric(r["metrics"][branch]["quality"]),
                previous.format_metric(
                    r["metrics"][branch]["diversity"], percentage=False
                ),
            ]
            for r in summary["configurations"]
            if r["study"] == "V13"
            for branch in old.BRANCHES
        ],
        [185, 55, 135, 135],
    )
    text(
        "Both declared contrasts: clean temperature minus raw-LOO temperature",
        "Heading2",
    )
    text(
        "Entries below are signed within-seed differences followed by their equal-seed mean and sample SD. Quality is in percentage points; diversity is in raw units. Primary is empirical CE; secondary is MASK-rich CE. Pairing is by seed, not matched molecular trajectories."
    )
    rows = [["Contrast / scoring", "Metric", "Seed 2100; 2101", "Mean ± sample SD"]]
    for contrast in summary["v13_temperature_space_contrasts"]:
        for branch in old.BRANCHES:
            for metric in ("quality", "diversity"):
                value = contrast["metrics"][branch][metric]
                scale = 1 if metric == "diversity" else 100
                pairs = "; ".join(
                    (
                        "unavailable"
                        if row["difference"] is None
                        else f"{scale * row['difference']:+.3f}"
                    )
                    for row in value["per_seed"]
                )
                formatted = previous.format_metric(
                    {
                        "mean": value["mean_difference"],
                        "sample_sd": value["sample_sd"],
                    },
                    percentage=metric != "diversity",
                    signed=True,
                )
                rows.append(
                    [
                        f"{contrast['contrast_id']} / {'repaired' if branch == 'released_comparable' else 'strict'}",
                        metric,
                        pairs,
                        formatted,
                    ]
                )
    table(rows, [160, 55, 150, 145])
    text("Definitions and limits", "Heading2")
    text(
        "Quality = within-seed unique valid molecules with QED >= 0.6 and SA <= 4 / all requested samples. QED measures drug-likeness (higher favored); SA estimates synthetic accessibility (lower favored). Validity = valid/requested; uniqueness = unique valid/valid. Diversity uses the released PyTDC metric on unique valid molecules. Repaired scoring applies SAFE repair and keeps the largest component; strict SAFE disables repair. Both remove tokenizer special tokens before decoding, so exact final control-token counts are recorded separately in overview.json."
    )
    text(
        "Every ± value is sample SD across two seeds, not a confidence interval. Missing or failed seeds withhold configuration and paired means/SD; retained partial source aggregates are explicitly separate. All four validity/uniqueness/quality/diversity outcomes and pair differences remain in the untouched V13 appendix and JSON/CSV."
    )
    text(
        "These are adaptive engineering pilots across 34 settings, not independent confirmatory tests. V5/V6 use 128 requests per setting; V9/V10/V12/V13 use 200. The local MDLM reference uses 3 × 1,000 requests, repaired quality 85.8%; strict quality is not directly comparable to that repaired reference. The baseline's 3,000 requests and failed V4 diagnostic's 32 requests are outside UDLM accepted totals. Different study seeds cannot support paired cross-study effects. Final UDLM seeds 0/1/2 remain reserved."
    )
    text(
        "V13 adds no training. Historical R/S/E adaptation used 1,000 updates × batch16 = 16,000 exposures. Empirical CE and MASK-rich CE each used a fresh MDLM 50k EMA initialization, 1,000 updates × batch128 = 128,000 exposures, seed1500; the changed prior changes denoising difficulty. V8 CT's failed post-exit controller and separate checkpoint audit, unstarted original CE arm, separate successful V8b CE, V11 prelaunch failure with no trained checkpoint, and successful V11b MASK CE remain preserved. Inference costs are not uniform across all studies: V10 includes 512 NFE, four times 128. V13's planned molecule-NFE budget is 102,400."
    )
    text(
        "Appendix order: untouched terminal V13 report, then the preserved 78-page overview through V12 (including its original reports, training/configuration details, prior/checkpoint identities, failures, and historical caveats). Older appendices retain the statements current when they were written. All original bytes, source relocations, inputs, and output hashes are recorded in the new manifest."
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        if synthetic:
            canvas.setFillColor(colors.red)
            canvas.drawString(36, 20, "SYNTHETIC TEST DATA — NOT V13 MOLECULAR RESULTS")
        else:
            canvas.drawString(
                36, 20, "Engineering pilot; sample SD is not a confidence interval"
            )
        canvas.drawRightString(A4[0] - 36, 20, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(
        stream,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=36,
        invariant=1,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()


def build_bundle(root, v13_directory, expected_v13_sha, *, workspace=old.WORKSPACE):
    root, directory = Path(root).resolve(strict=True), Path(v13_directory).resolve()
    if not (directory / "report.json").is_file():
        raise previous.NotReady(
            "V13 independent report is not available; no overview published"
        )
    inputs = old.Inputs(workspace)
    generator = inputs.read(Path(__file__), "Current V13 overview generator")
    for module in (
        previous,
        old,
        previous.resolution,
        previous.resolution.lexical,
        reporter,
    ):
        inputs.read(Path(module.__file__), "Current imported helper source")
    report, protocol, records, masks = load_v13(
        inputs, root, directory, expected_v13_sha
    )
    pdf12 = previous.load_preserved_bundle(
        inputs,
        root / V12_DIRECTORY,
        V12_MANIFEST_SHA,
        "study_overview.pdf",
        78,
        path_aliases=V12_SOURCE_ALIASES,
    )
    require(old.digest(pdf12) == V12_PDF_SHA, "Published V12 PDF differs")
    archived = inputs.read_json(
        root / V12_DIRECTORY / "overview.json", "Preserved V12 overview data"
    )
    require(
        archived["status"] == "complete"
        and archived["configuration_count"] == 30
        and archived["accounting"]["total"]
        == previous.accounting(archived["configurations"])["total"]
        and archived["accounting"]["total"]["accepted_requests"] == 4704
        and archived["superiority_established"] is False,
        "Preserved V12 accounting differs",
    )
    require(
        protocol["training"] == archived["v12_protocol"]["training"],
        "V13 changed the accepted training provenance",
    )
    summary = copy.deepcopy(archived)
    summary["configurations"] += records
    counts = previous.accounting(
        summary["configurations"], studies=("V5", "V6", "V9", "V10", "V12", "V13")
    )
    require(
        (
            counts["total"]["configurations"],
            counts["total"]["scheduled_runs"],
            counts["total"]["scheduled_requests"],
        )
        == (34, 68, 5504),
        "Combined scheduled scope differs",
    )
    summary.update(
        status=(
            "complete"
            if counts["total"]["accepted_requests"] == 5504
            else "terminal_with_failures"
        ),
        accounting=counts,
        configuration_count=34,
        v13_source_report_sha256=expected_v13_sha,
        v13_source_report_status=report["status"],
        v13_original_aggregates=report["aggregates"],
        v13_temperature_space_contrasts=report["paired_contrasts"],
        v13_runs=report["runs"],
        v13_token_counts=masks,
        v13_protocol=protocol,
        limitations=archived["limitations"] + protocol["limitations"],
        preserved_v12_manifest_sha256=V12_MANIFEST_SHA,
        preserved_v12_pdf_sha256=V12_PDF_SHA,
    )
    cover = make_cover(summary)
    writer, sections, cursor = PdfWriter(), [], 1
    for title, payload in (
        ("V13 update and combined accounting", cover),
        (
            "Untouched terminal V13 report",
            inputs.read(directory / "report.pdf", "V13 PDF appendix"),
        ),
        ("Preserved 78-page overview through V12", pdf12),
    ):
        reader = PdfReader(io.BytesIO(payload))
        count = len(reader.pages)
        writer.append(reader, outline_item=title)
        sections.append(
            dict(
                title=title,
                first_page=cursor,
                last_page=cursor + count - 1,
                page_count=count,
                input_pdf_sha256=old.digest(payload),
            )
        )
        cursor += count
    writer.add_metadata(
        {
            "/Title": "GenMol / UDLM engineering study through V13",
            "/Subject": "Engineering pilots; no superiority established",
        }
    )
    combined = io.BytesIO()
    writer.write(combined)
    summary["pdf_sections"] = sections
    outputs = {
        "study_overview.pdf": combined.getvalue(),
        "overview_cover.pdf": cover,
        "overview.json": old.json_bytes(summary),
        "all_configurations.csv": previous.configuration_csv(summary["configurations"]),
        "generate_study_overview_v13.py": generator,
    }
    # Preserve exact old bytes; future closure uses output copies, not the
    # frozen feature worktree from which these bytes were acquired.
    relocations = []
    acquired = set(V12_SOURCE_ALIASES.values())
    for original, relocated in V12_SOURCE_ALIASES.items():
        payload = inputs.payloads[(inputs.workspace / relocated).resolve()]
        filename = "preserved_v12_" + Path(original).name
        outputs[filename] = payload
        relocations.append(
            dict(
                original_workspace_relative_path=original,
                archived_output_filename=filename,
                acquired_from_workspace_relative_path=relocated,
                sha256=old.digest(payload),
                size_bytes=len(payload),
            )
        )
    manifest_inputs = [
        item
        for item in inputs.manifest()
        if item["workspace_relative_path"] not in acquired
    ]
    outputs["input_hash_manifest.json"] = old.json_bytes(
        dict(
            schema_version=1,
            status=summary["status"],
            inputs=manifest_inputs,
            generator_sha256=old.digest(generator),
            pdf_sections=sections,
            preserved_source_relocations=relocations,
            generation_mode="CPU saved-evidence aggregation; no GPU probes, checkpoint/model reads or new chemistry scoring",
            software={
                "pypdf": old.pypdf.__version__,
                "reportlab": old.reportlab.Version,
            },
            outputs={
                name: {"sha256": old.digest(payload), "size_bytes": len(payload)}
                for name, payload in outputs.items()
            },
        )
    )
    return outputs, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--v13-report-directory", type=Path, required=True)
    parser.add_argument("--v13-report-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    require(
        not args.output_directory.resolve().exists(), "Output directory must be fresh"
    )
    try:
        outputs, summary = build_bundle(
            args.input_root, args.v13_report_directory, args.v13_report_sha256
        )
    except (previous.NotReady, FileNotFoundError) as error:
        print(json.dumps({"status": "not_ready", "reason": str(error)}))
        return 2
    old.publish_bundle(args.output_directory, outputs)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "accounting": summary["accounting"]["total"],
                "pdf_sha256": old.digest(outputs["study_overview.pdf"]),
                "input_manifest_sha256": old.digest(
                    outputs["input_hash_manifest.json"]
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
