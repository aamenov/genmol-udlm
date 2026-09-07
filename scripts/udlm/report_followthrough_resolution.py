"""Independently rescore the resolution screen and report paired differences.

Run only after generation ends. A complete eight-run census is required;
partial output cannot select a winner. Existing reports are never overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.udlm import report_exploration as reports
from scripts.udlm.audit_molecular_failures import lexical_flags

PROTOCOL = Path("experiments/udlm/protocols/followthrough_resolution_r1.json")
OUTPUT = Path("output/udlm/followthrough_resolution_r1")


def paired_contrasts(report):
    """Require the complete, independently rescored paired factorial design."""
    if report.get("status") != "complete" or report.get("unexpected_run_directories"):
        raise ValueError("The independently rescored study must be complete")
    runs = report["runs"]
    expected = {(arm, nfe, seed) for arm in ("CT", "CE")
                for nfe in (128, 512) for seed in (17300, 17301)}
    indexed = {}
    for run in runs:
        if run["status"] != "completed":
            raise ValueError("All resolution runs must pass independent rescoring")
        key = (run["arm_id"], run["generation_protocol"]["nfe"], run["seed"])
        if key not in expected or key in indexed:
            raise ValueError("Unexpected or duplicate resolution run")
        indexed[key] = run
    if set(indexed) != expected:
        raise ValueError("Incomplete resolution comparison")
    contrasts = []
    for arm in ("CT", "CE"):
        for branch in reports.BRANCHES:
            for metric in reports.METRICS:
                differences = []
                for seed in (17300, 17301):
                    control = indexed[arm, 128, seed]["metrics"][branch][metric]
                    treatment = indexed[arm, 512, seed]["metrics"][branch][metric]
                    if control is None or treatment is None:
                        differences.append(None)
                    else:
                        differences.append(treatment - control)
                defined = all(value is not None for value in differences)
                contrasts.append({
                    "arm": arm, "decoding": branch, "metric": metric,
                    "seed_17300_difference": differences[0],
                    "seed_17301_difference": differences[1],
                    "mean_difference": statistics.mean(differences) if defined else None,
                    "sample_sd": statistics.stdev(differences) if defined else None,
                })
    return contrasts


def main():
    report = reports.build_report(PROTOCOL, OUTPUT)
    contrasts = paired_contrasts(report)
    syntax = []
    for run in report["runs"]:
        path = ROOT / run["run_directory"] / "raw_samples.csv"
        counts = {"odd_ring": 0, "bad_parentheses": 0, "either": 0}
        payload = path.read_bytes()
        expected = run["artifacts"]["raw_samples.csv"]["sha256"]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Raw molecule bytes changed after independent rescoring")
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
        if len(rows) != 100:
            raise ValueError("Resolution run does not have exactly 100 requested rows")
        for row in rows:
            flags = lexical_flags(row["raw_safe"])
            for key in ("odd_ring", "bad_parentheses"):
                counts[key] += int(flags[key])
            counts["either"] += int(any(flags.values()))
        syntax.append({"attempt_id": run["attempt_id"], "seed": run["seed"], **counts})
    report["resolution_contrasts"] = contrasts
    report["lexical_diagnostics"] = syntax
    directory = ROOT / "output/udlm/followthrough_resolution_r1_reports/complete"
    paths = reports.write_report(report, directory)
    with (directory / "paired_contrasts.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    styles = getSampleStyleSheet()
    story = [Paragraph("UDLM resolution: paired diagnostic", styles["Title"]),
             Paragraph("Differences are 512 minus 128 predictor evaluations. "
                       "Both objectives use temperature 1.0, identical checkpoint bytes, "
                       "and 100 requests for each of seeds 17300 and 17301. "
                       "512 steps cost four times the model calls. Two seeds provide "
                       "an engineering diagnostic, not a superiority claim.", styles["BodyText"]),
             Spacer(1, 12)]
    cells = [["Arm", "Decode", "Metric", "Mean delta", "Seed SD"]]
    for row in contrasts:
        values = ["undefined" if row[key] is None else f"{row[key]:+.5f}"
                  for key in ("mean_difference", "sample_sd")]
        cells.append([row["arm"], row["decoding"].replace("released_comparable", "repaired"),
                      row["metric"], *values])
    table = Table(cells, repeatRows=1)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story.extend([table, Spacer(1, 12), Paragraph(
        "The accompanying report.pdf contains configurations, checkpoint identities, "
        "per-run metrics, GPU mappings, timings, and the contextual MDLM comparator. "
        "report.json additionally records lexical error counts and both per-seed differences.",
        styles["BodyText"])])
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer).build(story)
    with (directory / "paired_contrasts.pdf").open("xb") as stream:
        stream.write(buffer.getvalue())
    print(json.dumps({"status": report["status"], "artifacts": paths,
                      "contrasts": contrasts}, sort_keys=True))


if __name__ == "__main__":
    main()
