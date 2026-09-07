"""Synthetic terminal evidence only; never opens actual V13 run outputs."""

from collections import Counter
import copy
import csv
import io
import json
import os
from pathlib import Path
import statistics

import pytest
from pypdf import PdfReader, PdfWriter
import yaml

from scripts.udlm import generate_study_overview_v13 as overview
from test_udlm_study_overview_v12 import (
    report_fixture,
    synthetic_cover_summary,
    write_json,
)


def refresh(root, report):
    """Rebind synthetic receipt bytes after a deliberate fixture change."""
    protocol = report["protocol"]["configuration"]
    protocol_ref = write_json(root, "protocol.json", protocol)
    report["protocol"] = dict(protocol_ref, configuration=protocol)
    for entry in protocol["entries"]:
        rows = [r for r in report["runs"] if r["config_id"] == entry["config_id"]]
        controller = rows[0]["controller"]
        receipt = controller["receipt"]
        receipt["identity"].update(
            entry=entry, protocol_sha256=protocol_ref["sha256"], seeds=protocol["seeds"]
        )
        artifacts = {}
        for row in rows:
            for name, ref in row["artifacts"].items():
                path = root / ref["relative_path"]
                ref.update(
                    sha256=overview.old.digest(path.read_bytes()),
                    size_bytes=path.stat().st_size,
                )
                artifacts[ref["relative_path"]] = ref["sha256"]
            if row["status"] == "completed":
                row["independent_rescore"].update(
                    summary_sha256=row["artifacts"]["summary.json"]["sha256"],
                    raw_samples_sha256=row["artifacts"]["raw_samples.csv"]["sha256"],
                )
        receipt["artifacts"] = artifacts
        controller["artifact"] = write_json(
            root,
            f"{protocol['output_root']}/controller_receipts/{entry['attempt_id']}.json",
            receipt,
        )
        for row in rows:
            row["controller"] = controller
    report["aggregates"] = []
    for entry in protocol["entries"]:
        rows = [
            r
            for r in report["runs"]
            if r["config_id"] == entry["config_id"] and r["status"] == "completed"
        ]
        agg = {
            k: entry[k] for k in ("config_id", "candidate_id", "arm_id", "attempt_id")
        }
        agg.update(
            complete=len(rows) == 2,
            completed_seeds=[r["seed"] for r in rows],
            expected_seeds=protocol["seeds"],
            metrics={},
        )
        for branch in overview.old.BRANCHES:
            agg["metrics"][branch] = {}
            for metric in overview.old.METRICS:
                values = [r["metrics"][branch][metric] for r in rows]
                defined = [v for v in values if v is not None]
                agg["metrics"][branch][metric] = dict(
                    mean=(
                        statistics.mean(defined)
                        if defined and len(defined) == len(values)
                        else None
                    ),
                    sample_sd=(
                        statistics.stdev(defined)
                        if len(defined) > 1 and len(defined) == len(values)
                        else None
                    ),
                    defined_seed_count=len(defined),
                )
        report["aggregates"].append(agg)
    counts = Counter(r["status"] for r in report["runs"])
    report.update(
        status="complete" if counts["completed"] == 8 else "incomplete",
        paired_contrasts=overview.reporter._paired_contrasts(protocol, report["runs"]),
        accounting=dict(
            scheduled_runs=8,
            scheduled_requests=800,
            status_counts=dict(counts),
            independently_rescored_requests=100 * counts["completed"],
        ),
    )
    return write_json(root, "report/report.json", report)["sha256"]


def v13_fixture(root):
    report, directory, _ = report_fixture(root)
    protocol = json.loads(
        (
            overview.ROOT
            / "experiments/udlm/protocols/engineering_v13_temperature_space.json"
        ).read_text()
    )
    original_rows = copy.deepcopy(report["runs"])
    report["protocol"]["configuration"] = protocol
    design = protocol["prospective_design"]
    path = root / design["relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((overview.ROOT / design["relative_path"]).read_bytes())
    report["runs"] = []
    for index, entry in enumerate(protocol["entries"]):
        config_bytes = (overview.ROOT / entry["config"]).read_bytes()
        path = root / entry["config"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(config_bytes)
        config = yaml.safe_load(config_bytes)
        sampling = {
            k: v for k, v in config.items() if k not in ("model_path", "num_samples")
        }
        clean = entry["temperature_space"] == "x0_denoiser"
        for offset, seed in enumerate(protocol["seeds"]):
            row = original_rows[index * 2 + offset]
            summary = json.loads(
                (root / row["artifacts"]["summary.json"]["relative_path"]).read_text()
            )
            row.update(
                {
                    k: entry[k]
                    for k in ("config_id", "candidate_id", "arm_id", "attempt_id")
                }
            )
            row.update(
                seed=seed,
                run_directory=f"{protocol['output_root']}/{entry['attempt_id']}/seed_{seed}",
                config=dict(
                    sha256=entry["config_sha256"],
                    source=config,
                    sampling=sampling,
                    effective=config,
                ),
                checkpoint=dict(
                    sha256=entry["checkpoint_sha256"],
                    udlm_prior_metadata=protocol["training"]["arms"][entry["arm_id"]][
                        "udlm_prior_metadata"
                    ],
                    udlm_denoiser_metadata=protocol["training"]["arms"][
                        entry["arm_id"]
                    ]["udlm_denoiser_metadata"],
                ),
                source={"commit": overview.V13_SOURCE},
            )
            mode = dict(overview.TEMPERATURE_PROTOCOL) if clean else {}
            row["generation_protocol"].update(mode)
            row["independent_rescore"]["identity"]["generation"] = dict(
                row["generation_protocol"]
            )
            row["independent_rescore"]["seed"] = seed
            count = [[30, 50], [40, 45], [60, 70], [55, 60]][index][offset]
            for branch in overview.old.BRANCHES:
                row["metrics"][branch].update(quality_count=count, quality=count / 100)
            row["independent_rescore"]["metrics"] = row["metrics"]
            summary.update(
                seed=seed,
                config=row["config"],
                checkpoint=row["checkpoint"],
                git=row["source"],
                run={"generation_protocol": row["generation_protocol"]},
            )
            summary_ref = write_json(
                root, row["run_directory"] + "/summary.json", summary
            )
            raw_path = root / row["run_directory"] / "raw_samples.csv"
            raw_path.write_text("raw_model_text\n" + "C\n" * 100)
            row["artifacts"] = {
                "summary.json": summary_ref,
                "raw_samples.csv": {"relative_path": str(raw_path.relative_to(root))},
            }
            row["controller"]["receipt"]["identity"]["source"] = {
                "head": overview.V13_SOURCE,
                "upstream": overview.V13_SOURCE,
            }
            report["runs"].append(row)
    return report, directory, refresh(root, report)


def load(root, report, directory):
    digest = refresh(root, report)
    return overview.load_v13(overview.old.Inputs(root), root, directory, digest)


@pytest.mark.parametrize(
    "failure,accepted", [(None, 800), ("failed", 600), ("invalid", 700)]
)
def test_terminal_panel_counts_both_signs_and_withholding(tmp_path, failure, accepted):
    report, directory, _ = v13_fixture(tmp_path)
    if failure:
        selected = report["runs"][2:4] if failure == "failed" else report["runs"][2:3]
        for row in selected:
            row.update(status=failure, metrics=None, raw_row_count=0)
            if failure == "failed":
                row["controller"].update(status="failed")
                row["controller"]["receipt"]["return_code"] = 7
    result, _, records, _ = load(tmp_path, report, directory)
    assert result["accounting"]["independently_rescored_requests"] == accepted
    assert sum(r["accepted_samples"] for r in records) == accepted
    primary, secondary = result["paired_contrasts"]
    assert primary["metrics"]["strict"]["quality"]["mean_difference"] == (
        None if failure else 0.025
    )
    assert secondary["metrics"]["strict"]["quality"]["mean_difference"] == -0.075
    if failure:
        assert records[1]["metrics"]["strict"]["quality"]["mean"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "order",
        "rescore_order",
        "bridge_boolean",
        "raw_mode",
        "raw_bytes",
        "protocol",
        "pending",
    ],
)
def test_rejects_unready_or_misbound_evidence(tmp_path, mutation):
    report, directory, _ = v13_fixture(tmp_path)
    selected = report["runs"][2]
    if mutation == "source":
        selected["controller"]["receipt"]["identity"]["source"] = dict(
            head="a" * 40, upstream="a" * 40
        )
    elif mutation == "order":
        selected["generation_protocol"]["temperature_application"] = "after"
    elif mutation == "rescore_order":
        selected["independent_rescore"]["identity"]["generation"][
            "temperature_application"
        ] = "after"
    elif mutation == "bridge_boolean":
        selected["independent_rescore"]["identity"]["generation"][
            "reverse_bridge_temperature"
        ] = True
    elif mutation == "raw_mode":
        report["runs"][0]["generation_protocol"].update(overview.TEMPERATURE_PROTOCOL)
    elif mutation == "raw_bytes":
        (
            tmp_path / selected["artifacts"]["raw_samples.csv"]["relative_path"]
        ).write_text("raw_model_text\nC\n")
    elif mutation == "protocol":
        report["protocol"]["configuration"]["claim"] = "different"
    else:
        selected["controller"]["status"] = "pending"
    with pytest.raises(ValueError):
        load(tmp_path, report, directory)


def fake_preserved_bundle(root, protocol, monkeypatch, workspace):
    directory = root / overview.V12_DIRECTORY
    directory.mkdir(parents=True)
    history = synthetic_cover_summary()
    history.update(
        status="complete",
        configuration_count=30,
        superiority_established=False,
        v12_protocol={"training": protocol["training"]},
        limitations=[],
    )
    writer = PdfWriter()
    for _ in range(78):
        writer.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    writer.write(stream)
    outputs = {
        "study_overview.pdf": stream.getvalue(),
        "overview.json": overview.old.json_bytes(history),
    }
    for name, data in outputs.items():
        (directory / name).write_bytes(data)
    original = root / "original/source.py"
    acquired = root / "acquisition/source.py"
    original.parent.mkdir()
    acquired.parent.mkdir()
    original.write_bytes(b"CHANGED CURRENT SOURCE")
    acquired.write_bytes(b"EXACT HISTORICAL SOURCE")
    a, b = str(original.relative_to(workspace)), str(acquired.relative_to(workspace))
    aliases = {a: b}
    manifest = dict(
        outputs={
            name: dict(sha256=overview.old.digest(data), size_bytes=len(data))
            for name, data in outputs.items()
        },
        inputs=[
            dict(
                workspace_relative_path=a,
                sha256=overview.old.digest(acquired.read_bytes()),
                size_bytes=acquired.stat().st_size,
            )
        ],
    )
    payload = overview.old.json_bytes(manifest)
    (directory / "input_hash_manifest.json").write_bytes(payload)
    monkeypatch.setattr(overview, "V12_SOURCE_ALIASES", aliases)
    monkeypatch.setattr(overview, "V12_MANIFEST_SHA", overview.old.digest(payload))
    monkeypatch.setattr(
        overview, "V12_PDF_SHA", overview.old.digest(outputs["study_overview.pdf"])
    )
    return directory, acquired


def test_complete_bundle_reproduces_bytes_preserves_appendices_and_archives_source(
    tmp_path, monkeypatch
):
    report, directory, digest = v13_fixture(tmp_path)
    # Replace only the synthetic placeholder PDF; keep its hash bound.
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    writer.write(stream)
    (directory / "report.pdf").write_bytes(stream.getvalue())
    report["report_artifacts"]["pdf_sha256"] = overview.old.digest(stream.getvalue())
    digest = refresh(tmp_path, report)
    workspace = Path(os.path.commonpath([tmp_path, overview.ROOT]))
    historical, acquired = fake_preserved_bundle(
        tmp_path, report["protocol"]["configuration"], monkeypatch, workspace
    )
    originals = {
        p: p.read_bytes()
        for p in (
            directory / "report.pdf",
            historical / "study_overview.pdf",
            historical / "input_hash_manifest.json",
        )
    }
    result, summary = overview.build_bundle(
        tmp_path, directory, digest, workspace=workspace
    )
    repeated, _ = overview.build_bundle(
        tmp_path, directory, digest, workspace=workspace
    )
    assert result == repeated
    assert summary["accounting"]["total"] == dict(
        configurations=34,
        scheduled_runs=68,
        scheduled_requests=5504,
        accepted_requests=5504,
        status_counts={"completed": 68},
    )
    assert (
        len(
            list(csv.DictReader(io.StringIO(result["all_configurations.csv"].decode())))
        )
        == 68
    )
    pages = PdfReader(io.BytesIO(result["study_overview.pdf"])).pages
    assert (
        len(pages)
        == len(PdfReader(io.BytesIO(result["overview_cover.pdf"])).pages) + 79
    )
    assert all(path.read_bytes() == value for path, value in originals.items())
    manifest = json.loads(result["input_hash_manifest.json"])
    relocation = manifest["preserved_source_relocations"][0]
    assert result[relocation["archived_output_filename"]] == acquired.read_bytes()
    assert not any(
        r["workspace_relative_path"] == str(acquired.relative_to(workspace))
        for r in manifest["inputs"]
    )
    for name, claim in manifest["outputs"].items():
        assert overview.old.digest(result[name]) == claim["sha256"]
    acquired.write_bytes(b"WRONG HISTORICAL SOURCE")
    with pytest.raises(ValueError, match="hash mismatch"):
        overview.build_bundle(tmp_path, directory, digest, workspace=workspace)


def test_cover_labels_synthetic_and_missing_mean(tmp_path):
    report, directory, _ = v13_fixture(tmp_path)
    _, _, records, _ = load(tmp_path, report, directory)
    summary = synthetic_cover_summary()
    summary["configurations"] += records
    summary.update(
        accounting=overview.previous.accounting(
            summary["configurations"], studies=("V5", "V6", "V9", "V10", "V12", "V13")
        ),
        v13_temperature_space_contrasts=report["paired_contrasts"],
    )
    summary["v13_temperature_space_contrasts"][0]["metrics"]["strict"]["quality"][
        "mean_difference"
    ] = None
    payload = overview.make_cover(summary, synthetic=True)
    reader = PdfReader(io.BytesIO(payload))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert all("SYNTHETIC TEST DATA" in page.extract_text() for page in reader.pages)
    for name in overview.V13_PANEL["config_ids"]:
        assert name in text
    for phrase in (
        "5,504",
        "unavailable",
        "primary",
        "secondary",
        "not a confidence interval",
        "V11 prelaunch",
        "V8 CT",
    ):
        assert phrase in text
    assert len(reader.pages) <= 4


def test_missing_future_report_publishes_nothing(tmp_path):
    output = tmp_path / "never_created"
    assert (
        overview.main(
            [
                "--input-root",
                str(tmp_path),
                "--v13-report-directory",
                str(tmp_path / "missing"),
                "--v13-report-sha256",
                "0" * 64,
                "--output-directory",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


@pytest.mark.parametrize("value", [None, "", "not-a-digest"])
def test_explicit_report_pin_is_required_before_reading(tmp_path, value):
    with pytest.raises(ValueError, match="explicit terminal"):
        overview.load_v13(
            overview.old.Inputs(tmp_path), tmp_path, tmp_path / "missing", value
        )
