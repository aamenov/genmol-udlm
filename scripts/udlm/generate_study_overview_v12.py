"""Combine terminal engineering evidence through V12 without sampling or model reads.

The historical appendices are hash-pinned and preserved. Missing or pending V12
inputs are not ready; terminal failures remain failures and never add accepted
requests. Every output directory must be new.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import copy
import csv
import io
import json
import os
from pathlib import Path
import struct
import sys

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "scripts/udlm/generate_study_overview_v9.py").is_file()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.udlm import generate_study_overview_v9 as old  # noqa: E402
from scripts.udlm import analyze_v10_resolution as resolution  # noqa: E402
from scripts.udlm import report_exploration as reporter  # noqa: E402

V9_DIRECTORY = Path("output/udlm/study_overview_v9_20260907")
V9_MANIFEST_SHA = "9f0614f5e84c01709ce920b7124b1b5151775e9d1eb73b2e7265fdb93564edec"
V10_DIRECTORY = Path("output/udlm/engineering_v10_reports/complete")
V10_REPORT_SHA = "8d9b7df5797338b6a768cfd3489af6abdadabe0cf48a8cec212086cfc7714d16"
RESOLUTION_DIRECTORY = Path("output/udlm/engineering_v10_reports/resolution_analysis")
RESOLUTION_MANIFEST_SHA = (
    "a25de263f7fb835133a04aa47e52ee3ddd2d79337c4ede7646fdf3932f8b8308"
)
require = old.require


class NotReady(ValueError):
    """A required future artifact or terminal controller is unavailable."""


def load_preserved_bundle(
    inputs, directory, expected_manifest, pdf_name, pages, *, path_aliases=None
):
    manifest = inputs.read_json(
        directory / "input_hash_manifest.json",
        "Preserved bundle manifest",
        expected_manifest,
    )
    for name, claim in manifest["outputs"].items():
        require(Path(name).name == name, "Unexpected preserved output path")
        payload = inputs.read(
            directory / name, "Preserved bundle output", claim["sha256"]
        )
        require(len(payload) == claim["size_bytes"], "Preserved output size differs")
    for claim in manifest["inputs"]:
        payload = inputs.read(
            inputs.workspace
            / (path_aliases or {}).get(
                claim["workspace_relative_path"], claim["workspace_relative_path"]
            ),
            "Preserved upstream evidence",
            claim["sha256"],
        )
        require(len(payload) == claim["size_bytes"], "Preserved input size differs")
    pdf = inputs.read(directory / pdf_name, "Preserved PDF appendix")
    require(
        len(PdfReader(io.BytesIO(pdf)).pages) == pages,
        "Preserved PDF page count differs",
    )
    return pdf


def metric_records(report, protocol, study):
    """Validate source aggregation, withholding cover means on incomplete pairs."""
    records = []
    require(
        len(report["aggregates"]) == len(protocol["entries"]),
        "Missing configuration aggregate",
    )
    for entry in protocol["entries"]:
        rows = sorted(
            (row for row in report["runs"] if row["config_id"] == entry["config_id"]),
            key=lambda row: row["seed"],
        )
        require(
            [row["seed"] for row in rows] == protocol["seeds"],
            "Duplicated or missing declared seed",
        )
        matches = [
            row
            for row in report["aggregates"]
            if row["config_id"] == entry["config_id"]
        ]
        require(len(matches) == 1, "Duplicate configuration aggregate")
        aggregate = copy.deepcopy(matches[0])
        successful = [row for row in rows if row["status"] == "completed"]
        complete = len(successful) == len(rows)
        require(
            aggregate["complete"] is complete
            and aggregate["completed_seeds"] == [row["seed"] for row in successful]
            and aggregate["expected_seeds"] == protocol["seeds"],
            "Aggregate completion differs",
        )
        old.validate_seed_aggregate(aggregate, successful)
        if not complete:
            for branch in old.BRANCHES:
                for metric in old.METRICS:
                    aggregate["metrics"][branch][metric].update(
                        mean=None, sample_sd=None
                    )
        config = next((row["config"]["sampling"] for row in successful), None)
        if config is None:
            config = protocol["_sampling_configs"][entry["config_id"]]
        aggregate.update(
            study=study,
            temperature=config["softmax_temp"],
            kernel=f"{config['num_steps']}P",
            gibbs_corrector=False,
            samples=len(rows) * protocol["num_samples"],
            accepted_samples=len(successful) * protocol["num_samples"],
            seeds=protocol["seeds"],
            statuses=dict(Counter(row["status"] for row in rows)),
            sampling_configuration=config,
        )
        records.append(aggregate)
    return records


def token_counts(summary, rescore_identity):
    """Count exact final editable occurrences, using the saved mask denominator."""
    audit = summary["sampled_token_control_audit"]
    rows, columns = audit["rows"], audit["columns"]
    require(
        type(rows) is int
        and type(columns) is int
        and 0 < rows <= 1000
        and 0 < columns <= 256,
        "Invalid token audit shape",
    )
    size = rows * columns

    def decoded(name):
        value = audit[name]
        raw = base64.b64decode(value["data_base64"], validate=True)
        require(
            old.digest(raw) == value["decoded_sha256"]
            and len(raw) == value["decoded_byte_count"],
            "Token audit byte hash/length differs",
        )
        return raw

    raw_ids, raw_mask = decoded("final_sampled_ids"), decoded("editable_mask")
    require(
        audit["final_sampled_ids"]["dtype"] == "uint16"
        and audit["final_sampled_ids"]["byte_order"] == "little"
        and len(raw_ids) == 2 * size
        and audit["editable_mask"]["bit_order"] == "msb0"
        and audit["editable_mask"]["logical_bit_count"] == size
        and len(raw_mask) == (size + 7) // 8,
        "Token audit encoding differs",
    )
    ids = struct.unpack(f"<{size}H", raw_ids)
    editable = [bool(raw_mask[i // 8] & (1 << (7 - i % 8))) for i in range(size)]
    controls = {"unk": 0, "bos": 1, "eos": 2, "pad": 3, "mask": 4}
    require(audit["control_token_ids"] == controls, "Control token identity differs")
    require(all(value < 1880 for value in ids), "Final token outside model vocabulary")
    counts = {
        name: sum(enabled and token == value for token, enabled in zip(ids, editable))
        for name, value in controls.items()
    }
    require(
        counts
        == audit["control_token_counts"]["final_sampled_editable_positions"]
        == rescore_identity["control_token_counts"]["final_sampled_editable_positions"],
        "Rescored control counts differ",
    )
    require(
        rescore_identity["final_sampled_ids_sha256"]
        == audit["final_sampled_ids"]["decoded_sha256"]
        and rescore_identity["editable_mask_sha256"]
        == audit["editable_mask"]["decoded_sha256"]
        and rescore_identity["exact_batch_decode_match"] is True,
        "Independent token audit identity differs",
    )
    return {
        "editable_positions": sum(editable),
        "final_control_counts": counts,
        "rows": rows,
        "rows_with_mask": sum(
            any(
                editable[i] and ids[i] == controls["mask"]
                for i in range(row * columns, (row + 1) * columns)
            )
            for row in range(rows)
        ),
    }


def load_v12(inputs, root, directory, expected_report_sha):
    """Preserve the original V12 entry point and fixed panel identity."""
    return load_terminal_panel(inputs, root, directory, expected_report_sha)


def load_terminal_panel(
    inputs,
    root,
    directory,
    expected_report_sha,
    *,
    study="V12",
    study_id="engineering-v12-ce-mask-prior",
    seeds=(2000, 2001),
    config_ids=("e_ce_t100", "e_ce_t050", "mask_ce_t100", "mask_ce_t050"),
    direction="MASK_minus_empirical",
):
    """Shared saved-evidence checks; callers still pin their concrete protocol."""
    if not (directory / "report.json").is_file():
        raise NotReady("V12 independent report is not available")
    report = inputs.read_json(
        directory / "report.json", "Terminal V12 report", expected_report_sha
    )
    require(
        report["superiority_established"] is False
        and not report["unexpected_run_directories"],
        "Unexpected V12 claim or extra runs",
    )
    require(
        report["baselines"] == reporter.BASELINES, "Contextual baseline values differ"
    )
    reference = report["protocol"]
    protocol = inputs.read_json(
        root / reference["relative_path"],
        "Checkpoint-bound V12 protocol",
        reference["sha256"],
    )
    require(
        protocol == reference["configuration"]
        and protocol["study_id"] == study_id
        and protocol["seeds"] == list(seeds)
        and protocol["num_samples"] == 100
        and protocol["nfe"] == 128,
        "Unexpected V12 protocol",
    )
    entries = {entry["config_id"]: entry for entry in protocol["entries"]}
    require(
        len(protocol["entries"]) == 4 and set(entries) == set(config_ids),
        "V12 must retain all four settings",
    )
    for extension in ("csv", "pdf"):
        inputs.read(
            directory / f"report.{extension}",
            "Original V12 report appendix",
            report["report_artifacts"][f"{extension}_sha256"],
        )
    inputs.read(
        directory / "paired_contrasts.csv",
        "V12 signed prior contrasts",
        report["report_artifacts"]["paired_contrasts_csv_sha256"],
    )
    design = protocol["prospective_design"]
    inputs.read(
        root / design["relative_path"], "Prospective V12 design", design["sha256"]
    )
    import yaml

    configs = {
        key: yaml.safe_load(
            inputs.read(
                root / entry["config"], "V12 inference config", entry["config_sha256"]
            )
        )
        for key, entry in entries.items()
    }
    slots = {
        (entry["config_id"], seed)
        for entry in entries.values()
        for seed in protocol["seeds"]
    }
    require(
        len(report["runs"]) == len(slots)
        and {(row["config_id"], row["seed"]) for row in report["runs"]} == slots,
        "V12 seed slots differ",
    )
    controls, masks = {}, []
    for run in report["runs"]:
        controller = run["controller"]
        if controller["status"] == "pending" or run["status"] == "pending":
            raise NotReady("V12 still has pending controller or seed results")
        entry = entries[run["config_id"]]
        require(
            all(
                run[key] == entry[key]
                for key in ("attempt_id", "candidate_id", "config_id", "arm_id")
            )
            and run["requested_samples"] == protocol["num_samples"]
            and run["run_directory"]
            == f"{protocol['output_root']}/{entry['attempt_id']}/seed_{run['seed']}"
            and controller["artifact"]["relative_path"]
            == f"{protocol['output_root']}/controller_receipts/{entry['attempt_id']}.json",
            "V12 run/controller namespace differs from its declared slot",
        )
        receipt = inputs.reference(
            root, controller["artifact"], "V12 terminal controller"
        )
        require(
            receipt == controller["receipt"]
            and type(receipt["return_code"]) is int
            and controller["status"]
            == ("completed" if receipt["return_code"] == 0 else "failed"),
            "Invalid V12 controller receipt",
        )
        identity = receipt["identity"]
        require(
            identity
            == {
                "entry": entry,
                "seeds": protocol["seeds"],
                "num_samples": protocol["num_samples"],
                "protocol_sha256": reference["sha256"],
                "source": identity["source"],
            }
            and identity["source"]["head"] == identity["source"]["upstream"],
            "V12 controller identity differs",
        )
        actual = controls.setdefault(
            entry["attempt_id"], {"receipt": receipt, "artifacts": {}}
        )
        require(
            run["status"] in {"completed", "failed", "invalid"}, "Nonterminal V12 run"
        )
        for name, artifact in run["artifacts"].items():
            require(
                name in {"summary.json", "raw_samples.csv", "failure_receipt.json"}
                and artifact["relative_path"] == f"{run['run_directory']}/{name}",
                "Unexpected V12 artifact path",
            )
            payload = inputs.read(
                root / artifact["relative_path"],
                "V12 original seed artifact",
                artifact["sha256"],
            )
            actual["artifacts"][artifact["relative_path"]] = old.digest(payload)
        if run["status"] != "completed":
            require(
                run.get("metrics") is None and run.get("raw_row_count", 0) == 0,
                "Failed/invalid seed was counted as accepted",
            )
            continue
        require(
            receipt["return_code"] == 0
            and set(run["artifacts"]) == {"summary.json", "raw_samples.csv"},
            "Accepted V12 run has failed evidence",
        )
        summary = inputs.reference(
            root, run["artifacts"]["summary.json"], "V12 accepted summary"
        )
        raw = inputs.read(
            root / run["artifacts"]["raw_samples.csv"]["relative_path"],
            "V12 accepted raw rows",
        )
        rescore = run["independent_rescore"]
        require(
            rescore["status"] == "exact_match"
            and rescore["summary_sha256"] == run["artifacts"]["summary.json"]["sha256"]
            and rescore["raw_samples_sha256"] == old.digest(raw),
            "V12 independent rescore binds different bytes",
        )
        require(
            run["raw_row_count"]
            == len(list(csv.DictReader(io.StringIO(raw.decode()))))
            == 100,
            "V12 raw row count differs",
        )
        old.validate_run_slot(run, summary, receipt, entry, protocol)
        require(
            summary["config"] == run["config"]
            and summary["checkpoint"] == run["checkpoint"]
            and summary["git"] == run["source"]
            and summary["runtime_seconds"] == run["runtime_seconds"]
            and summary["run"]["generation_protocol"] == run["generation_protocol"],
            "V12 report changed saved identity",
        )
        require(
            run["config"]["sha256"] == entry["config_sha256"]
            and run["checkpoint"]["sha256"] == entry["checkpoint_sha256"]
            and run["config"]["source"] == configs[run["config_id"]],
            "V12 checkpoint/config differs",
        )
        sampling = run["config"]["sampling"]
        require(
            all(
                sampling[key] == value
                for key, value in configs[run["config_id"]].items()
                if key not in {"model_path", "num_samples"}
            ),
            "Normalized sampling differs from frozen YAML",
        )
        trained = protocol["training"]["arms"][entry["arm_id"]]
        require(
            run["checkpoint"]["udlm_prior_metadata"] == trained["udlm_prior_metadata"]
            and run["checkpoint"]["udlm_denoiser_metadata"]
            == trained["udlm_denoiser_metadata"]
            and run["generation_protocol"]["inference_weights"]["source"] == "ema"
            and run["generation_protocol"]["inference_weights"]["ema_applied"] is True,
            "Generation prior/CE/EMA differs from training provenance",
        )
        require(
            sampling["parameterization"] == entry["parameterization"] == "x0_denoiser"
            and sampling["prior_variant"] == entry["prior_variant"]
            and sampling["prior_metadata_sha256"] == entry["prior_metadata_sha256"]
            and sampling["num_steps"] == run["generation_protocol"]["nfe"] == 128
            and not sampling.get("gibbs_corrector", False),
            "V12 sampling law differs",
        )
        for branch in old.BRANCHES:
            for metric in old.METRICS:
                require(
                    old.same_number(
                        run["metrics"][branch][metric],
                        rescore["metrics"][branch][metric],
                    ),
                    "V12 score differs from independent rescore",
                )
        masks.append(
            dict(
                config_id=run["config_id"],
                seed=run["seed"],
                **token_counts(
                    summary, rescore["identity"]["sampled_token_control_audit"]
                ),
            )
        )
    for control in controls.values():
        require(
            control["receipt"]["artifacts"] == control["artifacts"],
            "V12 controller artifact closure differs",
        )
    statuses = dict(Counter(row["status"] for row in report["runs"]))
    accounting = {
        "scheduled_runs": len(slots),
        "scheduled_requests": len(slots) * protocol["num_samples"],
        "status_counts": statuses,
        "independently_rescored_requests": sum(
            row.get("raw_row_count", 0)
            for row in report["runs"]
            if row["status"] == "completed"
        ),
    }
    require(
        report["accounting"] == accounting
        and report["status"]
        == ("complete" if statuses.get("completed") == len(slots) else "incomplete"),
        "V12 report accounting/status differs",
    )
    contrasts = reporter._paired_contrasts(protocol, report["runs"])
    require(
        report["paired_contrasts"] == contrasts
        and {row["contrast_id"] for row in contrasts} == {"primary", "secondary"}
        and all(row["direction"] == direction for row in contrasts),
        "V12 signed contrasts differ",
    )
    decorated = dict(protocol, _sampling_configs=configs)
    records = metric_records(report, decorated, study)
    return report, protocol, records, masks


def load_prior_training(inputs, root, protocol):
    training = protocol["training"]
    require(
        set(training["arms"]) == {"E_CE", "MASK_CE"}, "Unexpected V12 training arms"
    )
    accepted = {}
    for arm, spec in training["arms"].items():
        terminal = old.load_training_terminal(
            inputs, root, spec["terminal_receipt"], "completed"
        )
        plan, checkpoint = terminal["plan"], terminal["checkpoint"]
        train_protocol = inputs.reference(
            root, spec["protocol"], "V12 training protocol"
        )
        require(
            plan["protocol"] == train_protocol
            and plan["protocol_sha256"] == spec["protocol"]["sha256"],
            "Training protocol differs from terminal plan",
        )
        for name in ("request", "launch"):
            record = inputs.reference(
                root, spec[name + "_receipt"], "V12 training authority"
            )
            require(
                spec[name + "_receipt"]["sha256"] == terminal[name + "_sha256"]
                and record["source"] == terminal["source"]
                and record["plan"] == plan,
                "Training request/launch binding differs",
            )
            if name == "launch":
                require(
                    record["input_checkpoint"]["sha256"]
                    == reporter.BASELINES["local_mdlm_50000"]["checkpoint_sha256"],
                    "Training did not initialize from the pinned MDLM checkpoint",
                )
        require(
            terminal["source"]["head"] == spec["source_revision"]
            and plan["config_sha256"] == spec["resolved_config_sha256"]
            and checkpoint["sha256"] == spec["checkpoint_sha256"]
            and checkpoint["size_bytes"] == spec["checkpoint_size_bytes"],
            "V12 training/checkpoint identity differs",
        )
        require(
            checkpoint["global_step"] == 1000
            and checkpoint["finite_tensor_count"] > 0
            and terminal["completed_example_exposures"] == 128000
            and terminal["process_group_exit_grace"]["group_present_at_end"] is False,
            "V12 training completion is not accepted",
        )
        udlm = plan["config"]["training"]["udlm"]
        require(
            udlm["parameterization"] == "x0_denoiser"
            and udlm["prior_variant"] == spec["prior_variant"]
            and old.canonical_digest(spec["udlm_prior_metadata"])
            == spec["prior_metadata_sha256"]
            and spec["udlm_denoiser_metadata"]["parameterization"] == "x0_denoiser",
            "V12 prior/CE identity differs",
        )
        if arm == "MASK_CE":
            require(
                checkpoint["udlm_prior_metadata_sha256"]
                == spec["prior_metadata_sha256"],
                "V11b acceptance lacks the expected prior identity",
            )
        require(
            all(
                entry["checkpoint_sha256"] == spec["checkpoint_sha256"]
                and entry["prior_metadata_sha256"] == spec["prior_metadata_sha256"]
                for entry in protocol["entries"]
                if entry["arm_id"] == arm
            ),
            "Generation differs from its training checkpoint",
        )
        accepted[arm] = terminal
    expected = copy.deepcopy(accepted["E_CE"]["plan"]["config"])
    expected["callback"]["dirpath"] = accepted["MASK_CE"]["plan"]["config"]["callback"][
        "dirpath"
    ]
    expected["training"]["udlm"].update(
        prior_variant="mask_rich_empirical", mask_mixture_weight=0.9
    )
    require(
        expected == accepted["MASK_CE"]["plan"]["config"],
        "Prior training differs beyond declared prior and output",
    )
    failed = inputs.reference(
        root, training["original_v11_failure"], "Original V11 prelaunch failure"
    )
    require(
        failed["status"] == "failed"
        and failed["leases_release_authorized"] is True
        and all(
            failed.get(name) is None
            for name in (
                "training_pid",
                "launch_sha256",
                "training_return_code",
                "checkpoint",
                "completed_example_exposures",
                "end_to_end_training_examples_per_second",
            )
        ),
        "Original V11 failure was reclassified",
    )
    directory = Path(training["original_v11_failure"]["relative_path"]).parent
    request = inputs.read_json(
        root / directory / "request_manifest.json",
        "Original V11 failed request",
        failed["request_sha256"],
    )
    require(
        request["plan"] == failed["plan"] and request["source"] == failed["source"],
        "V11 failed request identity differs",
    )
    require(
        all(
            accepted["MASK_CE"]["plan"]["protocol"]["prelaunch_failure_evidence"][
                "terminal"
            ][key]
            == training["original_v11_failure"][key]
            for key in ("relative_path", "sha256")
        ),
        "V11b does not bind the retained V11 failure",
    )
    for path, digest in failed["artifact_hashes"].items():
        inputs.read(root / path, "Original V11 failure artifact", digest)
    return {
        "V11_prelaunch": failed,
        "V11b_MASK_CE": accepted["MASK_CE"],
        "V8b_empirical_CE_control": accepted["E_CE"],
    }


def accounting(records, *, studies=("V5", "V6", "V9", "V10", "V12")):
    study_ids = studies
    studies = []
    for study in study_ids:
        rows = [record for record in records if record["study"] == study]
        statuses = Counter()
        for row in rows:
            statuses.update(row.get("statuses", {"completed": len(row["seeds"])}))
        studies.append(
            {
                "study": study,
                "configurations": len(rows),
                "scheduled_runs": sum(len(row["seeds"]) for row in rows),
                "scheduled_requests": sum(row["samples"] for row in rows),
                "accepted_requests": sum(
                    row.get("accepted_samples", row["samples"]) for row in rows
                ),
                "status_counts": dict(statuses),
            }
        )
    require(
        len({(row["study"], row["config_id"]) for row in records}) == len(records),
        "Duplicated sampling configuration",
    )
    total = {
        field: sum(study[field] for study in studies)
        for field in (
            "configurations",
            "scheduled_runs",
            "scheduled_requests",
            "accepted_requests",
        )
    }
    total["status_counts"] = dict(
        sum((Counter(study["status_counts"]) for study in studies), Counter())
    )
    return {"studies": studies, "total": total}


def format_metric(value, *, percentage=True, signed=False):
    if value["mean"] is None:
        return "unavailable"
    scale = 100 if percentage else 1
    mean = format(scale * value["mean"], "+.3f" if signed else ".3f")
    sd = value["sample_sd"]
    return (
        f"{mean} +/- {scale * sd:.3f}" if sd is not None else f"{mean}; SD unavailable"
    )


def configuration_csv(records):
    """Extend the existing schema so unavailable means retain failure context."""
    reader = csv.DictReader(io.StringIO(old.configuration_csv(records).decode()))
    columns = reader.fieldnames + [
        "complete",
        "scheduled_samples",
        "accepted_samples",
        "completed_seeds",
        "status_counts",
    ]
    lookup = {(record["study"], record["config_id"]): record for record in records}
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in reader:
        record = lookup[(row["study"], row["config_id"])]
        row.update(
            complete=record["complete"],
            scheduled_samples=record["samples"],
            accepted_samples=record.get("accepted_samples", record["samples"]),
            completed_seeds=json.dumps(record.get("completed_seeds", record["seeds"])),
            status_counts=json.dumps(
                record.get("statuses", {"completed": len(record["seeds"])}),
                sort_keys=True,
            ),
        )
        writer.writerow(row)
    return stream.getvalue().encode()


def comparison_figure(records, *, synthetic=False):
    """All settings, with descriptive seed SD; missing pairs remain empty rows."""
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output/tmp/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    for axis, (branch, metric, title) in zip(
        axes,
        (
            ("released_comparable", "quality", "Repaired quality (%)"),
            ("strict", "validity", "Strict validity (%)"),
        ),
    ):
        for index, record in enumerate(records):
            value = record["metrics"][branch][metric]
            if value["mean"] is not None:
                axis.errorbar(
                    100 * value["mean"],
                    index,
                    xerr=None
                    if value["sample_sd"] is None
                    else 100 * value["sample_sd"],
                    fmt="o",
                    markersize=3,
                    capsize=2,
                    color="#16677b",
                )
        axis.set_title(title)
        # Mean +/- SD can extend beyond the [0,100] metric range.
        axis.set_xlim(-25, 125)
        axis.grid(axis="x", alpha=0.2)
    axes[0].axvline(
        85.8, color="#943c33", linestyle="--", label="Local MDLM mean (3 x 1000)"
    )
    axes[0].axvline(84.6, color="#555555", linestyle=":", label="Paper GenMol V1 mean")
    axes[0].set_yticks(
        range(len(records)),
        [f"{row['study']} {row['config_id']}" for row in records],
        fontsize=6,
    )
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=6, loc="lower left")
    figure.suptitle(
        ("SYNTHETIC TEST DATA: " if synthetic else "")
        + "All settings; two-seed sample SD, not confidence intervals",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=150, metadata={"Software": "Matplotlib"})
    plt.close(figure)
    return stream.getvalue()


def make_cover(summary, figure):
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 8
    styles["BodyText"].leading = 10
    story = []

    def text(value, style="BodyText"):
        story.append(Paragraph(value, styles[style]))
        story.append(Spacer(1, 6))

    def table(rows, widths):
        item = Table(
            [
                [Paragraph(escape(str(cell)), styles["BodyText"]) for cell in row]
                for row in rows
            ],
            colWidths=widths,
            repeatRows=1,
        )
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcecf0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f4f7f8")],
                    ),
                ]
            )
        )
        story.append(item)

    total = summary["accounting"]["total"]
    text("GenMol / UDLM engineering study through V12", "Title")
    text(
        f"<b>{total['configurations']} sampling configurations; {total['scheduled_runs']} scheduled runs; {total['accepted_requests']} independently rescored accepted requests out of {total['scheduled_requests']} planned.</b> Status: {escape(summary['status'])}. These are sampling configurations, not independent trained models."
    )
    text(
        "<b>No superiority established.</b> All engineering settings use two seeds: V5/V6 request 64 per seed, V9/V10/V12 request 100 per seed. The local MDLM reference uses 3 x 1,000; the paper reports 3 runs of 1,000. These different training and sampling budgets are context, not a matched superiority test. Final UDLM seeds 0/1/2 remain reserved."
    )
    table(
        [
            [
                "Study",
                "Configs",
                "Scheduled runs",
                "Accepted / planned requests",
                "Run status counts",
            ]
        ]
        + [
            [
                row["study"],
                row["configurations"],
                row["scheduled_runs"],
                f"{row['accepted_requests']} / {row['scheduled_requests']}",
                row["status_counts"],
            ]
            for row in summary["accounting"]["studies"]
        ],
        [55, 55, 100, 170, 390],
    )
    text(
        "The separate failed V4 diagnostic requested 32 samples and remains outside these accepted totals. Original V8 CT controller failure and V11 prelaunch failure remain failures; their follow-ups and the separate CT checkpoint audit retain distinct evidence."
    )
    text(
        "Quality counts within-seed unique valid molecules with QED >= 0.6 and SA <= 4 divided by all requested samples. QED measures drug-likeness (higher is favored); SA estimates synthetic accessibility (lower is favored). Validity uses requests; uniqueness uses valid samples; diversity is the released PyTDC metric on unique valid molecules. Strict SAFE decoding disables repair; released-compatible decoding repairs and selects the largest component. Both start after tokenizer special-token removal."
    )
    text(
        "Every mean and +/- value below is an equal-seed mean and sample SD, not a confidence interval. Means are withheld unless both declared seeds completed and defined the metric. Settings and temperatures were selected adaptively across studies; changed-seed comparisons are not paired treatment effects. Preserved appendices retain their original historical statements and dates."
    )
    story.append(PageBreak())
    text("All configurations: quality and strict validity", "Heading1")
    story.append(Image(io.BytesIO(figure), width=735, height=490))
    for study in ("V5", "V6", "V9", "V10", "V12"):
        story.append(PageBreak())
        text(f"{study}: every sampling configuration", "Heading1")
        rows = [
            [
                "Configuration / T / kernel",
                "Accepted / planned; seeds",
                "Branch",
                "Validity %",
                "Uniqueness %",
                "Quality %",
                "Diversity",
            ]
        ]
        for record in summary["configurations"]:
            if record["study"] != study:
                continue
            for branch in old.BRANCHES:
                rows.append(
                    [
                        f"{record['config_id']} / {record['temperature']} / {record['kernel']}",
                        f"{record.get('accepted_samples',record['samples'])}/{record['samples']}; {record['seeds']}",
                        "Repaired" if branch == "released_comparable" else "Strict",
                        *[
                            format_metric(
                                record["metrics"][branch][metric],
                                percentage=metric != "diversity",
                            )
                            for metric in old.METRICS
                        ],
                    ]
                )
        table(rows, [170, 100, 65, 105, 105, 105, 120])
        text(
            "Full inference dictionaries, checkpoint/source identities and per-seed outcomes remain in the appended study reports and overview.json. Missing means are not zero."
        )
    story.append(PageBreak())
    text("Training outcomes remain separate", "Heading1")
    table(
        [
            [
                "Training event",
                "Outcome",
                "Updates / configured exposures",
                "Evidence / interpretation",
            ]
        ]
        + summary["training_rows"],
        [135, 150, 170, 315],
    )
    text(
        "All continuations start from original MDLM 50k EMA with fresh optimizer/EMA. Historical R/S/E add 1,000 updates at batch 16 (16,000 exposures). V8 CT, V8b CE and V11b CE add 1,000 at batch 128 (128,000 exposures), eight times as many exposures. Exposures are not necessarily distinct molecules. Final checkpoint finiteness does not prove every update was finite. Original MDLM 50k training has a different budget."
    )
    story.append(PageBreak())
    text("V12: signed MASK-rich minus empirical CE contrasts", "Heading1")
    rows = [
        [
            "Contrast / T",
            "Branch / metric",
            "Seed2000 delta",
            "Seed2001 delta",
            "Mean +/- sample SD",
        ]
    ]
    for contrast in summary["v12_prior_contrasts"]:
        for branch in old.BRANCHES:
            for metric in old.METRICS:
                value = contrast["metrics"][branch][metric]
                scale = 1 if metric == "diversity" else 100
                deltas = [
                    "unavailable"
                    if row["difference"] is None
                    else f"{scale*row['difference']:+.3f}"
                    for row in value["per_seed"]
                ]
                rows.append(
                    [
                        f"{contrast['contrast_id']} / {contrast['temperature']}",
                        f"{branch} / {metric}",
                        *deltas,
                        format_metric(
                            {
                                "mean": value["mean_difference"],
                                "sample_sd": value["sample_sd"],
                            },
                            percentage=metric != "diversity",
                            signed=True,
                        ),
                    ]
                )
    table(rows, [145, 210, 115, 115, 185])
    text(
        "Percentage-point differences for validity, uniqueness and quality; raw units for diversity. Pairing is by declared seed, not molecular trajectory. A failed seed withholds the paired mean/SD. Temperature 1 is primary; 0.5 is secondary and acts after CE-to-LOO conversion."
    )
    story.append(PageBreak())
    text("V12: final editable MASK occurrences", "Heading1")
    rows = [
        [
            "Configuration",
            "Seed",
            "Final MASK / editable positions",
            "Rows with MASK / requests",
            "All generated control counts",
        ]
    ]
    for run in summary["v12_runs"]:
        match = next(
            (
                row
                for row in summary["v12_token_counts"]
                if (row["config_id"], row["seed"]) == (run["config_id"], run["seed"])
            ),
            None,
        )
        rows.append(
            [
                run["config_id"],
                run["seed"],
                f"{match['final_control_counts']['mask']} / {match['editable_positions']}"
                if match
                else "unavailable",
                f"{match['rows_with_mask']} / {match['rows']}"
                if match
                else run["status"],
                match["final_control_counts"]
                if match
                else run.get("failure_reason", "unavailable"),
            ]
        )
    table(rows, [155, 65, 160, 160, 230])
    text(
        "Counts come from hash-validated final uint16 IDs and the saved editable bitmask; padding/framing are excluded. CSV text skips special tokens, so high strict chemical validity does not certify zero generated controls. These are final occurrences, not survival trajectories. The ideal clean-content forward endpoint at lambda 0.9 has approximately 9 ppm MASK probability at positive epsilon 1e-5; learned predictions, finite-step factorization, temperature and independent-prior initialization prevent using that as an empirical guarantee."
    )
    text(
        "The manifest hashes every retained input, current imported helper source, archived generator and output. Large checkpoint bytes are not reloaded here: their digests and prior/CE/EMA acceptance are inherited from the bound CPU validation/training and independently rescored generation receipts. Original PDFs follow unchanged, beginning with the 43-page V9 overview."
    )
    stream = io.BytesIO()

    def page_notice(canvas, _document):
        if summary.get("synthetic_test_only"):
            canvas.setFillColor(colors.red)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.drawString(32, 12, "SYNTHETIC TEST DATA - NOT MOLECULAR RESULTS")

    SimpleDocTemplate(
        stream,
        pagesize=landscape(A4),
        leftMargin=32,
        rightMargin=32,
        topMargin=25,
        bottomMargin=25,
        invariant=1,
    ).build(story, onFirstPage=page_notice, onLaterPages=page_notice)
    return stream.getvalue()


def build_bundle(root, v12_directory, expected_v12_sha, *, workspace=old.WORKSPACE):
    root, v12_directory = Path(root).resolve(strict=True), Path(v12_directory).resolve()
    if not (v12_directory / "report.json").is_file():
        raise NotReady("V12 independent report is not available; no overview published")
    inputs = old.Inputs(workspace)
    generator = inputs.read(Path(__file__), "Current V12 overview generator")
    for module in (old, resolution, resolution.lexical, reporter):
        inputs.read(Path(module.__file__), "Current imported helper source")
    v12, protocol12, records12, masks = load_v12(
        inputs, root, v12_directory, expected_v12_sha
    )
    training12 = load_prior_training(inputs, root, protocol12)
    pdf9 = load_preserved_bundle(
        inputs, root / V9_DIRECTORY, V9_MANIFEST_SHA, "study_overview.pdf", 43
    )
    overview9 = inputs.read_json(
        root / V9_DIRECTORY / "overview.json", "Preserved 22-setting overview"
    )
    require(
        overview9["status"] == "complete"
        and overview9["configuration_count"] == 22
        and overview9["scheduled_runs"] == 44
        and overview9["independently_rescored_requests"] == 3104
        and overview9["superiority_established"] is False,
        "Historical overview accounting changed",
    )
    require(
        training12["V8b_empirical_CE_control"] == overview9["training"]["V8b_CE"],
        "V12 changed the archived empirical CE control",
    )
    records9 = copy.deepcopy(overview9["configurations"])
    require(
        len(records9) == 22
        and all(row["complete"] and len(row["seeds"]) == 2 for row in records9),
        "Historical configurations are incomplete",
    )
    v10, protocol10, _ = resolution.load_report(
        inputs, root, root / V10_DIRECTORY, V10_REPORT_SHA
    )
    records10 = metric_records(v10, protocol10, "V10")
    pdf10 = inputs.read(
        root / V10_DIRECTORY / "report.pdf", "Original V10 complete PDF"
    )
    pdf_resolution = load_preserved_bundle(
        inputs, root / RESOLUTION_DIRECTORY, RESOLUTION_MANIFEST_SHA, "analysis.pdf", 3
    )
    supplement = inputs.read_json(
        root / RESOLUTION_DIRECTORY / "analysis.json", "Preserved resolution contrasts"
    )
    require(
        supplement["source_report_sha256"] == V10_REPORT_SHA
        and supplement["accounting"] == v10["accounting"],
        "Resolution supplement binds another report",
    )
    historical = inputs.read_json(
        root / old.HISTORICAL / "overview.json",
        "Historical R/S/E training",
        "301768dfe226dbe92f46255e7fdc347ae680a3a57d9b26e6f686f868374aa7f9",
    )
    records = records9 + records10 + records12
    counts = accounting(records)
    require(
        counts["total"]["configurations"] == 30
        and counts["total"]["scheduled_runs"] == 60
        and counts["total"]["scheduled_requests"] == 4704,
        "Declared combined study scope differs",
    )
    training = {
        "historical_R_S_E": historical["training"],
        **overview9["training"],
        **training12,
    }
    rows = []
    for arm, evidence in historical["training"].items():
        data = evidence["accounting"]
        rows.append(
            [
                f"Historical {arm}",
                "Audited frozen training checkpoint",
                f"{data['optimizer_updates']} / {data['total_requested_example_exposures']}",
                f"Batch16; seed{data['training_seed']}; checkpoint {evidence['checkpoint'][:12]}",
            ]
        )
    for key, label in (
        ("V7", "V7 throughput pilot"),
        ("V8_CT", "V8 CT controller"),
        ("V8b_CE", "V8b empirical CE"),
        ("V11_prelaunch", "V11 MASK-rich CE"),
        ("V11b_MASK_CE", "V11b MASK-rich CE"),
    ):
        terminal = training[key]
        checkpoint = terminal.get("checkpoint")
        detail = f"Source {terminal['source']['head'][:12]}"
        if key == "V8_CT":
            audit = training["V8_CT_post_exit_audit"]
            detail += f"; separate post-exit audit accepted step{audit['checkpoint']['global_step']}, checkpoint {audit['checkpoint']['sha256'][:12]}; failed receipt unchanged"
        elif checkpoint:
            detail += f"; checkpoint {checkpoint['sha256'][:12]}; final finite tensors {checkpoint['finite_tensor_count']}"
        else:
            detail += "; no child or checkpoint"
        rows.append(
            [
                label,
                terminal["status"],
                f"{checkpoint['global_step'] if checkpoint else 'not accepted by controller'} / {terminal['completed_example_exposures']}",
                detail,
            ]
        )
    rows.append(
        [
            "Original V8 campaign",
            training["original_V8_campaign"]["status"],
            "CT then campaign stopped",
            "Original CE arm never launched; V8b is a separate fresh follow-up.",
        ]
    )
    incident_ref = v10["prior_v4_diagnostic_failure"]["artifact"]
    inputs.reference(root, incident_ref, "Separate failed V4 diagnostic receipt")
    incident_summary = inputs.read_json(
        root / Path(incident_ref["relative_path"]).parent / "summary.json",
        "Failed V4 diagnostic summary (not accepted)",
    )
    require(
        incident_summary["num_samples"] == 32,
        "Historical failed diagnostic sample count differs",
    )
    summary = {
        "schema_version": 1,
        "status": "complete"
        if counts["total"]["accepted_requests"] == counts["total"]["scheduled_requests"]
        else "terminal_with_failures",
        "readiness": "all_scheduled_controllers_terminal",
        "superiority_established": False,
        "formal_promotions": [],
        "reserved_udlm_final_seeds": [0, 1, 2],
        "accounting": counts,
        "configuration_count": len(records),
        "configurations": records,
        "baseline_requests": 3000,
        "baselines": v12["baselines"],
        "separate_failed_v4_diagnostic_requests": 32,
        "training": training,
        "training_rows": rows,
        "v12_source_report_status": v12["status"],
        "v12_original_aggregates": v12["aggregates"],
        "v12_prior_contrasts": v12["paired_contrasts"],
        "v12_runs": v12["runs"],
        "v12_token_counts": masks,
        "v12_protocol": protocol12,
        "v12_source_report_sha256": expected_v12_sha,
        "limitations": protocol12["limitations"],
    }
    figure = comparison_figure(records)
    cover = make_cover(summary, figure)
    parts = [
        ("Updated engineering overview through V12", cover),
        ("Preserved 43-page overview through V9", pdf9),
        ("Original V10 complete report", pdf10),
        ("Preserved 3-page V10 resolution supplement", pdf_resolution),
        (
            "Original terminal V12 report",
            inputs.read(v12_directory / "report.pdf", "V12 PDF appendix"),
        ),
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
                "input_pdf_sha256": old.digest(payload),
            }
        )
        cursor += count
    writer.add_metadata(
        {
            "/Title": "GenMol / UDLM engineering study through V12",
            "/Subject": "All settings; no superiority established",
        }
    )
    combined = io.BytesIO()
    writer.write(combined)
    summary["pdf_sections"] = sections
    outputs = {
        "study_overview.pdf": combined.getvalue(),
        "overview_cover.pdf": cover,
        "all_configurations.png": figure,
        "overview.json": old.json_bytes(summary),
        "all_configurations.csv": configuration_csv(records),
        "generate_study_overview_v12.py": generator,
    }
    outputs["input_hash_manifest.json"] = old.json_bytes(
        {
            "schema_version": 1,
            "status": summary["status"],
            "inputs": inputs.manifest(),
            "generator_sha256": old.digest(generator),
            "pdf_sections": sections,
            "generation_mode": "CPU aggregation of certified saved evidence; no generation, GPU probes, checkpoint/model reads or new chemistry scoring",
            "software": {
                "pypdf": old.pypdf.__version__,
                "reportlab": old.reportlab.Version,
                "matplotlib": sys.modules["matplotlib"].__version__,
            },
            "outputs": {
                name: {"sha256": old.digest(payload), "size_bytes": len(payload)}
                for name, payload in outputs.items()
            },
        }
    )
    return outputs, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--v12-report-directory", type=Path, required=True)
    parser.add_argument("--v12-report-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    require(
        not args.output_directory.resolve().exists(), "Output directory must be fresh"
    )
    try:
        outputs, summary = build_bundle(
            args.input_root, args.v12_report_directory, args.v12_report_sha256
        )
    except (NotReady, FileNotFoundError) as error:
        print(json.dumps({"status": "not_ready", "reason": str(error)}))
        return 2
    old.publish_bundle(args.output_directory, outputs)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_directory": str(args.output_directory.resolve()),
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
