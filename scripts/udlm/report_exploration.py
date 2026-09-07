"""Independently rescore and report a complete or partial engineering screen.

This CPU-only report never promotes a pilot to a final benchmark. A fresh
report directory is required for each snapshot; JSON is published last after
CSV and PDF, and existing evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

METRICS = ("validity", "uniqueness", "quality", "diversity")
BRANCHES = ("released_comparable", "strict")
BASELINES = {
    "local_mdlm_50000": {
        "validity": 1.0,
        "uniqueness": 0.9986666666666667,
        "quality": 0.858,
        "diversity": 0.8230213192558725,
        "seeds": [0, 1, 2],
        "requests_per_seed": 1000,
        "checkpoint_sha256": "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6",
        "source": "PROJECT_CONTEXT.md: frozen GenMol comparator",
    },
    "paper_genmol_v1_table1": {
        "validity": 1.0,
        "uniqueness": 0.997,
        "quality": 0.846,
        "diversity": 0.818,
        "requests_per_seed": 1000,
        "run_count": 3,
        "source": "scripts/exps/denovo/report.py: PAPER_REFERENCE",
    },
}
CAVEATS = [
    "Engineering pilots only: no superiority claim, final candidate lock, or final-seed evaluation.",
    "The local MDLM model completed 50,000 updates; each UDLM arm starts from its EMA weights and adds 1,000 updates. Training compute is not matched.",
    "The scheduler and FiLM conditioner were selected on E denoising loss and then shared by R/S/E. This is not independent optimization of each method.",
    "Small samples and configuration selection create sampling uncertainty and selection bias; a higher pilot mean is not evidence of superiority.",
    "Strict decoding uses fix=False. Released-compatible decoding uses SAFE repair and largest-component selection; report both because repairs can hide malformed outputs.",
    "Validity and quality divide by requested samples. Uniqueness divides by valid samples. Quality counts unique molecules with QED >= 0.6 and SA <= 4. Diversity is the released PyTDC fingerprint metric over unique valid molecules.",
    "Means and sample standard deviations average seeds equally within one configuration. Diversity is not pooled across seeds. Failed, pending, and invalid runs are excluded from means and remain disclosed.",
    "Paper seeds are undisclosed and hardware differs. Local timings are descriptive; no speed claim relative to the paper is made.",
]
V4_FAILURE = Path(
    "output/udlm/de_novo_candidate_campaign_v1/attempts/stage-d-e_t100_p100/seed_1100/failure_receipt.json"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def _scoped(path: Path, root: Path) -> Path:
    result = (path if path.is_absolute() else root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"path is outside repository: {path}")
    return result


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, payload


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    path = _scoped(path, root)
    payload = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha(payload),
        "size_bytes": len(payload),
    }


def rescore_summary(summary_path: Path, *, root: Path) -> dict[str, Any]:
    """Use the existing fresh CPU worker, pinning every identity in the summary."""
    from scripts.udlm.rescore_denovo_run import invoke_rescore_worker

    summary, payload = _read_json(summary_path)
    raw_path = summary_path.with_name("raw_samples.csv")
    inputs = summary["implementation_inputs"]
    return invoke_rescore_worker(
        summary_path=summary_path,
        raw_samples_path=raw_path,
        allowed_root=root,
        expected_summary_sha256=_sha(payload),
        expected_raw_samples_sha256=_sha(raw_path.read_bytes()),
        expected_seed=summary["seed"],
        expected_sample_count=summary["num_samples"],
        expected_checkpoint_sha256=summary["checkpoint"]["sha256"],
        expected_config_sha256=summary["config"]["sha256"],
        expected_source_revision=summary["git"]["commit"],
        expected_runner_sha256=summary["git"]["runner_sha256"],
        expected_sampler_source_sha256=inputs["sampler_source"]["sha256"],
        expected_ema_source_sha256=inputs["ema_source"]["sha256"],
        expected_implementation_inputs_sha256=_canonical_sha(inputs),
        expected_metric_inputs_sha256=_canonical_sha(summary["metric_inputs"]),
    )


def _gpu_mapping(summary: dict[str, Any]) -> dict[str, Any]:
    environment = summary.get("environment", {})
    launch = environment.get("launch_environment", {})
    snapshot = launch.get("GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT")
    if isinstance(snapshot, str):
        snapshot = json.loads(snapshot)
    return {
        "cuda_visible_devices": launch.get("CUDA_VISIBLE_DEVICES"),
        "physical_index": launch.get("GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX"),
        "logical_device": environment.get("cuda_device"),
        "final_probe": (snapshot or {}).get("physical_gpu_at_final_uuid_probe"),
        "launch_policy": (snapshot or {}).get("policy"),
    }


def _controller_receipt(
    entry: dict[str, Any],
    seeds: list[int],
    samples: int,
    protocol_sha256: str,
    output_root: Path,
    root: Path,
) -> dict[str, Any]:
    path = _scoped(
        output_root / "controller_receipts" / f"{entry['attempt_id']}.json", root
    )
    if not path.is_file():
        return {
            "status": "pending",
            "reason": "controller terminal receipt is not yet present",
        }
    try:
        receipt, _ = _read_json(path)
        identity = receipt["identity"]
        expected = {
            "entry": entry,
            "seeds": seeds,
            "num_samples": samples,
            "protocol_sha256": protocol_sha256,
            "source": identity["source"],
        }
        if identity != expected:
            raise ValueError("controller receipt differs from prospective protocol")
        source = identity["source"]
        if set(source) != {"head", "upstream"} or source["head"] != source["upstream"]:
            raise ValueError("controller receipt source is not clean pushed source")
        observed = {}
        for seed in seeds:
            for name in ("summary.json", "raw_samples.csv", "failure_receipt.json"):
                artifact = _scoped(
                    output_root / entry["attempt_id"] / f"seed_{seed}" / name, root
                )
                if artifact.is_file():
                    observed[artifact.relative_to(root).as_posix()] = _sha(
                        artifact.read_bytes()
                    )
        if observed != receipt["artifacts"]:
            raise ValueError("controller receipt output artifact hashes differ")
        if type(receipt["return_code"]) is not int:
            raise ValueError("controller receipt has invalid return code")
        return {
            "status": "completed" if receipt["return_code"] == 0 else "failed",
            "reason": f"attempt controller exited with status {receipt['return_code']}",
            "receipt": receipt,
            "artifact": _artifact(path, root),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {
            "status": "invalid",
            "reason": str(error),
            "artifact": _artifact(path, root),
        }


def _run_record(
    entry: dict[str, Any],
    seed: int,
    samples: int,
    output_root: Path,
    *,
    root: Path,
    rescore: Callable[..., dict[str, Any]],
    controller: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _scoped(output_root / entry["attempt_id"] / f"seed_{seed}", root)
    record = {
        key: entry.get(key)
        for key in ("attempt_id", "candidate_id", "arm_id", "config_id")
    }
    record.update(
        seed=seed,
        requested_samples=samples,
        status="pending",
        metrics=None,
        run_directory=run_dir.relative_to(root).as_posix(),
    )
    summary_path = run_dir / "summary.json"
    failure_path = run_dir / "failure_receipt.json"
    record["artifacts"] = {
        name: _artifact(run_dir / name, root)
        for name in ("summary.json", "raw_samples.csv", "failure_receipt.json")
        if (run_dir / name).is_file()
    }
    record["controller"] = controller
    if controller["status"] != "completed":
        record.update(status=controller["status"], failure_reason=controller["reason"])
        return record
    if failure_path.is_file():
        failure, _ = _read_json(failure_path)
        record.update(
            status="failed",
            failure_reason=failure.get("reason"),
            failure_stage=failure.get("stage"),
            process_exit_status=failure.get("process_exit_status"),
        )
        return record
    if not summary_path.is_file():
        record.update(
            status="invalid",
            failure_reason="successful controller receipt lacks seed summary",
        )
        return record
    try:
        summary, payload = _read_json(summary_path)
        if summary["seed"] != seed or summary["num_samples"] != samples:
            raise ValueError("summary seed/sample count differs from protocol slot")
        if summary["checkpoint"]["sha256"] != entry["checkpoint_sha256"]:
            raise ValueError("checkpoint digest differs from protocol")
        if summary["config"]["sha256"] != entry["config_sha256"]:
            raise ValueError("config digest differs from protocol")
        if (
            summary["git"]["commit"]
            != controller["receipt"]["identity"]["source"]["head"]
        ):
            raise ValueError("summary source differs from controller source")
        if summary["run"].get("final_protocol_eligible") is not False or seed < 1000:
            raise ValueError(
                "engineering screen contains final-eligible evidence or reserved seed"
            )
        result = rescore(summary_path, root=root)
        if result.get("status") != "exact_match" or result.get("seed") != seed:
            raise ValueError("independent rescore did not certify this seed")
        if result.get("summary_sha256") != _sha(payload):
            raise ValueError("independent rescore bound a different summary")
        raw_path = run_dir / "raw_samples.csv"
        raw_payload = raw_path.read_bytes()
        if result.get("raw_samples_sha256") != _sha(raw_payload):
            raise ValueError("independent rescore bound different raw rows")
        rows = list(csv.DictReader(io.StringIO(raw_payload.decode("utf-8"))))
        if len(rows) != samples:
            raise ValueError("raw row count differs from requested samples")
        if summary_path.read_bytes() != payload:
            raise ValueError("summary changed during reporting")
        record.update(
            status="completed",
            metrics=result["metrics"],
            raw_row_count=len(rows),
            checkpoint=summary["checkpoint"],
            config=summary["config"],
            source=summary["git"],
            runtime_seconds=summary["runtime_seconds"],
            gpu_mapping=_gpu_mapping(summary),
            generation_protocol=summary["run"]["generation_protocol"],
            started_at_utc=summary["run"].get("started_at_utc"),
            completed_at_utc=summary["run"].get("completed_at_utc"),
            independent_rescore=result,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        record.update(status="invalid", metrics=None, failure_reason=str(error))
    return record


def build_report(
    protocol_path: Path,
    output_root: Path,
    *,
    root: Path = REPOSITORY_ROOT,
    rescore: Callable[..., dict[str, Any]] = rescore_summary,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    protocol_path = _scoped(protocol_path, root)
    output_root = _scoped(output_root, root)
    protocol, protocol_payload = _read_json(protocol_path)
    entries, seeds, samples = (
        protocol["entries"],
        protocol["seeds"],
        protocol["num_samples"],
    )
    if not entries or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("protocol must schedule entries and distinct seeds")
    if (
        any(type(seed) is not int or seed < 1000 for seed in seeds)
        or not 1 <= samples <= 100
    ):
        raise ValueError(
            "protocol must be an engineering pilot with seeds >=1000 and 1..100 requests"
        )
    attempts = [entry["attempt_id"] for entry in entries]
    if len(attempts) != len(set(attempts)) or any(
        Path(value).name != value for value in attempts
    ):
        raise ValueError("attempt IDs must be distinct single path components")
    runs = []
    for entry in entries:
        controller = _controller_receipt(
            entry, seeds, samples, _sha(protocol_payload), output_root, root
        )
        runs.extend(
            _run_record(
                entry,
                seed,
                samples,
                output_root,
                root=root,
                rescore=rescore,
                controller=controller,
            )
            for seed in seeds
        )
    scheduled = {record["run_directory"] for record in runs}
    discovered = {
        path.parent.relative_to(root).as_posix()
        for pattern in ("*/seed_*/summary.json", "*/seed_*/failure_receipt.json")
        for path in output_root.glob(pattern)
    }
    unexpected = sorted(discovered - scheduled)
    aggregates = []
    for entry in entries:
        selected = [run for run in runs if run["attempt_id"] == entry["attempt_id"]]
        successful = [run for run in selected if run["status"] == "completed"]
        item = {
            key: entry.get(key)
            for key in ("attempt_id", "candidate_id", "arm_id", "config_id")
        }
        item.update(
            completed_seeds=[run["seed"] for run in successful],
            expected_seeds=seeds,
            complete=len(successful) == len(seeds),
            metrics={},
        )
        for branch in BRANCHES:
            item["metrics"][branch] = {}
            for metric in METRICS:
                values = [run["metrics"][branch][metric] for run in successful]
                defined = [value for value in values if value is not None]
                item["metrics"][branch][metric] = {
                    "mean": (
                        statistics.mean(defined)
                        if defined and len(defined) == len(values)
                        else None
                    ),
                    "sample_sd": (
                        statistics.stdev(defined)
                        if len(defined) > 1 and len(defined) == len(values)
                        else None
                    ),
                    "defined_seed_count": len(defined),
                }
        aggregates.append(item)
    counts = Counter(run["status"] for run in runs)
    incident_path = _scoped(V4_FAILURE, root)
    incident = {
        "status": "historical_failed_diagnostic",
        "disposition": "V4 remains campaign_incomplete; its failed D attempt is never reclassified or reused for ranking.",
        "reason": "The generation process exited 0, but completion validation omitted the raw_loo_top_p=1.0 default from expected effective config.",
    }
    if incident_path.is_file():
        failure, _ = _read_json(incident_path)
        incident.update(
            artifact=_artifact(incident_path, root),
            reason=failure.get("reason"),
            stage=failure.get("stage"),
            failed_at_utc=failure.get("failed_at_utc"),
        )
    else:
        incident["artifact_available"] = False
    status = (
        "complete"
        if counts["completed"] == len(runs) and not unexpected
        else "incomplete"
    )
    result = {
        "schema_version": 1,
        "study_id": protocol.get("study_id"),
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim": "engineering_screen_only_no_superiority_claim",
        "superiority_established": False,
        "protocol": {
            "relative_path": protocol_path.relative_to(root).as_posix(),
            "sha256": _sha(protocol_payload),
            "configuration": protocol,
        },
        "accounting": {
            "scheduled_runs": len(runs),
            "scheduled_requests": samples * len(runs),
            "status_counts": dict(counts),
            "independently_rescored_requests": sum(
                run.get("raw_row_count", 0) for run in runs
            ),
        },
        "runs": runs,
        "aggregates": aggregates,
        "unexpected_run_directories": unexpected,
        "prior_v4_diagnostic_failure": incident,
        "baselines": BASELINES,
        "caveats": CAVEATS,
    }
    if _has_denoiser_design(result):
        result["caveats"] = [
            CAVEATS[0],
            _objective_training_description(protocol),
            "CT and clean CE share the training settings recorded in this protocol. "
            "Equal updates and examples do not imply equal compute or independently "
            "optimized objectives; training losses have different scales and targets.",
            *CAVEATS[3:],
        ]
    return result


def _objective_training_description(protocol: dict[str, Any]) -> str:
    training = protocol.get("training", {})
    settings = [
        f"{label}: {training[key]}"
        for key, label in (
            ("initialization", "initialization"),
            ("optimizer_updates", "optimizer updates per arm"),
            ("global_batch_size", "global batch"),
            ("training_seed", "training seed"),
            ("gpu_count", "training GPUs"),
            ("example_exposures_per_arm", "requested example exposures per arm"),
            ("common_mask_policy", "common clean-target mask"),
        )
        if key in training
    ]
    description = (
        "Prospective training settings: " + "; ".join(settings) + "."
        if settings
        else "Training settings are not recorded in this protocol."
    )
    return description + " The MDLM and paper baselines remain contextual comparisons."


def _has_gibbs_design(report: dict[str, Any]) -> bool:
    protocol = report["protocol"]["configuration"]
    return "gibbs_treatment" in protocol.get("design", {}) or any(
        run.get("generation_protocol", {}).get("gibbs_corrector") is True
        for run in report["runs"]
    )


def _has_denoiser_design(report: dict[str, Any]) -> bool:
    protocol = report["protocol"]["configuration"]
    return (
        "objective_comparison" in protocol.get("design", {})
        or any(
            entry.get("parameterization") == "x0_denoiser"
            for entry in protocol.get("entries", [])
        )
        or any(
            run.get("config", {}).get("sampling", {}).get("parameterization")
            == "x0_denoiser"
            for run in report["runs"]
        )
    )


def _objective_evidence(run: dict[str, Any]) -> dict[str, Any]:
    """Describe completed UDLM runs using independently certified identity."""
    if (
        run.get("status") != "completed"
        or run.get("generation_protocol", {}).get("diffusion_type") != "udlm"
    ):
        return {}
    parameterization = run["config"]["sampling"].get("parameterization", "raw_loo")
    identity = run["independent_rescore"].get("identity", {})
    generation = identity.get("generation", {})
    evidence = {
        "objective": "clean CE"
        if parameterization == "x0_denoiser"
        else "CT / raw-LOO",
        "parameterization": parameterization,
        "inference_weights": generation.get("inference_weights"),
    }
    if parameterization == "x0_denoiser":
        evidence.update(
            udlm_denoiser_metadata=generation.get("udlm_denoiser_metadata"),
            denoiser_source_sha256=identity.get("source", {}).get(
                "denoiser_source_sha256"
            ),
        )
    return evidence


def _sampling_budget(run: dict[str, Any]) -> dict[str, Any]:
    """Expose certified NFE; recover unchanged predictor counts for old schemas."""
    protocol = run.get("generation_protocol", {})
    if run.get("status") != "completed" or protocol.get("diffusion_type") not in {"udlm", "mdlm"}:
        return {}
    corrector = protocol.get("gibbs_corrector", False)
    return {
        "nfe": protocol["nfe"],
        "gibbs_corrector": corrector,
        "predictor_transitions_per_molecule": (
            protocol["predictor_transitions_per_molecule"]
            if corrector
            else protocol["nfe"]
        ),
        "corrector_steps_per_molecule": (
            protocol["corrector_steps_per_molecule"] if corrector else 0
        ),
    }


def _csv_bytes(report: dict[str, Any]) -> bytes:
    fields = [
        "attempt_id",
        "config_id",
        "arm_id",
        "seed",
        "status",
        "requested_samples",
        "raw_row_count",
        *[f"{branch}_{metric}" for branch in BRANCHES for metric in METRICS],
        "generation_seconds",
        "checkpoint_sha256",
        "source_revision",
        "gpu_uuid",
        "raw_sha256",
        "failure_reason",
    ]
    include_denoiser = _has_denoiser_design(report)
    include_sampling = _has_gibbs_design(report) or include_denoiser
    if include_sampling:
        fields.extend(
            [
                "nfe",
                "gibbs_corrector",
                "predictor_transitions_per_molecule",
                "corrector_steps_per_molecule",
                "generation_protocol",
                "corrector_source_sha256",
                "protocol_design",
                "protocol_limitations",
            ]
        )
    if include_denoiser:
        fields.extend(
            [
                "planned_parameterization",
                "objective",
                "parameterization",
                "inference_weights",
                "udlm_denoiser_metadata",
                "denoiser_source_sha256",
                "protocol_training",
            ]
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for run in report["runs"]:
        row = {key: run.get(key) for key in fields}
        for branch in BRANCHES:
            for metric in METRICS:
                row[f"{branch}_{metric}"] = (
                    (run.get("metrics") or {}).get(branch, {}).get(metric)
                )
        row.update(
            generation_seconds=run.get("runtime_seconds", {}).get("generation"),
            checkpoint_sha256=run.get("checkpoint", {}).get("sha256"),
            source_revision=run.get("source", {}).get("commit"),
            gpu_uuid=run.get("gpu_mapping", {}).get("cuda_visible_devices"),
            raw_sha256=run["artifacts"].get("raw_samples.csv", {}).get("sha256"),
        )
        if include_sampling:
            protocol = report["protocol"]["configuration"]
            source = (
                run.get("independent_rescore", {}).get("identity", {}).get("source", {})
            )
            row.update(
                **_sampling_budget(run),
                generation_protocol=json.dumps(
                    run.get("generation_protocol", {}), sort_keys=True
                ),
                corrector_source_sha256=source.get("corrector_source_sha256"),
                protocol_design=json.dumps(protocol.get("design", {}), sort_keys=True),
                protocol_limitations=json.dumps(protocol.get("limitations", [])),
            )
        if include_denoiser:
            objective = _objective_evidence(run)
            metadata = objective.get("udlm_denoiser_metadata")
            weights = objective.get("inference_weights")
            row.update(
                **objective,
                planned_parameterization=next(
                    (
                        entry.get("parameterization")
                        for entry in protocol.get("entries", [])
                        if entry["attempt_id"] == run["attempt_id"]
                    ),
                    None,
                ),
                protocol_training=json.dumps(
                    protocol.get("training", {}), sort_keys=True
                ),
            )
            row["udlm_denoiser_metadata"] = (
                json.dumps(metadata, sort_keys=True) if metadata is not None else None
            )
            row["inference_weights"] = (
                json.dumps(weights, sort_keys=True) if weights is not None else None
            )
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _pdf_bytes(report: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    stream = io.BytesIO()
    styles = getSampleStyleSheet()
    story = [
        Paragraph("GenMol UDLM engineering exploration", styles["Title"]),
        Paragraph(
            escape(
                f"{report['study_id']} | {report['status']} | {report['created_at_utc']}"
            ),
            styles["Normal"],
        ),
        Paragraph(
            "Engineering pilots only. Superiority has not been established.",
            styles["Heading2"],
        ),
    ]
    for caveat in report["caveats"]:
        story.append(Paragraph(escape(caveat), styles["BodyText"]))
    include_denoiser = _has_denoiser_design(report)
    include_sampling = _has_gibbs_design(report) or include_denoiser
    if include_sampling:
        protocol = report["protocol"]["configuration"]
        story.append(Paragraph("Prospective sampling design", styles["Heading2"]))
        story.append(
            Paragraph(
                escape(json.dumps(protocol.get("design", {}), sort_keys=True)),
                styles["BodyText"],
            )
        )
        for caveat in protocol.get("limitations", []):
            story.append(Paragraph(escape(caveat), styles["BodyText"]))
    if include_denoiser:
        story.append(Paragraph("Objective comparison", styles["Heading2"]))
        story.append(
            Paragraph(
                "CT / raw-LOO predicts clean leave-one-out probabilities. Clean CE "
                "predicts clean denoiser probabilities, converted to LOO logits at "
                "the current noisy state and time before temperature or top-p. "
                "The conversion adds no backbone evaluation. Shared sampling "
                "settings do not imply independently optimized objectives. "
                "Objective labels and checkpoint evidence below describe completed "
                "runs only; the prospective design also covers pending runs.",
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(
                escape(
                    "Training provenance: "
                    + json.dumps(protocol.get("training", {}), sort_keys=True)
                ),
                styles["BodyText"],
            )
        )
    story.append(Paragraph("Reference means (context only)", styles["Heading2"]))
    rows = [["Reference", *METRICS]]
    for name, baseline in report["baselines"].items():
        rows.append([name, *[f"{baseline[metric]:.6f}" for metric in METRICS]])

    def add_table(data: list[list[Any]]) -> None:
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce7ef")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ]
            )
        )
        story.extend([table, Spacer(1, 10)])

    add_table(rows)
    if include_sampling:
        story.append(Paragraph("Sampling evaluation budget", styles["Heading2"]))
        story.append(
            Paragraph(
                "Counts are per molecule. Each predictor transition and each fresh "
                "Gibbs correction consumes one backbone evaluation. Predictor-only "
                "counts equal recorded NFE; missing runs have no observed budget.",
                styles["BodyText"],
            )
        )
        rows = [["Config", "Seed", "Total NFE", "Predictors", "Correctors"]]
        for run in report["runs"]:
            budget = _sampling_budget(run)
            rows.append(
                [
                    run["config_id"],
                    run["seed"],
                    budget.get("nfe", "--"),
                    budget.get("predictor_transitions_per_molecule", "--"),
                    budget.get("corrector_steps_per_molecule", "--"),
                ]
            )
        add_table(rows)
    if include_denoiser:
        story.append(Paragraph("Checkpoint logit interpretation", styles["Heading2"]))
        rows = [["Config", "Seed", "Objective", "Parameterization", "Weights"]]
        for run in report["runs"]:
            evidence = _objective_evidence(run)
            rows.append(
                [
                    run["config_id"],
                    run["seed"],
                    evidence.get("objective", "--"),
                    evidence.get("parameterization", "--"),
                    (evidence.get("inference_weights") or {}).get("source", "--"),
                ]
            )
        add_table(rows)
    story.append(
        Paragraph("Per-configuration mean, equal seed weights", styles["Heading2"])
    )
    for branch in BRANCHES:
        story.append(Paragraph(escape(branch), styles["Heading3"]))
        rows = [["Config", "Seeds complete", *METRICS]]
        for item in report["aggregates"]:
            values = []
            for metric in METRICS:
                summary = item["metrics"][branch][metric]
                values.append(
                    "--"
                    if summary["mean"] is None
                    else f"{summary['mean']:.4f}"
                    + (
                        f" +/- {summary['sample_sd']:.4f}"
                        if summary["sample_sd"] is not None
                        else ""
                    )
                )
            rows.append(
                [
                    item["config_id"],
                    f"{len(item['completed_seeds'])}/{len(item['expected_seeds'])}",
                    *values,
                ]
            )
        add_table(rows)
    story.append(Paragraph("Every scheduled seed", styles["Heading2"]))
    rows = [
        [
            "Config",
            "Seed",
            "Status",
            "Requests",
            "Raw rows",
            "Repair Q",
            "Strict Q",
            "Generation s",
        ]
    ]
    for run in report["runs"]:
        metrics = run.get("metrics") or {}
        rows.append(
            [
                run["config_id"],
                run["seed"],
                run["status"],
                run["requested_samples"],
                run.get("raw_row_count", "--"),
                *[
                    f"{metrics[branch]['quality']:.4f}" if metrics else "--"
                    for branch in BRANCHES
                ],
                (
                    f"{run['runtime_seconds']['generation']:.2f}"
                    if "runtime_seconds" in run
                    else "--"
                ),
            ]
        )
    add_table(rows)
    story.append(Paragraph("Preserved v4 diagnostic failure", styles["Heading2"]))
    incident = report["prior_v4_diagnostic_failure"]
    story.append(Paragraph(escape(incident["disposition"]), styles["BodyText"]))
    story.append(Paragraph(escape(incident["reason"]), styles["BodyText"]))
    story.append(Paragraph("Configuration and artifact provenance", styles["Heading2"]))
    story.append(
        Paragraph(
            escape("Protocol SHA-256: " + report["protocol"]["sha256"]),
            styles["BodyText"],
        )
    )
    for run in report["runs"]:
        if run["status"] == "completed":
            detail = {
                "config": run["config"]["sampling"],
                "checkpoint": run["checkpoint"]["sha256"],
                "source": run["source"]["commit"],
                "gpu": run["gpu_mapping"],
                "raw": run["artifacts"]["raw_samples.csv"],
            }
            if include_sampling:
                detail["generation_protocol"] = run["generation_protocol"]
                detail["sampling_budget"] = _sampling_budget(run)
                source = (
                    run.get("independent_rescore", {})
                    .get("identity", {})
                    .get("source", {})
                )
                if "corrector_source_sha256" in source:
                    detail["corrector_source_sha256"] = source[
                        "corrector_source_sha256"
                    ]
            if include_denoiser:
                detail.update(_objective_evidence(run))
            story.append(
                Paragraph(
                    escape(
                        f"{run['attempt_id']} / seed {run['seed']}: "
                        + json.dumps(detail, sort_keys=True)
                    ),
                    styles["BodyText"],
                )
            )
        elif run.get("failure_reason"):
            story.append(
                Paragraph(
                    escape(
                        f"{run['attempt_id']} / seed {run['seed']}: {run['failure_reason']}"
                    ),
                    styles["BodyText"],
                )
            )
    SimpleDocTemplate(
        stream,
        pagesize=landscape(A4),
        leftMargin=32,
        rightMargin=32,
        topMargin=30,
        bottomMargin=30,
    ).build(story)
    return stream.getvalue()


def write_report(
    report: dict[str, Any], report_dir: Path, *, root: Path = REPOSITORY_ROOT
) -> dict[str, str]:
    from scripts import artifact_io

    root = root.resolve(strict=True)
    directory = _scoped(report_dir, root)
    directory.mkdir(parents=True, exist_ok=True)
    relative = directory.relative_to(root).as_posix()
    csv_payload, pdf_payload = _csv_bytes(report), _pdf_bytes(report)
    report = {
        **report,
        "report_artifacts": {
            "csv_sha256": _sha(csv_payload),
            "pdf_sha256": _sha(pdf_payload),
        },
    }
    artifact_io.publish_bundle_exclusive(
        root,
        [
            artifact_io.PublishItem(f"{relative}/report.csv", csv_payload),
            artifact_io.PublishItem(f"{relative}/report.pdf", pdf_payload),
        ],
        completion=artifact_io.PublishItem(
            f"{relative}/report.json",
            (
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode(),
        ),
    )
    return {kind: str(directory / f"report.{kind}") for kind in ("json", "csv", "pdf")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("output/udlm/engineering_v5")
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.protocol, args.output_root)
    paths = write_report(report, args.report_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "accounting": report["accounting"],
                "artifacts": paths,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
