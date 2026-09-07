"""Execute a finite, hash-pinned PMO panel in fresh one/two-GPU waves.

This controller selects no scientific arms and never resumes or retries a child.
Run inside tmux; --dry-run performs CPU metadata preflight without resource probes.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(import_root))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import benchmark, launch_benchmark as resources  # noqa: E402
from scripts.exps.pmo import run_ablation as runner, udlm_sampling  # noqa: E402
from scripts.exps.pmo.main.genmol import experiment_io  # noqa: E402
from scripts.udlm import launch_engineering_training as cleanup  # noqa: E402

POLICY = {"max_utilization_percent": 10, "min_free_memory_mib": 30000}
RUNNER_KEYS = {
    "max_iterations",
    "population_size",
    "warmup",
    "legacy_warmup_off_by_one",
    "reporting_frequency",
    "checkpoint_every",
    "guidance_scale",
    "min_mol_size",
    "max_mol_size",
}
ENTRY_KEYS = {
    "id",
    "seed",
    "checkpoint",
    "checkpoint_sha256",
    "sampling_config",
    "sampling_config_sha256",
    "vocabulary",
    "vocabulary_sha256",
    "max_oracle_calls",
}
PANEL_KEYS = {
    "schema_version",
    "campaign_id",
    "oracle",
    "scientific_status",
    "output_root",
    "log_root",
    "child_timeout_seconds",
    "capacity_wait_seconds",
    "runner",
    "entries",
    "input_files",
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def integer(value, name, minimum=0, maximum=None):
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def positive_real(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]{0,79}", value
    ):
        raise ValueError(f"invalid {name}")
    return value


def relative(value):
    if not isinstance(value, str) or not value or Path(value).as_posix() != value:
        raise ValueError("paths must be canonical repository-relative strings")
    path = Path(value)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("paths must remain inside the repository")
    return value


def fingerprint(path, expected=None, *, capture=False):
    base = ROOT
    input_relative = path
    if isinstance(path, str) and Path(path).is_absolute():
        if Path(path).as_posix() != path:
            raise ValueError("absolute inputs must use canonical paths")
        base = resources._project_root()
        try:
            input_relative = str(Path(path).relative_to(base))
        except ValueError as error:
            raise ValueError(
                "absolute inputs must remain inside the project"
            ) from error
    claim, payload = artifact_io.snapshot_file(
        base, relative(input_relative), capture_bytes=capture
    )
    if expected is not None and (
        not isinstance(expected, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
        or claim.sha256 != expected
    ):
        raise ValueError(f"input SHA-256 differs: {path}")
    return {
        "path": path,
        "sha256": claim.sha256,
        "size_bytes": claim.size_bytes,
    }, payload


def _read_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)


def command_for(panel, entry, panel_record):
    command = [
        str(resources._project_venv_python()),
        "-u",
        str(ROOT / "scripts/exps/pmo/run_ablation.py"),
        "--oracle",
        panel["oracle"],
        "--variant",
        "released",
        "--gamma",
        "0",
        "--model-path",
        str(ROOT / entry["checkpoint"]),
        "--sampling-config",
        str(ROOT / entry["sampling_config"]),
        "--vocab-path",
        str(ROOT / entry["vocabulary"]),
        "--device",
        "cuda:0",
        "--seed",
        str(entry["seed"]),
        "--max-oracle-calls",
        str(entry["max_oracle_calls"]),
        "--experiment-id",
        entry["id"],
        "--scientific-status",
        panel["scientific_status"],
        "--output-root",
        str(ROOT / panel["output_root"] / "runs"),
        "--matrix-path",
        str(ROOT / panel_record["path"]),
        "--matrix-sha256",
        panel_record["sha256"],
        "--durable-events",
    ]
    for key, value in panel["runner"].items():
        if key == "legacy_warmup_off_by_one":
            if value:
                command.append("--legacy-warmup-off-by-one")
        else:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def resolved_command(command):
    previous = sys.argv
    try:
        sys.argv = command[2:]
        args = runner._parse_args()
    finally:
        sys.argv = previous
    runner._validate_args(args)
    return runner._resolved_config(args)


def build_plan(panel_path, panel_sha256, gpu_count):
    """Validate every finite request and checkpoint metadata before any Popen."""
    integer(gpu_count, "gpu-count", 1, 2)
    panel_record, payload = fingerprint(panel_path, panel_sha256, capture=True)
    panel = _read_json(payload)
    if (
        not isinstance(panel, dict)
        or set(panel) != PANEL_KEYS
        or type(panel["schema_version"]) is not int
        or panel["schema_version"] != 1
    ):
        raise ValueError("PMO panel must use exactly the schema-1 fields")
    identifier(panel["campaign_id"], "campaign_id")
    if panel["oracle"] not in runner.ORACLES:
        raise ValueError("oracle must be an explicit released PMO task")
    if (
        not isinstance(panel["scientific_status"], str)
        or not panel["scientific_status"].strip()
    ):
        raise ValueError("scientific_status must disclose the study's status")
    for key in ("output_root", "log_root"):
        relative(panel[key])
    if not panel["output_root"].startswith("output/udlm/") or not panel[
        "log_root"
    ].startswith("output/logs/"):
        raise ValueError("use fresh output/udlm and output/logs namespaces")
    positive_real(panel["child_timeout_seconds"], "child_timeout_seconds")
    integer(panel["capacity_wait_seconds"], "capacity_wait_seconds", 0, 21600)
    options = panel["runner"]
    if not isinstance(options, dict) or set(options) != RUNNER_KEYS:
        raise ValueError("runner settings must use exactly the declared fields")
    for key in RUNNER_KEYS - {"legacy_warmup_off_by_one", "guidance_scale"}:
        integer(options[key], key, 0 if key == "warmup" else 1)
    if options["population_size"] < 2 or options["max_iterations"] <= options["warmup"]:
        raise ValueError(
            "population must contain two fragments and iterations must exceed warmup"
        )
    if type(options["legacy_warmup_off_by_one"]) is not bool:
        raise ValueError("legacy_warmup_off_by_one must be Boolean")
    positive_real(options["guidance_scale"], "guidance_scale")
    if not isinstance(panel["entries"], list) or not panel["entries"]:
        raise ValueError("panel requires a finite nonempty entries list")
    records = {panel_record["path"]: panel_record}
    if not isinstance(panel["input_files"], list) or not panel["input_files"]:
        raise ValueError("panel requires auxiliary input_files")
    for value in panel["input_files"]:
        if (
            not isinstance(value, dict)
            or set(value) != {"path", "sha256"}
            or value["path"] in records
        ):
            raise ValueError("auxiliary inputs require unique path/sha256 pairs")
        record, _ = fingerprint(value["path"], value["sha256"])
        records[record["path"]] = record
    if panel["oracle"] == "gsk3b" and "oracle/gsk3b_current.pkl" not in records:
        raise ValueError(
            "pin the existing local oracle/gsk3b_current.pkl; no oracle download"
        )
    jobs, checkpoints, identities = [], {}, set()
    for entry in panel["entries"]:
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise ValueError("entry must use exactly the declared fields")
        identifier(entry["id"], "entry id")
        if entry["id"] in identities:
            raise ValueError("entry ids must be unique")
        identities.add(entry["id"])
        integer(entry["seed"], "seed")
        integer(entry["max_oracle_calls"], "max_oracle_calls", 1001)
        for field in ("checkpoint", "sampling_config", "vocabulary"):
            record, _ = fingerprint(entry[field], entry[field + "_sha256"])
            if record["path"] in records and records[record["path"]] != record:
                raise ValueError("one input path cannot have conflicting identities")
            records[record["path"]] = record
        command = command_for(panel, entry, panel_record)
        config = resolved_command(command)
        contract = config["pmo_sampling"]
        if contract["checkpoint_sha256"] != entry["checkpoint_sha256"]:
            raise ValueError("sampling YAML and panel checkpoint hashes differ")
        if entry["checkpoint"] not in checkpoints:
            checkpoints[entry["checkpoint"]] = benchmark.checkpoint_metadata(
                ROOT / entry["checkpoint"], expected_sha256=entry["checkpoint_sha256"]
            )
        metadata = checkpoints[entry["checkpoint"]]
        sampling = contract["configuration"]
        if metadata["diffusion_type"] != sampling["diffusion_type"]:
            raise ValueError("checkpoint diffusion type differs from sampling YAML")
        benchmark.validate_denoiser_sampling_identity(metadata, sampling)
        if sampling["diffusion_type"] == "udlm":
            for checkpoint_key, sampling_key in (
                ("udlm_inference_eps", "inference_eps"),
                ("udlm_exclude_special_tokens", "exclude_special_tokens"),
                ("udlm_prior_variant", "prior_variant"),
                ("udlm_prior_metadata_sha256", "prior_metadata_sha256"),
            ):
                if metadata[checkpoint_key] != sampling[sampling_key]:
                    raise ValueError(f"checkpoint differs from sampling {sampling_key}")
        if (
            len(
                runner.FragmentPopulation.from_csv(
                    config["vocab_path"],
                    capacity=config["population_size"],
                    mode="released",
                    fragmenter=runner.cut,
                ).active_fragments
            )
            < 2
        ):
            raise ValueError("vocabulary needs at least two active fragments")
        jobs.append(
            {
                "entry": entry,
                "command": command,
                "config": config,
                "config_sha256": experiment_io.sha256_config(config),
                "checkpoint": metadata,
                "run_relative": str(
                    Path(panel["output_root"])
                    / "runs"
                    / entry["id"]
                    / panel["oracle"]
                    / "released"
                    / f"seed_{entry['seed']}"
                ),
                "log_relative": str(Path(panel["log_root"]) / (entry["id"] + ".log")),
            }
        )
    return {
        "schema_version": 1,
        "panel": panel,
        "panel_input": panel_record,
        "gpu_count": gpu_count,
        "inputs": list(records.values()),
        "jobs": jobs,
    }


def validate_inputs(plan):
    for wanted in plan["inputs"]:
        observed, _ = fingerprint(wanted["path"], wanted["sha256"])
        if observed != wanted:
            raise RuntimeError(f"input size changed: {wanted['path']}")


def select_gpus(states, count):
    integer(count, "gpu-count", 1, 2)
    if len({state.uuid for state in states}) != len(states) or len(
        {state.index for state in states}
    ) != len(states):
        raise ValueError("GPU inventory has duplicate UUIDs or physical indices")
    eligible = [state for state in states if not state.rejection_reasons(**POLICY)]
    eligible.sort(
        key=lambda state: (
            state.utilization_percent,
            state.memory_used_mib,
            state.index,
        )
    )
    return eligible[:count]


def child_environment(seed, gpu):
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("PYTHON") or key.startswith("GENMOL_BENCHMARK_"):
            env.pop(key)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu.uuid,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONOPTIMIZE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HOME": str(ROOT / ".cache/huggingface"),
            "TORCH_HOME": str(ROOT / ".cache/torch"),
            "TMPDIR": str(ROOT / "output/tmp"),
        }
    )
    return env


def validate_completion(job, source):
    records, values = {}, {}
    for name in ("manifest.json", "summary.json", "events.jsonl", "state/latest.pkl"):
        record, payload = fingerprint(
            job["run_relative"] + "/" + name, capture=name.endswith(".json")
        )
        records[name] = record
        if payload is not None:
            values[name] = _read_json(payload)
    manifest, summary = values["manifest.json"], values["summary.json"]
    if (
        manifest["status"] != "completed"
        or summary["status"] != "completed"
        or summary["checkpoint_consistent"] is not True
    ):
        raise RuntimeError("PMO child did not complete its full declared oracle budget")
    if (
        manifest["config"] != job["config"]
        or manifest["config_sha256"] != job["config_sha256"]
        or summary["config_sha256"] != job["config_sha256"]
    ):
        raise RuntimeError("PMO completion config binding differs")
    if (
        manifest["model"]["sha256"] != job["checkpoint"]["sha256"]
        or summary["model_sha256"] != job["checkpoint"]["sha256"]
    ):
        raise RuntimeError("PMO completion checkpoint binding differs")
    budget = job["entry"]["max_oracle_calls"]
    if (
        type(manifest["oracle_calls"]) is not int
        or manifest["oracle_calls"] != budget
        or summary["scores"]["all_charged_molecules"]["oracle_calls"] != budget
    ):
        raise RuntimeError("PMO completion oracle count differs from full budget")
    if (
        manifest["extra"]["git"]["commit"] != source["head"]
        or summary["run_id"] != manifest["run_id"]
    ):
        raise RuntimeError("PMO completion source/run identity differs")
    config = job["config"]
    expected_run_id = (
        f"{config['experiment_id']}:{config['oracle']}:released:seed{config['seed']}"
    )
    if manifest["run_id"] != expected_run_id or any(
        manifest[key] != value
        for key, value in (
            ("seed", config["seed"]),
            ("task", config["oracle"]),
            ("variant", "released"),
            ("oracle_budget", budget),
        )
    ):
        raise RuntimeError("PMO completion run slot differs from the request")
    receipt = manifest["extra"]["sampling"]
    if receipt.get("oracle_call_protocol") != udlm_sampling.ORACLE_CALL_PROTOCOL:
        raise RuntimeError("PMO acceptance lacks strict oracle-error propagation")
    if (
        summary["sampling"]["identity"] != receipt
        or receipt["contract"] != job["config"]["pmo_sampling"]
        or receipt["checkpoint"] != job["checkpoint"]
    ):
        raise RuntimeError("PMO sampling acceptance differs from preflight")
    benchmark.validate_inference_weights(receipt["inference_weights"], require_ema=True)
    for record in receipt["implementation_inputs"].values():
        recorded_path = Path(record["path"])
        path = (
            str(recorded_path.relative_to(ROOT))
            if ROOT in recorded_path.parents
            else str(recorded_path)
        )
        actual, _ = fingerprint(path, record["sha256"])
        if actual["size_bytes"] != record["size_bytes"]:
            raise RuntimeError("PMO sampling source/input size changed")
    return {
        "artifacts": records,
        "observed_sampling": summary["sampling"]["observed"],
        "oracle_calls": budget,
        "scores": summary["scores"],
    }


def stop_owned_process(process):
    """Signal only a process group created by this controller's own Popen."""
    result = {"pid": process.pid, "term_sent": False, "kill_sent": False}
    for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if not cleanup.process_group_exists(process.pid):
            break
        try:
            os.killpg(process.pid, sig)
            result["term_sent" if sig == signal.SIGTERM else "kill_sent"] = True
        except ProcessLookupError:
            break
        deadline = time.monotonic() + seconds
        while cleanup.process_group_exists(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.25)
    process.poll()
    result["return_code"] = process.returncode
    result["group_present_at_end"] = cleanup.process_group_exists(process.pid)
    return result


def execute(plan, source):
    panel = plan["panel"]
    output, logs = panel["output_root"], panel["log_root"]
    for path in (output, logs):
        if os.path.lexists(ROOT / path):
            raise FileExistsError(f"campaign namespace already exists: {path}")
        resources._ensure_repository_directory(
            (ROOT / path).parent, label="campaign parent"
        )
    resources._ensure_repository_directory(
        ROOT / "output/tmp", label="temporary directory"
    )
    lease = resources._acquire_generation_lease(source_revision=source["head"])
    owned_output = None
    jobs, active = [], []
    started = time.monotonic()
    terminal = {
        "schema_version": 1,
        "status": "failed",
        "source": source,
        "panel_sha256": plan["panel_input"]["sha256"],
        "started_at": stamp(),
        "jobs": jobs,
        "error": None,
    }
    telemetry_index = 0

    def publish(name, value):
        return artifact_io.publish_bytes_exclusive(
            ROOT, output + "/" + name, encode(value)
        )

    def event(value):
        nonlocal telemetry_index
        publish(f"telemetry/{telemetry_index:06d}.json", {"at": stamp(), **value})
        telemetry_index += 1

    def guard():
        if resources._require_clean_pushed_source() != source:
            raise RuntimeError("source changed during PMO campaign")
        resources._revalidate_generation_lease(lease)

    try:
        owned_output = artifact_io.create_directory_exclusive(ROOT, output)
        artifact_io.create_directory_exclusive(ROOT, logs)
        artifact_io.create_directory_exclusive(ROOT, output + "/telemetry")
        artifact_io.create_directory_exclusive(ROOT, output + "/runs")
        request = publish(
            "request_manifest.json",
            {
                "source": source,
                "plan": plan,
                "gpu_policy": {
                    **POLICY,
                    "max_concurrent": plan["gpu_count"],
                    "active_processes_allowed": True,
                },
                "started_at": terminal["started_at"],
            },
        )
        terminal["request_sha256"] = request.sha256
        first_popen = False
        capacity_deadline = time.monotonic() + panel["capacity_wait_seconds"]
        for start in range(0, len(plan["jobs"]), plan["gpu_count"]):
            wave = plan["jobs"][start : start + plan["gpu_count"]]
            validate_inputs(plan)
            while True:
                guard()
                try:
                    states = resources._snapshot()
                except Exception as error:
                    event(
                        {
                            "phase": "inventory",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    raise
                selected = select_gpus(states, len(wave))
                event(
                    {
                        "phase": "inventory",
                        "wave_start": start,
                        "gpus": [state.as_dict() for state in states],
                    }
                )
                if len(selected) == len(wave):
                    break
                if first_popen or time.monotonic() >= capacity_deadline:
                    raise RuntimeError(
                        "insufficient qualifying GPU capacity; no child retry"
                    )
                time.sleep(min(30, capacity_deadline - time.monotonic()))
            with ExitStack() as stack:
                active = []
                for job, initial in zip(wave, selected):
                    validate_inputs(plan)
                    guard()
                    log_owner = artifact_io.reserve_log_exclusive(
                        ROOT, job["log_relative"]
                    )
                    log = stack.enter_context(
                        artifact_io.open_log_append_exact(ROOT, log_owner)
                    )
                    try:
                        current = resources._probe_gpu(initial.uuid)
                    except Exception as error:
                        event(
                            {
                                "phase": "immediately_before_launch",
                                "entry_id": job["entry"]["id"],
                                "requested_uuid": initial.uuid,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                        raise
                    rejection = current.rejection_reasons(**POLICY)
                    if (current.uuid, current.index) != (initial.uuid, initial.index):
                        rejection.append("GPU identity changed")
                    event(
                        {
                            "phase": "immediately_before_launch",
                            "entry_id": job["entry"]["id"],
                            "requested_uuid": initial.uuid,
                            "gpus": [current.as_dict()],
                            "rejections": rejection,
                        }
                    )
                    if rejection:
                        raise RuntimeError(
                            f"final GPU probe rejected launch: {rejection}"
                        )
                    env = child_environment(job["entry"]["seed"], current)
                    record = {
                        "entry_id": job["entry"]["id"],
                        "status": "launching",
                        "command": job["command"],
                        "started_at": stamp(),
                        "gpu": current.as_dict(),
                        "environment": {
                            key: env[key]
                            for key in (
                                "CUDA_VISIBLE_DEVICES",
                                "CUDA_DEVICE_ORDER",
                                "PYTHONHASHSEED",
                                "PYTHONPATH",
                                "TMPDIR",
                                "OMP_NUM_THREADS",
                                "MKL_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS",
                                "TOKENIZERS_PARALLELISM",
                            )
                        },
                        "pid": None,
                        "return_code": None,
                    }
                    jobs.append(record)
                    launch = publish(job["entry"]["id"] + ".launch.json", record)
                    record["launch_sha256"] = launch.sha256
                    launch_time = time.monotonic()
                    process = subprocess.Popen(
                        job["command"],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    first_popen = True
                    record.update(pid=process.pid, status="running")
                    active.append((job, process, record, launch_time))
                pending = list(active)
                while pending:
                    guard()
                    for item in list(pending):
                        job, process, record, launch_time = item
                        code = process.poll()
                        elapsed = time.monotonic() - launch_time
                        if elapsed >= panel["child_timeout_seconds"]:
                            record["timed_out"] = True
                            raise RuntimeError(
                                f"child wall timeout: {record['entry_id']}"
                            )
                        if code is None:
                            continue
                        record.update(
                            return_code=code,
                            subprocess_seconds=elapsed,
                            finished_at=stamp(),
                        )
                        record["process_group_exit_grace"] = (
                            cleanup.wait_for_process_group_exit(process.pid)
                        )
                        if record["process_group_exit_grace"]["group_present_at_end"]:
                            raise RuntimeError(
                                "child process group remained after bounded exit grace"
                            )
                        if code != 0:
                            raise RuntimeError(
                                f"child exited {code}: {record['entry_id']}"
                            )
                        record["acceptance"] = validate_completion(job, source)
                        record["status"] = "completed"
                        pending.remove(item)
                    if pending:
                        time.sleep(1)
            active = []
        guard()
        validate_inputs(plan)
        terminal["status"] = "completed"
    except BaseException as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
        for record in jobs:
            if record["status"] == "launching":
                record["status"] = "failed_before_popen"
        for _, process, record, launch_time in active:
            if record["status"] != "completed":
                record["status"] = "failed"
            try:
                record["cleanup"] = stop_owned_process(process)
            except Exception as cleanup_error:
                record["cleanup"] = {
                    "error": f"{type(cleanup_error).__name__}: {cleanup_error}"
                }
            record["return_code"] = process.returncode
            record["subprocess_seconds"] = time.monotonic() - launch_time
    finally:
        safe = all(
            not cleanup.process_group_exists(process.pid) for _, process, _, _ in active
        )
        try:
            guard()
            validate_inputs(plan)
            terminal["final_input_validation"] = "unchanged"
        except Exception as error:
            terminal["status"] = "failed"
            terminal["final_input_validation"] = f"{type(error).__name__}: {error}"
        try:
            resources._revalidate_generation_lease(lease)
        except Exception as error:
            safe = False
            terminal["lease_error"] = f"{type(error).__name__}: {error}"
        if not safe:
            terminal["status"] = "failed"
        terminal.update(
            finished_at=stamp(),
            controller_seconds=time.monotonic() - started,
            lease_release_authorized=safe,
            telemetry_records=telemetry_index,
            unlaunched_entry_ids=[
                job["entry"]["id"]
                for job in plan["jobs"]
                if not any(
                    record["entry_id"] == job["entry"]["id"]
                    and record["pid"] is not None
                    for record in jobs
                )
            ],
        )
        if owned_output is not None:
            terminal["logs"] = [
                fingerprint(job["log_relative"])[0]
                for job in plan["jobs"]
                if os.path.lexists(ROOT / job["log_relative"])
            ]
            publish("terminal_manifest.json", terminal)
        if safe:
            resources._release_generation_lease_exact(lease)
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "terminal_manifest": output + "/terminal_manifest.json",
            }
        ),
        flush=True,
    )
    return 0 if terminal["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--gpu-count", type=int, choices=(1, 2), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    resources._require_project_virtual_environment()
    source = None if args.dry_run else resources._require_clean_pushed_source()
    plan = build_plan(args.panel, args.panel_sha256, args.gpu_count)
    if args.dry_run:
        print(encode(plan).decode(), end="")
        return 0
    resources._require_tmux_for_execution()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("unset CUDA_VISIBLE_DEVICES on the dynamic controller")
    if resources._require_clean_pushed_source() != source:
        raise RuntimeError("source changed during CPU preflight")
    return execute(plan, source)


if __name__ == "__main__":
    raise SystemExit(main())
