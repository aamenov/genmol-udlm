"""Launch only the frozen v7 20-update CT-E throughput pilot, inside tmux.

Run --dry-run from any development checkout. Live runs require the canonical
artifact checkout, clean pushed source and the project virtual environment.
Existing generation and training locks are both held until terminal evidence
is durable and the child process group has exited. Nothing resumes or retries.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parents[1]
CANONICAL_ROOT = PROJECT_ROOT / "run_sources/udlm_genmol_worktree"
PROTOCOL = "experiments/udlm/protocols/engineering_v7_throughput.json"
LEASE_PATHS = (
    "output/.single_generation_job.lock",
    "output/udlm/.single_training_job.lock",
)
POLICY = {
    "max_gpus": 2,
    "max_utilization_percent": 10,
    "min_free_memory_mib": 30000,
    "active_compute_processes_allowed": True,
}
for import_root in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(import_root))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import launch_benchmark as benchmark  # noqa: E402
from scripts.udlm import launch_train_pilot as audited  # noqa: E402


def stamp():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_count(gpu_count):
    if type(gpu_count) is not int or gpu_count not in (1, 2):
        raise ValueError("gpu-count must be 1 or 2")
    return gpu_count


def ensure_directory(root, relative):
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise ValueError("directory must be relative to the repository")
    for depth in range(1, len(parts) + 1):
        subpath = Path(*parts[:depth])
        try:
            artifact_io.create_directory_exclusive(root, subpath)
        except FileExistsError:
            path = root / subpath
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"artifact parent is not a direct directory: {path}")


def build_plan(gpu_count, *, root=ROOT):
    """Compose all fixed task settings without CUDA discovery or mutations."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    validate_count(gpu_count)
    protocol_claim, payload = artifact_io.snapshot_file(
        root, PROTOCOL, capture_bytes=True
    )
    protocol = json.loads(payload)
    fixed = {
        "schema_version": 1,
        "study_id": "engineering-v7-ct-e-throughput20",
        "config": "configs/udlm_e_throughput20.yaml",
        "objective": "existing_categorical_ct_raw_loo",
        "seed": 1400,
        "optimizer_updates": 20,
        "global_batch_size": 128,
        "micro_batch_size": 16,
        "gpu_policy": POLICY,
    }
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise ValueError("the frozen 20-update protocol changed")
    config_claim, _ = artifact_io.snapshot_file(root, protocol["config"])
    if config_claim.sha256 != protocol["config_sha256"]:
        raise ValueError("protocol config digest mismatch")
    attempt = f"ct_e_throughput20_b128_w{gpu_count}"
    output = f"output/udlm/engineering_v7/{attempt}"
    checkpoint = PROJECT_ROOT / protocol["checkpoint"]
    if not checkpoint.is_relative_to(PROJECT_ROOT):
        raise ValueError("initialization checkpoint escapes project")
    overrides = [
        f"trainer.devices={gpu_count}",
        f"trainer.accumulate_grad_batches={audited.exact_accumulation_steps(128, 16, gpu_count)}",
        f"training.init_from_mdlm_checkpoint={checkpoint}",
        f"training.init_from_mdlm_checkpoint_sha256={protocol['checkpoint_sha256']}",
        f"callback.dirpath={root / output / 'checkpoints'}",
        f"hydra.run.dir={root / output / 'hydra'}",
    ]
    resolvers = {
        "cwd": lambda: str(root),
        "device_count": lambda: gpu_count,
        "eval": lambda value: eval(value, {"__builtins__": {}}, {}),
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        config = OmegaConf.to_container(
            compose(config_name="udlm_e_throughput20", overrides=overrides),
            resolve=True,
        )
    trainer, loader, training = config["trainer"], config["loader"], config["training"]
    if (
        config["seed"] != 1400
        or trainer["max_steps"] != 20
        or trainer["num_nodes"] != 1
        or loader["global_batch_size"] != 128
        or loader["batch_size"] != 16
        or 16 * gpu_count * trainer["accumulate_grad_batches"] != 128
        or training["diffusion"] != "udlm"
        or training["udlm"]["prior_variant"] != "empirical_frequency"
        or training["udlm"]["conditioning_variant"] != "film_adaln"
        or training["udlm"]["empirical_uniform_mix"] != 0.0002
        or training["udlm"]["exclude_special_tokens"] is not False
        or training["udlm"].get("parameterization", "raw_loo") != "raw_loo"
        or training["reseed_after_model_initialization"] is not False
        or training["init_from_mdlm_ema"] is not True
        or training["udlm"]["zero_init_conditioning"] is not False
        or config["optim"]["lr"] != 0.0003
        or config["optim"]["scheduler"]
        != {
            "name": "half_cosine_with_linear_warmup_and_floor",
            "warmup_updates": 50,
            "horizon_updates": 1000,
            "decay_floor_lr": 0.000003,
        }
    ):
        raise ValueError("resolved config differs from the fixed CT-E throughput pilot")
    command = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        "-u",
        str(root / "scripts/train.py"),
        "--config-name",
        "udlm_e_throughput20",
        *overrides,
    ]
    return {
        "schema_version": 1,
        "kind": "manual_engineering_training",
        "attempt_id": attempt,
        "protocol": protocol,
        "protocol_sha256": protocol_claim.sha256,
        "config_source_sha256": config_claim.sha256,
        "config": config,
        "config_sha256": canonical_digest(config),
        "training_argv": command,
        "argv_sha256": canonical_digest(command),
        "gpu_count": gpu_count,
        "output_relative": output,
        "log_relative": f"output/logs/engineering_v7/{attempt}.training.log",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "example_exposures": 2560,
    }


def select_gpus(states, gpu_count):
    validate_count(gpu_count)
    if len({s.uuid for s in states}) != len(states) or len(
        {s.physical_index for s in states}
    ) != len(states):
        raise ValueError("GPU inventory contains duplicate identities")
    if audited.ACTIVE_COMPUTE_PROCESSES_ALLOWED is not True:
        raise RuntimeError(
            "audited GPU helper does not implement the authorized policy"
        )
    return audited.select_idle_gpus(
        states,
        gpu_count=gpu_count,
        max_utilization_percent=10,
        min_free_memory_mib=30000,
    )


def recheck_gpus(selected, emit):
    """Keep every final probe, including rejected or failed UUID queries."""
    checked, rejected = [], {}
    for initial in selected:
        event = {
            "phase": "immediately_before_launch",
            "at": stamp(),
            "requested_uuid": initial.uuid,
        }
        try:
            current = audited.probe_gpu_uuid(initial.uuid)
            event["gpus"] = [asdict(current)]
            reasons = current.rejection_reasons(
                max_utilization_percent=10, min_free_memory_mib=30000
            )
            if (current.uuid, current.physical_index) != (
                initial.uuid,
                initial.physical_index,
            ):
                reasons.append("selected GPU identity changed")
            if reasons:
                rejected[initial.uuid] = reasons
            checked.append(current)
        except Exception as error:
            event["error"] = f"{type(error).__name__}: {error}"
            rejected[initial.uuid] = [event["error"]]
        emit(event)
    if rejected:
        raise RuntimeError(f"final GPU probe rejected launch: {rejected}")
    return tuple(checked)


def acquire_leases(root, identity):
    """Block both old generation and training launchers; rollback only our claims."""
    claims = []
    payload = encode(
        {
            "schema_version": 1,
            "purpose": "manual_engineering_training_excludes_generation_and_training",
            "identity": identity,
            "owner_token": secrets.token_hex(32),
            "pid": os.getpid(),
            "acquired_at": stamp(),
            "stale_policy": "never_automatically_recover",
        }
    )
    try:
        for relative in LEASE_PATHS:
            claims.append(artifact_io.acquire_lock_exclusive(root, relative, payload))
    except BaseException:
        release_leases(root, claims)
        raise
    return claims


def release_leases(root, claims):
    for claim in reversed(claims):
        artifact_io.release_lock_exact(root, claim)


def check_leases(root, claims):
    for claim in claims:
        current, _ = artifact_io.snapshot_file(root, claim.relative_path)
        if current != claim:
            raise RuntimeError("owned resource lease changed")


def child_environment(uuids):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GENMOL_", "SLURM_", "OMPI_", "PMI_", "PMIX_"))
        and key not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "NODE_RANK", "GROUP_RANK"}
    }
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(uuids),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="1400",
        PYTHONOPTIMIZE="0",
    )
    return env


def verify_checkpoint_input(plan):
    relative = Path(plan["checkpoint_path"]).relative_to(PROJECT_ROOT)
    claim, _ = artifact_io.snapshot_file(PROJECT_ROOT, relative)
    if claim.sha256 != plan["checkpoint_sha256"]:
        raise RuntimeError("MDLM initialization checkpoint digest mismatch")
    return asdict(claim)


def validate_checkpoint_output(path, config):
    """Inspect local, trusted checkpoint on CPU; do not trust exit status alone."""
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(1)
    before, _ = artifact_io.snapshot_file(ROOT, path.relative_to(ROOT))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") != 20:
        raise RuntimeError("final checkpoint is not at optimizer step 20")
    saved = checkpoint.get("hyper_parameters", {}).get("config")
    if OmegaConf.is_config(saved):
        saved = OmegaConf.to_container(saved, resolve=True)
    if saved != config:
        raise RuntimeError("final checkpoint configuration differs from launch")
    tensor_count = 0

    def check(value):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            if not torch.isfinite(value).all():
                raise RuntimeError("final checkpoint contains non-finite tensors")
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)

    for key in ("state_dict", "optimizer_states", "ema"):
        if not checkpoint.get(key):
            raise RuntimeError(f"final checkpoint lacks {key}")
        check(checkpoint[key])
    if checkpoint["ema"].get("num_updates") != 20 or tensor_count == 0:
        raise RuntimeError("final checkpoint lacks twenty EMA updates or model tensors")
    after, _ = artifact_io.snapshot_file(ROOT, path.relative_to(ROOT))
    if before != after:
        raise RuntimeError("final checkpoint changed during CPU validation")
    return {**asdict(after), "global_step": 20, "finite_tensor_count": tensor_count}


def process_group_exists(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def execute(plan, source):
    """Run the sole child and retain failure evidence before releasing resources."""
    output = plan["output_relative"]
    ensure_directory(ROOT, str(Path(output).parent))
    ensure_directory(ROOT, str(Path(plan["log_relative"]).parent))
    artifact_io.create_directory_exclusive(ROOT, output)

    def publish(name, value):
        return artifact_io.publish_bytes_exclusive(
            ROOT, f"{output}/{name}", encode(value)
        )

    request = {"created_at": stamp(), "source": source, "plan": plan}
    request_claim = publish("request_manifest.json", request)
    leases, process = [], None
    terminal = {
        "schema_version": 1,
        "kind": "manual_engineering_training_terminal",
        "status": "failed",
        "request_sha256": request_claim.sha256,
        "started_at": stamp(),
        "source": source,
        "plan": plan,
        "training_return_code": None,
        "completed_example_exposures": None,
        "end_to_end_training_examples_per_second": None,
        "checkpoint": None,
        "telemetry_errors": [],
    }
    controller_start = time.monotonic()
    training_start = None
    safe_to_release = True
    try:
        leases = acquire_leases(
            ROOT, {"attempt_id": plan["attempt_id"], "source": source}
        )
        terminal["leases"] = [asdict(claim) for claim in leases]
        log_owner = artifact_io.reserve_log_exclusive(ROOT, plan["log_relative"])
        telemetry_relative = f"{output}/gpu_telemetry.jsonl"
        telemetry_owner = artifact_io.reserve_log_exclusive(ROOT, telemetry_relative)
        with (
            artifact_io.open_log_append_exact(ROOT, log_owner) as log,
            artifact_io.open_log_append_exact(ROOT, telemetry_owner) as telemetry,
        ):
            probes = []

            def emit(event):
                telemetry.write(
                    (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
                )

            def record_probe(event):
                probes.append(event)
                emit(event)

            inventory = audited.probe_all_gpus()
            record_probe(
                {
                    "phase": "selection",
                    "at": stamp(),
                    "gpus": [asdict(s) for s in inventory],
                }
            )
            selected = select_gpus(inventory, plan["gpu_count"])
            input_claim = verify_checkpoint_input(plan)
            if (
                benchmark._require_clean_pushed_source() != source
                or build_plan(plan["gpu_count"]) != plan
            ):
                raise RuntimeError(
                    "source or fixed configuration changed before launch"
                )
            selected = recheck_gpus(selected, record_probe)
            check_leases(ROOT, leases)
            uuids = [state.uuid for state in selected]
            env = child_environment(uuids)
            launch = {
                **request,
                "input_checkpoint": input_claim,
                "probes": probes,
                "selected_gpu_uuids": uuids,
                "leases": terminal["leases"],
                "training_environment": {
                    key: env[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "CUDA_DEVICE_ORDER",
                        "PYTHONPATH",
                        "PYTHONHASHSEED",
                        "PYTHONNOUSERSITE",
                        "PYTHONDONTWRITEBYTECODE",
                        "PYTHONOPTIMIZE",
                    )
                },
            }
            launch_claim = publish("launch_manifest.json", launch)
            terminal["launch_sha256"] = launch_claim.sha256
            terminal["selected_gpu_uuids"] = uuids
            training_start = time.monotonic()
            process = subprocess.Popen(
                plan["training_argv"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            terminal["training_pid"] = process.pid
            maximum_used = {state.uuid: state.memory_used_mib for state in selected}
            while process.poll() is None:
                for uuid in uuids:
                    try:
                        state = audited.probe_gpu_uuid(uuid)
                        emit(
                            {
                                "phase": "during_training",
                                "at": stamp(),
                                "requested_uuid": uuid,
                                "gpus": [asdict(state)],
                            }
                        )
                        maximum_used[state.uuid] = max(
                            maximum_used[state.uuid], state.memory_used_mib
                        )
                    except Exception as error:
                        event = {
                            "at": stamp(),
                            "requested_uuid": uuid,
                            "error": f"{type(error).__name__}: {error}",
                        }
                        terminal["telemetry_errors"].append(event)
                        emit(event)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            terminal["training_return_code"] = process.returncode
            terminal["training_subprocess_seconds"] = time.monotonic() - training_start
            terminal["max_observed_aggregate_gpu_used_mib"] = maximum_used
            terminal["memory_measurement"] = plan["protocol"]["memory_measurement"]
            if process_group_exists(process.pid):
                safe_to_release = False
                raise RuntimeError(
                    "child exited but its process group remains; leases retained"
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"training process exited with status {process.returncode}"
                )
        terminal["checkpoint"] = validate_checkpoint_output(
            ROOT / output / "checkpoints/20.ckpt", plan["config"]
        )
        if benchmark._require_clean_pushed_source() != source:
            raise RuntimeError("source changed during training")
        check_leases(ROOT, leases)
        terminal["status"] = "completed"
        terminal["completed_example_exposures"] = 2560
        terminal["end_to_end_training_examples_per_second"] = (
            2560 / terminal["training_subprocess_seconds"]
        )
    except BaseException as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                safe_to_release = False
            terminal["training_return_code"] = process.poll()
        if process is not None and process_group_exists(process.pid):
            safe_to_release = False
    finally:
        if process is not None and process_group_exists(process.pid):
            safe_to_release = False
            terminal["status"] = "failed"
            terminal.setdefault("error", "training process group has not exited")
        try:
            check_leases(ROOT, leases)
        except Exception as error:
            safe_to_release = False
            terminal["lease_validation_error"] = f"{type(error).__name__}: {error}"
            terminal["status"] = "failed"
        terminal["finished_at"] = stamp()
        if terminal["status"] != "completed":
            terminal["completed_example_exposures"] = None
            terminal["end_to_end_training_examples_per_second"] = None
        terminal["controller_seconds"] = time.monotonic() - controller_start
        terminal["leases_release_authorized"] = safe_to_release
        terminal["artifact_hashes"] = {}
        artifact_paths = [plan["log_relative"], f"{output}/gpu_telemetry.jsonl"]
        artifact_paths.extend(
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / output / "checkpoints").glob("*.ckpt"))
        )
        for relative in artifact_paths:
            if (ROOT / relative).is_file():
                claim, _ = artifact_io.snapshot_file(ROOT, relative)
                terminal["artifact_hashes"][relative] = claim.sha256
        publish("terminal_manifest.json", terminal)
        if safe_to_release:
            release_leases(ROOT, leases)
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "terminal_manifest": f"{output}/terminal_manifest.json",
            }
        ),
        flush=True,
    )
    return 0 if terminal["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    benchmark._require_project_virtual_environment()
    if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
        raise RuntimeError(
            "controller Python prefix is not the project virtual environment"
        )
    source = benchmark._require_clean_pushed_source()
    plan = build_plan(args.gpu_count)
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", PROTOCOL, plan["protocol"]["config"]],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    verify_checkpoint_input(plan)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "source": source,
                    "plan": plan,
                    "gpu_queries": 0,
                    "artifact_mutations": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if ROOT != CANONICAL_ROOT:
        raise RuntimeError(
            "merge into the canonical artifact worktree before live execution"
        )
    benchmark._require_tmux_for_execution()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            "unset CUDA_VISIBLE_DEVICES on the controller; selection is dynamic"
        )
    return execute(plan, source)


if __name__ == "__main__":
    raise SystemExit(main())
