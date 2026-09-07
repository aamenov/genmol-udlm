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
PROCESS_GROUP_EXIT_GRACE_SECONDS = 15.0
PROCESS_GROUP_EXIT_POLL_SECONDS = 0.25
GPU_AVAILABILITY_WAIT_SECONDS = 21600
GPU_AVAILABILITY_POLL_SECONDS = 30
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


class GPUCapacityUnavailable(RuntimeError):
    """Verified GPU identities currently fail the unchanged resource thresholds."""


def recheck_gpus(selected, emit, *, allow_capacity_wait=False):
    """Keep every final probe, including rejected or failed UUID queries."""
    checked, rejected, invalid_probe = [], {}, False
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
                invalid_probe = True
            if reasons:
                rejected[initial.uuid] = reasons
            checked.append(current)
        except Exception as error:
            invalid_probe = True
            event["error"] = f"{type(error).__name__}: {error}"
            rejected[initial.uuid] = [event["error"]]
        emit(event)
    if rejected:
        if allow_capacity_wait and not invalid_probe:
            raise GPUCapacityUnavailable(f"final GPU probe rejected launch: {rejected}")
        raise RuntimeError(f"final GPU probe rejected launch: {rejected}")
    return tuple(checked)


def gpu_availability_wait_policy(plan):
    """Only an explicit matching plan/protocol pair enables the bounded wait."""
    fields = {
        "gpu_availability_wait_seconds": GPU_AVAILABILITY_WAIT_SECONDS,
        "gpu_availability_poll_seconds": GPU_AVAILABILITY_POLL_SECONDS,
    }
    if not any(key in plan or key in plan["protocol"] for key in fields):
        return None
    if any(
        type(plan.get(key)) is not int
        or plan.get(key) != value
        or type(plan["protocol"].get(key)) is not int
        or plan["protocol"].get(key) != value
        for key, value in fields.items()
    ):
        raise ValueError(
            "GPU capacity wait requires matching 21600-second/30-second plan and protocol"
        )
    return fields


def wait_for_gpu_capacity(plan, source, leases, emit, record):
    """Wait only before the first child; malformed probes/source/leases fail closed."""
    policy = gpu_availability_wait_policy(plan)
    if policy is None:
        raise ValueError("GPU capacity waiting is not enabled by this plan")
    started = time.monotonic()
    deadline = started + policy["gpu_availability_wait_seconds"]
    record.update(policy, started_at=stamp(), rounds=0, outcome="waiting")

    def timeout():
        record["outcome"] = "timed_out"
        raise TimeoutError(
            "GPU launch capacity did not remain eligible within 21600 seconds"
        )

    try:
        while True:
            if benchmark._require_clean_pushed_source() != source:
                raise RuntimeError("source changed while waiting for GPU capacity")
            check_leases(ROOT, leases)
            if time.monotonic() >= deadline:
                timeout()
            record["rounds"] += 1
            event = {
                "phase": "selection",
                "at": stamp(),
                "availability_round": record["rounds"],
            }
            try:
                inventory = audited.probe_all_gpus()
                event["gpus"] = [asdict(state) for state in inventory]
                event["rejection_reasons"] = [
                    {
                        "uuid": state.uuid,
                        "reasons": state.rejection_reasons(
                            max_utilization_percent=10, min_free_memory_mib=30000
                        ),
                    }
                    for state in inventory
                ]
            except Exception as error:
                event["error"] = f"{type(error).__name__}: {error}"
                emit(event)
                raise
            emit(event)
            # Invalid identities or policy implementations are defects, not
            # insufficient capacity, and must not be hidden by waiting.
            validate_count(plan["gpu_count"])
            if len({state.uuid for state in inventory}) != len(inventory) or len(
                {state.physical_index for state in inventory}
            ) != len(inventory):
                raise ValueError("GPU inventory contains duplicate identities")
            if audited.ACTIVE_COMPUTE_PROCESSES_ALLOWED is not True:
                raise RuntimeError(
                    "audited GPU helper does not implement the authorized policy"
                )
            rejections = {
                state.uuid: state.rejection_reasons(
                    max_utilization_percent=10, min_free_memory_mib=30000
                )
                for state in inventory
            }
            eligible_count = sum(not reasons for reasons in rejections.values())
            try:
                if eligible_count < plan["gpu_count"]:
                    raise GPUCapacityUnavailable(
                        f"requested {plan['gpu_count']} GPUs; only {eligible_count} eligible: {rejections}"
                    )
                selected = select_gpus(inventory, plan["gpu_count"])
                selected = recheck_gpus(selected, emit, allow_capacity_wait=True)
            except GPUCapacityUnavailable as error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timeout()
                sleep_seconds = min(policy["gpu_availability_poll_seconds"], remaining)
                emit(
                    {
                        "phase": "availability_wait",
                        "at": stamp(),
                        "availability_round": record["rounds"],
                        "error": str(error),
                        "sleep_seconds": sleep_seconds,
                    }
                )
                time.sleep(sleep_seconds)
                continue
            check_leases(ROOT, leases)
            if time.monotonic() >= deadline:
                timeout()
            record["outcome"] = "ready"
            record["selected_gpu_uuids"] = [state.uuid for state in selected]
            return selected
    except BaseException as error:
        if record["outcome"] == "waiting":
            record["outcome"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_at"] = stamp()
        record["elapsed_seconds"] = time.monotonic() - started


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


def validate_checkpoint_output(path, config, *, expected_steps=20):
    """Inspect local, trusted checkpoint on CPU; do not trust exit status alone."""
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(1)
    before, _ = artifact_io.snapshot_file(ROOT, path.relative_to(ROOT))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") != expected_steps:
        raise RuntimeError(
            f"final checkpoint is not at optimizer step {expected_steps}"
        )
    saved = checkpoint.get("hyper_parameters", {}).get("config")
    if OmegaConf.is_config(saved):
        saved = OmegaConf.to_container(saved, resolve=True)
    if saved != config:
        raise RuntimeError("final checkpoint configuration differs from launch")
    from genmol.model import (
        UDLM_DENOISER_CHECKPOINT_KEY,
        UDLM_DENOISER_METADATA,
        UDLM_DENOISER_STATE_KEY,
        _exact_nested_data_equal,
    )

    parameterization = (
        config.get("training", {}).get("udlm", {}).get("parameterization", "raw_loo")
    )
    state_dict = checkpoint.get("state_dict", {})
    marker = state_dict.get(UDLM_DENOISER_STATE_KEY)
    if parameterization == "x0_denoiser":
        if not _exact_nested_data_equal(
            checkpoint.get(UDLM_DENOISER_CHECKPOINT_KEY), UDLM_DENOISER_METADATA
        ):
            raise RuntimeError("final CE checkpoint lacks valid denoiser metadata")
        if (
            not isinstance(marker, torch.Tensor)
            or marker.shape != torch.Size([])
            or marker.dtype != torch.int64
            or marker.item() != 1
        ):
            raise RuntimeError("final CE checkpoint lacks its denoiser state marker")
    elif (
        parameterization != "raw_loo"
        or UDLM_DENOISER_CHECKPOINT_KEY in checkpoint
        or UDLM_DENOISER_STATE_KEY in state_dict
    ):
        raise RuntimeError(
            "final raw-LOO checkpoint contains incompatible denoiser identity"
        )
    prior_audit = {}
    training = config.get("training", {})
    udlm = training.get("udlm", {})
    if str(udlm.get("prior_variant", "")).lower() == "mask_rich_empirical":
        from scripts.exps.denovo import benchmark as checkpoint_audit

        if training.get("diffusion") != "udlm":
            raise RuntimeError("final mask-rich checkpoint must use UDLM")
        if (
            type(training.get("antithetic_sampling")) is not bool
            or type(udlm.get("exclude_special_tokens", False)) is not bool
        ):
            raise RuntimeError(
                "final mask-rich checkpoint has invalid Boolean settings"
            )
        metadata = checkpoint_audit.validate_udlm_prior_metadata_record(
            checkpoint.get(checkpoint_audit.UDLM_PRIOR_CHECKPOINT_KEY),
            expected_variant="mask_rich_empirical",
            expected_full_vocab_size=checkpoint_audit._strict_integer(
                config.get("model", {}).get("vocab_size"),
                "final model.vocab_size",
                minimum=2,
            ),
            expected_exclude_special_tokens=udlm.get("exclude_special_tokens", False),
            expected_sampling_eps=checkpoint_audit._strict_probability(
                training.get("sampling_eps"), "final training.sampling_eps"
            ),
            expected_noise_eps=checkpoint_audit._strict_probability(
                udlm.get("noise_eps", 1e-3), "final training.udlm.noise_eps"
            ),
            expected_antithetic_sampling=training["antithetic_sampling"],
            expected_uniform_mixture_weight=checkpoint_audit._strict_probability(
                udlm.get("empirical_uniform_mix"), "final empirical_uniform_mix"
            ),
            expected_mask_mixture_weight=checkpoint_audit._strict_mask_mixture_weight(
                udlm.get("mask_mixture_weight"), "final mask_mixture_weight"
            ),
            state_dict=state_dict,
        )
        prior_audit["udlm_prior_metadata_sha256"] = canonical_digest(metadata)
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
    if checkpoint["ema"].get("num_updates") != expected_steps or tensor_count == 0:
        raise RuntimeError(
            "final checkpoint EMA updates or model tensors are incomplete"
        )
    after, _ = artifact_io.snapshot_file(ROOT, path.relative_to(ROOT))
    if before != after:
        raise RuntimeError("final checkpoint changed during CPU validation")
    return {
        **asdict(after),
        "global_step": expected_steps,
        "finite_tensor_count": tensor_count,
        **prior_audit,
    }


def process_group_exists(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_group_exit(pid):
    """Observe descendant teardown after the parent is reaped; never signal it."""
    started_at = stamp()
    start = time.monotonic()
    deadline = start + PROCESS_GROUP_EXIT_GRACE_SECONDS
    probes = 0
    initially_present = None
    while True:
        present = process_group_exists(pid)
        probes += 1
        if initially_present is None:
            initially_present = present
        now = time.monotonic()
        if not present or now >= deadline:
            return {
                "process_group_id": pid,
                "started_at": started_at,
                "finished_at": stamp(),
                "grace_seconds": PROCESS_GROUP_EXIT_GRACE_SECONDS,
                "poll_interval_seconds": PROCESS_GROUP_EXIT_POLL_SECONDS,
                "elapsed_seconds": now - start,
                "probe_count": probes,
                "initially_present": initially_present,
                "group_present_at_end": present,
                "outcome": (
                    "timed_out"
                    if present
                    else (
                        "exited_during_grace" if initially_present else "already_exited"
                    )
                ),
            }
        time.sleep(min(PROCESS_GROUP_EXIT_POLL_SECONDS, deadline - now))


def execute(plan, source, *, plan_builder=None):
    """Run the sole child and retain failure evidence before releasing resources."""
    output = plan["output_relative"]
    plan_builder = build_plan if plan_builder is None else plan_builder
    expected_steps = plan["protocol"]["optimizer_updates"]
    exposures = plan["example_exposures"]
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
        "process_group_exit_grace": None,
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

            waiting = gpu_availability_wait_policy(plan)
            if waiting is None:
                # Preserve the historical immediate-launch sequence exactly.
                inventory = audited.probe_all_gpus()
                record_probe(
                    {
                        "phase": "selection",
                        "at": stamp(),
                        "gpus": [asdict(s) for s in inventory],
                    }
                )
                selected = select_gpus(inventory, plan["gpu_count"])
            # Expensive initialization hashing and Hydra/source validation run
            # before the opt-in availability loop, never between its successful
            # inventory and final UUID checks.
            input_claim = verify_checkpoint_input(plan)
            if (
                benchmark._require_clean_pushed_source() != source
                or plan_builder(plan["gpu_count"]) != plan
            ):
                raise RuntimeError(
                    "source or fixed configuration changed before launch"
                )
            if waiting is None:
                selected = recheck_gpus(selected, record_probe)
            else:
                terminal["gpu_availability_wait"] = {}
                selected = wait_for_gpu_capacity(
                    plan,
                    source,
                    leases,
                    record_probe,
                    terminal["gpu_availability_wait"],
                )
            check_leases(ROOT, leases)
            uuids = [state.uuid for state in selected]
            env = child_environment(uuids)
            env["PYTHONHASHSEED"] = str(plan["config"].get("seed", 1400))
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
            if waiting is not None:
                launch["gpu_availability_wait"] = terminal["gpu_availability_wait"]
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
            # Lightning's rank/data-worker descendants can finish just after the
            # reaped parent. Bound this observation window without releasing a
            # lease, changing the process return code, or signaling descendants.
            terminal["process_group_exit_grace"] = wait_for_process_group_exit(
                process.pid
            )
            if terminal["process_group_exit_grace"]["group_present_at_end"]:
                safe_to_release = False
                raise RuntimeError(
                    "child exited but its process group remains after cleanup grace; "
                    "leases retained"
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"training process exited with status {process.returncode}"
                )
        validated_checkpoint = validate_checkpoint_output(
            ROOT / output / "checkpoints" / f"{expected_steps}.ckpt",
            plan["config"],
            expected_steps=expected_steps,
        )
        if (
            "expected_prior_metadata_sha256" in plan
            and validated_checkpoint.get("udlm_prior_metadata_sha256")
            != plan["expected_prior_metadata_sha256"]
        ):
            raise RuntimeError(
                "final checkpoint prior differs from the prospective plan"
            )
        terminal["checkpoint"] = validated_checkpoint
        if benchmark._require_clean_pushed_source() != source:
            raise RuntimeError("source changed during training")
        check_leases(ROOT, leases)
        terminal["status"] = "completed"
        terminal["completed_example_exposures"] = exposures
        terminal["end_to_end_training_examples_per_second"] = (
            exposures / terminal["training_subprocess_seconds"]
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
