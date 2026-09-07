"""Launch reproducible de novo benchmark runs on dynamically selected GPUs.

The controller is intended to run inside ``tmux``. The caller supplies only a
GPU count from one through three. Before each child starts, the controller inventories
every NVIDIA GPU and dynamically chooses a policy-eligible device under the configured
utilization, memory, and compute-mode guards. It then re-probes that exact UUID
and maps it into the child as logical ``cuda:0``. Active compute processes are
allowed only when the utilization and free-memory guards still pass; their
complete NVIDIA telemetry is retained in both probes and never interrupted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_SRC = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, REPOSITORY_SRC):
    while str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import benchmark as benchmark_runner  # noqa: E402


FINAL_BENCHMARK_SAMPLES_PER_SEED = 1_000
REGISTERED_SELECTION_PILOT_SEEDS = (1000, 1001)
REGISTERED_SELECTION_PILOT_SAMPLES_PER_SEED = 256
ATTEMPT_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
CANDIDATE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,95}\Z")
PILOT_FAILURE_RECEIPT_SCHEMA_VERSION = 1
PILOT_FAILURE_RECEIPT_KIND = "pilot_failure"
PILOT_FAILURE_RECEIPT_FILENAME = "failure_receipt.json"
PILOT_MODES = {"engineering", "registered_selection"}
MAX_CONCURRENT_GENERATION_JOBS = 3
GENERATION_LEASE_RELATIVE_PATH = "output/.single_generation_job.lock"
GENERATION_LEASE_SCHEMA_VERSION = 1
LAUNCH_AUTHORITY_SCHEMA_VERSION = 1
ARTIFACT_IO_RELATIVE_PATH = "scripts/artifact_io.py"
CANDIDATE_LOCK_RELATIVE_PATH = "experiments/udlm/candidates/candidate_lock.json"
BENCHMARK_RUNNER_RELATIVE_PATH = "scripts/exps/denovo/benchmark.py"
SAMPLER_SOURCE_RELATIVE_PATH = "src/genmol/sampler.py"
FINAL_CONFIG_ID_PATTERN = re.compile(r"[rse]_t(?:050|070|085|100)_p(?:095|098|100)\Z")
FINAL_CANDIDATE_ID_PATTERN = re.compile(r"[rse]-w1-1000u-dcb271453411\Z")
FINAL_INFERENCE_WEIGHTS = {
    "source": "ema",
    "ema_applied": True,
    "ema": {
        "shadow_parameter_count": 230,
        "decay": 0.9999,
        "num_updates": 1_000,
    },
}


class CompletionArtifactError(RuntimeError):
    """Raised when a seed directory cannot safely be skipped or relaunched."""


class FinalCandidateLockError(RuntimeError):
    """Raised when final-tier authority is absent, stale, or inconsistent."""


@dataclass(frozen=True)
class GPUState:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int
    compute_mode: str
    compute_processes: tuple[dict[str, Any], ...]

    def rejection_reasons(
        self,
        *,
        max_utilization_percent: int,
        min_free_memory_mib: int,
    ) -> list[str]:
        reasons = []
        free_memory_mib = self.memory_total_mib - self.memory_used_mib
        if free_memory_mib < min_free_memory_mib:
            reasons.append(
                f"{free_memory_mib} MiB free is below the required "
                f"{min_free_memory_mib} MiB"
            )
        if self.utilization_percent >= max_utilization_percent:
            reasons.append(
                f"{self.utilization_percent}% utilization is not strictly below "
                f"{max_utilization_percent}%"
            )
        if self.compute_mode.lower() == "prohibited":
            reasons.append("compute mode is prohibited")
        return reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "utilization_percent": self.utilization_percent,
            "compute_mode": self.compute_mode,
            "compute_processes": list(self.compute_processes),
        }


@dataclass
class RunningJob:
    seed: int
    gpu: GPUState
    process: subprocess.Popen
    log_handle: Any
    log_path: Path
    command: tuple[str, ...]
    started_at_utc: str
    log_owner: artifact_io.OwnedLog | None = None
    log_context: Any | None = None
    output_directory_owner: artifact_io.OwnedDirectory | None = None


@dataclass(frozen=True)
class GenerationLease:
    """Exact-owner capability for the repository-global generation lease."""

    claim: artifact_io.FileClaim
    owner_token: str
    payload: bytes
    artifact_io_source: artifact_io.FileClaim


@dataclass(frozen=True)
class FinalCandidateAuthority:
    """Retained identity and normalized contents of the pre-final lock."""

    claim: artifact_io.FileClaim
    payload: bytes
    source_revision: str
    normalized_lock: Mapping[str, Any]
    candidate_id: str
    config_id: str
    output_root: Path


@dataclass(frozen=True)
class ExpectedRunIdentity:
    """Inputs that must match before an existing seed is considered complete."""

    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_global_step: int
    checkpoint_size_bytes: int
    checkpoint_diffusion_type: str
    checkpoint_udlm_inference_eps: float | None
    checkpoint_udlm_exclude_special_tokens: bool | None
    checkpoint_udlm_prior_variant: str | None
    checkpoint_udlm_prior_metadata: Mapping[str, Any] | None
    checkpoint_udlm_prior_metadata_sha256: str | None
    config_path: Path
    source_config: Mapping[str, Any]
    source_config_sha256: str
    config_git_tracking: Mapping[str, Any] | None
    sampling_config: Mapping[str, Any]
    sampling_config_sha256: str
    effective_config: Mapping[str, Any]
    effective_config_sha256: str
    benchmark_runner_sha256: str
    implementation_inputs: Mapping[str, Any]
    metric_inputs: Mapping[str, Any]
    num_samples: int
    source_revision: str | None = None
    device: str = "cuda:0"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1_000)
    tier = parser.add_mutually_exclusive_group()
    tier.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Allow a deliberately small run of at most 100 samples. Without this "
            "flag, the final protocol requires exactly 1000 samples per seed."
        ),
    )
    tier.add_argument(
        "--selection-pilot",
        action="store_true",
        help=(
            "Run the registered candidate-selection tier: exactly 256 samples "
            "for each of the ordered seeds 1000 and 1001. This mode requires a "
            "fresh --attempt-id plus --candidate-id and remains "
            "pilot/final-ineligible evidence."
        ),
    )
    parser.add_argument(
        "--attempt-id",
        help=(
            "Normalized identifier for one auditable pilot attempt. It is required "
            "with --pilot or --selection-pilot, forbidden for final runs, and "
            "appended to the output and log roots."
        ),
    )
    parser.add_argument(
        "--candidate-id",
        help=(
            "Normalized candidate identity required with either pilot mode, "
            "forbidden for final runs, and bound into producer-authored failure "
            "receipts."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-lock",
        type=Path,
        help=(
            "Required only for the final tier and required to name exactly "
            f"{CANDIDATE_LOCK_RELATIVE_PATH}. Pilot modes forbid this option."
        ),
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        required=True,
        help=(
            "Maximum concurrent benchmark GPUs, selected dynamically from the "
            "full NVIDIA inventory; must be 1, 2, or 3."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-utilization-percent", type=int, default=10)
    parser.add_argument("--min-free-memory-mib", type=int, default=30_000)
    parser.add_argument("--log-root", type=Path, default=Path("output/logs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_gpu_count(gpu_count: int) -> int:
    """Validate the user-selected concurrency without accepting physical IDs."""

    if (
        type(gpu_count) is not int
        or not 1 <= gpu_count <= MAX_CONCURRENT_GENERATION_JOBS
    ):
        raise ValueError(
            f"gpu-count must be between 1 and {MAX_CONCURRENT_GENERATION_JOBS}"
        )
    return gpu_count


def _validate_sample_tier(
    num_samples: int,
    *,
    pilot: bool,
    selection_pilot: bool = False,
) -> str:
    """Prevent a pilot from being mistaken for the final benchmark."""

    if type(num_samples) is not int:
        raise ValueError("num-samples must be an integer")
    if num_samples <= 0:
        raise ValueError("num-samples must be positive")
    if pilot and selection_pilot:
        raise ValueError("pilot and selection-pilot modes are mutually exclusive")
    if selection_pilot:
        if num_samples != REGISTERED_SELECTION_PILOT_SAMPLES_PER_SEED:
            raise ValueError(
                "selection-pilot runs require exactly "
                f"{REGISTERED_SELECTION_PILOT_SAMPLES_PER_SEED} samples per seed"
            )
        # benchmark.py schema 7 deliberately labels every non-1000 run as
        # pilot and final_protocol_eligible=false.
        return "pilot"
    if pilot:
        if num_samples > 100:
            raise ValueError("pilot runs are capped at 100 samples per seed")
        return "pilot"
    if num_samples != 1_000:
        raise ValueError(
            "final benchmark runs require exactly 1000 samples per seed; "
            "pass --pilot for a run of at most 100"
        )
    return "final"


def _validate_attempt_id(attempt_id: Any) -> str:
    """Require the candidate-ledger identifier in its canonical path-safe form."""

    if (
        not isinstance(attempt_id, str)
        or ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None
    ):
        raise ValueError(
            "attempt-id must already be normalized as 1-96 lowercase ASCII "
            "letters, digits, dots, underscores, or hyphens, beginning with an "
            "ASCII letter or digit"
        )
    return attempt_id


def _validate_attempt_scope(
    *,
    pilot: bool = False,
    selection_pilot: bool,
    attempt_id: Any,
    seeds: list[int],
) -> str | None:
    """Bind every auditable pilot to safe seeds and a unique attempt identity."""

    if not pilot and not selection_pilot:
        if attempt_id is not None:
            raise ValueError("attempt-id is allowed only with a pilot mode")
        return None
    normalized = _validate_attempt_id(attempt_id)
    if selection_pilot and seeds != list(REGISTERED_SELECTION_PILOT_SEEDS):
        raise ValueError(
            "selection-pilot runs require exactly the ordered seeds "
            f"{list(REGISTERED_SELECTION_PILOT_SEEDS)}"
        )
    if pilot and any(type(seed) is not int or seed < 1000 for seed in seeds):
        raise ValueError("engineering pilot seeds must be integers >=1000")
    return normalized


def _validate_candidate_scope(
    *,
    pilot: bool = False,
    selection_pilot: bool,
    candidate_id: Any,
) -> str | None:
    """Require a gate-compatible candidate identity for either pilot mode."""

    if not pilot and not selection_pilot:
        if candidate_id is not None:
            raise ValueError("candidate-id is allowed only with a pilot mode")
        return None
    if (
        not isinstance(candidate_id, str)
        or CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None
    ):
        raise ValueError(
            "candidate-id must already be normalized as 3-96 lowercase ASCII "
            "letters, digits, dots, underscores, or hyphens, beginning with an "
            "ASCII letter or digit"
        )
    return candidate_id


def _validate_candidate_lock_scope(
    candidate_lock: Path | None,
    *,
    pilot: bool,
    selection_pilot: bool,
) -> Path | None:
    """Require the one fixed lock only for non-pilot final evaluation."""

    if pilot or selection_pilot:
        if candidate_lock is not None:
            raise ValueError("candidate-lock is forbidden with either pilot mode")
        return None
    if candidate_lock is None:
        raise ValueError(
            "final benchmark runs require --candidate-lock "
            f"{CANDIDATE_LOCK_RELATIVE_PATH}"
        )
    raw = os.fspath(candidate_lock)
    fixed_relative = CANDIDATE_LOCK_RELATIVE_PATH
    fixed_absolute = os.fspath(REPOSITORY_ROOT / fixed_relative)
    if raw not in {fixed_relative, fixed_absolute}:
        raise ValueError(
            "candidate-lock must name the fixed canonical path exactly: "
            f"{fixed_relative}"
        )
    return REPOSITORY_ROOT / fixed_relative


def _resolve_attempt_keyed_roots(
    output_root: Path,
    log_root: Path,
    *,
    attempt_id: str | None,
) -> tuple[Path, Path]:
    """Resolve roots, adding one unambiguous path component per selection attempt."""

    resolved_output_base = _resolve_in_repo(output_root)
    resolved_log_base = _resolve_in_repo(log_root)
    if attempt_id is None:
        return resolved_output_base, resolved_log_base
    normalized = _validate_attempt_id(attempt_id)
    keyed_output = _resolve_in_repo(resolved_output_base / normalized)
    keyed_logs = _resolve_in_repo(resolved_log_base / normalized)
    if keyed_output == keyed_logs:
        raise ValueError("pilot output and log attempt roots must differ")
    if keyed_output.name != normalized or keyed_logs.name != normalized:
        raise ValueError("pilot paths are not keyed by the exact attempt-id")
    return keyed_output, keyed_logs


def _path_exists_without_following_final_symlink(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _require_fresh_pilot_attempt_paths(output_root: Path, log_root: Path) -> None:
    """Reject reuse: a retry must receive a new candidate-ledger attempt ID."""

    existing = [
        path
        for path in (output_root, log_root)
        if _path_exists_without_following_final_symlink(path)
    ]
    if existing:
        raise FileExistsError(
            "pilot attempt paths must be fresh; retries require a new "
            f"attempt-id. Existing path(s): {', '.join(map(str, existing))}"
        )


def _reserve_pilot_attempt_paths(output_root: Path, log_root: Path) -> None:
    """Atomically reserve the output identity before any GPU is queried."""

    repository = Path(os.path.abspath(os.fspath(REPOSITORY_ROOT)))
    if Path(os.path.abspath(os.fspath(output_root.parent))) != repository:
        _ensure_repository_directory(output_root.parent, label="pilot output parent")
    output_relative = _repository_relative(output_root, label="pilot output attempt")
    try:
        output_owner = artifact_io.create_directory_exclusive(
            REPOSITORY_ROOT, output_relative
        )
    except FileExistsError as error:
        raise FileExistsError(
            "pilot output attempt was concurrently claimed; use a new "
            f"attempt-id: {output_root}"
        ) from error
    try:
        if Path(os.path.abspath(os.fspath(log_root.parent))) != repository:
            _ensure_repository_directory(log_root.parent, label="pilot log parent")
        log_relative = _repository_relative(log_root, label="pilot log attempt")
        try:
            artifact_io.create_directory_exclusive(REPOSITORY_ROOT, log_relative)
        except FileExistsError as error:
            raise FileExistsError(
                "pilot log attempt path already exists; use a new attempt-id: "
                f"{log_root}"
            ) from error
    except BaseException as error:
        try:
            artifact_io.remove_empty_directory_exact(REPOSITORY_ROOT, output_owner)
        except BaseException as rollback_error:
            raise artifact_io.RollbackError(error, [rollback_error]) from error
        raise


def _selection_policy(
    *,
    requested_gpu_count: int,
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> dict[str, Any]:
    """Return the policy schema embedded in controller and child provenance."""

    requested_gpu_count = _validate_gpu_count(requested_gpu_count)
    return {
        "selection_method": "dynamic_idle_discovery",
        "inventory_scope": "all_nvidia_gpus",
        "requested_gpu_count": requested_gpu_count,
        "max_utilization_percent": max_utilization_percent,
        "utilization_comparison": "strictly_less_than",
        "min_free_memory_mib": min_free_memory_mib,
        "active_compute_processes_allowed": True,
    }


def _resolve_in_repo(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPOSITORY_ROOT / path
    resolved = resolved.resolve()
    if resolved != REPOSITORY_ROOT and REPOSITORY_ROOT not in resolved.parents:
        raise ValueError(f"path escapes repository root: {resolved}")
    return resolved


def _resolve_checkpoint(path: Path) -> Path:
    """Allow shared checkpoints in the containing project, never outside it."""

    resolved = path if path.is_absolute() else REPOSITORY_ROOT / path
    resolved = resolved.resolve()
    project_root = _project_root()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"checkpoint escapes project root: {resolved}")
    return resolved


def _project_root() -> Path:
    return (
        REPOSITORY_ROOT.parent.parent
        if REPOSITORY_ROOT.parent.name == "run_sources"
        else REPOSITORY_ROOT
    )


def _project_venv_python() -> Path:
    return _project_root() / ".venv/bin/python"


def _require_project_virtual_environment() -> Path:
    """Require the shared project ``.venv`` for controller and child execution."""

    expected_python = _project_venv_python()
    if not expected_python.is_file():
        raise RuntimeError(
            f"project virtual-environment Python is missing: {expected_python}"
        )
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(
            "benchmark launcher must run with the project virtual environment: "
            f"{expected_python}"
        )
    return expected_python


def _run_nvidia_smi(
    gpu_identifier: int | str,
    query: str,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_identifier),
            query,
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _physical_gpu_indices() -> list[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    indices = [
        int(line.strip()) for line in completed.stdout.splitlines() if line.strip()
    ]
    if (
        not indices
        or any(index < 0 for index in indices)
        or len(indices) != len(set(indices))
    ):
        raise RuntimeError(
            "nvidia-smi returned no GPUs or invalid/duplicate physical indices"
        )
    return sorted(indices)


def _probe_gpu(gpu_identifier: int | str) -> GPUState:
    """Probe one physical GPU by enumerated index or exact NVIDIA UUID."""

    if isinstance(gpu_identifier, bool) or not isinstance(gpu_identifier, (int, str)):
        raise TypeError("GPU identifier must be a physical index or NVIDIA UUID")
    expected_index = gpu_identifier if isinstance(gpu_identifier, int) else None
    expected_uuid = gpu_identifier if isinstance(gpu_identifier, str) else None
    if expected_index is not None and expected_index < 0:
        raise ValueError("physical GPU index must be non-negative")
    if expected_uuid is not None and not expected_uuid.startswith("GPU-"):
        raise ValueError(f"invalid NVIDIA GPU UUID: {expected_uuid!r}")
    context = f"GPU {gpu_identifier}"
    status = _run_nvidia_smi(
        gpu_identifier,
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,compute_mode",
    )
    if status.returncode or status.stderr.strip():
        detail = status.stderr.strip() or status.stdout.strip()
        raise RuntimeError(f"{context} status query failed: {detail}")
    rows = [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(status.stdout), skipinitialspace=True)
        if any(field.strip() for field in row)
    ]
    if len(rows) != 1 or len(rows[0]) != 7:
        raise RuntimeError(f"{context} status query returned an unexpected row")
    (
        index_text,
        uuid,
        name,
        memory_used_text,
        memory_total_text,
        utilization_text,
        compute_mode,
    ) = rows[0]
    try:
        physical_index = int(index_text)
    except ValueError as error:
        raise RuntimeError(
            f"{context} status query returned a non-integer physical index"
        ) from error
    if (
        physical_index < 0
        or (expected_index is not None and physical_index != expected_index)
        or (expected_uuid is not None and uuid != expected_uuid)
        or not uuid.startswith("GPU-")
        or not name
        or not compute_mode
    ):
        raise RuntimeError(f"{context} status query returned inconsistent identity")
    try:
        memory_used_mib = int(memory_used_text)
        memory_total_mib = int(memory_total_text)
        utilization_percent = int(utilization_text)
    except ValueError as error:
        raise RuntimeError(
            f"{context} status query returned non-integer telemetry"
        ) from error
    if (
        memory_used_mib < 0
        or memory_total_mib <= 0
        or memory_used_mib > memory_total_mib
        or not 0 <= utilization_percent <= 100
    ):
        raise RuntimeError(f"{context} status query returned invalid telemetry")

    processes = _run_nvidia_smi(
        gpu_identifier,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    )
    if processes.returncode or processes.stderr.strip():
        detail = processes.stderr.strip() or processes.stdout.strip()
        raise RuntimeError(f"{context} process query failed: {detail}")
    raw_process_rows = [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(processes.stdout), skipinitialspace=True)
        if any(field.strip() for field in row)
    ]
    no_process_markers = [
        fields
        for fields in raw_process_rows
        if fields[0].lower().startswith("no running")
    ]
    if no_process_markers and (
        len(raw_process_rows) != 1 or len(no_process_markers[0]) != 1
    ):
        raise RuntimeError(
            f"{context} process query returned ambiguous no-process telemetry"
        )

    process_rows = []
    seen_process_pids: set[int] = set()
    for fields in raw_process_rows:
        if fields[0].lower().startswith("no running"):
            continue
        if len(fields) != 4 or not fields[1].isdigit():
            raise RuntimeError(f"{context} process query returned an unexpected row")
        process_uuid = fields[0]
        process_pid = int(fields[1])
        process_name = fields[2]
        try:
            process_memory_mib = int(fields[3])
        except ValueError as error:
            raise RuntimeError(
                f"{context} process query returned non-integer memory telemetry"
            ) from error
        if (
            process_uuid != uuid
            or process_pid <= 0
            or not process_name
            or process_memory_mib < 0
            or process_pid in seen_process_pids
        ):
            raise RuntimeError(f"{context} process query returned invalid telemetry")
        seen_process_pids.add(process_pid)
        process_rows.append(
            {
                "pid": process_pid,
                "process_name": process_name,
                "used_memory_mib": process_memory_mib,
            }
        )
    return GPUState(
        index=physical_index,
        uuid=uuid,
        name=name,
        memory_used_mib=memory_used_mib,
        memory_total_mib=memory_total_mib,
        utilization_percent=utilization_percent,
        compute_mode=compute_mode,
        compute_processes=tuple(process_rows),
    )


def _snapshot() -> list[GPUState]:
    """Enumerate and inspect the complete NVIDIA GPU inventory."""

    available_indices = _physical_gpu_indices()
    states = []
    errors = {}
    for index in available_indices:
        try:
            states.append(_probe_gpu(index))
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            errors[index] = str(error)
    if errors:
        raise RuntimeError(
            "complete NVIDIA GPU inventory is unverifiable; refusing dynamic "
            f"selection: {errors}"
        )
    if len({state.uuid for state in states}) != len(states):
        raise RuntimeError(
            "complete NVIDIA GPU inventory contains duplicate UUIDs; refusing "
            "dynamic selection"
        )
    return states


def _require_tmux_for_execution() -> None:
    """Refuse a disconnect-fragile controller for every non-dry GPU run."""

    if not os.environ.get("TMUX"):
        raise RuntimeError(
            "benchmark execution must run inside tmux; dry-run and completed-artifact "
            "validation remain available outside tmux"
        )


def _eligible(
    state: GPUState,
    *,
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> bool:
    return not state.rejection_reasons(
        max_utilization_percent=max_utilization_percent,
        min_free_memory_mib=min_free_memory_mib,
    )


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_relative(path: Path, *, label: str) -> str:
    """Return one canonical root-relative path without using it for mutation."""

    root = Path(os.path.abspath(os.fspath(REPOSITORY_ROOT)))
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute == root or root not in absolute.parents:
        raise ValueError(f"{label} must be strictly inside the repository")
    relative = absolute.relative_to(root).as_posix()
    if str(root / relative) != str(absolute):
        raise ValueError(f"{label} is not canonical")
    return relative


def _ensure_repository_directory(path: Path, *, label: str) -> None:
    """Create missing directory components through ``artifact_io`` only."""

    relative = _repository_relative(path, label=label)
    parts = Path(relative).parts
    for length in range(1, len(parts) + 1):
        component = "/".join(parts[:length])
        try:
            artifact_io.create_directory_exclusive(REPOSITORY_ROOT, component)
        except FileExistsError:
            current = REPOSITORY_ROOT.joinpath(*parts[:length])
            try:
                state = current.stat(follow_symlinks=False)
            except OSError as error:
                raise RuntimeError(
                    f"cannot verify existing {label} component: {current}"
                ) from error
            if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
                raise RuntimeError(
                    f"existing {label} component is not a direct directory: "
                    f"{current}"
                )


def _artifact_io_source_claim() -> artifact_io.FileClaim:
    claim, _payload = artifact_io.snapshot_file(
        REPOSITORY_ROOT, ARTIFACT_IO_RELATIVE_PATH, capture_bytes=False
    )
    return claim


def _generation_lease_payload(
    *, source_revision: str, owner_token: str, artifact_source: artifact_io.FileClaim
) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("generation lease source revision must be 40 lowercase hex")
    if not re.fullmatch(r"[0-9a-f]{64}", owner_token):
        raise ValueError("generation lease owner token must be 64 lowercase hex")
    record = {
        "schema_version": GENERATION_LEASE_SCHEMA_VERSION,
        "status": "held",
        "purpose": "enforce_one_repository_generation_controller_at_a_time",
        "source_revision": source_revision,
        "owner_token": owner_token,
        "launcher_pid_at_acquisition": os.getpid(),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_io": {
            "relative_path": artifact_source.relative_path,
            "device": artifact_source.device,
            "inode": artifact_source.inode,
            "sha256": artifact_source.sha256,
        },
        "owner_process_exit_does_not_make_lock_stale": True,
        "stale_lock_policy": "fail_closed_and_require_manual_review",
        "release_policy": (
            "exact_owner_only_after_all_handed_off_children_terminal_and_required_"
            "ignored_decisions_or_failure_receipts_are_durable"
        ),
    }
    return (
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _acquire_generation_lease(*, source_revision: str) -> GenerationLease:
    """Acquire the global lease; an existing or stale lease always blocks."""

    artifact_source = _artifact_io_source_claim()
    owner_token = secrets.token_hex(32)
    payload = _generation_lease_payload(
        source_revision=source_revision,
        owner_token=owner_token,
        artifact_source=artifact_source,
    )
    try:
        claim = artifact_io.acquire_lock_exclusive(
            REPOSITORY_ROOT, GENERATION_LEASE_RELATIVE_PATH, payload
        )
    except FileExistsError as error:
        raise RuntimeError(
            "another or stale generation lease exists; fail closed and review it "
            f"manually: {REPOSITORY_ROOT / GENERATION_LEASE_RELATIVE_PATH}"
        ) from error
    return GenerationLease(
        claim=claim,
        owner_token=owner_token,
        payload=payload,
        artifact_io_source=artifact_source,
    )


def _revalidate_generation_lease(lease: GenerationLease) -> None:
    """Revalidate lease bytes, token, hash, dev/inode, and implementation source."""

    if not isinstance(lease, GenerationLease):
        raise TypeError("generation lease capability is invalid")
    try:
        current, payload = artifact_io.snapshot_file(
            REPOSITORY_ROOT,
            GENERATION_LEASE_RELATIVE_PATH,
            capture_bytes=True,
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise RuntimeError("generation lease cannot be revalidated") from error
    if current != lease.claim or payload != lease.payload:
        raise RuntimeError("generation lease identity or bytes changed")
    if hashlib.sha256(lease.payload).hexdigest() != lease.claim.sha256:
        raise RuntimeError("generation lease payload hash changed")
    try:
        record = json.loads(lease.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:  # pragma: no cover
        raise RuntimeError("generation lease payload became unreadable") from error
    if (
        not isinstance(record, Mapping)
        or record.get("owner_token") != lease.owner_token
        or record.get("schema_version") != GENERATION_LEASE_SCHEMA_VERSION
        or record.get("status") != "held"
    ):
        raise RuntimeError("generation lease owner token or schema changed")
    try:
        source_now, _payload = artifact_io.snapshot_file(
            REPOSITORY_ROOT, ARTIFACT_IO_RELATIVE_PATH, capture_bytes=False
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise RuntimeError("artifact_io source cannot be revalidated") from error
    if source_now != lease.artifact_io_source:
        raise RuntimeError("artifact_io source identity or bytes changed")


def _release_generation_lease_exact(lease: GenerationLease) -> None:
    _revalidate_generation_lease(lease)
    artifact_io.release_lock_exact(REPOSITORY_ROOT, lease.claim)


def _launch_authority(
    *,
    lease: GenerationLease,
    output_directory: artifact_io.OwnedDirectory,
    command: list[str],
) -> dict[str, Any]:
    """Build the child-verifiable authority record for one exact launch."""

    _revalidate_generation_lease(lease)
    if len(command) != 24:
        raise ValueError("launch authority requires the exact 24-string child argv")
    return {
        "schema_version": LAUNCH_AUTHORITY_SCHEMA_VERSION,
        "generation_lease": {
            "path": str(REPOSITORY_ROOT / GENERATION_LEASE_RELATIVE_PATH),
            "relative_path": GENERATION_LEASE_RELATIVE_PATH,
            "sha256": lease.claim.sha256,
            "device": lease.claim.device,
            "inode": lease.claim.inode,
            "owner_token": lease.owner_token,
        },
        "artifact_io_source": {
            "path": str(REPOSITORY_ROOT / ARTIFACT_IO_RELATIVE_PATH),
            "sha256": lease.artifact_io_source.sha256,
            "device": lease.artifact_io_source.device,
            "inode": lease.artifact_io_source.inode,
        },
        "output_directory": {
            "path": str(REPOSITORY_ROOT / output_directory.relative_path),
            "relative_path": output_directory.relative_path,
            "device": output_directory.device,
            "inode": output_directory.inode,
        },
        "command": command,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":"), ensure_ascii=True).encode(
                "ascii"
            )
        ).hexdigest(),
    }


def _stable_file_reference(
    path: Path,
    *,
    label: str,
    required: bool,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, Any] | None:
    """Hash one regular file through its descriptor and reject pathname races."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(f"{label} is missing: {absolute}") from None
        return None
    except OSError as error:
        raise RuntimeError(
            f"cannot open {label} as a regular file: {absolute}: {error}"
        ) from error
    try:
        descriptor_state = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_state.st_mode):
            raise RuntimeError(f"{label} must be a regular file: {absolute}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, chunk_size)
            if not chunk:
                break
            digest.update(chunk)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        pathname_state = os.stat(absolute, follow_symlinks=False)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{label} disappeared while it was hashed: {absolute}"
        ) from error
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(
        getattr(descriptor_state, field) != getattr(descriptor_after, field)
        or getattr(descriptor_after, field) != getattr(pathname_state, field)
        for field in identity_fields
    ):
        raise RuntimeError(f"{label} changed while it was hashed: {absolute}")
    return {
        "path": str(absolute),
        "sha256": digest.hexdigest(),
        "size_bytes": descriptor_after.st_size,
    }


def _validate_failure_timestamps(started_at_utc: str, failed_at_utc: str) -> None:
    parsed: list[datetime] = []
    for label, value in (
        ("started_at_utc", started_at_utc),
        ("failed_at_utc", failed_at_utc),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a nonempty ISO-8601 timestamp")
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
        if timestamp.tzinfo is None:
            raise ValueError(f"{label} must include a timezone")
        parsed.append(timestamp)
    if parsed[1] < parsed[0]:
        raise ValueError("failure timestamp cannot predate process start")


def _build_pilot_failure_receipt(
    *,
    output_root: Path,
    job: RunningJob,
    expected: ExpectedRunIdentity,
    attempt_id: str,
    candidate_id: str,
    pilot_mode: str,
    source_revision: Mapping[str, str],
    stage: str,
    reason: str,
    process_exit_status: int | None,
    failed_at_utc: str,
) -> dict[str, Any]:
    """Build one fully bound producer receipt for a failed auditable pilot seed."""

    attempt_id = _validate_attempt_id(attempt_id)
    if pilot_mode not in PILOT_MODES:
        raise ValueError("pilot failure receipt has an invalid pilot mode")
    candidate_id_value = _validate_candidate_scope(
        pilot=pilot_mode == "engineering",
        selection_pilot=pilot_mode == "registered_selection",
        candidate_id=candidate_id,
    )
    if output_root.name != attempt_id:
        raise ValueError("failure receipt output root is not keyed by attempt-id")
    if pilot_mode == "engineering":
        if job.seed < 1000 or not 1 <= expected.num_samples <= 100:
            raise ValueError(
                "engineering failure receipt requires seed >=1000 and 1..100 samples"
            )
    elif (
        job.seed not in REGISTERED_SELECTION_PILOT_SEEDS
        or expected.num_samples != REGISTERED_SELECTION_PILOT_SAMPLES_PER_SEED
    ):
        raise ValueError(
            "registered-selection failure receipt has an invalid seed or sample count"
        )
    if stage == "benchmark_child_process":
        if type(process_exit_status) is not int or process_exit_status == 0:
            raise ValueError("child-process failure receipt requires a nonzero status")
    elif stage == "completion_validation":
        if process_exit_status is not None:
            raise ValueError(
                "completion-validation failure receipt requires a null process status"
            )
    else:
        raise ValueError("pilot failure stage is invalid")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("pilot failure reason must be nonempty")
    if (
        not isinstance(job.command, tuple)
        or not job.command
        or any(not isinstance(value, str) or not value for value in job.command)
    ):
        raise ValueError("pilot failure command must be a nonempty string tuple")
    if job.output_directory_owner is None:
        raise ValueError("pilot failure lacks its controller-owned output directory")
    expected_command = tuple(
        _command(
            checkpoint=expected.checkpoint_path,
            expected_checkpoint_sha256=expected.checkpoint_sha256,
            expected_source_revision=str(expected.source_revision),
            config=expected.config_path,
            expected_config_sha256=expected.source_config_sha256,
            num_samples=expected.num_samples,
            seed=job.seed,
            output_dir=Path(os.path.abspath(output_root / f"seed_{job.seed}")),
            expected_output_directory_device=(job.output_directory_owner.device),
            expected_output_directory_inode=(job.output_directory_owner.inode),
        )
    )
    if expected.source_revision is None or job.command != expected_command:
        raise ValueError(
            "pilot failure command differs from the launched child command"
        )
    if dict(source_revision) != {
        "head": expected.source_revision,
        "upstream": expected.source_revision,
    }:
        raise ValueError("pilot failure source revision differs from expected source")
    _validate_failure_timestamps(job.started_at_utc, failed_at_utc)

    run_dir = output_root / f"seed_{job.seed}"
    launcher_source_path = Path(__file__).resolve(strict=True)
    launcher_source = _stable_file_reference(
        launcher_source_path,
        label="benchmark launcher source",
        required=True,
    )
    log_reference = _stable_file_reference(
        job.log_path,
        label="pilot benchmark log",
        required=True,
    )
    assert launcher_source is not None
    assert log_reference is not None
    partial_artifacts = {
        "summary_json": _stable_file_reference(
            run_dir / benchmark_runner.SUMMARY_FILENAME,
            label="partial benchmark summary",
            required=False,
        ),
        "raw_samples_csv": _stable_file_reference(
            run_dir / benchmark_runner.RAW_SAMPLES_FILENAME,
            label="partial benchmark raw samples",
            required=False,
        ),
    }
    return {
        "schema_version": PILOT_FAILURE_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": PILOT_FAILURE_RECEIPT_KIND,
        "status": "failed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id_value,
        "pilot_seed": job.seed,
        "pilot_mode": pilot_mode,
        "requested_samples": expected.num_samples,
        "stage": stage,
        "reason": reason,
        "started_at_utc": job.started_at_utc,
        "failed_at_utc": failed_at_utc,
        "process_exit_status": process_exit_status,
        "checkpoint": {
            "path": str(expected.checkpoint_path),
            "sha256": expected.checkpoint_sha256,
            "size_bytes": expected.checkpoint_size_bytes,
            "global_step": expected.checkpoint_global_step,
        },
        "config": {
            "path": str(expected.config_path),
            "sha256": expected.source_config_sha256,
            "sampling": dict(expected.sampling_config),
            "sampling_sha256": expected.sampling_config_sha256,
        },
        "command": list(job.command),
        "source_revision": dict(source_revision),
        "launcher_source": launcher_source,
        "log": log_reference,
        "partial_artifacts": partial_artifacts,
    }


def _atomic_write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON through the common no-clobber artifact primitive."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    repository_root = Path(os.path.abspath(os.fspath(REPOSITORY_ROOT)))
    if (
        absolute == repository_root
        or repository_root not in absolute.parents
        or absolute.suffix != ".json"
    ):
        raise ValueError("failure receipt must be an in-repository JSON path")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        artifact_io.publish_bytes_exclusive(
            REPOSITORY_ROOT,
            _repository_relative(absolute, label="failure receipt"),
            encoded,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace pilot failure receipt: {absolute}"
        ) from error


def _write_pilot_failure_receipt(
    *,
    output_root: Path,
    job: RunningJob,
    expected: ExpectedRunIdentity,
    attempt_id: str,
    candidate_id: str,
    pilot_mode: str,
    source_revision: Mapping[str, str],
    stage: str,
    reason: str,
    process_exit_status: int | None,
    failed_at_utc: str,
) -> Path:
    receipt = _build_pilot_failure_receipt(
        output_root=output_root,
        job=job,
        expected=expected,
        attempt_id=attempt_id,
        candidate_id=candidate_id,
        pilot_mode=pilot_mode,
        source_revision=source_revision,
        stage=stage,
        reason=reason,
        process_exit_status=process_exit_status,
        failed_at_utc=failed_at_utc,
    )
    receipt_path = output_root / f"seed_{job.seed}" / PILOT_FAILURE_RECEIPT_FILENAME
    _atomic_write_json_exclusive(receipt_path, receipt)
    return receipt_path


def _git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _committed_regular_file_snapshot(
    relative_path: str,
    *,
    source_revision: str,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[artifact_io.FileClaim, bytes]:
    """Require stable live bytes to equal one non-executable regular Git blob."""

    path = Path(relative_path)
    if (
        not relative_path
        or path.is_absolute()
        or path.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in relative_path.split("/"))
    ):
        raise FinalCandidateLockError(f"{label} path is not canonical")
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise FinalCandidateLockError(
            "benchmark source revision is not 40 lowercase hex"
        )
    if (
        expected_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise FinalCandidateLockError(f"{label} expected SHA-256 is invalid")

    try:
        listing = _git_bytes(
            "ls-tree", "-z", "--full-tree", source_revision, "--", relative_path
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalCandidateLockError(
            f"cannot inspect committed {label} at benchmark HEAD"
        ) from error
    records = [record for record in listing.split(b"\0") if record]
    if len(records) != 1 or b"\t" not in records[0]:
        raise FinalCandidateLockError(
            f"benchmark HEAD does not contain exactly one {label} blob"
        )
    metadata, recorded_path = records[0].split(b"\t", 1)
    fields = metadata.split(b" ")
    if (
        fields[:2] != [b"100644", b"blob"]
        or len(fields) != 3
        or recorded_path != relative_path.encode("utf-8")
        or re.fullmatch(rb"[0-9a-f]{40,64}", fields[2]) is None
    ):
        raise FinalCandidateLockError(
            f"committed {label} must be the exact non-executable regular file"
        )
    try:
        committed_payload = _git_bytes("cat-file", "blob", fields[2].decode("ascii"))
        claim, live_payload = artifact_io.snapshot_file(
            REPOSITORY_ROOT, relative_path, capture_bytes=True
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        artifact_io.ArtifactIOError,
    ) as error:
        raise FinalCandidateLockError(f"cannot read stable {label} bytes") from error
    if live_payload is None:  # pragma: no cover - artifact_io capture contract
        raise AssertionError("stable artifact snapshot did not retain bytes")
    if live_payload != committed_payload:
        raise FinalCandidateLockError(
            f"live {label} bytes differ from the exact benchmark HEAD blob"
        )
    if (
        claim.size_bytes != len(live_payload)
        or claim.sha256 != hashlib.sha256(live_payload).hexdigest()
    ):
        raise FinalCandidateLockError(f"stable {label} identity is inconsistent")
    if expected_sha256 is not None and claim.sha256 != expected_sha256:
        raise FinalCandidateLockError(f"{label} digest differs from locked authority")
    return claim, live_payload


def _require_exact_final_lock_commit(
    *, source_revision: str, lock_payload: bytes
) -> str:
    """Require benchmark HEAD itself to be the one-path lock publication."""

    try:
        parent_record = (
            _git_bytes("rev-list", "--parents", "-n", "1", source_revision)
            .decode("ascii", errors="strict")
            .strip()
            .split()
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        raise FinalCandidateLockError(
            "cannot inspect final candidate-lock publication commit"
        ) from error
    if (
        len(parent_record) != 2
        or parent_record[0] != source_revision
        or re.fullmatch(r"[0-9a-f]{40}", parent_record[1]) is None
    ):
        raise FinalCandidateLockError(
            "final benchmark HEAD must be the single-parent lock-only publication"
        )
    parent_revision = parent_record[1]
    try:
        changed = _git_bytes(
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--no-renames",
            "-r",
            "-z",
            parent_revision,
            source_revision,
            "--",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalCandidateLockError(
            "cannot inspect final candidate-lock publication diff"
        ) from error
    expected_change = b"A\0" + CANDIDATE_LOCK_RELATIVE_PATH.encode("utf-8") + b"\0"
    if changed != expected_change:
        raise FinalCandidateLockError(
            "final benchmark HEAD must add only the fixed candidate lock"
        )
    absence = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "cat-file",
            "-e",
            f"{parent_revision}:{CANDIDATE_LOCK_RELATIVE_PATH}",
        ],
        capture_output=True,
        check=False,
    )
    if absence.returncode == 0:
        raise FinalCandidateLockError(
            "candidate lock existed before the lock-only publication"
        )
    if absence.returncode not in {1, 128}:
        raise FinalCandidateLockError(
            "cannot prove candidate lock absent before publication"
        )
    try:
        committed = _git_bytes(
            "show", f"{source_revision}:{CANDIDATE_LOCK_RELATIVE_PATH}"
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalCandidateLockError(
            "cannot read final candidate lock from publication commit"
        ) from error
    if committed != lock_payload:
        raise FinalCandidateLockError(
            "candidate-lock publication bytes differ from retained lock"
        )
    return parent_revision


def _implementation_source_digest(
    expected: ExpectedRunIdentity,
    *,
    key: str,
    relative_path: str,
) -> str:
    record = expected.implementation_inputs.get(key)
    if not isinstance(record, Mapping):
        raise FinalCandidateLockError(
            f"implementation inputs lack locked {key} provenance"
        )
    digest = record.get("sha256")
    size_bytes = record.get("size_bytes")
    recorded_path = record.get("path")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or type(size_bytes) is not int
        or size_bytes <= 0
        or not isinstance(recorded_path, str)
    ):
        raise FinalCandidateLockError(f"{key} provenance is malformed")
    expected_path = REPOSITORY_ROOT / relative_path
    if Path(recorded_path) != expected_path:
        raise FinalCandidateLockError(f"{key} provenance path is not the fixed source")
    claim, _payload = _committed_regular_file_snapshot(
        relative_path,
        source_revision=str(expected.source_revision),
        label=key.replace("_", " "),
        expected_sha256=digest,
    )
    if claim.size_bytes != size_bytes:
        raise FinalCandidateLockError(f"{key} size differs from generation provenance")
    return digest


def _bind_final_candidate_authority(
    *,
    normalized_lock: Mapping[str, Any],
    expected: ExpectedRunIdentity,
    checkpoint: Path,
    config: Path,
    seeds: list[int],
    num_samples: int,
    output_root: Path,
    source_revision: str,
) -> tuple[str, str, Path]:
    """Bind CLI inputs and live generation inputs to one normalized lock."""

    if seeds != [0, 1, 2]:
        raise FinalCandidateLockError(
            "final benchmark seeds must be exactly the ordered list 0 1 2"
        )
    if num_samples != FINAL_BENCHMARK_SAMPLES_PER_SEED:
        raise FinalCandidateLockError("final benchmark requires exactly 1000 samples")
    if expected.source_revision != source_revision:
        raise FinalCandidateLockError(
            "final run identity uses a different source revision"
        )
    candidate_id = normalized_lock.get("candidate_id")
    if (
        not isinstance(candidate_id, str)
        or FINAL_CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None
    ):
        raise FinalCandidateLockError("candidate lock has an unregistered candidate ID")

    locked_checkpoint = normalized_lock.get("checkpoint")
    if not isinstance(locked_checkpoint, Mapping):
        raise FinalCandidateLockError("candidate lock checkpoint is malformed")
    locked_checkpoint_path = _resolve_in_repo(
        REPOSITORY_ROOT / Path(str(locked_checkpoint.get("relative_path")))
    )
    if checkpoint != locked_checkpoint_path or expected.checkpoint_path != checkpoint:
        raise FinalCandidateLockError("CLI checkpoint path differs from candidate lock")
    locked_checkpoint_identity = {
        "sha256": locked_checkpoint.get("sha256"),
        "size_bytes": locked_checkpoint.get("size_bytes"),
        "global_step": locked_checkpoint.get("global_step"),
    }
    observed_checkpoint_identity = {
        "sha256": expected.checkpoint_sha256,
        "size_bytes": expected.checkpoint_size_bytes,
        "global_step": expected.checkpoint_global_step,
    }
    if locked_checkpoint_identity != observed_checkpoint_identity:
        raise FinalCandidateLockError(
            "checkpoint bytes or training step differ from lock"
        )
    if expected.checkpoint_diffusion_type != "udlm":
        raise FinalCandidateLockError("final locked checkpoint must be UDLM")

    locked_config_path = _resolve_in_repo(
        REPOSITORY_ROOT
        / Path(str(normalized_lock.get("evaluation_config_relative_path")))
    )
    if config != locked_config_path or expected.config_path != config:
        raise FinalCandidateLockError("CLI evaluation config path differs from lock")
    if expected.source_config_sha256 != normalized_lock.get("evaluation_config_sha256"):
        raise FinalCandidateLockError("evaluation config bytes differ from lock")
    if dict(expected.sampling_config) != normalized_lock.get("sampling_config"):
        raise FinalCandidateLockError(
            "normalized sampling configuration differs from lock"
        )
    if expected.sampling_config_sha256 != normalized_lock.get(
        "sampling_sha256"
    ) or expected.sampling_config_sha256 != _canonical_json_sha256(
        expected.sampling_config
    ):
        raise FinalCandidateLockError("normalized sampling digest differs from lock")
    if (
        expected.sampling_config.get("diffusion_type") != "udlm"
        or expected.sampling_config.get("num_steps") != 128
        or normalized_lock.get("inference_weights") != FINAL_INFERENCE_WEIGHTS
    ):
        raise FinalCandidateLockError("final inference is not the locked EMA UDLM run")

    if expected.benchmark_runner_sha256 != normalized_lock.get(
        "benchmark_runner_sha256"
    ):
        raise FinalCandidateLockError("benchmark runner source differs from lock")
    sampler_sha256 = _implementation_source_digest(
        expected, key="sampler_source", relative_path=SAMPLER_SOURCE_RELATIVE_PATH
    )
    _implementation_source_digest(
        expected, key="artifact_io_source", relative_path=ARTIFACT_IO_RELATIVE_PATH
    )
    if sampler_sha256 != normalized_lock.get("sampler_source_sha256"):
        raise FinalCandidateLockError("sampler source differs from candidate lock")
    if _canonical_json_sha256(expected.implementation_inputs) != normalized_lock.get(
        "implementation_inputs_sha256"
    ):
        raise FinalCandidateLockError("implementation-input map differs from lock")
    if _canonical_json_sha256(expected.metric_inputs) != normalized_lock.get(
        "metric_inputs_sha256"
    ):
        raise FinalCandidateLockError("metric-input map differs from lock")

    runner_path = Path(benchmark_runner.__file__).resolve()
    if runner_path != REPOSITORY_ROOT / BENCHMARK_RUNNER_RELATIVE_PATH:
        raise FinalCandidateLockError(
            "benchmark runner was imported from outside the repo"
        )
    _committed_regular_file_snapshot(
        BENCHMARK_RUNNER_RELATIVE_PATH,
        source_revision=source_revision,
        label="benchmark runner source",
        expected_sha256=expected.benchmark_runner_sha256,
    )

    final_directories = normalized_lock.get("final_run_directories")
    if not isinstance(final_directories, Mapping):
        raise FinalCandidateLockError("candidate lock final directories are malformed")
    directory_values = [final_directories.get(seed) for seed in (0, 1, 2)]
    if any(not isinstance(value, Path) for value in directory_values):
        raise FinalCandidateLockError("candidate lock lacks every final seed directory")
    parents = {value.parent for value in directory_values if isinstance(value, Path)}
    if len(parents) != 1:
        raise FinalCandidateLockError("final seed directories do not share one root")
    locked_output_relative = parents.pop()
    parts = locked_output_relative.parts
    if (
        len(parts) != 5
        or parts[:3] != ("output", "udlm", "final")
        or parts[3] != candidate_id
        or FINAL_CONFIG_ID_PATTERN.fullmatch(parts[4]) is None
        or parts[4][0] != candidate_id[0]
    ):
        raise FinalCandidateLockError("locked final output root has invalid identity")
    config_id = parts[4]
    for seed in (0, 1, 2):
        expected_directory = locked_output_relative / f"seed_{seed}"
        if final_directories.get(seed) != expected_directory:
            raise FinalCandidateLockError(
                "candidate lock final seed directories are not the fixed layout"
            )
    locked_output_root = Path(
        os.path.abspath(os.fspath(REPOSITORY_ROOT / locked_output_relative))
    )
    if output_root != locked_output_root:
        raise FinalCandidateLockError("CLI output root differs from candidate lock")

    temperature_code = int(config_id.split("_t", 1)[1].split("_p", 1)[0])
    top_p_code = int(config_id.rsplit("_p", 1)[1])
    if (
        expected.sampling_config.get("softmax_temp") != temperature_code / 100
        or expected.sampling_config.get("raw_loo_top_p") != top_p_code / 100
    ):
        raise FinalCandidateLockError("config ID disagrees with locked operating point")
    return candidate_id, config_id, locked_output_root


def _preflight_final_candidate_lock(
    *,
    candidate_lock: Path,
    expected: ExpectedRunIdentity,
    checkpoint: Path,
    config: Path,
    seeds: list[int],
    num_samples: int,
    output_root: Path,
    source_revision: str,
) -> FinalCandidateAuthority:
    """Validate the committed schema-2 lock before any final-tier mutation/GPU use."""

    from scripts.udlm import superiority_gate

    if candidate_lock != REPOSITORY_ROOT / CANDIDATE_LOCK_RELATIVE_PATH:
        raise FinalCandidateLockError("candidate lock path is not the fixed path")
    lock_claim, lock_payload = _committed_regular_file_snapshot(
        CANDIDATE_LOCK_RELATIVE_PATH,
        source_revision=source_revision,
        label="candidate lock",
    )
    _require_exact_final_lock_commit(
        source_revision=source_revision, lock_payload=lock_payload
    )
    protocol_claim, protocol_payload = _committed_regular_file_snapshot(
        superiority_gate.PROTOCOL_RELATIVE_PATH.as_posix(),
        source_revision=source_revision,
        label="superiority protocol",
        expected_sha256=superiority_gate.PROTOCOL_SHA256,
    )
    del protocol_claim
    try:
        protocol = superiority_gate.strict_json_loads(
            protocol_payload, label="superiority protocol"
        )
        lock = superiority_gate.strict_json_loads(lock_payload, label="candidate lock")
        if not isinstance(protocol, Mapping) or not isinstance(lock, Mapping):
            raise FinalCandidateLockError("protocol and candidate lock must be objects")
        superiority_gate.validate_protocol(protocol)
        normalized_lock = superiority_gate.validate_candidate_lock(lock, protocol)
    except FinalCandidateLockError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise FinalCandidateLockError(
            "candidate lock fails the frozen protocol/schema-2 contract"
        ) from error

    candidate_id, config_id, locked_output_root = _bind_final_candidate_authority(
        normalized_lock=normalized_lock,
        expected=expected,
        checkpoint=checkpoint,
        config=config,
        seeds=seeds,
        num_samples=num_samples,
        output_root=output_root,
        source_revision=source_revision,
    )
    analysis_sources = {
        "scripts/udlm/superiority_gate.py": normalized_lock["gate_source_sha256"],
        "scripts/exps/denovo/report.py": normalized_lock["report_source_sha256"],
        "scripts/udlm/rescore_denovo_run.py": normalized_lock["rescore_source_sha256"],
        "scripts/udlm/rescore_mdlm_baseline.py": normalized_lock[
            "rescore_dependency_sha256"
        ],
        "scripts/exps/denovo/launch_benchmark.py": normalized_lock[
            "benchmark_launcher_source_sha256"
        ],
        "scripts/udlm/write_pilot_evidence.py": normalized_lock[
            "pilot_evidence_writer_source_sha256"
        ],
    }
    for relative_path, digest in analysis_sources.items():
        _committed_regular_file_snapshot(
            relative_path,
            source_revision=source_revision,
            label=f"locked analysis source {relative_path}",
            expected_sha256=digest,
        )
    artifact_source = expected.implementation_inputs.get("artifact_io_source")
    if not isinstance(artifact_source, Mapping):
        raise FinalCandidateLockError("artifact_io generation provenance is absent")
    try:
        superiority_gate.validate_analysis_runtime(
            normalized_lock,
            expected_artifact_io_source_sha256=str(artifact_source.get("sha256")),
        )
    except (OSError, TypeError, ValueError) as error:
        raise FinalCandidateLockError(
            "runtime analysis sources differ from candidate lock"
        ) from error
    return FinalCandidateAuthority(
        claim=lock_claim,
        payload=lock_payload,
        source_revision=source_revision,
        normalized_lock=normalized_lock,
        candidate_id=candidate_id,
        config_id=config_id,
        output_root=locked_output_root,
    )


def _revalidate_final_candidate_lock(authority: FinalCandidateAuthority) -> None:
    """Fail closed if the retained lock path, bytes, or committed blob changes."""

    if not isinstance(authority, FinalCandidateAuthority):
        raise TypeError("final candidate authority capability is invalid")
    claim, payload = _committed_regular_file_snapshot(
        CANDIDATE_LOCK_RELATIVE_PATH,
        source_revision=authority.source_revision,
        label="candidate lock",
    )
    if claim != authority.claim or payload != authority.payload:
        raise FinalCandidateLockError(
            "candidate lock identity or bytes changed after final preflight"
        )


def _require_clean_pushed_source() -> dict[str, str]:
    """Require immutable, upstream-backed code while allowing output artifacts."""

    status = _git_text("status", "--porcelain=v1", "--untracked-files=normal")
    disallowed = []
    for line in status.splitlines():
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if path_text != "output" and not path_text.startswith("output/"):
            disallowed.append(line)
    if disallowed:
        raise RuntimeError(
            "benchmark source worktree is dirty outside output/: "
            + "; ".join(disallowed)
        )

    head = _git_text("rev-parse", "HEAD")
    try:
        upstream = _git_text("rev-parse", "@{upstream}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "benchmark branch has no upstream; commit and push it before launch"
        ) from exc
    if head != upstream:
        raise RuntimeError(
            f"benchmark HEAD {head} is not the pushed upstream commit {upstream}"
        )
    return {"head": head, "upstream": upstream}


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_expected_run_identity(
    checkpoint: Path,
    config: Path,
    num_samples: int,
    *,
    source_revision: str | None = None,
    checkpoint_info: Mapping[str, Any] | None = None,
    metric_inputs: Mapping[str, Any] | None = None,
    implementation_inputs: Mapping[str, Any] | None = None,
    benchmark_runner_sha256: str | None = None,
) -> ExpectedRunIdentity:
    """Resolve and fingerprint the exact inputs passed to every child run."""

    # Quality depends on this ignored binary artifact. Verify it before the
    # checkpoint inspection and well before any GPU selection/model startup.
    metric_inputs = (
        benchmark_runner.metric_input_provenance()
        if metric_inputs is None
        else dict(metric_inputs)
    )
    checkpoint_info = (
        benchmark_runner.checkpoint_metadata(checkpoint)
        if checkpoint_info is None
        else dict(checkpoint_info)
    )
    source_config_sha256 = _sha256_file(config)
    config_git_tracking = (
        benchmark_runner.tracked_source_file_provenance(
            config,
            expected_revision=source_revision,
            expected_sha256=source_config_sha256,
        )
        if source_revision is not None
        else None
    )
    source_config = benchmark_runner.load_yaml_config(config)
    if _sha256_file(config) != source_config_sha256:
        raise RuntimeError(f"Inference config changed while being inspected: {config}")
    sampling_config = benchmark_runner.validate_sampling_config(source_config)
    checkpoint_diffusion_type = str(
        checkpoint_info.get("diffusion_type", "mdlm")
    ).lower()
    if sampling_config["diffusion_type"] != checkpoint_diffusion_type:
        raise ValueError(
            "config diffusion_type does not match checkpoint metadata: "
            f"{sampling_config['diffusion_type']!r} != {checkpoint_diffusion_type!r}"
        )
    checkpoint_udlm_inference_eps = checkpoint_info.get("udlm_inference_eps")
    checkpoint_udlm_exclude_special_tokens = checkpoint_info.get(
        "udlm_exclude_special_tokens"
    )
    checkpoint_udlm_prior_variant = checkpoint_info.get(
        "udlm_prior_variant",
        "release_uniform" if checkpoint_diffusion_type == "udlm" else None,
    )
    checkpoint_udlm_prior_metadata = checkpoint_info.get("udlm_prior_metadata")
    checkpoint_udlm_prior_metadata_sha256 = checkpoint_info.get(
        "udlm_prior_metadata_sha256"
    )
    if checkpoint_diffusion_type == "udlm":
        if checkpoint_udlm_inference_eps is None or not math.isclose(
            float(checkpoint_udlm_inference_eps),
            float(sampling_config["inference_eps"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "config inference_eps does not match checkpoint metadata: "
                f"{sampling_config['inference_eps']} != "
                f"{checkpoint_udlm_inference_eps}"
            )
        if (
            checkpoint_udlm_exclude_special_tokens
            is not sampling_config["exclude_special_tokens"]
        ):
            raise ValueError(
                "config exclude_special_tokens does not match checkpoint metadata"
            )
        if checkpoint_udlm_prior_variant != sampling_config["prior_variant"]:
            raise ValueError(
                "config prior_variant does not match checkpoint metadata: "
                f"{sampling_config['prior_variant']!r} != "
                f"{checkpoint_udlm_prior_variant!r}"
            )
        if (
            checkpoint_udlm_prior_metadata_sha256
            != sampling_config["prior_metadata_sha256"]
        ):
            raise ValueError(
                "config prior_metadata_sha256 does not match checkpoint metadata"
            )
        if checkpoint_udlm_prior_variant in (
            benchmark_runner.UDLM_CATEGORICAL_PRIOR_VARIANTS
        ) and not isinstance(checkpoint_udlm_prior_metadata, Mapping):
            raise ValueError(
                "categorical checkpoint metadata inspection did not return the full "
                "immutable prior record"
            )
    implementation_inputs = (
        (
            benchmark_runner.implementation_input_provenance(gibbs_corrector=True)
            if sampling_config.get("gibbs_corrector", False)
            else benchmark_runner.implementation_input_provenance()
        )
        if implementation_inputs is None
        else dict(implementation_inputs)
    )
    benchmark_runner_path = Path(benchmark_runner.__file__).resolve()
    benchmark_runner_sha256 = (
        _sha256_file(benchmark_runner_path)
        if benchmark_runner_sha256 is None
        else benchmark_runner_sha256
    )
    effective_config = dict(source_config)
    effective_config.update(
        {
            "model_path": str(checkpoint),
            "num_samples": num_samples,
            "device": "cuda:0",
        }
    )
    if checkpoint_diffusion_type == "udlm":
        # Match the child's effective configuration, including historical YAMLs
        # that omit the normalized identity (no-truncation) top-p default.
        effective_config["raw_loo_top_p"] = sampling_config["raw_loo_top_p"]
    return ExpectedRunIdentity(
        checkpoint_path=checkpoint,
        checkpoint_sha256=str(checkpoint_info["sha256"]),
        checkpoint_global_step=int(checkpoint_info["global_step"]),
        checkpoint_size_bytes=int(checkpoint_info["size_bytes"]),
        checkpoint_diffusion_type=checkpoint_diffusion_type,
        checkpoint_udlm_inference_eps=(
            float(checkpoint_udlm_inference_eps)
            if checkpoint_udlm_inference_eps is not None
            else None
        ),
        checkpoint_udlm_exclude_special_tokens=(
            bool(checkpoint_udlm_exclude_special_tokens)
            if checkpoint_udlm_exclude_special_tokens is not None
            else None
        ),
        checkpoint_udlm_prior_variant=(
            str(checkpoint_udlm_prior_variant)
            if checkpoint_udlm_prior_variant is not None
            else None
        ),
        checkpoint_udlm_prior_metadata=(
            dict(checkpoint_udlm_prior_metadata)
            if isinstance(checkpoint_udlm_prior_metadata, Mapping)
            else None
        ),
        checkpoint_udlm_prior_metadata_sha256=(
            str(checkpoint_udlm_prior_metadata_sha256)
            if checkpoint_udlm_prior_metadata_sha256 is not None
            else None
        ),
        config_path=config,
        source_config=source_config,
        source_config_sha256=source_config_sha256,
        config_git_tracking=config_git_tracking,
        sampling_config=sampling_config,
        sampling_config_sha256=_canonical_json_sha256(sampling_config),
        effective_config=effective_config,
        effective_config_sha256=_canonical_json_sha256(effective_config),
        benchmark_runner_sha256=benchmark_runner_sha256,
        implementation_inputs=implementation_inputs,
        metric_inputs=metric_inputs,
        num_samples=num_samples,
        source_revision=source_revision,
    )


def _recorded_path(value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _validate_raw_csv(
    samples_path: Path,
    *,
    expected_row_count: int,
    expected_fields: tuple[str, ...],
) -> tuple[str, int, tuple[str, ...], list[str]]:
    """Return the CSV digest and shape while retaining every schema error."""

    errors: list[str] = []
    actual_fields: tuple[str, ...] = ()
    actual_row_count = 0
    try:
        with samples_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                actual_fields = tuple(next(reader))
            except StopIteration:
                errors.append("raw_samples.csv is empty and has no header")
                return _sha256_file(samples_path), 0, actual_fields, errors

            sample_index_column = (
                actual_fields.index("sample_index")
                if "sample_index" in actual_fields
                else None
            )
            for expected_index, row in enumerate(reader):
                actual_row_count += 1
                if len(row) != len(actual_fields):
                    errors.append(
                        "raw_samples.csv row "
                        f"{actual_row_count} has {len(row)} columns; expected "
                        f"{len(actual_fields)}"
                    )
                    continue
                if sample_index_column is not None:
                    try:
                        observed_index = int(row[sample_index_column])
                    except ValueError:
                        errors.append(
                            f"raw_samples.csv row {actual_row_count} has a non-integer "
                            "sample_index"
                        )
                    else:
                        if observed_index != expected_index:
                            errors.append(
                                f"raw_samples.csv row {actual_row_count} has sample_index "
                                f"{observed_index}; expected {expected_index}"
                            )
    except (OSError, csv.Error, UnicodeError) as error:
        errors.append(f"raw_samples.csv could not be parsed: {error}")

    if actual_fields != expected_fields:
        errors.append(
            "raw_samples.csv header/schema differs from benchmark RAW_SAMPLE_FIELDS"
        )
    if actual_row_count != expected_row_count:
        errors.append(
            f"raw_samples.csv has {actual_row_count} data rows; expected "
            f"{expected_row_count}"
        )
    try:
        digest = _sha256_file(samples_path)
    except OSError as error:
        errors.append(f"raw_samples.csv could not be hashed: {error}")
        digest = ""
    return digest, actual_row_count, actual_fields, errors


def _completed(
    output_root: Path,
    seed: int,
    expected: ExpectedRunIdentity,
) -> bool:
    """Return true only for complete artifacts matching every requested input.

    No artifacts means the seed is safe to launch.  Any partial, unreadable, or
    mismatched state is an error because the child runner intentionally refuses
    to overwrite it; treating that state as merely pending would waste a GPU and
    fail only after model startup.
    """

    run_dir = output_root / f"seed_{seed}"
    summary_path = run_dir / "summary.json"
    samples_path = run_dir / "raw_samples.csv"
    lock_path = run_dir / benchmark_runner.LOCK_FILENAME
    summary_exists = summary_path.exists()
    samples_exist = samples_path.exists()

    if lock_path.exists():
        raise CompletionArtifactError(
            f"Seed {seed} output is locked by {lock_path}. Verify whether another "
            "benchmark is running before removing a stale lock or choosing a new "
            "output root."
        )
    if not summary_exists and not samples_exist:
        return False
    if summary_exists != samples_exist:
        present = summary_path if summary_exists else samples_path
        missing = samples_path if summary_exists else summary_path
        raise CompletionArtifactError(
            f"Seed {seed} has partial benchmark artifacts: {present} exists but "
            f"{missing} is missing. The summary is the completion marker and the "
            "launcher will not write into this directory; inspect/recover the run "
            "or choose a fresh output root."
        )
    if not summary_path.is_file() or not samples_path.is_file():
        raise CompletionArtifactError(
            f"Seed {seed} artifact paths must be regular files: {summary_path}, "
            f"{samples_path}"
        )

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionArtifactError(
            f"Seed {seed} summary cannot be read as JSON: {summary_path}: {error}"
        ) from error
    if not isinstance(summary, Mapping):
        raise CompletionArtifactError(
            f"Seed {seed} summary root must be a JSON object: {summary_path}"
        )

    errors: list[str] = []

    def expect(label: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            errors.append(f"{label}={actual!r}; expected {wanted!r}")

    summary_schema_version = summary.get("schema_version")
    supported_schema_versions = {7, benchmark_runner.SCHEMA_VERSION}
    if summary_schema_version not in supported_schema_versions:
        errors.append(
            "schema_version is neither historical schema 7 nor the current child "
            f"schema: {summary_schema_version!r}"
        )
    if (
        expected.num_samples == FINAL_BENCHMARK_SAMPLES_PER_SEED
        and expected.checkpoint_diffusion_type == "udlm"
        and summary_schema_version != benchmark_runner.SCHEMA_VERSION
    ):
        errors.append("locked UDLM final evidence must use current benchmark schema 8")
    expected_top_level = {
        "schema_version",
        "status",
        "seed",
        "num_samples",
        "run",
        "checkpoint",
        "config",
        "metrics",
        "failure_counts",
        "runtime_seconds",
        "environment",
        "git",
        "implementation_inputs",
        "metric_inputs",
        "tokenizer",
        "artifacts",
    }
    if summary_schema_version != 7:
        expected_top_level.add("sampled_token_control_audit")
    if set(summary) != expected_top_level:
        errors.append(
            "summary top-level fields differ from the current child contract: "
            f"found {sorted(summary)}, expected {sorted(expected_top_level)}"
        )

    expect("status", summary.get("status"), "completed")
    expect("seed", summary.get("seed"), seed)
    expect("num_samples", summary.get("num_samples"), expected.num_samples)

    run = _mapping(summary.get("run"))
    expect("run.seed", run.get("seed"), seed)
    expect(
        "run.requested_sample_count",
        run.get("requested_sample_count"),
        expected.num_samples,
    )
    expect(
        "run.evaluation_tier",
        run.get("evaluation_tier"),
        "final" if expected.num_samples == 1_000 else "pilot",
    )
    expect(
        "run.final_protocol_eligible",
        run.get("final_protocol_eligible"),
        expected.num_samples == 1_000,
    )
    expect("run.one_seed_per_invocation", run.get("one_seed_per_invocation"), True)
    expect("run.single_generation_batch", run.get("single_generation_batch"), True)
    expected_seed_configuration = {
        "seed": seed,
        "seed_applied_immediately_before_generation": True,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda_all": True,
        "python_hash_seed": str(seed),
    }
    expect(
        "run.seed_configuration",
        run.get("seed_configuration"),
        expected_seed_configuration,
    )
    for timestamp_field in ("started_at_utc", "completed_at_utc"):
        timestamp = run.get(timestamp_field)
        try:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"run.{timestamp_field} is not an ISO-8601 timestamp")
        else:
            if parsed.tzinfo is None:
                errors.append(f"run.{timestamp_field} lacks a timezone")

    if expected.source_revision is not None:
        if summary_schema_version == 7:
            expected_command = _legacy_schema7_command(
                checkpoint=expected.checkpoint_path,
                expected_checkpoint_sha256=expected.checkpoint_sha256,
                expected_source_revision=expected.source_revision,
                config=expected.config_path,
                expected_config_sha256=expected.source_config_sha256,
                num_samples=expected.num_samples,
                seed=seed,
                output_dir=run_dir.resolve(),
            )
        else:
            try:
                output_state = run_dir.stat(follow_symlinks=False)
            except OSError as error:
                errors.append(f"run output directory cannot be inspected: {error}")
                output_state = None
            if output_state is not None and not stat.S_ISDIR(output_state.st_mode):
                errors.append("run output directory is not a direct directory")
                output_state = None
            expected_command = (
                None
                if output_state is None
                else _command(
                    checkpoint=expected.checkpoint_path,
                    expected_checkpoint_sha256=expected.checkpoint_sha256,
                    expected_source_revision=expected.source_revision,
                    config=expected.config_path,
                    expected_config_sha256=expected.source_config_sha256,
                    num_samples=expected.num_samples,
                    seed=seed,
                    output_dir=Path(os.path.abspath(run_dir)),
                    expected_output_directory_device=int(output_state.st_dev),
                    expected_output_directory_inode=int(output_state.st_ino),
                )
            )
        expect("run.command", run.get("command"), expected_command)
    protocol = _mapping(run.get("generation_protocol"))
    sampling = expected.sampling_config
    for key, wanted in {
        "diffusion_type": sampling["diffusion_type"],
        "num_steps": sampling["num_steps"],
        "inference_eps": sampling["inference_eps"],
        "exclude_special_tokens": sampling["exclude_special_tokens"],
        "prior_variant": sampling["prior_variant"],
        "prior_metadata_sha256": sampling["prior_metadata_sha256"],
        "temperature": sampling["softmax_temp"],
        "randomness": sampling["randomness"],
        "randomness_used_by_sampler": sampling["diffusion_type"] == "mdlm",
    }.items():
        expect(f"run.generation_protocol.{key}", protocol.get(key), wanted)
    nfe = protocol.get("nfe")
    if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe <= 0:
        errors.append("run.generation_protocol.nfe must be a positive integer")
    elif sampling["diffusion_type"] == "udlm" and nfe != sampling["num_steps"]:
        errors.append("run.generation_protocol.nfe must equal UDLM num_steps")
    if sampling.get("gibbs_corrector", False):
        expect("schema_version for Gibbs correction", summary_schema_version, 8)
        for key, wanted in {
            "gibbs_corrector": True,
            "predictor_transitions_per_molecule": sampling["num_steps"] // 2,
            "corrector_steps_per_molecule": sampling["num_steps"] // 2,
            "num_steps_source": benchmark_runner.GIBBS_CORRECTOR_NUM_STEPS_SOURCE,
            "nfe_definition": benchmark_runner.GIBBS_CORRECTOR_NFE_DEFINITION,
        }.items():
            expect(f"run.generation_protocol.{key}", protocol.get(key), wanted)
            if type(protocol.get(key)) is not type(wanted):
                errors.append(f"run.generation_protocol.{key} has an invalid type")
    elif any(
        key in protocol
        for key in (
            "gibbs_corrector",
            "predictor_transitions_per_molecule",
            "corrector_steps_per_molecule",
        )
    ):
        errors.append("run.generation_protocol declares an unconfigured Gibbs corrector")

    checkpoint = _mapping(summary.get("checkpoint"))
    expect(
        "checkpoint.path",
        _recorded_path(checkpoint.get("path")),
        expected.checkpoint_path,
    )
    expect(
        "checkpoint.sha256",
        checkpoint.get("sha256"),
        expected.checkpoint_sha256,
    )
    expect(
        "checkpoint.global_step",
        checkpoint.get("global_step"),
        expected.checkpoint_global_step,
    )
    expect(
        "checkpoint.size_bytes",
        checkpoint.get("size_bytes"),
        expected.checkpoint_size_bytes,
    )
    expect(
        "checkpoint.byte_identity_verified_before_and_after_load",
        checkpoint.get("byte_identity_verified_before_and_after_load"),
        True,
    )
    expect(
        "checkpoint.diffusion_type",
        checkpoint.get("diffusion_type"),
        expected.checkpoint_diffusion_type,
    )
    expect(
        "checkpoint.udlm_inference_eps",
        checkpoint.get("udlm_inference_eps"),
        expected.checkpoint_udlm_inference_eps,
    )
    expect(
        "checkpoint.udlm_exclude_special_tokens",
        checkpoint.get("udlm_exclude_special_tokens"),
        expected.checkpoint_udlm_exclude_special_tokens,
    )
    expect(
        "checkpoint.udlm_prior_variant",
        checkpoint.get("udlm_prior_variant"),
        expected.checkpoint_udlm_prior_variant,
    )
    expect(
        "checkpoint.udlm_prior_metadata",
        checkpoint.get("udlm_prior_metadata"),
        expected.checkpoint_udlm_prior_metadata,
    )
    expect(
        "checkpoint.udlm_prior_metadata_sha256",
        checkpoint.get("udlm_prior_metadata_sha256"),
        expected.checkpoint_udlm_prior_metadata_sha256,
    )

    config = _mapping(summary.get("config"))
    expect("config.path", _recorded_path(config.get("path")), expected.config_path)
    expect("config.sha256", config.get("sha256"), expected.source_config_sha256)
    expect(
        "config.git_tracking", config.get("git_tracking"), expected.config_git_tracking
    )
    expect("config.source", config.get("source"), expected.source_config)
    expect("config.sampling", config.get("sampling"), expected.sampling_config)
    expect(
        "config.sampling_sha256",
        config.get("sampling_sha256"),
        expected.sampling_config_sha256,
    )
    expect("config.effective", config.get("effective"), expected.effective_config)
    expect(
        "config.effective_sha256",
        config.get("effective_sha256"),
        expected.effective_config_sha256,
    )

    git = _mapping(summary.get("git"))
    expect(
        "git.runner_sha256",
        git.get("runner_sha256"),
        expected.benchmark_runner_sha256,
    )
    if expected.source_revision is not None:
        expect("git.commit", git.get("commit"), expected.source_revision)
        expect("git.upstream", git.get("upstream"), expected.source_revision)
        expect(
            "git.expected_source_revision",
            git.get("expected_source_revision"),
            expected.source_revision,
        )
        expect("git.dirty", git.get("dirty"), False)
        expect(
            "git.clean_pushed_source_verified_before_and_after_run",
            git.get("clean_pushed_source_verified_before_and_after_run"),
            True,
        )
    expect(
        "implementation_inputs",
        summary.get("implementation_inputs"),
        expected.implementation_inputs,
    )
    expect(
        "metric_inputs",
        summary.get("metric_inputs"),
        expected.metric_inputs,
    )

    metrics = _mapping(summary.get("metrics"))
    if set(metrics) != {"released_comparable", "strict"} or any(
        not isinstance(metrics.get(branch), Mapping) or not metrics.get(branch)
        for branch in ("released_comparable", "strict")
    ):
        errors.append(
            "metrics must contain non-empty released_comparable and strict branches"
        )
    expected_failure_fields = {
        "raw_safe_conversion_failed",
        "strict_decode_failed",
        "released_decode_failed",
        "released_recovered_strict_failure",
        "strict_valid_but_released_failed",
        "released_largest_component_applied",
        "strict_duplicates",
        "released_duplicates",
    }
    failure_counts = _mapping(summary.get("failure_counts"))
    if set(failure_counts) != expected_failure_fields or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in failure_counts.values()
    ):
        errors.append(
            "failure_counts does not match the current nonnegative count schema"
        )
    expected_runtime_fields = {
        "model_load_and_device_move",
        "model_sampling_and_tokenizer",
        "released_postprocessing",
        "generation",
        "decode_and_metrics",
        "total_before_summary_write",
    }
    if summary_schema_version != 7:
        expected_runtime_fields.add("sampled_token_control_audit")
    runtime = _mapping(summary.get("runtime_seconds"))
    if set(runtime) != expected_runtime_fields or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in runtime.values()
    ):
        errors.append("runtime_seconds does not match the current finite timing schema")
    environment = _mapping(summary.get("environment"))
    expect(
        "environment.requested_device", environment.get("requested_device"), "cuda:0"
    )
    expect(
        "environment.resolved_model_device",
        environment.get("resolved_model_device"),
        "cuda:0",
    )
    expect(
        "environment.torch_cuda_available",
        environment.get("torch_cuda_available"),
        True,
    )
    if not _mapping(summary.get("tokenizer")):
        errors.append("tokenizer provenance must be a non-empty mapping")
    if expected.source_revision is not None:
        launch_environment = _mapping(environment.get("launch_environment"))
        snapshot_text = launch_environment.get(
            "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"
        )
        try:
            snapshot = json.loads(snapshot_text)
        except (TypeError, json.JSONDecodeError):
            errors.append("launch environment lacks a valid GPU selection snapshot")
        else:
            snapshot_mapping = _mapping(snapshot)
            if not snapshot_mapping:
                errors.append("launch GPU selection snapshot must be a JSON object")
            expect(
                "launch snapshot.command",
                snapshot_mapping.get("command"),
                expected_command,
            )
            expect(
                "launch snapshot.source_revision",
                snapshot_mapping.get("source_revision"),
                {
                    "head": expected.source_revision,
                    "upstream": expected.source_revision,
                },
            )

    artifacts = _mapping(summary.get("artifacts"))
    raw_artifact = _mapping(artifacts.get("raw_samples_csv"))
    summary_artifact = _mapping(artifacts.get("summary_json"))
    expected_fields = tuple(benchmark_runner.RAW_SAMPLE_FIELDS)
    expect(
        "artifacts.raw_samples_csv.path",
        _recorded_path(raw_artifact.get("path")),
        samples_path.resolve(),
    )
    expect(
        "artifacts.raw_samples_csv.row_count",
        raw_artifact.get("row_count"),
        expected.num_samples,
    )
    expect(
        "artifacts.raw_samples_csv.fields",
        raw_artifact.get("fields"),
        list(expected_fields),
    )
    expect(
        "artifacts.summary_json.path",
        _recorded_path(summary_artifact.get("path")),
        summary_path.resolve(),
    )

    actual_sha256, actual_rows, actual_fields, csv_errors = _validate_raw_csv(
        samples_path,
        expected_row_count=expected.num_samples,
        expected_fields=expected_fields,
    )
    errors.extend(csv_errors)
    expect("raw_samples.csv actual row_count", actual_rows, expected.num_samples)
    expect("raw_samples.csv actual fields", actual_fields, expected_fields)
    expect(
        "artifacts.raw_samples_csv.sha256", raw_artifact.get("sha256"), actual_sha256
    )

    if errors:
        detail = "\n  - ".join(errors)
        raise CompletionArtifactError(
            f"Seed {seed} has benchmark artifacts that do not match the requested "
            f"run or fail integrity checks:\n  - {detail}\nRefusing to skip or "
            "relaunch into the existing directory. Inspect the artifacts or choose "
            "a fresh output root."
        )
    if expected.num_samples in {
        FINAL_BENCHMARK_SAMPLES_PER_SEED,
        REGISTERED_SELECTION_PILOT_SAMPLES_PER_SEED,
    }:
        # Registered final and candidate-selection runs must be consumable by
        # the actual report validator, not merely resemble a child summary
        # structurally. Generic <=100 smoke pilots intentionally retain only
        # the lighter launcher completion contract.
        if expected.source_revision is None:
            raise CompletionArtifactError(
                f"Seed {seed} registered completion lacks an expected source revision"
            )
        try:
            current_source_revision = _require_clean_pushed_source()
        except RuntimeError as error:
            raise CompletionArtifactError(
                f"Seed {seed} cannot validate completion against stable report code: "
                f"{error}"
            ) from error
        if current_source_revision["head"] != expected.source_revision:
            raise CompletionArtifactError(
                f"Seed {seed} report code revision changed while the controller ran"
            )

        from scripts.exps.denovo import report as benchmark_report

        expected_tier = (
            "final"
            if expected.num_samples == FINAL_BENCHMARK_SAMPLES_PER_SEED
            else "pilot"
        )
        try:
            benchmark_report.validate_run_evidence(
                run_dir,
                seed,
                expected_samples=expected.num_samples,
                expected_tier=expected_tier,
                final_protocol_eligible=expected_tier == "final",
            )
        except benchmark_report.ReportValidationError as error:
            raise CompletionArtifactError(
                f"Seed {seed} artifacts fail registered report validation: {error}. "
                "Refusing to skip or relaunch into the existing directory."
            ) from error
    return True


def _command(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    expected_source_revision: str,
    config: Path,
    expected_config_sha256: str,
    num_samples: int,
    seed: int,
    output_dir: Path,
    expected_output_directory_device: int,
    expected_output_directory_inode: int,
) -> list[str]:
    if not output_dir.is_absolute():
        raise ValueError("benchmark child output directory must be absolute")
    for label, value in (
        ("output directory device", expected_output_directory_device),
        ("output directory inode", expected_output_directory_inode),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"expected {label} must be a nonnegative integer")
    command = [
        str(_project_venv_python()),
        str(REPOSITORY_ROOT / "scripts/exps/denovo/benchmark.py"),
        "--checkpoint",
        str(checkpoint),
        "--expected-checkpoint-sha256",
        expected_checkpoint_sha256,
        "--expected-source-revision",
        expected_source_revision,
        "--config",
        str(config),
        "--expected-config-sha256",
        expected_config_sha256,
        "--num-samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--output-dir",
        str(output_dir),
        "--expected-output-directory-device",
        str(expected_output_directory_device),
        "--expected-output-directory-inode",
        str(expected_output_directory_inode),
    ]
    if len(command) != 24 or any(not isinstance(value, str) for value in command):
        raise AssertionError("benchmark child argv must contain exactly 24 strings")
    return command


def _legacy_schema7_command(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    expected_source_revision: str,
    config: Path,
    expected_config_sha256: str,
    num_samples: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    """Reconstruct only the historical schema-7 exact-20 provenance argv."""

    command = _command(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_source_revision=expected_source_revision,
        config=config,
        expected_config_sha256=expected_config_sha256,
        num_samples=num_samples,
        seed=seed,
        output_dir=output_dir,
        expected_output_directory_device=0,
        expected_output_directory_inode=0,
    )[:20]
    if len(command) != 20:
        raise AssertionError("historical schema-7 argv must contain 20 strings")
    return command


def _reserve_child_paths(
    *, output_dir: Path, log_path: Path
) -> tuple[artifact_io.OwnedDirectory, artifact_io.OwnedLog]:
    """Exclusively reserve the exact child output directory and log."""

    output_owner = artifact_io.create_directory_exclusive(
        REPOSITORY_ROOT,
        _repository_relative(output_dir, label="benchmark child output directory"),
    )
    try:
        log_owner = artifact_io.reserve_log_exclusive(
            REPOSITORY_ROOT,
            _repository_relative(log_path, label="benchmark child log"),
        )
    except BaseException as error:
        try:
            artifact_io.remove_empty_directory_exact(REPOSITORY_ROOT, output_owner)
        except BaseException as rollback_error:
            raise artifact_io.RollbackError(error, [rollback_error]) from error
        raise
    return output_owner, log_owner


def _open_job_log(
    owner: artifact_io.OwnedLog, selection: Mapping[str, Any]
) -> tuple[Any, Any]:
    context = artifact_io.open_log_append_exact(REPOSITORY_ROOT, owner)
    handle = context.__enter__()
    try:
        handle.write(
            (json.dumps(selection, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            )
        )
    except BaseException:
        context.__exit__(*sys.exc_info())
        raise
    return context, handle


def _close_job_log(job: RunningJob) -> None:
    if job.log_context is not None:
        job.log_context.__exit__(None, None, None)
    elif job.log_handle is not None and not job.log_handle.closed:
        # Compatibility for synthetic/historical RunningJob instances.
        job.log_handle.close()


def _snapshot_signature(states: list[GPUState]) -> str:
    return json.dumps([state.as_dict() for state in states], sort_keys=True)


def _recheck_gpu_for_launch(
    candidate: GPUState,
    *,
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> tuple[Optional[GPUState], list[str]]:
    """Probe the exact chosen UUID and reject changed or occupied devices."""

    try:
        rechecked = _probe_gpu(candidate.uuid)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        return None, [f"final GPU probe failed: {error}"]
    reasons = rechecked.rejection_reasons(
        max_utilization_percent=max_utilization_percent,
        min_free_memory_mib=min_free_memory_mib,
    )
    if rechecked.uuid != candidate.uuid:
        reasons.append(
            f"UUID identity changed from {candidate.uuid} to {rechecked.uuid}"
        )
    if rechecked.index != candidate.index:
        reasons.append(
            f"physical index identity changed from {candidate.index} to {rechecked.index}"
        )
    if reasons:
        return None, reasons
    return rechecked, []


def _child_environment(
    *,
    seed: int,
    gpu: GPUState,
    selection: Mapping[str, Any],
    run_label: str,
    generation_lease: GenerationLease,
    launch_authority: Mapping[str, Any],
) -> dict[str, str]:
    """Map the verified physical GPU UUID into the child environment."""

    environment = os.environ.copy()
    required_python_paths = [str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)]
    for inherited_key in tuple(environment):
        if inherited_key.startswith("PYTHON") or inherited_key.startswith(
            "GENMOL_BENCHMARK_"
        ):
            environment.pop(inherited_key)
    _revalidate_generation_lease(generation_lease)
    authority = dict(launch_authority)
    if authority.get("schema_version") != LAUNCH_AUTHORITY_SCHEMA_VERSION:
        raise ValueError("child launch authority has an invalid schema")
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu.uuid,
            "PYTHONHASHSEED": str(seed),
            "HF_HOME": str(REPOSITORY_ROOT / ".cache/huggingface"),
            "TORCH_HOME": str(REPOSITORY_ROOT / ".cache/torch"),
            "PIP_CACHE_DIR": str(REPOSITORY_ROOT / ".cache/pip"),
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            # Do not inherit arbitrary startup code or dependency shadowing via
            # sitecustomize/user paths from the controller shell.
            "PYTHONPATH": os.pathsep.join(required_python_paths),
            "PYTHONNOUSERSITE": "1",
            "PYTHONOPTIMIZE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX": str(gpu.index),
            "GENMOL_BENCHMARK_GPU_UUID": gpu.uuid,
            "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT": json.dumps(
                selection, separators=(",", ":"), sort_keys=True
            ),
            "GENMOL_BENCHMARK_RUN_LABEL": run_label,
            "GENMOL_BENCHMARK_GENERATION_LEASE_PATH": str(
                REPOSITORY_ROOT / GENERATION_LEASE_RELATIVE_PATH
            ),
            "GENMOL_BENCHMARK_EXPECTED_GENERATION_LEASE_SHA256": (
                generation_lease.claim.sha256
            ),
            "GENMOL_BENCHMARK_GENERATION_LEASE_OWNER_TOKEN": (
                generation_lease.owner_token
            ),
            "GENMOL_BENCHMARK_LAUNCH_AUTHORITY_JSON": json.dumps(
                authority, separators=(",", ":"), sort_keys=True
            ),
        }
    )
    return environment


def _finalize_finished_job(
    *,
    job: RunningJob,
    return_code: int,
    output_root: Path,
    expected: ExpectedRunIdentity,
    pilot_mode: str | None,
    attempt_id: str | None,
    candidate_id: str | None,
    source_revision: Mapping[str, str],
) -> tuple[bool, list[str]]:
    """Validate one child result and publish registered failure evidence if needed."""

    if type(return_code) is not int:
        raise TypeError("benchmark child return code must be an integer")
    failure_details: list[str] = []
    failure_stage: str | None = None
    failure_reason: str | None = None
    receipt_process_status: int | None = None
    if return_code != 0:
        failure_stage = "benchmark_child_process"
        failure_reason = f"benchmark child exited with status {return_code}"
        receipt_process_status = return_code
        failure_details.append(
            f"seed {job.seed} exited with status {return_code}; see {job.log_path}"
        )
    else:
        try:
            child_completed = _completed(output_root, job.seed, expected)
        except CompletionArtifactError as error:
            child_completed = False
            failure_reason = f"completion validation failed: {error}"
            failure_details.append(f"seed {job.seed}: {error}")
        if not child_completed:
            failure_stage = "completion_validation"
            if failure_reason is None:
                failure_reason = (
                    "completion validation failed: child exited successfully but "
                    "produced no completion artifacts"
                )
            failure_details.append(
                f"seed {job.seed} exited successfully but produced no valid "
                "completion artifacts"
            )

    failed = failure_stage is not None
    if failed and pilot_mode is not None:
        if pilot_mode not in PILOT_MODES:
            raise RuntimeError("finished pilot job has an invalid pilot mode")
        if attempt_id is None or candidate_id is None or failure_reason is None:
            raise RuntimeError("pilot failure lacks attempt/candidate provenance")
        failed_at_utc = datetime.now(timezone.utc).isoformat()
        try:
            receipt_path = _write_pilot_failure_receipt(
                output_root=output_root,
                job=job,
                expected=expected,
                attempt_id=attempt_id,
                candidate_id=candidate_id,
                pilot_mode=pilot_mode,
                source_revision=source_revision,
                stage=failure_stage,
                reason=failure_reason,
                process_exit_status=receipt_process_status,
                failed_at_utc=failed_at_utc,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failure_details.append(
                f"seed {job.seed} failure receipt could not be published: {error}"
            )
        else:
            failure_details.append(
                f"seed {job.seed} failure receipt published at {receipt_path}"
            )
    return failed, failure_details


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise RuntimeError(f"run from repository root: {REPOSITORY_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            "controller inherited CUDA_VISIBLE_DEVICES; start it from a shell without "
            "a pre-existing GPU visibility mask"
        )
    _require_project_virtual_environment()
    source_revision = _require_clean_pushed_source()
    evaluation_tier = _validate_sample_tier(
        args.num_samples,
        pilot=args.pilot,
        selection_pilot=args.selection_pilot,
    )
    candidate_lock = _validate_candidate_lock_scope(
        args.candidate_lock,
        pilot=args.pilot,
        selection_pilot=args.selection_pilot,
    )
    pilot_mode = (
        "registered_selection"
        if args.selection_pilot
        else ("engineering" if args.pilot else None)
    )
    attempt_id = _validate_attempt_scope(
        pilot=args.pilot,
        selection_pilot=args.selection_pilot,
        attempt_id=args.attempt_id,
        seeds=args.seeds,
    )
    candidate_id = _validate_candidate_scope(
        pilot=args.pilot,
        selection_pilot=args.selection_pilot,
        candidate_id=args.candidate_id,
    )
    gpu_count = _validate_gpu_count(args.gpu_count)
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if args.min_free_memory_mib < 30_000:
        raise ValueError("min-free-memory-mib must be at least 30000")
    if not 1 <= args.max_utilization_percent <= 10:
        raise ValueError("max-utilization-percent must lie in [1, 10]")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be unique")
    selection_policy = _selection_policy(
        requested_gpu_count=gpu_count,
        max_utilization_percent=args.max_utilization_percent,
        min_free_memory_mib=args.min_free_memory_mib,
    )

    checkpoint = _resolve_checkpoint(args.checkpoint)
    config = _resolve_in_repo(args.config)
    output_root, log_root = _resolve_attempt_keyed_roots(
        args.output_root,
        args.log_root,
        attempt_id=attempt_id,
    )
    if pilot_mode is not None:
        _require_fresh_pilot_attempt_paths(output_root, log_root)
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("checkpoint and config must both exist")

    expected = _build_expected_run_identity(
        checkpoint,
        config,
        args.num_samples,
        source_revision=source_revision["head"],
    )
    final_authority = (
        _preflight_final_candidate_lock(
            candidate_lock=candidate_lock,
            expected=expected,
            checkpoint=checkpoint,
            config=config,
            seeds=args.seeds,
            num_samples=args.num_samples,
            output_root=output_root,
            source_revision=source_revision["head"],
        )
        if candidate_lock is not None
        else None
    )
    pending = []
    completed_at_start = []
    for seed in args.seeds:
        if _completed(output_root, seed, expected):
            completed_at_start.append(seed)
        else:
            pending.append(seed)
    if final_authority is not None:
        _revalidate_final_candidate_lock(final_authority)
    print(
        json.dumps(
            {
                "event": "controller_start",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": expected.checkpoint_sha256,
                "checkpoint_global_step": expected.checkpoint_global_step,
                "checkpoint_udlm_prior_variant": (
                    expected.checkpoint_udlm_prior_variant
                ),
                "checkpoint_udlm_prior_metadata": (
                    expected.checkpoint_udlm_prior_metadata
                ),
                "checkpoint_udlm_prior_metadata_sha256": (
                    expected.checkpoint_udlm_prior_metadata_sha256
                ),
                "config": str(config),
                "config_sha256": expected.source_config_sha256,
                "sampling_config": expected.sampling_config,
                "sampling_config_sha256": expected.sampling_config_sha256,
                "num_samples": args.num_samples,
                "evaluation_tier": evaluation_tier,
                "benchmark_mode": (
                    "selection_pilot"
                    if args.selection_pilot
                    else ("generic_pilot" if args.pilot else "final")
                ),
                "attempt_id": attempt_id,
                "candidate_id": (
                    final_authority.candidate_id
                    if final_authority is not None
                    else candidate_id
                ),
                "candidate_config_id": (
                    final_authority.config_id if final_authority is not None else None
                ),
                "candidate_lock": (
                    {
                        "relative_path": CANDIDATE_LOCK_RELATIVE_PATH,
                        "sha256": final_authority.claim.sha256,
                        "size_bytes": final_authority.claim.size_bytes,
                        "exact_regular_blob_at_source_revision": True,
                    }
                    if final_authority is not None
                    else None
                ),
                "pilot_mode": pilot_mode,
                "seeds": args.seeds,
                "completed_seeds": completed_at_start,
                "pending_seeds": pending,
                "user_requested_gpu_count": gpu_count,
                "selection_policy": selection_policy,
                "source_revision": source_revision,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not pending:
        print(
            "All requested de novo benchmark seeds already have matching, "
            "integrity-checked artifacts.",
            flush=True,
        )
        return
    if args.dry_run:
        print(
            json.dumps(
                {
                    "event": "dry_run_preflight_completed_no_launch",
                    "generation_lease_acquired": False,
                    "gpu_query_performed": False,
                    "artifact_mutation_performed": False,
                    "tmux_operation_performed": False,
                    "child_argv_deferred_until_output_directory_reservation": True,
                },
                sort_keys=True,
            )
        )
        for seed in pending:
            print(
                "DRY RUN",
                json.dumps(
                    {
                        "seed": seed,
                        "predicted_output_directory": str(output_root / f"seed_{seed}"),
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": expected.checkpoint_sha256,
                        "config": str(config),
                        "config_sha256": expected.source_config_sha256,
                        "num_samples": args.num_samples,
                    },
                    sort_keys=True,
                ),
            )
        return

    if final_authority is not None:
        _revalidate_final_candidate_lock(final_authority)
    _require_tmux_for_execution()
    if final_authority is not None:
        _revalidate_final_candidate_lock(final_authority)
    global_output = REPOSITORY_ROOT / "output"
    if not global_output.is_dir():
        _ensure_repository_directory(global_output, label="global output directory")
    lease = _acquire_generation_lease(source_revision=source_revision["head"])
    running: dict[str, RunningJob] = {}
    release_authorized = True
    try:
        if pilot_mode is not None:
            _reserve_pilot_attempt_paths(output_root, log_root)
        else:
            _ensure_repository_directory(output_root, label="benchmark output root")
            _ensure_repository_directory(log_root, label="benchmark log root")

        failure_seen = False
        failure_details: list[str] = []
        last_signature: Optional[str] = None
        unchanged_polls = 0

        while pending or running:
            for gpu_uuid, job in list(running.items()):
                return_code = job.process.poll()
                if return_code is None:
                    continue
                del running[gpu_uuid]
                try:
                    _close_job_log(job)
                except BaseException as error:
                    release_authorized = False
                    failure_seen = True
                    failure_details.append(
                        f"seed {job.seed} log finalization failed: {error}"
                    )
                print(
                    f"FINISHED seed={job.seed} physical_gpu={job.gpu.index} "
                    f"uuid={gpu_uuid} exit={return_code} log={job.log_path}",
                    flush=True,
                )
                job_failed, job_failure_details = _finalize_finished_job(
                    job=job,
                    return_code=return_code,
                    output_root=output_root,
                    expected=expected,
                    pilot_mode=pilot_mode,
                    attempt_id=attempt_id,
                    candidate_id=candidate_id,
                    source_revision=source_revision,
                )
                failure_seen = failure_seen or job_failed
                failure_details.extend(job_failure_details)
                if final_authority is not None:
                    _revalidate_final_candidate_lock(final_authority)
                if any(
                    "failure receipt could not be published" in detail
                    for detail in job_failure_details
                ):
                    release_authorized = False
                try:
                    _revalidate_generation_lease(lease)
                except BaseException:
                    release_authorized = False
                    raise

            if failure_seen:
                if not running:
                    details = "\n  - ".join(failure_details)
                    raise RuntimeError(
                        "A benchmark child failed completion validation; no further "
                        f"seeds were launched:\n  - {details}"
                    )
                time.sleep(args.poll_seconds)
                continue

            # Every inventory and every launch is independently authorized by
            # the exact lease.  A new inventory is used after each child launch.
            if final_authority is not None:
                _revalidate_final_candidate_lock(final_authority)
            _revalidate_generation_lease(lease)
            states = _snapshot()
            inventory_snapshot_completed_at_utc = datetime.now(timezone.utc).isoformat()
            signature = _snapshot_signature(states)
            unchanged_polls = unchanged_polls + 1 if signature == last_signature else 0
            last_signature = signature
            if unchanged_polls == 0 or unchanged_polls % 20 == 0:
                eligible_uuids = [
                    state.uuid
                    for state in states
                    if _eligible(
                        state,
                        max_utilization_percent=args.max_utilization_percent,
                        min_free_memory_mib=args.min_free_memory_mib,
                    )
                ]
                print(
                    json.dumps(
                        {
                            "event": "gpu_poll",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "eligible_gpu_uuids": eligible_uuids,
                            "pending_seeds": pending,
                            "running_seeds": [job.seed for job in running.values()],
                            "gpu_states": [state.as_dict() for state in states],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            free_slots = gpu_count - len(running)
            candidates = (
                sorted(
                    (
                        state
                        for state in states
                        if state.uuid not in running
                        and _eligible(
                            state,
                            max_utilization_percent=args.max_utilization_percent,
                            min_free_memory_mib=args.min_free_memory_mib,
                        )
                    ),
                    key=lambda state: (
                        -(state.memory_total_mib - state.memory_used_mib),
                        state.utilization_percent,
                        state.uuid,
                    ),
                )[:1]
                if free_slots > 0
                else []
            )

            launched_job = False
            for candidate in candidates:
                if not pending:
                    break
                seed = pending[0]
                output_dir = Path(os.path.abspath(output_root / f"seed_{seed}"))
                run_label = benchmark_runner.benchmark_run_label(
                    expected.checkpoint_global_step,
                    expected.checkpoint_sha256,
                    seed,
                )
                log_path = Path(os.path.abspath(log_root / f"{run_label}.log"))
                if _completed(output_root, seed, expected):
                    pending.pop(0)
                    print(
                        f"SKIPPED seed={seed}; matching artifacts appeared while "
                        "waiting",
                        flush=True,
                    )
                    continue

                rechecked, rejection_reasons = _recheck_gpu_for_launch(
                    candidate,
                    max_utilization_percent=args.max_utilization_percent,
                    min_free_memory_mib=args.min_free_memory_mib,
                )
                if rechecked is None:
                    print(
                        json.dumps(
                            {
                                "event": "gpu_final_probe_rejected",
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "physical_index": candidate.index,
                                "initial_uuid": candidate.uuid,
                                "reasons": rejection_reasons,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                if final_authority is not None:
                    _revalidate_final_candidate_lock(final_authority)
                final_uuid_probe_completed_at_utc = datetime.now(
                    timezone.utc
                ).isoformat()
                output_owner, log_owner = _reserve_child_paths(
                    output_dir=output_dir, log_path=log_path
                )
                command = _command(
                    checkpoint=checkpoint,
                    expected_checkpoint_sha256=expected.checkpoint_sha256,
                    expected_source_revision=source_revision["head"],
                    config=config,
                    expected_config_sha256=expected.source_config_sha256,
                    num_samples=args.num_samples,
                    seed=seed,
                    output_dir=output_dir,
                    expected_output_directory_device=output_owner.device,
                    expected_output_directory_inode=output_owner.inode,
                )
                authority = _launch_authority(
                    lease=lease,
                    output_directory=output_owner,
                    command=command,
                )
                selection = {
                    "event": "launch",
                    "gpu_selection_schema_version": 3,
                    "timestamp_utc": final_uuid_probe_completed_at_utc,
                    "inventory_snapshot_completed_at_utc": (
                        inventory_snapshot_completed_at_utc
                    ),
                    "final_uuid_probe_completed_at_utc": (
                        final_uuid_probe_completed_at_utc
                    ),
                    "source_revision": source_revision,
                    "gpu_inventory_at_selection": [state.as_dict() for state in states],
                    "running_gpu_uuids_at_selection": sorted(running),
                    "physical_gpu_at_final_uuid_probe": rechecked.as_dict(),
                    "policy": selection_policy,
                    "launch_authority": authority,
                    "command": command,
                }
                log_context, log_handle = _open_job_log(log_owner, selection)
                environment = _child_environment(
                    seed=seed,
                    gpu=rechecked,
                    selection=selection,
                    run_label=run_label,
                    generation_lease=lease,
                    launch_authority=authority,
                )
                _revalidate_generation_lease(lease)
                pending.pop(0)
                started_at_utc = datetime.now(timezone.utc).isoformat()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=REPOSITORY_ROOT,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log_context.__exit__(*sys.exc_info())
                    raise
                running[rechecked.uuid] = RunningJob(
                    seed=seed,
                    gpu=rechecked,
                    process=process,
                    log_handle=log_handle,
                    log_path=log_path,
                    command=tuple(command),
                    started_at_utc=started_at_utc,
                    log_owner=log_owner,
                    log_context=log_context,
                    output_directory_owner=output_owner,
                )
                launched_job = True
                print(
                    f"LAUNCHED seed={seed} pid={process.pid} "
                    f"physical_gpu={rechecked.index} uuid={rechecked.uuid} "
                    f"log={log_path}",
                    flush=True,
                )
                break

            if launched_job and pending and len(running) < gpu_count:
                continue
            if pending or running:
                time.sleep(args.poll_seconds)

        for seed in args.seeds:
            if not _completed(output_root, seed, expected):
                raise RuntimeError(
                    f"Seed {seed} is missing completion artifacts after the "
                    "controller finished"
                )
        if final_authority is not None:
            _revalidate_final_candidate_lock(final_authority)
    except BaseException:
        if release_authorized and not running:
            _release_generation_lease_exact(lease)
        raise
    else:
        _release_generation_lease_exact(lease)
    print("All de novo benchmark seeds completed.", flush=True)


if __name__ == "__main__":
    main()
