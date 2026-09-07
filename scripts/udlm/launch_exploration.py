"""Run a prospective engineering screen with the audited benchmark launcher.

Each entry is a fresh, disclosed pilot attempt. The child launcher owns GPU
discovery and the generation lease; this controller runs one launcher at a time
with at most two concurrent seeds. Completed entries may be resumed only after
their exact recorded inputs and output hashes have been checked. Failed entries
are retained and never automatically retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(_path))

from scripts.exps.denovo import launch_benchmark as launcher  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def in_repo(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"path escapes worktree: {value}")
    return path


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported engineering protocol schema")
    policy = protocol["gpu_policy"]
    if policy != {
        "max_gpus": 2,
        "max_utilization_percent": 10,
        "min_free_memory_mib": 30000,
    }:
        raise ValueError("screen requires at most two GPUs below 10% utilization")
    seeds = protocol["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 1000 for seed in seeds)
    ):
        raise ValueError("screen seeds must be distinct integers >=1000")
    count = protocol["num_samples"]
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("engineering pilots require 1..100 requested samples")
    entries = protocol["entries"]
    if not entries or len({e["attempt_id"] for e in entries}) != len(entries):
        raise ValueError("entries require unique attempt IDs")
    for entry in entries:
        launcher._validate_attempt_id(entry["attempt_id"])
        launcher._validate_candidate_scope(
            pilot=True, selection_pilot=False, candidate_id=entry["candidate_id"]
        )
        for field in ("config", "checkpoint"):
            in_repo(entry[field])
            if len(entry[f"{field}_sha256"]) != 64:
                raise ValueError(f"missing pinned {field} digest")
    return protocol


def command_for(entry: dict, protocol: dict, output: Path, logs: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/exps/denovo/launch_benchmark.py"),
        "--checkpoint",
        str(in_repo(entry["checkpoint"])),
        "--config",
        str(in_repo(entry["config"])),
        "--pilot",
        "--attempt-id",
        entry["attempt_id"],
        "--candidate-id",
        entry["candidate_id"],
        "--num-samples",
        str(protocol["num_samples"]),
        "--seeds",
        *map(str, protocol["seeds"]),
        "--gpu-count",
        str(protocol["gpu_policy"]["max_gpus"]),
        "--max-utilization-percent",
        "10",
        "--min-free-memory-mib",
        "30000",
        "--poll-seconds",
        "10",
        "--output-root",
        str(output),
        "--log-root",
        str(logs),
    ]


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def output_hashes(output: Path, entry: dict, seeds: list[int]) -> dict:
    artifacts = {}
    for seed in seeds:
        directory = output / entry["attempt_id"] / f"seed_{seed}"
        for filename in ("summary.json", "raw_samples.csv", "failure_receipt.json"):
            path = directory / filename
            if path.is_file():
                artifacts[str(path.relative_to(ROOT))] = digest(path)
    return artifacts


def validate_success_artifacts(
    output: Path, entry: dict, seeds: list[int], artifacts: dict
) -> None:
    for seed in seeds:
        directory = output / entry["attempt_id"] / f"seed_{seed}"
        if str((directory / "failure_receipt.json").relative_to(ROOT)) in artifacts:
            raise RuntimeError("successful entry has a seed failure receipt")
        for filename in ("summary.json", "raw_samples.csv"):
            if str((directory / filename).relative_to(ROOT)) not in artifacts:
                raise RuntimeError("successful entry lacks a required seed artifact")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("output/udlm/engineering_v5")
    )
    parser.add_argument(
        "--log-root", type=Path, default=Path("output/logs/engineering_v5")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = in_repo(args.protocol)
    protocol = load_protocol(protocol_path)
    output, logs = in_repo(args.output_root), in_repo(args.log_root)
    if output == logs or output.is_relative_to(logs) or logs.is_relative_to(output):
        raise ValueError("log/output roots must be separate")
    launcher._require_project_virtual_environment()
    source = launcher._require_clean_pushed_source()
    protocol_digest = digest(protocol_path)
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(protocol_path.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # Hash each distinct checkpoint once before any launch. The audited child
    # independently rechecks checkpoint/config bytes and source at each entry.
    verified = {}
    for entry in protocol["entries"]:
        for field in ("config", "checkpoint"):
            path = in_repo(entry[field])
            if path not in verified:
                verified[path] = digest(path)
            if verified[path] != entry[f"{field}_sha256"]:
                raise ValueError(f"{field} bytes differ from protocol: {path}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "event": "engineering_preview",
                    "protocol_sha256": protocol_digest,
                    "source": source,
                    "gpu_queries": 0,
                    "artifact_mutations": 0,
                    "commands": [
                        command_for(e, protocol, output, logs)
                        for e in protocol["entries"]
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    launcher._require_tmux_for_execution()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("unset CUDA_VISIBLE_DEVICES on the controller")
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    # Exclusive orchestration lock prevents two resumptions racing between
    # sequential child launchers. The audited child owns the GPU lease.
    import fcntl

    with (output / ".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts = output / "controller_receipts"
        receipts.mkdir(exist_ok=True)
        any_failed = False
        for entry in protocol["entries"]:
            if (
                launcher._require_clean_pushed_source() != source
                or digest(protocol_path) != protocol_digest
            ):
                raise RuntimeError(
                    "source or prospective protocol changed during screen"
                )
            identity = {
                "source": source,
                "protocol_sha256": protocol_digest,
                "entry": entry,
                "seeds": protocol["seeds"],
                "num_samples": protocol["num_samples"],
            }
            receipt_path = receipts / f"{entry['attempt_id']}.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["identity"] != identity:
                    raise RuntimeError("existing receipt belongs to different inputs")
                if receipt["artifacts"] != output_hashes(
                    output, entry, protocol["seeds"]
                ):
                    raise RuntimeError("existing output bytes changed")
                if type(receipt["return_code"]) is not int:
                    raise RuntimeError("existing receipt lacks an integer exit status")
                if receipt["return_code"] == 0:
                    validate_success_artifacts(
                        output, entry, protocol["seeds"], receipt["artifacts"]
                    )
                any_failed |= receipt["return_code"] != 0
                print(
                    json.dumps(
                        {
                            "event": "entry_already_terminal",
                            "attempt": entry["attempt_id"],
                        }
                    ),
                    flush=True,
                )
                continue
            if (output / entry["attempt_id"]).exists():
                raise RuntimeError(
                    "unfinished attempt preserved; adjudicate before resuming"
                )
            for field in ("checkpoint", "config"):
                if digest(in_repo(entry[field])) != entry[f"{field}_sha256"]:
                    raise RuntimeError(f"pinned {field} changed before entry launch")
            command = command_for(entry, protocol, output, logs)
            started = stamp()
            print(
                json.dumps(
                    {
                        "event": "entry_start",
                        "attempt": entry["attempt_id"],
                        "time": started,
                    }
                ),
                flush=True,
            )
            with (logs / f"{entry['attempt_id']}-controller.log").open("x") as log:
                result = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
            any_failed |= result.returncode != 0
            artifacts = output_hashes(output, entry, protocol["seeds"])
            validation_error = None
            return_code = result.returncode
            if return_code == 0:
                try:
                    validate_success_artifacts(
                        output, entry, protocol["seeds"], artifacts
                    )
                except RuntimeError as error:
                    return_code = 1
                    validation_error = str(error)
                    any_failed = True
            write_once(
                receipt_path,
                {
                    "identity": identity,
                    "command": command,
                    "started_at_utc": started,
                    "completed_at_utc": stamp(),
                    "return_code": return_code,
                    "process_return_code": result.returncode,
                    "validation_error": validation_error,
                    "artifacts": artifacts,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "entry_terminal",
                        "attempt": entry["attempt_id"],
                        "return_code": return_code,
                    }
                ),
                flush=True,
            )
        print(
            json.dumps(
                {"event": "screen_terminal", "any_failed": any_failed, "time": stamp()}
            ),
            flush=True,
        )
        return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
