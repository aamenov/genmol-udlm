"""Execute the separately pinned four-artifact V15 CPU export panel once."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def record(path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(PROJECT):
        raise ValueError("artifact outside project")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "size_bytes": path.stat().st_size}


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encode(value))


def now():
    return datetime.now(timezone.utc).isoformat()


def source(root, expected=None):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    head = git("rev-parse", "HEAD")
    if git("status", "--porcelain") or git("rev-parse", "@{upstream}") != head:
        raise ValueError("source must remain clean and pushed")
    if expected is not None and head != expected:
        raise ValueError("source revision changed")
    return head


def check_inputs(plan, own_head):
    source(ROOT, own_head)
    source(plan["exporter_root"], plan["exporter_revision"])
    for expected in plan["inputs"]:
        if record(expected["path"]) != expected:
            raise ValueError(f"input changed: {expected['path']}")


def group_exists(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_group(process, seconds):
    start = time.monotonic()
    while group_exists(process.pid) and time.monotonic() - start < seconds:
        process.poll()
        time.sleep(0.25)
    process.poll()
    return {"process_group_id": process.pid, "elapsed_seconds": time.monotonic() - start,
            "group_present_at_end": group_exists(process.pid)}


def stop(process):
    result = {"pid": process.pid, "term_sent": False, "kill_sent": False}
    for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if not group_exists(process.pid):
            break
        try:
            os.killpg(process.pid, sig)
            result["term_sent" if sig == signal.SIGTERM else "kill_sent"] = True
        except ProcessLookupError:
            break
        wait_group(process, seconds)
    process.poll()
    return {**result, "returncode": process.returncode,
            "group_present_at_end": group_exists(process.pid)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    if Path(sys.prefix).resolve() != PROJECT / ".venv":
        raise ValueError("use project .venv")
    own_head = source(ROOT)
    plan_record = record(plan_path)
    plan = json.loads(plan_path.read_text())
    check_inputs(plan, own_head)
    output = Path(plan["output_root"]).resolve()
    if not output.is_relative_to(PROJECT):
        raise ValueError("output outside project")
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    cpu_environment = {
        "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4", "PYTHONHASHSEED": "0",
    }
    environment.update(cpu_environment)
    request = {
        "schema_version": 1, "operation": "fixed_v15_cpu_transfer_exports",
        "source_revision": own_head, "plan": plan_record,
        "wrapper": record(__file__), "started_at": now(),
        "cpu_environment": cpu_environment, "gpu_jobs": 0,
        "model_forwards": 0, "optimizer_updates": 0, "molecular_oracle_calls": 0,
    }
    write(output / "request.json", request)
    terminal = {"schema_version": 1, "status": "failed", "request": record(output / "request.json"), "entries": []}
    started = time.monotonic()
    current = None
    process = None
    try:
        backbone_state = None
        for entry in plan["entries"]:
            check_inputs(plan, own_head)
            if record(plan_path) != plan_record:
                raise ValueError("plan changed")
            current = {"entry_id": entry["entry_id"], "status": "failed", "started_at": now()}
            terminal["entries"].append(current)
            target = output / entry["entry_id"]
            command = [sys.executable, "-u", str(Path(plan["exporter_root"]) / "scripts/udlm/export_zero_update_transfer.py"),
                       "--config", entry["config"], "--source-checkpoint", plan["source_checkpoint"],
                       "--expected-source-sha256", plan["source_checkpoint_sha256"],
                       "--seed", str(plan["initialization_seed"]), "--output-dir", str(target)]
            current["command"] = command
            write(output / f"{entry['entry_id']}.intent.json", {**current, "cpu_environment": cpu_environment})
            job_start = time.monotonic()
            with (output / f"{entry['entry_id']}.log").open("xb") as log:
                process = subprocess.Popen(command, cwd=plan["exporter_root"], env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                current["pid"] = process.pid
                try:
                    current["returncode"] = process.wait(timeout=plan["child_timeout_seconds"])
                except BaseException:
                    current["cleanup"] = stop(process)
                    current["returncode"] = process.returncode
                    raise
                finally:
                    current["runtime_seconds"] = time.monotonic() - job_start
            current["process_group_exit_grace"] = wait_group(process, 15)
            if current["process_group_exit_grace"]["group_present_at_end"]:
                raise RuntimeError("owned export process group remained after exit grace")
            current["log"] = record(output / f"{entry['entry_id']}.log")
            if current["returncode"] != 0:
                raise RuntimeError(f"export child failed: {entry['entry_id']}")
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            metadata = manifest["transfer_metadata"]
            if (manifest["status"] != "completed" or metadata["udlm_optimizer_updates"] != 0
                    or metadata["udlm_example_exposures"] != 0 or metadata["trained_as_ct_or_ce"] is not False
                    or metadata["selected_parameterization"] != entry["parameterization"]
                    or metadata["initialization_seed"] != plan["initialization_seed"]
                    or metadata["source_mdlm"]["checkpoint"]["sha256"] != plan["source_checkpoint_sha256"]
                    or metadata["exporter_source"]["head"] != plan["exporter_revision"]
                    or manifest["validation"]["ema"]["num_updates"] != 0):
                raise ValueError("completed export identity differs from fixed panel")
            config = json.loads((target / "resolved_config.json").read_text())
            if config != json.loads(Path(entry["config"]).read_text()):
                raise ValueError("resolved export config differs")
            for artifact in manifest["outputs"].values():
                if record(artifact["path"]) != artifact:
                    raise ValueError("export output changed")
            state = {name: value for name, value in manifest["validation"]["state"].items() if name.startswith("backbone.")}
            if not state or (backbone_state is not None and backbone_state != state):
                raise ValueError("all four complete backbone tensor tables must be identical")
            backbone_state = state
            check_inputs(plan, own_head)
            current.update(status="completed", manifest=record(manifest_path), checkpoint=manifest["checkpoint"], finished_at=now())
            write(output / f"{entry['entry_id']}.exit.json", current)
            print(json.dumps({"entry_id": entry["entry_id"], "status": "completed", "checkpoint": manifest["checkpoint"]}), flush=True)
        if record(plan_path) != plan_record:
            raise ValueError("plan changed after exports")
        terminal.update(status="completed", identical_all_four_backbone_state=True,
                        backbone_state_sha256=hashlib.sha256(encode(backbone_state)).hexdigest(),
                        acceptance_scope="inference_artifact_consistency_only_no_molecular_evidence")
    except BaseException as error:
        if process is not None:
            cleanup = stop(process)
            if current is not None:
                current.setdefault("cleanup", cleanup)
        terminal["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        done = {entry["entry_id"] for entry in terminal["entries"]}
        terminal["entries"].extend({"entry_id": entry["entry_id"], "status": "unlaunched"} for entry in plan["entries"] if entry["entry_id"] not in done)
        terminal.update(finished_at=now(), runtime_seconds=time.monotonic() - started)
        write(output / "terminal.json", terminal)


if __name__ == "__main__":
    main()
