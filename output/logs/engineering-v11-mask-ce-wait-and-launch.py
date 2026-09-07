"""Wait for the authorized two-GPU launch capacity, then invoke V11 once."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))
from scripts.udlm import launch_engineering_training as engine

EXPECTED_SOURCE = sys.argv[1]
STATUS = ROOT / "output/logs/engineering-v11-mask-ce-wait-status.json"
PROBES = ROOT / "output/logs/engineering-v11-mask-ce-availability.jsonl"

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def check_source():
    source = engine.benchmark._require_clean_pushed_source()
    if source != {"head": EXPECTED_SOURCE, "upstream": EXPECTED_SOURCE}:
        raise RuntimeError("V11 waiting source no longer equals its clean pushed launch revision")

check_source()
engine.benchmark._require_project_virtual_environment()
engine.benchmark._require_tmux_for_execution()
started = now()
status = {"started_at_utc": started, "expected_source": EXPECTED_SOURCE,
          "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "controller_invocations": 0, "training_returncode": None}
try:
    with PROBES.open("x") as stream:
        while True:
            check_source()
            states = engine.audited.probe_all_gpus()
            eligible = [s for s in states if not s.rejection_reasons(
                max_utilization_percent=10, min_free_memory_mib=30000)]
            event = {"at": now(), "eligible_count": len(eligible),
                     "gpus": [engine.asdict(s) for s in states]}
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps({"at": event["at"], "eligible_count": len(eligible),
                              "event": "capacity_probe"}), flush=True)
            if len(eligible) >= 2:
                engine.select_gpus(states, 2)
                break
            time.sleep(30)
    check_source()
    status["controller_started_at_utc"] = now()
    status["controller_invocations"] = 1
    result = subprocess.run([sys.executable, "-u", "scripts/udlm/launch_mask_prior_training.py",
                             "--gpu-count", "2"])
    status["training_returncode"] = result.returncode
    status["status"] = "controller_completed" if result.returncode == 0 else "controller_failed"
except BaseException as error:
    status["status"] = "failed"
    status["error"] = f"{type(error).__name__}: {error}"
    raise
finally:
    status["finished_at_utc"] = now()
    with STATUS.open("x") as stream:
        json.dump(status, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(status, sort_keys=True), flush=True)
if status["training_returncode"]:
    raise SystemExit(status["training_returncode"])
