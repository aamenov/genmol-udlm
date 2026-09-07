"""Run the actual child launcher's CPU-only preflight for every configuration."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.udlm.launch_exploration import command_for, digest, load_protocol
from scripts.exps.denovo.launch_benchmark import _require_clean_pushed_source


def main():
    source = _require_clean_pushed_source()
    protocol_path = ROOT / "experiments/udlm/protocols/followthrough_resolution_r1.json"
    protocol = load_protocol(protocol_path)
    for entry in protocol["entries"]:
        for field in ("config", "checkpoint"):
            if digest(ROOT / entry[field]) != entry[field + "_sha256"]:
                raise ValueError(f"{field} digest differs from the protocol")
        command = command_for(entry, protocol,
                              ROOT / "output/udlm/followthrough_resolution_r1",
                              ROOT / "output/logs/followthrough_resolution_r1")
        result = subprocess.run([*command, "--dry-run"], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Child preflight failed: {entry['attempt_id']}")
        print(json.dumps({"event": "child_preflight_passed", "attempt": entry["attempt_id"],
                          "source": source, "gpu_queries": 0}), flush=True)


if __name__ == "__main__":
    main()
