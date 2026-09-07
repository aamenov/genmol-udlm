"""CPU diagnostic of frozen independently rescored V9 rows; JSON to stdout."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.udlm.audit_molecular_failures import (
    DEFINITIONS, Inputs, parse_rows, row_diagnostics, tally,
)

inputs = Inputs(ROOT)
inputs.read(Path(__file__), role="V9 audit wrapper")
inputs.read(ROOT / "scripts/udlm/audit_molecular_failures.py", role="tested diagnostic definitions")
report = json.loads(inputs.read(
    ROOT / "output/udlm/engineering_v9_reports/complete/report.json",
    expected="b880c8bbad2e5e1d5f3639ba8f6848da934c87c2ec44d7e9c466f46a941c7fb6",
    role="complete independent V9 report",
))
assert report["status"] == "complete"
assert report["accounting"]["independently_rescored_requests"] == 800
per_seed, grouped = [], {}
for run in report["runs"]:
    assert run["independent_rescore"]["status"] == "exact_match"
    artifact = run["artifacts"]["raw_samples.csv"]
    rows = parse_rows(inputs.read(ROOT / artifact["relative_path"], expected=artifact["sha256"], role="V9 raw rows"), 100)
    diagnostics = [row_diagnostics(row) for row in rows]
    counts = tally(diagnostics)
    for branch, prefix in (("released_comparable", "released"), ("strict", "strict")):
        assert counts[prefix + "_quality"] / 100 == run["metrics"][branch]["quality"]
    assert counts["strict_valid"] / 100 == run["metrics"]["strict"]["validity"]
    grouped.setdefault(run["config_id"], []).extend(diagnostics)
    per_seed.append({"config_id": run["config_id"], "seed": run["seed"], "counts": counts})
result = {
    "schema_version": 1,
    "study_id": report["study_id"],
    "scope": "Four settings, two seeds of 100 each. Configuration counts are descriptive; no cross-setting model estimate or causal claim.",
    "definitions": {key: DEFINITIONS[key] for key in ("ring_parity", "parentheses", "syntax_scope", "strict", "repaired", "quality")},
    "per_seed": per_seed,
    "per_configuration": {key: tally(value) for key, value in grouped.items()},
    "input_manifest": inputs.manifest(),
}
print(json.dumps(result, indent=2, sort_keys=True))
