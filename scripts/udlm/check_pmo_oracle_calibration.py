"""Fixed CPU sanity check for V14's deterministic oracle; no optimization feedback."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "experiments/udlm/diagnostics/pmo_oracle_inputs_20260907/manifest.json"
OUTPUT = ROOT / "experiments/udlm/diagnostics/pmo_oracle_inputs_20260907/calibration.json"


def fingerprint(path):
    payload = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(MANIFEST.read_text())
    for wanted in manifest["files"].values():
        if fingerprint(wanted["path"]) != wanted:
            raise RuntimeError(f"Pinned oracle input changed: {wanted['path']}")
    from tdc import Oracle
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors
    from scripts.exps.pmo.udlm_sampling import singleton_list_oracle

    reference = manifest["task_definition"]["reference_smiles"]
    cases = [("ethanol", "CCO"), ("benzene", "c1ccccc1"), ("fixed_reference", reference)]
    evaluator = Oracle(name="fexofenadine_mpo")
    strict = singleton_list_oracle(evaluator)
    ref_fp = AllChem.GetAtomPairFingerprint(Chem.MolFromSmiles(reference), maxLength=10)
    rows = []
    started = time.monotonic()
    for label, smiles in cases:
        molecule = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetAtomPairFingerprint(molecule, maxLength=10)
        similarity = float(DataStructs.TanimotoSimilarity(fp, ref_fp))
        tpsa = float(Descriptors.TPSA(molecule))
        logp = float(Descriptors.MolLogP(molecule))
        factors = [min(similarity / 0.8, 1.0),
                   math.exp(-0.5 * (min(tpsa - 90.0, 0.0) / 10.0) ** 2),
                   math.exp(-0.5 * max(logp - 4.0, 0.0) ** 2)]
        expected = math.prod(factors) ** (1.0 / 3.0)
        scalar, observed = float(evaluator(smiles)), strict(smiles)
        if not all(math.isfinite(value) for value in (expected, scalar, observed)):
            raise RuntimeError("Non-finite calibration score")
        if scalar != observed or abs(expected - observed) > 1e-12:
            raise RuntimeError(f"Calibration mismatch: {label}")
        rows.append({"label": label, "smiles": smiles, "similarity": similarity,
                     "TPSA": tpsa, "logP": logp, "factors": factors,
                     "manual_formula": expected, "tdc_scalar": scalar,
                     "strict_singleton_list": observed,
                     "absolute_formula_error": abs(expected - observed)})
    for wanted in manifest["files"].values():
        if fingerprint(wanted["path"]) != wanted:
            raise RuntimeError("Oracle source changed during calibration")
    result = {"schema_version": 1, "status": "passed", "purpose":
              "fixed-input oracle calibration only; no model or optimization and no tuning feedback",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source": fingerprint(__file__), "manifest": fingerprint(MANIFEST),
              "sampler_adapter": fingerprint(ROOT / "scripts/exps/pmo/udlm_sampling.py"),
              "cases": rows, "fixed_input_count": 3, "tdc_evaluator_calls": 6,
              "manual_descriptor_checks": 3, "online_optimization_calls": 0,
              "seed": None, "device": "CPU; CUDA_VISIBLE_DEVICES empty",
              "elapsed_seconds": time.monotonic() - started,
              "maximum_absolute_formula_error": max(row["absolute_formula_error"] for row in rows)}
    with OUTPUT.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(OUTPUT),
                      "sha256": fingerprint(OUTPUT)["sha256"],
                      "maximum_absolute_formula_error": result["maximum_absolute_formula_error"]}))


if __name__ == "__main__":
    main()
