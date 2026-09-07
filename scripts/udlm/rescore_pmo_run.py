"""Independently verify a completed saved Fexofenadine PMO run on CPU.

Verification oracle calls occur after the entire optimization campaign, outside
its online budget, with no feedback. Never reads a model or pickle checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
from importlib import metadata
import json
import math
import numbers
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = next(p for p in ROOT.parents if (p / ".venv").is_dir())
TOLERANCE = 1e-12
ORACLE = "fexofenadine_mpo"
PACKAGES = ("pytdc", "rdkit", "numpy", "scipy", "pandas", "scikit-learn")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def finite_score(value):
    require(
        not isinstance(value, bool) and isinstance(value, numbers.Real),
        "Score must be a real scalar",
    )
    value = float(value)
    require(
        math.isfinite(value) and 0 <= value <= 1,
        "Fexofenadine score must be finite and in [0,1]",
    )
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid integer {name}")
    return value


def inside(path, workspace=WORKSPACE):
    path = Path(path).expanduser().resolve()
    require(
        path.is_relative_to(workspace),
        "Evidence/output must stay inside the project workspace",
    )
    return path


class Evidence:
    def __init__(self, workspace=WORKSPACE):
        self.workspace = Path(workspace).resolve()
        self.records = {}

    def read(self, name, path, expected_sha=None, expected_size=None, *, capture=True):
        path = inside(path, self.workspace)
        hasher, chunks = hashlib.sha256(), []
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
                if capture:
                    chunks.append(chunk)
            after = os.fstat(handle.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            all(getattr(before, k) == getattr(after, k) for k in fields),
            f"Input changed while reading: {path}",
        )
        record = dict(
            path=str(path), sha256=hasher.hexdigest(), size_bytes=after.st_size
        )
        require(
            expected_sha is None or record["sha256"] == expected_sha,
            f"Input hash differs: {path}",
        )
        require(
            expected_size is None or record["size_bytes"] == expected_size,
            f"Input size differs: {path}",
        )
        self.records[name] = record
        return b"".join(chunks) if capture else None

    def json(self, name, path, expected_sha=None, expected_size=None):
        def reject_constant(value):
            raise ValueError(f"Nonfinite JSON constant {value}")

        return json.loads(
            self.read(name, path, expected_sha, expected_size),
            parse_constant=reject_constant,
        )

    def recheck(self):
        for name, record in list(self.records.items()):
            self.read(
                name,
                record["path"],
                record["sha256"],
                record["size_bytes"],
                capture=False,
            )


def _path(root, value):
    return Path(value) if Path(value).is_absolute() else root / value


def _one(rows, key, value):
    selected = [row for row in rows if row.get(key) == value]
    require(len(selected) == 1, f"Expected one {key}={value}")
    return selected[0]


def load_completed(inputs, root, protocol_path, entry_id, run_directory):
    """Bind completed controller/config/artifacts before any scoring."""
    protocol = inputs.json("protocol", protocol_path)
    require(
        protocol["schema_version"] == 1 and protocol["oracle"] == ORACLE,
        "Only the declared Fexofenadine PMO panel is supported",
    )
    entry = _one(protocol["entries"], "id", entry_id)
    budget = integer(entry["max_oracle_calls"], "online budget", 1)
    seed = integer(entry["seed"], "seed")
    options = protocol["runner"]
    require(
        options["min_mol_size"] == 20 and options["max_mol_size"] == 40,
        "Fexofenadine verification requires 20..40 atoms",
    )
    controller_root = _path(root, protocol["output_root"])
    terminal = inputs.json(
        "controller_terminal", controller_root / "terminal_manifest.json"
    )
    request = inputs.json(
        "controller_request",
        controller_root / "request_manifest.json",
        terminal["request_sha256"],
    )
    require(
        terminal["status"] == "completed"
        and terminal["final_input_validation"] == "unchanged"
        and terminal["lease_release_authorized"] is True,
        "Campaign is not fully completed with accepted final inputs and resource release",
    )
    require(
        terminal["panel_sha256"] == inputs.records["protocol"]["sha256"],
        "Controller protocol hash differs",
    )
    require(
        request["plan"]["panel"] == protocol
        and request["plan"]["panel_input"]["sha256"] == terminal["panel_sha256"],
        "Controller request panel differs",
    )
    source = terminal["source"]
    require(
        source == request["source"] and source["head"] == source["upstream"],
        "Controller source differs",
    )
    planned_rows = [
        row
        for row in request["plan"]["jobs"]
        if row.get("entry", {}).get("id") == entry_id
    ]
    require(len(planned_rows) == 1, "Expected one planned entry")
    planned = planned_rows[0]
    require(planned["entry"] == entry, "Controller planned entry differs")
    finished = _one(terminal["jobs"], "entry_id", entry_id)
    require(
        finished["status"] == "completed"
        and type(finished["return_code"]) is int
        and finished["return_code"] == 0,
        "PMO job did not complete successfully",
    )
    expected_directory = inside(
        controller_root / "runs" / entry_id / ORACLE / "released" / f"seed_{seed}",
        inputs.workspace,
    )
    require(
        run_directory == expected_directory
        and inside(root / planned["run_relative"], inputs.workspace) == run_directory,
        "Run directory does not match the declared entry",
    )
    values = {}
    artifacts = finished["acceptance"]["artifacts"]
    for name in ("manifest", "summary", "events"):
        filename = "events.jsonl" if name == "events" else name + ".json"
        claim = artifacts[filename]
        path = inside(root / claim["path"], inputs.workspace)
        require(path == run_directory / filename, "Controller artifact path differs")
        values[name] = (
            inputs.read(name, path, claim["sha256"], claim["size_bytes"])
            if name == "events"
            else inputs.json(name, path, claim["sha256"], claim["size_bytes"])
        )
    manifest, summary = values["manifest"], values["summary"]
    config = manifest["config"]
    digest = sha(canonical_bytes(config))
    require(
        config == planned["config"]
        and digest
        == planned["config_sha256"]
        == manifest["config_sha256"]
        == summary["config_sha256"],
        "Saved resolved configuration differs",
    )
    require(
        manifest["status"] == summary["status"] == "completed"
        and summary["checkpoint_consistent"] is True,
        "Saved run is not complete",
    )
    expected_run_id = f"{entry_id}:{ORACLE}:released:seed{seed}"
    require(
        manifest["run_id"] == summary["run_id"] == expected_run_id,
        "Saved run identity differs",
    )
    for key, value in dict(
        experiment_id=entry_id,
        oracle=ORACLE,
        variant="released",
        policy_mode="released",
        parent_control=False,
        gamma=0.0,
        seed=seed,
        max_oracle_calls=budget,
        **options,
    ).items():
        require(
            config[key] == value and type(config[key]) is type(value),
            f"Saved setting differs: {key}",
        )
    require(
        config["matrix_sha256"] == terminal["panel_sha256"],
        "Saved protocol identity differs",
    )
    require(
        manifest["model"]["sha256"]
        == summary["model_sha256"]
        == entry["checkpoint_sha256"],
        "Saved checkpoint digest differs",
    )
    require(
        manifest["extra"]["git"]["commit"] == source["head"],
        "Saved generation source differs",
    )
    for key, value in dict(
        task=ORACLE, variant="released", seed=seed, oracle_budget=budget
    ).items():
        require(manifest[key] == value, f"Manifest {key} differs")
    require(
        integer(manifest["oracle_calls"], "manifest calls")
        == budget
        == finished["acceptance"]["oracle_calls"],
        "Online call budget was not completed",
    )
    require(
        summary["scores"]["all_charged_molecules"]["oracle_calls"] == budget
        and summary["scores"]["all_charged_molecules"]["oracle_budget"] == budget,
        "Summary online budget differs",
    )
    inputs.read(
        "sampling_config",
        _path(root, entry["sampling_config"]),
        entry["sampling_config_sha256"],
        capture=False,
    )
    inputs.read(
        "vocabulary",
        _path(root, entry["vocabulary"]),
        entry["vocabulary_sha256"],
        capture=False,
    )
    sampling = config["pmo_sampling"]
    require(
        sampling["source"]["sha256"] == entry["sampling_config_sha256"]
        and sampling["checkpoint_sha256"] == entry["checkpoint_sha256"],
        "Saved sampling contract differs",
    )
    require(manifest.get("resume_count") == 0, "Verification supports fresh runs only")
    require(
        summary["sampling"]["identity"] == manifest["extra"]["sampling"],
        "Terminal sampling identity differs",
    )
    call_protocol = manifest["extra"]["sampling"]["oracle_call_protocol"]
    require(
        call_protocol
        == {
            "input": "singleton list containing the CachedOracle canonical SMILES",
            "output": "exactly one finite real scalar; booleans rejected",
            "exception_policy": "propagate ordinary non-docking TDC list-path evaluator failures",
            "budget": "CachedOracle charges only after a finite successful return",
        },
        "Original oracle exception/call policy differs",
    )
    for index, claim in enumerate(protocol["input_files"]):
        inputs.read(
            f"oracle_input_{index:03d}",
            _path(root, claim["path"]),
            claim["sha256"],
            capture=False,
        )
    return protocol, entry, manifest, summary, values["events"], source


def reconstruct(events_payload, summary, budget):
    """Rebuild the online score cache from molecular event rows independently."""
    from rdkit import Chem

    ledger, rows, duplicates = {}, [], 0
    lines = events_payload.splitlines()
    require(
        integer(summary["events"], "event count") == len(lines),
        "Saved event count differs",
    )
    require(
        summary["iterations_completed"] == len(lines),
        "Completed iteration count differs",
    )
    for index, line in enumerate(lines):
        event = json.loads(line)
        require(
            type(event["event_index"]) is int
            and event["event_index"] == index
            and type(event["iteration"]) is int
            and event["iteration"] == index,
            "Event/iteration indices are not contiguous",
        )
        require(event["parent_oracle"] is None, "Released panel must not score parents")
        outcome = event["child_oracle"]
        require(
            isinstance(outcome, dict)
            and outcome["valid"] is True
            and type(outcome["charged"]) is bool,
            "Child oracle row is not a valid scored outcome",
        )
        smiles = outcome["canonical_smiles"]
        require(isinstance(smiles, str) and smiles, "Missing canonical SMILES")
        mol = Chem.MolFromSmiles(smiles)
        require(
            mol is not None and Chem.MolToSmiles(mol) == smiles,
            "Charged/cache-hit SMILES is invalid or noncanonical",
        )
        atoms = mol.GetNumAtoms()
        require(20 <= atoms <= 40, "Charged/cache-hit molecule is outside 20..40 atoms")
        require(
            outcome["raw_smiles"] == event["child_smiles"] == smiles,
            "Event raw/canonical child identity differs",
        )
        require(
            event["atom_count"] == event["child_atom_count"] == atoms,
            "Event atom counts differ",
        )
        score = finite_score(outcome["score"])
        call = integer(outcome["call_index"], "call index", 1)
        if outcome["charged"]:
            require(
                outcome["reason"] == "scored"
                and smiles not in ledger
                and call == len(rows) + 1,
                "First-charge order/uniqueness is inconsistent",
            )
            row = dict(
                call_index=call,
                canonical_smiles=smiles,
                atom_count=atoms,
                original_score=score,
            )
            rows.append(row)
            ledger[smiles] = row
        else:
            require(
                outcome["reason"] == "cache_hit" and smiles in ledger,
                "Duplicate has no earlier charge",
            )
            require(
                call == ledger[smiles]["call_index"]
                and score == ledger[smiles]["original_score"],
                "Duplicate does not reference its exact first charged score/index",
            )
            duplicates += 1
        require(
            type(event["oracle_calls"]) is int
            and event["oracle_calls"] == len(rows) <= budget,
            "Event cumulative oracle calls differ",
        )
    require(len(rows) == budget, "Events do not reconstruct the complete online budget")
    return rows, duplicates


def _new_oracle(inputs):
    from tdc import Oracle
    from rdkit.Chem import Crippen, MolSurf
    from rdkit import rdBase

    # Bind the modules actually imported by this verifier to the frozen panel's
    # installed-code pins; hashing another checkout's package files is insufficient.
    pinned = {
        record["path"]
        for name, record in inputs.records.items()
        if name.startswith("oracle_input_")
    }
    require(
        str(Path(inspect.getfile(Oracle)).resolve()) in pinned,
        "Imported TDC Oracle is not pinned by the panel",
    )
    evaluator = Oracle(name=ORACLE)
    require(
        evaluator.name == ORACLE and evaluator.evaluator_func.__name__ == ORACLE,
        "Unexpected oracle implementation",
    )
    for value in (evaluator.evaluator_func, Crippen, MolSurf, rdBase):
        require(
            str(Path(inspect.getfile(value)).resolve()) in pinned,
            "Imported oracle dependency is not pinned by the panel",
        )
    return evaluator


def _git_source():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_run(
    input_root,
    protocol_path,
    entry_id,
    run_directory,
    *,
    evaluator_factory=None,
    workspace=WORKSPACE,
):
    """Return a terminal receipt; synthetic injection never claims real scoring."""
    started = time.perf_counter()
    inputs = Evidence(workspace)
    receipt = dict(
        schema_version=1,
        status="failed",
        error=None,
        evaluation_mode="synthetic_test"
        if evaluator_factory is not None
        else "tdc_fresh_oracle_singleton_list",
        call_scope="postoptimization verification outside the online budget; no oracle feedback to optimization",
        inputs=inputs.records,
        run=dict(entry_id=entry_id, oracle=ORACLE),
        verification=dict(
            unique_charged_count=0,
            verification_calls=0,
            completed_verification_calls=0,
            duplicates_checked=0,
            mismatch_count=0,
            tolerance=TOLERANCE,
            max_abs_error=None,
            rows=[],
        ),
        source=dict(
            git_commit=_git_source(),
            python_executable=sys.executable,
            python_version=platform.python_version(),
        ),
        packages={name: metadata.version(name) for name in PACKAGES},
        runtime=dict(
            started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid()
        ),
    )
    lock = None
    try:
        root, directory = (
            inside(input_root, inputs.workspace),
            inside(run_directory, inputs.workspace),
        )
        inputs.read("verifier_source", Path(__file__), capture=False)
        lock = (directory / ".run.lock").open("rb")
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        protocol, entry, manifest, summary, events, generation_source = load_completed(
            inputs, root, _path(root, protocol_path), entry_id, directory
        )
        receipt["run"].update(
            run_id=manifest["run_id"],
            seed=entry["seed"],
            optimization_status=summary["status"],
            online_budget=entry["max_oracle_calls"],
            generation_source=generation_source,
            checkpoint_sha256=entry["checkpoint_sha256"],
        )
        rows, duplicates = reconstruct(events, summary, entry["max_oracle_calls"])
        verification = receipt["verification"]
        verification.update(
            unique_charged_count=len(rows), duplicates_checked=duplicates
        )
        # All original identities and molecule/ledger checks precede this factory.
        evaluator = (
            evaluator_factory()
            if evaluator_factory is not None
            else _new_oracle(inputs)
        )
        for row in rows:
            verification["verification_calls"] += 1
            result = evaluator([row["canonical_smiles"]])
            require(
                type(result) is list and len(result) == 1,
                "Verification oracle must return one score",
            )
            score = finite_score(result[0])
            error = abs(score - row["original_score"])
            verification["completed_verification_calls"] += 1
            verification["max_abs_error"] = max(
                verification["max_abs_error"] or 0.0, error
            )
            verification["rows"].append(
                dict(row, rescored_score=score, absolute_error=error)
            )
            verification["mismatch_count"] += int(error > TOLERANCE)
        inputs.recheck()
        require(
            verification["mismatch_count"] == 0,
            "Saved oracle score differs beyond absolute tolerance 1e-12",
        )
        receipt["status"] = "verified"
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        if lock is not None:
            lock.close()
        receipt["runtime"].update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            seconds=time.perf_counter() - started,
        )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = inside(args.output)
    require(not output.exists(), "Verification output must be a fresh file")
    require(
        not output.is_relative_to(args.run_directory.resolve()),
        "Verification output must not modify the original run directory",
    )
    # Reserve the output before any evaluation; repeated invocation cannot incur
    # verification calls and then discover an occupied output namespace.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        receipt = verify_run(
            args.input_root, args.protocol, args.entry_id, args.run_directory
        )
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            dict(
                status=receipt["status"],
                verification_calls=receipt["verification"]["verification_calls"],
                output=str(output),
            )
        )
    )
    return 0 if receipt["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
