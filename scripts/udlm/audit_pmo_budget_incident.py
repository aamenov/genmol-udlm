"""Postterminal V14 ledger accounting, without model/pickle/oracle evaluation.

This addendum counts only validated newline-terminated event records. It never
accepts a partial budget, computes a ranking, or estimates unrecorded work.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.udlm import report_pmo_pilot as historical  # noqa: E402

PANEL_SHA256 = "8f42ce172b392d27260317f3db7e07ce5c5cbe528eb448234388fcc34e0f4fda"
REPLAY_SHA256 = "01e6de744f500dda8a4f4996052bd8461227fd547edafcd8c477b2d2178174ed"
LEASE = Path("output/.single_generation_job.lock")
require, same = historical.require, historical.same


def encode(value):
    return historical.encoded(value, pretty=True)


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON number")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def pin(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        "invalid SHA-256",
    )
    return value


class Inputs(historical.Inputs):
    def read(self, path, expected=None):
        resolved = (self.root / path).resolve(strict=True)
        require(resolved.is_relative_to(PROJECT), "input escapes project")
        before = resolved.stat()
        data = super().read(path, expected)
        after = resolved.stat()
        require(
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "input changed while read",
        )
        return data

    def json(self, path, expected=None):
        return strict_json(self.read(path, expected))


def prefix_counts(data, config, population):
    """Validate complete lines; preserve an unterminated tail by hash and size."""
    boundary = data.rfind(b"\n") + 1
    prefix, tail = data[:boundary], data[boundary:]
    lines = prefix.split(b"\n")[:-1]
    result = {
        "status": "unvalidated_prefix",
        "complete_jsonl_lines": len(lines),
        "complete_line_prefix_sha256": historical.digest(prefix),
        "complete_line_prefix_bytes": len(prefix),
        "unterminated_tail_bytes": len(tail),
        "unterminated_tail_sha256": historical.digest(tail),
        "durable_unique_charges": None,
        "cache_hit_events": None,
        "accepted_event_sampling": None,
        "last_event_elapsed_seconds": None,
        "count_scope": "validated complete-line ledger only; lower bound on work actually performed",
    }
    try:
        events = [strict_json(line) for line in lines]
        require(
            len(events) <= config["max_iterations"],
            "event prefix exceeds iteration budget",
        )
        replay = historical.replay(events, config, population, complete=False)
        # The preserved historical helper also computes curves internally; none
        # of its AUC/score/ranking fields are included in this incident artifact.
        result.update(
            status="validated_complete_line_prefix",
            durable_unique_charges=len(replay["charged_rows"]),
            cache_hit_events=replay["duplicate_events"],
            accepted_proposal_attempts=replay["accepted_proposal_attempts"],
            accepted_event_sampling=replay["accepted_event_sampling"],
            remasking_events=replay["remasking_events"],
            charged_remasking_events=replay["charged_remasking_events"],
            charged_remasking_with_nfe=replay["charged_remasking_with_nfe"],
            last_event_elapsed_seconds=events[-1]["elapsed_seconds"]
            if events
            else None,
            validated_charged_identity_sha256=replay["cache_sha256"],
        )
    except (ValueError, KeyError, TypeError, IndexError, UnicodeDecodeError) as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def bind_manifest(inputs, manifest, job, outcome, panel_sha, source):
    entry, config = job["entry"], job["config"]
    same(manifest["config"], config, "resolved run config")
    config_sha = historical.digest(historical.encoded(config))
    same(job["config_sha256"], config_sha, "planned config digest")
    same(manifest["config_sha256"], config_sha, "manifest config digest")
    expected = {
        "experiment_id": entry["id"],
        "matrix_sha256": panel_sha,
        "oracle": "fexofenadine_mpo",
        "seed": entry["seed"],
        "variant": "released",
        "policy_mode": "released",
        "parent_control": False,
        "gamma": 0.0,
        "warmup": 1000,
        "legacy_warmup_off_by_one": True,
        "population_size": 100,
        "reporting_frequency": 100,
        "max_oracle_calls": 2000,
        "max_iterations": 5000,
        "min_mol_size": 20,
        "max_mol_size": 40,
    }
    for key, value in expected.items():
        same(config[key], value, "fixed config " + key)
    same(manifest["resume_count"], 0, "fresh run")
    same(
        manifest["run_id"],
        f"{entry['id']}:fexofenadine_mpo:released:seed{entry['seed']}",
        "run identity",
    )
    same(manifest["extra"]["git"]["commit"], source["head"], "child source")
    same(manifest["extra"]["git"]["dirty"], False, "clean child source")
    same(manifest["model"]["sha256"], entry["checkpoint_sha256"], "child checkpoint")
    same(job["checkpoint"]["sha256"], entry["checkpoint_sha256"], "planned checkpoint")
    receipt = manifest["extra"]["sampling"]
    same(receipt["checkpoint"], job["checkpoint"], "sampling checkpoint receipt")
    same(receipt["contract"], config["pmo_sampling"], "sampling contract")
    same(
        receipt["oracle_call_protocol"], historical.ORACLE_PROTOCOL, "scoring protocol"
    )
    same(
        receipt["contract"]["source"]["sha256"],
        entry["sampling_config_sha256"],
        "sampling YAML",
    )
    inputs.record(receipt["contract"]["source"])
    law = receipt["contract"]["configuration"]
    same(
        receipt["contract"]["configuration_sha256"],
        historical.digest(historical.encoded(law)),
        "sampling configuration digest",
    )
    arm, _seed = historical.slot(entry)
    expected_law = {
        "diffusion_type": "mdlm" if arm == "mdlm" else "udlm",
        "softmax_temp": 1.2 if arm == "mdlm" else 0.5,
        "randomness": 2.0 if arm == "mdlm" else 0.0,
        "min_add_len": 18,
    }
    if arm != "mdlm":
        expected_law.update(
            num_steps=128,
            raw_loo_top_p=1.0,
            inference_eps=1e-5,
            exclude_special_tokens=False,
            prior_variant="mask_rich_empirical"
            if arm == "mask_ce"
            else "schedule_uniform",
        )
    for key, value in expected_law.items():
        same(law[key], value, "sampling law " + key)
    same(
        law.get("parameterization", "raw_loo"),
        "x0_denoiser" if arm == "mask_ce" else "raw_loo",
        "logit interpretation",
    )
    require(
        not law.get("gibbs_corrector", False)
        and law.get("temperature_space", "raw_loo") == "raw_loo",
        "undeclared sampling mode",
    )
    for record in receipt["implementation_inputs"].values():
        inputs.record(record)
    same(
        manifest["extra"]["vocabulary"]["sha256"],
        entry["vocabulary_sha256"],
        "vocabulary",
    )
    environment = outcome["environment"]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        same(environment[key], "1", "thread environment")
    same(environment["PYTHONHASHSEED"], "0", "hash seed")
    same(environment["CUDA_VISIBLE_DEVICES"], outcome["gpu"]["uuid"], "GPU mapping")
    same(
        manifest["extra"]["runtime"]["cuda_visible_devices"],
        outcome["gpu"]["uuid"],
        "child GPU mapping",
    )
    same(manifest["extra"]["runtime"]["python_hash_seed"], "0", "child hash seed")
    return config


def build_audit(
    input_root, panel_path, panel_sha, terminal_sha, pipeline_path, pipeline_sha
):
    for value in (panel_sha, terminal_sha, pipeline_sha):
        pin(value)
    require(panel_sha == PANEL_SHA256, "not the pinned V14 panel")
    inputs = Inputs(input_root)
    require(not os.path.lexists(inputs.root / LEASE), "generation lease remains")
    # These gates precede every child ledger read.
    pipeline = inputs.json(pipeline_path, pipeline_sha)
    same(pipeline["source"], historical.GENERATION_SOURCE, "pipeline source")
    require(
        type(pipeline["return_code"]) is int, "pipeline has no terminal return code"
    )
    panel = inputs.json(panel_path, panel_sha)
    same(panel["oracle"], "fexofenadine_mpo", "fixed task")
    require(
        len(panel["entries"]) == 6 and len({e["id"] for e in panel["entries"]}) == 6,
        "six unique slots required",
    )
    require(
        {historical.slot(e) for e in panel["entries"]}
        == {(a, s) for a in historical.CHECKPOINTS for s in historical.SEEDS},
        "panel slots differ",
    )
    output = Path(panel["output_root"])
    terminal = inputs.json(output / "terminal_manifest.json", terminal_sha)
    require(terminal["status"] in ("completed", "failed"), "controller is not terminal")
    same(
        pipeline["return_code"] == 0,
        terminal["status"] == "completed",
        "pipeline/controller outcome",
    )
    same(terminal["panel_sha256"], panel_sha, "terminal panel")
    same(
        terminal["lease_release_authorized"],
        True,
        "controller cleanup did not authorize release",
    )
    request = inputs.json(output / "request_manifest.json", terminal["request_sha256"])
    same(request["source"], terminal["source"], "request source")
    for key in ("head", "upstream"):
        same(request["source"][key], historical.GENERATION_SOURCE, "frozen source")
    same(request["plan"]["panel"], panel, "requested panel")
    same(request["plan"]["panel_input"]["sha256"], panel_sha, "requested panel hash")
    planned = request["plan"]["jobs"]
    same([job["entry"] for job in planned], panel["entries"], "planned slots")
    outcomes = {row["entry_id"]: row for row in terminal["jobs"]}
    require(
        len(outcomes) == len(terminal["jobs"])
        and set(outcomes) <= {e["id"] for e in panel["entries"]},
        "unknown/duplicate outcomes",
    )
    unlaunched = [
        j["entry"]["id"]
        for j in planned
        if outcomes.get(j["entry"]["id"], {}).get("pid") is None
    ]
    same(terminal["unlaunched_entry_ids"], unlaunched, "unlaunched accounting")
    inputs.read(Path(__file__))
    inputs.read(Path(historical.__file__), REPLAY_SHA256)
    inputs.read(ROOT / "tests/test_udlm_pmo_budget_incident.py")
    runs = []
    for job in planned:
        entry = job["entry"]
        arm, seed = historical.slot(entry)
        outcome = outcomes.get(entry["id"])
        directory = (
            output / "runs" / entry["id"] / "fexofenadine_mpo/released" / f"seed_{seed}"
        )
        same(job["run_relative"], str(directory), "run namespace")
        row = {
            "entry_id": entry["id"],
            "arm": arm,
            "seed": seed,
            "requested_oracle_budget": 2000,
            "controller_status": outcome["status"] if outcome else "unlaunched",
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "ledger": None,
            "addendum_grants_acceptance": False,
            "controller_claimed_accepted_calls": (
                historical.integer(
                    outcome["acceptance"]["oracle_calls"], "controller accepted calls"
                )
                if outcome and outcome["status"] == "completed"
                else 0
            ),
            "ranking_eligible": False,
            "artifacts": {},
        }
        runs.append(row)
        if entry["id"] in unlaunched:
            require(
                not (inputs.root / directory).exists(),
                "unlaunched slot has child evidence",
            )
            row["recorded_work_status"] = "not_launched_no_child_work_recorded"
            row.update(
                durable_unique_charges=0, cache_hit_events=0, complete_jsonl_lines=0
            )
            continue
        historical.integer(outcome["pid"], "child PID", 1)
        require(outcome["status"] in ("failed", "completed"), "child not terminal")
        same(outcome["command"], job["command"], "child command")
        launch = inputs.json(
            output / (entry["id"] + ".launch.json"), outcome["launch_sha256"]
        )
        for key in ("entry_id", "command", "gpu", "environment"):
            same(launch[key], outcome[key], "launch " + key)
        row.update(
            pid=outcome["pid"],
            return_code=outcome["return_code"],
            timed_out=outcome.get("timed_out", False),
            subprocess_seconds=historical.finite(outcome["subprocess_seconds"]),
            gpu=outcome["gpu"],
            environment=outcome["environment"],
        )
        retained = {}
        for name in ("manifest.json", "summary.json", "events.jsonl"):
            path = directory / name
            if (inputs.root / path).is_file():
                expected = (
                    outcome.get("acceptance", {})
                    .get("artifacts", {})
                    .get(name, {})
                    .get("sha256")
                )
                retained[name] = inputs.read(path, expected)
                row["artifacts"][name] = inputs.records[
                    str((inputs.root / path).resolve())
                ]
        if not {"manifest.json", "events.jsonl"} <= retained.keys():
            row["recorded_work_status"] = "missing_manifest_or_event_ledger"
            continue
        try:
            manifest = strict_json(retained["manifest.json"])
            config = bind_manifest(
                inputs, manifest, job, outcome, panel_sha, request["source"]
            )
            vocab = inputs.read(entry["vocabulary"], entry["vocabulary_sha256"])
            row["ledger"] = prefix_counts(
                retained["events.jsonl"],
                config,
                historical.initial_population(vocab, 100),
            )
            row["recorded_work_status"] = row["ledger"]["status"]
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
            row["recorded_work_status"] = "unvalidated_run_identity"
            row["error"] = {"type": type(error).__name__, "message": str(error)}
    inputs.recheck()
    require(not os.path.lexists(inputs.root / LEASE), "generation lease reappeared")
    valid = [
        r["ledger"]
        for r in runs
        if r["ledger"] and r["ledger"]["status"] == "validated_complete_line_prefix"
    ]
    audit = {
        "schema_version": 1,
        "artifact_kind": "v14_durable_budget_incident_addendum",
        "status": "diagnostic_only_unrankable",
        "source": request["source"],
        "controller_status": terminal["status"],
        "controller_error": terminal.get("error"),
        "controller_final_input_validation": terminal.get("final_input_validation"),
        "pipeline_return_code": pipeline["return_code"],
        "runs": runs,
        "accounting": {
            "declared_slots": 6,
            "requested_oracle_budget": 12000,
            "unlaunched_slots": len(unlaunched),
            "validated_ledgers": len(valid),
            "durable_unique_charge_lower_bound": sum(
                r["durable_unique_charges"] for r in valid
            ),
            "controller_claimed_accepted_calls": sum(
                r["controller_claimed_accepted_calls"] for r in runs
            ),
        },
        "caveats": [
            "Counts describe durable newline-terminated records with validated event/cache/population arithmetic; no chemistry or oracle was reevaluated.",
            "Termination may lose an in-flight oracle event or compute work. Durable charge counts are lower bounds, never exact actual oracle totals; no upper bound is inferred.",
            "Accepted-event NFE omits rejected proposals and in-flight work. Last-event time differs from controller subprocess wall time.",
            "An unterminated tail is excluded with its exact byte count and SHA-256; malformed complete lines invalidate the ledger count instead of being silently skipped.",
            "No partial-budget AUC, performance ranking, new benchmark acceptance, model checkpoint read, or state-pickle read belongs to this addendum.",
        ],
    }
    return audit, inputs


def publish(audit, inputs, output):
    output = Path(output).resolve()
    require(output.is_relative_to(PROJECT), "output escapes project")
    csv_text = io.StringIO(newline="")
    fields = [
        "entry_id",
        "arm",
        "seed",
        "controller_status",
        "recorded_work_status",
        "durable_unique_charges",
        "cache_hit_events",
        "complete_jsonl_lines",
        "unterminated_tail_bytes",
        "generation_calls",
        "backbone_evaluations",
        "subprocess_seconds",
        "last_event_elapsed_seconds",
    ]
    writer = csv.DictWriter(csv_text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in audit["runs"]:
        ledger = row.get("ledger") or {}
        values = {**row, **ledger, **(ledger.get("accepted_event_sampling") or {})}
        writer.writerow({key: values.get(key) for key in fields})
    files = {
        "audit.json": encode(audit),
        "runs.csv": csv_text.getvalue().encode(),
        "audit_source.py": Path(__file__).read_bytes(),
        "replay_source.py": Path(historical.__file__).read_bytes(),
        "audit_tests.py": (
            ROOT / "tests/test_udlm_pmo_budget_incident.py"
        ).read_bytes(),
    }
    inputs.recheck()
    require(
        not os.path.lexists(inputs.root / LEASE),
        "generation lease reappeared before publication",
    )
    manifest = {
        "schema_version": 1,
        "inputs": sorted(inputs.records.values(), key=lambda r: r["path"]),
        "outputs": {
            name: {"sha256": historical.digest(data), "size_bytes": len(data)}
            for name, data in files.items()
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in {**files, "manifest.json": encode(manifest)}.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input-root",
        "panel",
        "panel-sha256",
        "terminal-sha256",
        "pipeline",
        "pipeline-sha256",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    audit, inputs = build_audit(
        args.input_root,
        args.panel,
        args.panel_sha256,
        args.terminal_sha256,
        args.pipeline,
        args.pipeline_sha256,
    )
    manifest = publish(audit, inputs, args.output)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "audit_sha256": manifest["outputs"]["audit.json"]["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
