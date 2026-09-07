"""Audit saved V14 evidence and render a report without models or chemistry.

State pickles and model checkpoints are never deserialized. The controller's
pinned checkpoint receipts supply model identity. A separate verification
receipt supplies chemistry/oracle agreement; event replay alone cannot do so.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import importlib.metadata
import platform
import json
import math
from pathlib import Path
import statistics
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
GENERATION_SOURCE = "48473d4febbd06d9bc07986ca96ceb93926c91ec"
CHECKPOINTS = {
    "mdlm": "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6",
    "mask_ce": "62299de99d8c003e3776215efce9643cca196091a6284f351de2b404a22c2e8d",
    "s_ct": "100f467b94766f2c87cc734398c8590bd14a1c8e0722ce31dbca7b446d986e72",
}
SEEDS = (2300, 2301)
COUNTERS = ("generation_calls", "backbone_evaluations", "pre_generation_fallbacks")
ORACLE_PROTOCOL = {
    "input": "singleton list containing the CachedOracle canonical SMILES",
    "output": "exactly one finite real scalar; booleans rejected",
    "exception_policy": "propagate ordinary non-docking TDC list-path evaluator failures",
    "budget": "CachedOracle charges only after a finite successful return",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value, *, pretty=False):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ).encode() + (b"\n" if pretty else b"")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite(value, label="number"):
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {label}")
    return float(value)


def integer(value, label="counter", minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {label}")
    return value


def same(actual, wanted, label):
    """Exact structure/identities, tolerance only for finite float arithmetic."""
    if isinstance(wanted, dict):
        require(
            isinstance(actual, dict) and actual.keys() == wanted.keys(),
            f"{label} keys differ",
        )
        for key in wanted:
            same(actual[key], wanted[key], f"{label}.{key}")
    elif isinstance(wanted, list):
        require(
            isinstance(actual, list) and len(actual) == len(wanted),
            f"{label} length differs",
        )
        for a, b in zip(actual, wanted):
            same(a, b, label)
    elif isinstance(wanted, float):
        require(abs(finite(actual, label) - wanted) <= 1e-12, f"{label} differs")
    else:
        require(type(actual) is type(wanted) and actual == wanted, f"{label} differs")


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.records = {}

    def read(self, path, expected=None):
        resolved = (self.root / path).resolve(strict=True)
        data = resolved.read_bytes()
        record = {
            "path": str(resolved),
            "sha256": digest(data),
            "size_bytes": len(data),
        }
        if expected is not None:
            require(record["sha256"] == expected, f"input hash changed: {path}")
        previous = self.records.setdefault(str(resolved), record)
        require(previous == record, f"input changed during report: {path}")
        return data

    def record(self, record):
        data = self.read(record["path"], record["sha256"])
        if "size_bytes" in record:
            require(len(data) == record["size_bytes"], "input size differs")
        return data

    def recheck(self):
        for record in list(self.records.values()):
            self.record(record)

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))


def score_metrics(scores, budget, frequency, *, pad=False):
    """Independent call-axis trapezoids; unpadded curves end at observed calls."""
    require(len(scores) <= budget, "oracle budget exceeded")
    scores = [finite(value, "score") for value in scores]
    result = {
        "oracle_calls": len(scores),
        "oracle_budget": budget,
        "reporting_frequency": frequency,
    }
    for k in (1, 10, 100):
        counts = list(range(frequency, len(scores) + 1, frequency))
        if scores and (not counts or counts[-1] != len(scores)):
            counts.append(len(scores))
        points = [{"oracle_calls": 0, "top_k_mean": 0.0}]
        for count in counts:
            points.append(
                {
                    "oracle_calls": count,
                    "top_k_mean": math.fsum(sorted(scores[:count], reverse=True)[:k])
                    / min(k, count),
                }
            )
        if pad and points[-1]["oracle_calls"] < budget:
            points.append(
                {"oracle_calls": budget, "top_k_mean": points[-1]["top_k_mean"]}
            )
        area = math.fsum(
            (b["oracle_calls"] - a["oracle_calls"])
            * (a["top_k_mean"] + b["top_k_mean"])
            / 2
            for a, b in zip(points, points[1:])
        )
        result.update(
            {
                f"top_{k}": (
                    None
                    if not scores
                    else math.fsum(sorted(scores, reverse=True)[:k])
                    / min(k, len(scores))
                ),
                f"auc_top_{k}": area / budget,
                f"trajectory_top_{k}": points,
            }
        )
    return result


def population_step(population, score, event):
    """Replay released admission from SAVED fragment strings, not chemistry."""
    update = event["population_update"]
    fragments = update["observed_fragments"]
    require(
        isinstance(fragments, list)
        and all(isinstance(f, str) and f for f in fragments),
        "bad fragment observation",
    )
    require(
        fragments == sorted(set(fragments)), "fragment observations not unique/sorted"
    )
    before = {fragment for _, fragment in population}
    selected = event["selected_fragments"]
    require(
        len(selected) == 2 and len(set(selected)) == 2 and set(selected) <= before,
        "selected fragments absent from population",
    )
    wanted = {
        "updated": False,
        "reason": "score_below_cutoff",
        "observed_fragments": [],
        "admitted": [],
        "displaced": [],
    }
    if score > population[-1][0]:
        wanted["reason"] = "no_fragments"
        if fragments:
            capacity = len(population)
            population = sorted(
                population + [(score, f) for f in fragments if f not in before],
                reverse=True,
            )[:capacity]
            after = {f for _, f in population}
            wanted.update(
                updated=before != after,
                reason="updated" if before != after else "no_new_fragments",
                observed_fragments=fragments,
                admitted=sorted(after - before),
                displaced=sorted(before - after),
            )
    same(update, wanted, "population update")
    same(
        event["population_cutoff_after"], float(population[-1][0]), "population cutoff"
    )
    same(event["population_size_after"], len(population), "population size")
    same(event["fragment_statistics_after"], {}, "released statistics")
    return population


def replay(events, config, initial_population, *, complete):
    """Reconstruct unique cache, score indices, population and accepted proposal NFE."""
    budget = config["max_oracle_calls"]
    cache, scores, charged, warmup = {}, [], [], []
    population = list(initial_population)
    counts = dict.fromkeys(COUNTERS, 0)
    duplicate_count = remasking = charged_remasking = charged_model = proposals = 0
    elapsed = 0.0
    for index, event in enumerate(events):
        same(event["event_index"], index, "event index")
        same(event["iteration"], index, "iteration")
        require(
            event["parent_oracle"] is None and event["attribution"] is None,
            "unexpected parent/control scoring",
        )
        remask = (
            index > config["warmup"]
            if config["legacy_warmup_off_by_one"]
            else index >= config["warmup"]
        )
        same(event["remask_enabled"], remask, "warmup schedule")
        counters = event["sampling"]
        require(set(counters) == set(COUNTERS), "sampling counter fields differ")
        for key in COUNTERS:
            counts[key] += integer(counters[key], key)
        generation, nfe, fallback = [counters[key] for key in COUNTERS]
        require(
            generation in (0, 1) and fallback in (0, 1),
            "invalid accepted proposal counters",
        )
        require(
            (generation == 1 and fallback == 0 and nfe > 0)
            or (generation == 0 and nfe == 0),
            "NFE/call inconsistency",
        )
        if remask:
            require(generation + fallback == 1, "unaccounted remasking proposal")
            remasking += 1
        else:
            require(not any(counters.values()), "warmup used diffusion")
            warmup.append(
                {
                    k: v
                    for k, v in event.items()
                    if k not in ("sampling", "elapsed_seconds")
                }
            )
        if (
            config["pmo_sampling"]["configuration"]["diffusion_type"] == "udlm"
            and generation
        ):
            same(nfe, 128, "accepted UDLM NFE")
        proposals += integer(event["proposal_attempts"], "proposal attempts", 1)
        require(event["proposal_attempts"] <= 1000, "proposal bound exceeded")
        atoms = integer(event["child_atom_count"], "child atoms")
        require(
            config["min_mol_size"] <= atoms <= config["max_mol_size"],
            "recorded child size outside bounds",
        )
        same(event["atom_count"], atoms, "atom count")
        value = event["child_oracle"]
        require(
            value["valid"] is True
            and value["canonical_smiles"] == event["child_smiles"]
            and value["raw_smiles"] == event["child_smiles"]
            and isinstance(event["child_smiles"], str)
            and event["child_smiles"],
            "invalid child/cache identity",
        )
        score = finite(value["score"], "fexofenadine score")
        require(0 <= score <= 1, "task score outside [0,1]")
        molecule = value["canonical_smiles"]
        if molecule not in cache:
            require(
                value["charged"] is True and value["reason"] == "scored",
                "new molecule uncharged",
            )
            same(value["call_index"], len(scores) + 1, "new call index")
            scores.append(score)
            cache[molecule] = (score, len(scores))
            charged.append(
                {
                    "call_index": len(scores),
                    "canonical_smiles": molecule,
                    "atom_count": atoms,
                    "original_score": score,
                }
            )
            charged_remasking += int(remask)
            charged_model += int(remask and nfe > 0)
        else:
            require(
                value["charged"] is False and value["reason"] == "cache_hit",
                "duplicate charged",
            )
            require(score == cache[molecule][0], "duplicate score differs")
            same(
                atoms,
                charged[cache[molecule][1] - 1]["atom_count"],
                "duplicate atom count",
            )
            same(value["call_index"], cache[molecule][1], "duplicate call index")
            duplicate_count += 1
        require(len(scores) <= budget, "budget exceeded")
        require(
            len(scores) < budget or index == len(events) - 1,
            "events continue after full budget",
        )
        same(event["oracle_calls"], len(scores), "event oracle counter")
        for k in (1, 10, 100):
            same(
                event[f"top_{k}"],
                math.fsum(sorted(scores, reverse=True)[:k]) / min(k, len(scores)),
                "event top mean",
            )
        population = population_step(population, score, event)
        current = finite(event["elapsed_seconds"], "event runtime")
        require(current >= elapsed, "event time reversed")
        elapsed = current
    if complete:
        same(len(scores), budget, "completed budget")
        require(len(events) <= config["max_iterations"], "iteration limit exceeded")
    return {
        "metrics": score_metrics(scores, budget, config["reporting_frequency"]),
        "runner_padded_metrics": score_metrics(
            scores, budget, config["reporting_frequency"], pad=True
        ),
        "charged_rows": charged,
        "cache_sha256": digest(encoded(charged)),
        "population": [[float(score), fragment] for score, fragment in population],
        "warmup_sha256": digest(encoded(warmup)),
        "warmup_events": len(warmup),
        "events": len(events),
        "accepted_proposal_attempts": proposals,
        "duplicate_events": duplicate_count,
        "remasking_events": remasking,
        "charged_remasking_events": charged_remasking,
        "charged_remasking_with_nfe": charged_model,
        "charged_remasking_with_nfe_fraction": (
            None if not charged_remasking else charged_model / charged_remasking
        ),
        "accepted_event_sampling": counts,
    }


def validate_verification(inputs, reference, replayed, bindings, *, run_id, entry):
    receipt = json.loads(inputs.record(reference))
    require(
        receipt["evaluation_mode"] == "tdc_fresh_oracle_singleton_list",
        "verification not scientifically accepted",
    )
    require(receipt["status"] in ("verified", "failed"), "verification not terminal")
    if receipt["status"] == "failed":
        same(receipt["run"]["entry_id"], entry["id"], "failed verification entry")
        for key, record in receipt["inputs"].items():
            if key in bindings:
                same(
                    record["sha256"],
                    bindings[key],
                    f"failed verification {key} binding",
                )
            inputs.record(record)
        calls = integer(
            receipt["verification"]["verification_calls"], "failed verification calls"
        )
        require(calls <= 2000, "verification budget exceeded")
        return {
            "status": "failed",
            "receipt": reference,
            "verification_calls": calls,
            "error": receipt.get("error"),
            "diagnostics": receipt["verification"],
        }
    run = receipt["run"]
    for key, expected in {
        "entry_id": entry["id"],
        "run_id": run_id,
        "oracle": "fexofenadine_mpo",
        "seed": entry["seed"],
        "optimization_status": "completed",
        "online_budget": 2000,
    }.items():
        same(run[key], expected, f"verification {key}")
    for key, sha in bindings.items():
        same(receipt["inputs"][key]["sha256"], sha, f"verification {key} binding")
    for record in receipt["inputs"].values():
        inputs.record(record)
    verify = receipt["verification"]
    for key in (
        "unique_charged_count",
        "verification_calls",
        "completed_verification_calls",
    ):
        same(verify[key], 2000, key)
    same(
        verify["duplicates_checked"],
        replayed["duplicate_events"],
        "verified duplicate count",
    )
    same(verify["tolerance"], 1e-12, "verification tolerance")
    same(verify["mismatch_count"], 0, "verification mismatches")
    require(len(verify["rows"]) == 2000, "verification rows missing")
    errors = []
    for actual, expected in zip(verify["rows"], replayed["charged_rows"]):
        for key, wanted in expected.items():
            same(actual[key], wanted, f"verification row {key}")
        error = abs(finite(actual["rescored_score"]) - expected["original_score"])
        require(error <= 1e-12, "verification score mismatch")
        same(actual["absolute_error"], error, "verification error")
        errors.append(error)
    same(verify["max_abs_error"], max(errors), "verification maximum error")
    return {
        "status": "verified",
        "receipt": reference,
        "verification_calls": 2000,
        "max_abs_error": max(errors),
    }


def initial_population(data, capacity):
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    population = [(finite(float(row["score"])), row["frag"]) for row in rows[:capacity]]
    require(
        len(population) == capacity and len({f for _, f in population}) == capacity,
        "initial population invalid",
    )
    return population


def slot(entry):
    matches = [
        name for name, sha in CHECKPOINTS.items() if entry["checkpoint_sha256"] == sha
    ]
    require(len(matches) == 1 and entry["seed"] in SEEDS, "unknown fixed panel slot")
    same(entry["max_oracle_calls"], 2000, "fixed oracle budget")
    return matches[0], entry["seed"]


def load_completed_run(
    inputs,
    panel,
    request,
    terminal,
    job,
    outcome,
    verification,
    panel_sha,
    terminal_sha,
):
    entry, config = job["entry"], job["config"]
    role, seed = slot(entry)
    artifacts = outcome["acceptance"]["artifacts"]
    data = {name: inputs.record(record) for name, record in artifacts.items()}
    require(
        set(data)
        == {"manifest.json", "summary.json", "events.jsonl", "state/latest.pkl"},
        "completion artifacts differ",
    )
    manifest, summary = [
        json.loads(data[name]) for name in ("manifest.json", "summary.json")
    ]
    require(
        outcome["return_code"] == 0
        and outcome["pid"] is not None
        and manifest["status"] == summary["status"] == "completed"
        and summary["checkpoint_consistent"] is True,
        "completion status mismatch",
    )
    same(manifest["config"], config, "resolved configuration")
    same(config["experiment_id"], entry["id"], "entry experiment id")
    same(config["matrix_sha256"], panel_sha, "resolved panel binding")
    same(manifest["resume_count"], 0, "fresh run")
    run_path = (inputs.root / job["run_relative"]).resolve()
    expected_path = (
        inputs.root
        / panel["output_root"]
        / "runs"
        / entry["id"]
        / panel["oracle"]
        / "released"
        / f"seed_{seed}"
    ).resolve()
    same(str(run_path), str(expected_path), "run directory")
    for name, claim in artifacts.items():
        same(
            str((inputs.root / claim["path"]).resolve()),
            str(run_path / name),
            "artifact location",
        )
    config_sha = digest(encoded(config))
    for value in (
        job["config_sha256"],
        manifest["config_sha256"],
        summary["config_sha256"],
    ):
        same(value, config_sha, "configuration hash")
    for key, wanted in {
        "oracle": "fexofenadine_mpo",
        "seed": seed,
        "variant": "released",
        "policy_mode": "released",
        "parent_control": False,
        "gamma": 0.0,
        "warmup": 1000,
        "checkpoint_every": 100,
        "guidance_scale": 2.0,
        "legacy_warmup_off_by_one": True,
        "population_size": 100,
        "reporting_frequency": 100,
        "max_oracle_calls": 2000,
        "max_iterations": 5000,
        "min_mol_size": 20,
        "max_mol_size": 40,
    }.items():
        same(config[key], wanted, f"fixed {key}")
    same(manifest["extra"]["git"]["commit"], request["source"]["head"], "child source")
    require(manifest["extra"]["git"]["dirty"] is False, "child source was dirty")
    environment = outcome["environment"]
    for key, wanted in {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }.items():
        same(environment[key], wanted, f"launch environment {key}")
    same(
        environment["CUDA_VISIBLE_DEVICES"],
        outcome["gpu"]["uuid"],
        "launch GPU identity",
    )
    same(manifest["extra"]["runtime"]["python_hash_seed"], "0", "runtime hash seed")
    same(
        manifest["extra"]["runtime"]["cuda_visible_devices"],
        outcome["gpu"]["uuid"],
        "runtime GPU identity",
    )
    same(summary["run_id"], manifest["run_id"], "run id")
    same(
        manifest["run_id"],
        f"{config['experiment_id']}:fexofenadine_mpo:released:seed{seed}",
        "requested run id",
    )
    for key, expected in {
        "task": "fexofenadine_mpo",
        "seed": seed,
        "variant": "released",
        "oracle_budget": 2000,
    }.items():
        same(manifest[key], expected, f"manifest {key}")
    for sha in (
        manifest["model"]["sha256"],
        summary["model_sha256"],
        job["checkpoint"]["sha256"],
    ):
        same(sha, entry["checkpoint_sha256"], "checkpoint identity")
    sampling = manifest["extra"]["sampling"]
    same(sampling, summary["sampling"]["identity"], "sampling receipt")
    same(sampling["contract"], config["pmo_sampling"], "sampling contract")
    same(sampling["checkpoint"], job["checkpoint"], "checkpoint receipt")
    same(sampling["oracle_call_protocol"], ORACLE_PROTOCOL, "oracle exception policy")
    require(
        sampling["inference_weights"]["source"] == "ema"
        and sampling["inference_weights"]["ema_applied"] is True
        and sampling["inference_weights"].get("ema"),
        "actual EMA missing",
    )
    same(
        sampling["contract"]["source"]["sha256"],
        entry["sampling_config_sha256"],
        "sampling YAML identity",
    )
    inputs.record(sampling["contract"]["source"])
    for record in sampling["implementation_inputs"].values():
        inputs.record(record)
    vocab = inputs.read(entry["vocabulary"], entry["vocabulary_sha256"])
    same(
        manifest["extra"]["vocabulary"]["sha256"],
        entry["vocabulary_sha256"],
        "vocabulary identity",
    )
    law = sampling["contract"]["configuration"]
    same(
        sampling["contract"]["configuration_sha256"],
        digest(encoded(law)),
        "normalized sampling hash",
    )
    expected_law = {
        "diffusion_type": "mdlm" if role == "mdlm" else "udlm",
        "softmax_temp": 1.2 if role == "mdlm" else 0.5,
        "randomness": 2.0 if role == "mdlm" else 0.0,
        "min_add_len": 18,
    }
    if role != "mdlm":
        expected_law.update(
            num_steps=128,
            raw_loo_top_p=1.0,
            inference_eps=1e-5,
            exclude_special_tokens=False,
            prior_variant=(
                "mask_rich_empirical" if role == "mask_ce" else "schedule_uniform"
            ),
        )
    for key, wanted in expected_law.items():
        same(law[key], wanted, f"sampling {key}")
    same(
        law.get("parameterization", "raw_loo"),
        "x0_denoiser" if role == "mask_ce" else "raw_loo",
        "parameterization",
    )
    require(
        not law.get("gibbs_corrector", False)
        and law.get("temperature_space", "raw_loo") == "raw_loo",
        "undeclared sampling mode",
    )
    events = [json.loads(line) for line in data["events.jsonl"].splitlines()]
    result = replay(events, config, initial_population(vocab, 100), complete=True)
    same(
        summary["scores"]["all_charged_molecules"],
        result["runner_padded_metrics"],
        "runner curves",
    )
    same(outcome["acceptance"]["scores"], summary["scores"], "controller scores")
    for key in ("events", "iterations_completed", "recoverable_events"):
        same(summary[key], len(events), key)
    for calls in (
        manifest["oracle_calls"],
        summary["recoverable_oracle_calls"],
        outcome["acceptance"]["oracle_calls"],
    ):
        same(calls, 2000, "recorded call count")
    same(
        summary["population"],
        {"size": 100, "active_rows": result["population"]},
        "final population",
    )
    observed = summary["sampling"]["observed"]
    same(
        observed, outcome["acceptance"]["observed_sampling"], "observed counter receipt"
    )
    for key in (*COUNTERS, "modification_attempts"):
        integer(observed[key], key)
    require(
        observed["modification_attempts"]
        == observed["generation_calls"] + observed["pre_generation_fallbacks"],
        "total modification accounting differs",
    )
    for key in COUNTERS:
        require(
            observed[key] >= result["accepted_event_sampling"][key],
            "accepted counters exceed all proposal counters",
        )
    if role != "mdlm":
        same(
            observed["backbone_evaluations"],
            128 * observed["generation_calls"],
            "total UDLM NFE",
        )
    bindings = {
        "protocol": panel_sha,
        "controller_request": terminal["request_sha256"],
        "controller_terminal": terminal_sha,
        "manifest": artifacts["manifest.json"]["sha256"],
        "summary": artifacts["summary.json"]["sha256"],
        "events": artifacts["events.jsonl"]["sha256"],
        "sampling_config": entry["sampling_config_sha256"],
        "vocabulary": entry["vocabulary_sha256"],
    }
    verification_result = {"status": "pending_verification", "verification_calls": 0}
    if verification is not None:
        verification_result = validate_verification(
            inputs,
            verification,
            result,
            bindings,
            run_id=manifest["run_id"],
            entry=entry,
        )
    for key in ("runner_padded_metrics", "population", "charged_rows"):
        result.pop(key)
    result.update(
        entry_id=entry["id"],
        arm=role,
        seed=seed,
        status="completed",
        recorded_oracle_calls=2000,
        verification=verification_result,
        observed_sampling=observed,
        unaccepted_proposal_sampling={
            key: observed[key] - result["accepted_event_sampling"][key]
            for key in COUNTERS
        },
        subprocess_seconds=finite(outcome["subprocess_seconds"]),
        runner_seconds=finite(summary["elapsed_seconds"]),
        checkpoint_sha256=entry["checkpoint_sha256"],
        sampling=sampling,
        config=config,
        gpu=outcome["gpu"],
        environment=outcome["environment"],
        run_id=manifest["run_id"],
        artifacts=artifacts,
        generation_identity=job["checkpoint"],
    )
    return result


def comparisons(runs, *, validated, verified):
    by_slot = {(run["arm"], run["seed"]): run for run in runs}
    result = []
    for name, arm in (("primary", "mask_ce"), ("secondary", "s_ct")):
        metrics = {}
        for metric in ("auc_top_10", "top_10"):
            values = []
            for seed in SEEDS:
                control, treatment = by_slot[("mdlm", seed)], by_slot[(arm, seed)]
                value = (
                    (treatment["metrics"][metric] - control["metrics"][metric])
                    if validated
                    else None
                )
                values.append({"seed": seed, "difference": value})
            defined = [v["difference"] for v in values if v["difference"] is not None]
            metrics[metric] = {
                "per_seed": values,
                "mean": statistics.mean(defined) if len(defined) == 2 else None,
                "sample_sd": statistics.stdev(defined) if len(defined) == 2 else None,
            }
        arithmetic_pass = (
            validated
            and all(row["difference"] > 0 for row in metrics["auc_top_10"]["per_seed"])
            and metrics["top_10"]["mean"] >= 0
        )
        result.append(
            {
                "name": name,
                "treatment": arm,
                "control": "mdlm",
                "direction": f"{arm}_minus_mdlm",
                "status": (
                    "withheld_incomplete_or_unaudited"
                    if not validated
                    else (
                        "verified_descriptive_pilot"
                        if verified
                        else "descriptive_pending_verification"
                    )
                ),
                "metrics": metrics,
                "arithmetic_gate_pass": bool(arithmetic_pass),
                "engineering_gate_qualified": bool(arithmetic_pass and verified),
                "formal_promotion": False,
            }
        )
    return result


def build_report(input_root, panel_path, panel_sha, terminal_sha, verifications=None):
    inputs = Inputs(input_root)
    panel = inputs.json(panel_path, panel_sha)
    require(
        panel["oracle"] == "fexofenadine_mpo" and len(panel["entries"]) == 6,
        "not the fixed V14 panel",
    )
    require(
        {slot(e) for e in panel["entries"]}
        == {(arm, seed) for arm in CHECKPOINTS for seed in SEEDS},
        "missing/duplicate panel slots",
    )
    require(
        len({e["id"] for e in panel["entries"]}) == 6, "duplicate entry identifiers"
    )
    output = Path(panel["output_root"])
    terminal = inputs.json(output / "terminal_manifest.json", terminal_sha)
    require(terminal["status"] in ("completed", "failed"), "campaign not terminal")
    same(terminal["panel_sha256"], panel_sha, "terminal panel hash")
    request = inputs.json(output / "request_manifest.json", terminal["request_sha256"])
    same(request["source"], terminal["source"], "controller source")
    same(request["source"]["head"], GENERATION_SOURCE, "frozen source")
    same(request["source"]["upstream"], GENERATION_SOURCE, "pushed source")
    same(request["plan"]["panel"], panel, "requested panel")
    same(request["plan"]["panel_input"]["sha256"], panel_sha, "request panel hash")
    planned = request["plan"]["jobs"]
    same([job["entry"] for job in planned], panel["entries"], "planned entries")
    outcomes = {outcome["entry_id"]: outcome for outcome in terminal["jobs"]}
    require(
        len(outcomes) == len(terminal["jobs"])
        and set(outcomes) <= {e["id"] for e in panel["entries"]},
        "duplicate/unknown terminal job",
    )
    verifications = verifications or {}
    require(
        set(verifications) <= {e["id"] for e in panel["entries"]},
        "unknown verification entry",
    )
    # Checkpoints are hashed by the original frozen controller; do not reload them.
    checkpoint_paths = {
        str((inputs.root / e["checkpoint"]).resolve()) for e in panel["entries"]
    }
    checkpoint_receipts = []
    for record in request["plan"]["inputs"]:
        if str((inputs.root / record["path"]).resolve()) in checkpoint_paths:
            require(
                record["sha256"] in CHECKPOINTS.values(),
                "checkpoint input receipt differs",
            )
            checkpoint_receipts.append(record)
        else:
            inputs.record(record)
    require(
        len({r["path"] for r in checkpoint_receipts}) == 3,
        "missing controller checkpoint receipts",
    )
    for reference in panel.get("input_files", []):
        inputs.record(reference)
    runs = []
    for job in planned:
        entry = job["entry"]
        outcome = outcomes.get(entry["id"])
        if outcome is not None and outcome["status"] == "completed":
            runs.append(
                load_completed_run(
                    inputs,
                    panel,
                    request,
                    terminal,
                    job,
                    outcome,
                    verifications.get(entry["id"]),
                    panel_sha,
                    terminal_sha,
                )
            )
        else:
            # Retain partial artifacts and padded runner summaries, but never rank them.
            diagnostics = {}
            for filename in (
                "manifest.json",
                "summary.json",
                "events.jsonl",
                "state/latest.pkl",
            ):
                path = Path(job["run_relative"]) / filename
                if (inputs.root / path).is_file():
                    data = inputs.read(path)
                    diagnostics[filename] = inputs.records[
                        str((inputs.root / path).resolve())
                    ]
                    if filename == "summary.json":
                        try:
                            diagnostics["saved_summary"] = json.loads(data)
                        except (ValueError, UnicodeDecodeError) as error:
                            diagnostics["summary_parse_error"] = str(error)
            arm, seed = slot(entry)
            runs.append(
                {
                    "entry_id": entry["id"],
                    "arm": arm,
                    "seed": seed,
                    "status": "not_launched" if outcome is None else outcome["status"],
                    "metrics": None,
                    "recorded_oracle_calls": None,
                    "verification": {"status": "unrankable", "verification_calls": 0},
                    "diagnostics": diagnostics,
                    "controller_outcome": outcome,
                }
            )
    warmups = []
    for seed in SEEDS:
        selected = [
            run for run in runs if run["seed"] == seed and run["status"] == "completed"
        ]
        matches = (
            len(selected) == 3
            and len({r["warmup_sha256"] for r in selected}) == 1
            and all(r["warmup_events"] == 1001 for r in selected)
        )
        if len(selected) == 3:
            require(matches, "same-seed attaching warmup differs")
        warmups.append(
            {
                "seed": seed,
                "identical": matches,
                "events_per_arm": 1001 if matches else None,
                "prefix_sha256": selected[0]["warmup_sha256"] if matches else None,
            }
        )
    completed = sum(run["status"] == "completed" for run in runs)
    if terminal["status"] == "completed":
        require(
            completed == 6
            and terminal["final_input_validation"] == "unchanged"
            and terminal["lease_release_authorized"] is True,
            "campaign completion incomplete",
        )
    validated = (
        terminal["status"] == "completed"
        and completed == 6
        and all(
            run["charged_remasking_with_nfe"] > 0
            and run["verification"]["status"] != "failed"
            for run in runs
        )
    )
    verified = validated and all(
        run["verification"]["status"] == "verified" for run in runs
    )
    report = {
        "schema_version": 1,
        "study": "V14",
        "task": "fexofenadine_mpo",
        "status": (
            "verified_complete"
            if verified
            else ("pending_verification" if validated else "incomplete_or_unrankable")
        ),
        "panel_sha256": panel_sha,
        "controller_terminal_sha256": terminal_sha,
        "source": request["source"],
        "panel": panel,
        "planned_runs": 6,
        "planned_online_calls": 12000,
        "completed_runs": completed,
        "accepted_online_calls": 2000 * completed,
        "verified_online_calls": sum(
            2000 for run in runs if run["verification"]["status"] == "verified"
        ),
        "outside_budget_verification_calls": sum(
            run["verification"]["verification_calls"] for run in runs
        ),
        "runs": runs,
        "warmup_prefix_identity": warmups,
        "contrasts": comparisons(runs, validated=validated, verified=verified),
        "checkpoint_identity_from_controller_only": checkpoint_receipts,
        "controller_terminal": terminal,
        "count_definition": "accepted_online_calls counts controller-completed and event-ledger-replayed calls; chemistry/score verification is counted separately as verified_online_calls",
        "metric_definitions": {
            "top_k": "arithmetic mean of best min(k,c) saved scores after c unique calls",
            "auc_top_10": "origin(0,0); call-count grid100..2000; trapezoidal integral divided by2000",
            "partial_padding": "runner padded summaries retained only as diagnostics; no full-budget ranking",
            "warmup": "iterations0..1000 inclusive; exact semantic event-prefix hashes exclude only elapsed_seconds and sampling",
            "nfe": "observed backbone forward calls; all-proposal and accepted-event denominators remain distinct",
        },
        "audit_boundary": "No TDC/RDKit evaluation, fragmentation, checkpoint load or pickle deserialization. Canonical strings and fragments are recorded claims; separate hashed verification establishes molecule/score agreement. Population admission and score/cache arithmetic independently replayed.",
    }
    inputs.recheck()
    return report, list(inputs.records.values())


def report_csv(report):
    stream = io.StringIO(newline="")
    fields = [
        "arm",
        "seed",
        "status",
        "verification_status",
        "oracle_calls",
        "auc_top_10",
        "top_1",
        "top_10",
        "top_100",
        "events",
        "duplicate_events",
        "charged_remasking_events",
        "charged_remasking_with_nfe",
        "charged_remasking_with_nfe_fraction",
        "generation_calls",
        "backbone_evaluations",
        "pre_generation_fallbacks",
        "subprocess_seconds",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for run in report["runs"]:
        row = {key: run.get(key) for key in fields}
        row.update(run.get("metrics") or {})
        row.update(run.get("observed_sampling") or {})
        row["verification_status"] = run["verification"]["status"]
        writer.writerow({key: row.get(key) for key in fields})
    return stream.getvalue().encode()


def contrasts_csv(report):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "comparison",
            "direction",
            "metric",
            "seed",
            "difference",
            "mean",
            "sample_sd",
            "status",
            "engineering_gate_qualified",
        ]
    )
    for contrast in report["contrasts"]:
        for name, metric in contrast["metrics"].items():
            for row in metric["per_seed"]:
                writer.writerow(
                    [
                        contrast["name"],
                        contrast["direction"],
                        name,
                        row["seed"],
                        row["difference"],
                        metric["mean"],
                        metric["sample_sd"],
                        contrast["status"],
                        contrast["engineering_gate_qualified"],
                    ]
                )
    return stream.getvalue().encode()


def render_pdf(report):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=(8.27 * inch, 11.69 * inch),
        rightMargin=35,
        leftMargin=35,
        topMargin=32,
        bottomMargin=32,
        invariant=1,
        title="V14 saved-evidence PMO pilot",
    )
    styles = getSampleStyleSheet()
    story = []

    def paragraph(value, style="BodyText"):
        story.append(Paragraph(escape(value), styles[style]))
        story.append(Spacer(1, 7))

    def table(rows, widths=None):
        cells = [
            [Paragraph(escape(str(value)), styles["BodyText"]) for value in row]
            for row in rows
        ]
        widget = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        widget.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(widget)
        story.append(Spacer(1, 10))

    def number(value):
        return "NA" if value is None else f"{value:.6f}"

    paragraph("V14: saved-evidence PMO pilot", "Title")
    paragraph(
        f"Status: {report['status']}. Six scheduled runs, 12,000 planned online calls; {report['completed_runs']} completed runs, {report['accepted_online_calls']} accepted online calls, {report['verified_online_calls']} calls covered by independent verification. Separate verification evaluations: {report['outside_budget_verification_calls']} (outside optimization budget)."
    )
    paragraph(
        "One fexofenadine_mpo task; seeds 2300/2301; 2,000 unique canonical oracle calls per run. Candidate checkpoints and sampling choices were selected adaptively using earlier de novo studies. This two-seed engineering pilot does not establish superiority or reproduce the paper's 23-task, 10,000-call, three-run PMO benchmark."
    )
    rows = [
        [
            "Arm / seed",
            "Status / verification",
            "Calls",
            "Top-10 AUC",
            "Top-10",
            "Top-1 / top-100",
        ]
    ]
    for run in report["runs"]:
        metric = run.get("metrics") or {}
        rows.append(
            [
                f"{run['arm']} / {run['seed']}",
                f"{run['status']} / {run['verification']['status']}",
                metric.get("oracle_calls", "NA"),
                number(metric.get("auc_top_10")),
                number(metric.get("top_10")),
                f"{number(metric.get('top_1'))} / {number(metric.get('top_100'))}",
            ]
        )
    table(rows, [76, 117, 36, 77, 65, 151])
    paragraph(
        "Top-k is the arithmetic mean of the best min(k,c) saved scores after c unique calls. Top-10 AUC uses origin (0,0), points every 100 calls through 2,000, trapezoidal integration divided by 2,000. Duplicate canonical cache hits consume no new calls; zero is a legitimate finite score. Partial-run padded AUC values are retained only in diagnostic JSON and never ranked."
    )
    paragraph("Both signed treatment-minus-MDLM comparisons", "Heading2")
    rows = [
        [
            "Comparison",
            "Metric",
            "Seed 2300 / 2301",
            "Mean / sample SD",
            "Engineering gate",
        ]
    ]
    for contrast in report["contrasts"]:
        for name, metric in contrast["metrics"].items():
            rows.append(
                [
                    contrast["direction"],
                    name,
                    " / ".join(number(row["difference"]) for row in metric["per_seed"]),
                    f"{number(metric['mean'])} / {number(metric['sample_sd'])}",
                    str(contrast["engineering_gate_qualified"]),
                ]
            )
    table(rows, [135, 68, 113, 127, 79])
    paragraph(
        "Sample SD describes two seed differences; it is not a confidence interval or significance test. All six runs must complete and pass identity, shared warmup, event replay and independent verification. Each must have a charged accepted post-warmup event with positive observed NFE. Eligibility further requires both paired AUC differences >0 and mean terminal top-10 difference >=0. Both arms remain visible, including negative effects. Passing means only eligible for a separately declared replication; no formal promotion."
    )
    story.append(PageBreak())
    paragraph("Compute, warmup and scientific interpretation", "Title")
    rows = [
        [
            "Arm / seed",
            "Events / duplicates",
            "Charged model events / post-warmup",
            "All-proposal calls / NFE / fallbacks",
            "Child seconds",
        ]
    ]
    for run in report["runs"]:
        counters = run.get("observed_sampling", {})
        rows.append(
            [
                f"{run['arm']} / {run['seed']}",
                f"{run.get('events', 'NA')} / {run.get('duplicate_events', 'NA')}",
                f"{run.get('charged_remasking_with_nfe', 'NA')} / {run.get('charged_remasking_events', 'NA')}",
                " / ".join(str(counters.get(key, "NA")) for key in COUNTERS),
                number(run.get("subprocess_seconds")),
            ]
        )
    table(rows, [75, 88, 127, 149, 83])
    paragraph(
        "Counters distinguish accepted event proposals from all attempted model modifications, including rejected size/chemistry proposals. Report JSON/CSV retain the charged model-event fraction and unaccepted-proposal residual counters. NFE counts backbone forward calls, not molecules or oracle evaluations. Cached duplicates and discarded proposals may consume model compute. Shared GPUs and unequal NFE prevent a controlled speed claim."
    )
    paragraph(
        "The attaching-only prefix is iterations 0..1000 inclusive (1,001 events); remasking begins at iteration 1,001. Per-seed selected fragments, molecules, scores and complete population-update records must match across arms, ignoring only timestamps and explicit zero sampler counters. Matched prefixes: "
        + "; ".join(
            f"seed {row['seed']}: {row['identical']}"
            for row in report["warmup_prefix_identity"]
        )
        + "."
    )
    paragraph(
        "All arms use population 100, gamma 0, released duplicate update/admission policy, frozen scored vocabulary, effective completion minimum 18 added tokens, size 20..40 atoms, inner proposal limit 1,000, maximum 5,000 iterations and child timeout 3,600 seconds. Offline vocabulary scoring, the prelaunch three-input calibration and postoptimization verification are separate from the 2,000-call online budget."
    )
    paragraph(
        "MDLM uses the local 50,000-step EMA checkpoint, temperature 1.2 and confidence randomness 2, with adaptive observed NFE. MASK CE uses 1,000 batch-128 updates (128,000 configured exposures), MASK prior weight 0.9, raw-LOO temperature 0.5 and 128 predictor NFE. S CT uses 1,000 batch-16 updates (16,000 configured exposures), schedule-consistent uniform prior and the same UDLM sampling controls. Both warm-started from the MDLM EMA; no new training belongs to V14. Training amounts and sampling laws differ: this tests full configured pipelines at matched oracle budget, not an isolated architecture effect."
    )
    paragraph("Task definition", "Heading2")
    paragraph(
        "For atom-pair count-fingerprint Tanimoto similarity S to the fixed fexofenadine reference, TPSA P and MolLogP L, the score is [min(S/0.8,1) × exp(-0.5(min(P-90,0)/10)^2) × exp(-0.5 max(L-4,0)^2)]^(1/3). Fingerprints use maxLength=10; the installed PyTDC/RDKit/SciPy implementations and reference are pinned by the panel. It is deterministic descriptor scoring, without a learned oracle checkpoint, QED threshold or SA filter."
    )
    paragraph(
        "This report replays saved canonical-string cache and population-admission arithmetic. It does not recompute chemistry, cut fragments, load model checkpoints or deserialize state/latest.pkl; the pickle is hash-bound only. Separate verification must reproduce canonical identity, atom bounds and all saved charged scores at absolute tolerance 1e-12. Verification cannot feed back into optimization. Original failures, settings, YAML/source/checkpoint receipts and input hashes are retained in report JSON/manifest."
    )
    paragraph(
        "Primary references: GenMol paper https://arxiv.org/html/2501.06158v3 ; installed PyTDC definition documented at https://tdc.readthedocs.io/en/latest/_modules/tdc/chem_utils/oracle/oracle.html#fexofenadine_mpo . The preserved earlier 93-page study overview remains separate unless explicitly appended later; its de novo metric is a different endpoint from this PMO score."
    )

    def footer(canvas, document):
        if report.get("synthetic_preview"):
            canvas.saveState()
            canvas.setFillColor(colors.red)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.drawString(
                35, 17, "SYNTHETIC TEST DATA — NO PMO EXPERIMENT OR ORACLE EVALUATION"
            )
            canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def publish(report, inputs, output):
    output = Path(output)
    files = {
        "report.json": encoded(report, pretty=True),
        "runs.csv": report_csv(report),
        "report.pdf": render_pdf(report),
    }
    source = Path(__file__)
    files["contrasts.csv"] = contrasts_csv(report)
    source_bytes = source.read_bytes()
    files["report_source.py"] = source_bytes
    manifest = {
        "schema_version": 1,
        "inputs": sorted(inputs, key=lambda r: r["path"]),
        "source": {"path": str(source), "sha256": digest(source_bytes)},
        "packages": {
            "python": platform.python_version(),
            "reportlab": importlib.metadata.version("reportlab"),
        },
        "outputs": {
            name: {"sha256": digest(data), "size_bytes": len(data)}
            for name, data in files.items()
        },
        "reproduction": "Saved bytes only; reportlab invariant PDF. No clocks included in generated content.",
    }
    files["manifest.json"] = encoded(manifest, pretty=True)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        with (output / name).open("xb") as handle:
            handle.write(data)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--terminal-sha256", required=True)
    parser.add_argument(
        "--verification",
        action="append",
        nargs=3,
        default=[],
        metavar=("ENTRY", "PATH", "SHA256"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipts = {
        entry: {"path": path, "sha256": sha} for entry, path, sha in args.verification
    }
    require(len(receipts) == len(args.verification), "duplicate verification argument")
    report, inputs = build_report(
        args.input_root, args.panel, args.panel_sha256, args.terminal_sha256, receipts
    )
    manifest = publish(report, inputs, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "report_pdf_sha256": manifest["outputs"]["report.pdf"]["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
