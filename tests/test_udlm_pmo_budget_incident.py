"""Saved synthetic ledgers only; no model, chemistry, or pickle loads."""

import builtins
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.udlm import audit_pmo_budget_incident as audit
from test_udlm_pmo_report import config, make_events, population, write


def jsonl(events):
    return b"".join(audit.historical.encoded(event) + b"\n" for event in events)


def test_partial_prefix_counts_cache_and_model_work_without_auc():
    cfg = config(budget=5, warmup=1)
    events, _scores = make_events(cfg, duplicate=True)
    result = audit.prefix_counts(jsonl(events[:5]), cfg, population())
    assert result["status"] == "validated_complete_line_prefix"
    assert result["durable_unique_charges"] == 4
    assert result["cache_hit_events"] == 1
    assert result["accepted_event_sampling"] == {
        "generation_calls": 3,
        "backbone_evaluations": 384,
        "pre_generation_fallbacks": 0,
    }
    assert result["last_event_elapsed_seconds"] == 4
    assert not any("auc" in key or "metric" in key for key in result)


@pytest.mark.parametrize(
    "tail", [b'{"event_index":2', b"\xff\x00", b'{"complete_json_but_no_newline":true}']
)
def test_unterminated_tail_is_excluded_and_hash_bound(tail):
    cfg = config(budget=5)
    events, _scores = make_events(cfg)
    prefix = jsonl(events[:2])
    result = audit.prefix_counts(prefix + tail, cfg, population())
    assert result["durable_unique_charges"] == 2
    assert result["complete_jsonl_lines"] == 2
    assert result["complete_line_prefix_sha256"] == audit.historical.digest(prefix)
    assert result["unterminated_tail_bytes"] == len(tail)
    assert result["unterminated_tail_sha256"] == audit.historical.digest(tail)


def test_carriage_return_cannot_create_an_extra_committed_event_record():
    cfg = config(budget=5)
    events, _scores = make_events(cfg)
    data = (
        audit.historical.encoded(events[0])
        + b"\r"
        + audit.historical.encoded(events[1])
        + b"\n"
    )
    result = audit.prefix_counts(data, cfg, population())
    assert result["complete_jsonl_lines"] == 1
    assert result["status"] == "unvalidated_prefix"
    assert result["durable_unique_charges"] is None


@pytest.mark.parametrize(
    "corruption",
    [
        "invalid_json",
        "blank",
        "duplicate_field",
        "new_charge_index",
        "cache_score",
        "event_index",
        "nfe",
        "runtime",
        "after_budget",
    ],
)
def test_malformed_complete_line_or_bad_invariant_never_silently_skips(corruption):
    cfg = config(budget=5, warmup=1)
    events, _scores = make_events(cfg, duplicate=True)
    if corruption == "invalid_json":
        data = jsonl(events[:2]) + b"{broken}\n"
    elif corruption == "blank":
        data = jsonl(events[:2]) + b"\n"
    elif corruption == "duplicate_field":
        data = b'{"event_index":0,"event_index":1}\n'
    else:
        if corruption == "new_charge_index":
            events[1]["child_oracle"]["call_index"] = 4
        elif corruption == "cache_score":
            events[4]["child_oracle"]["score"] = 0.1
        elif corruption == "event_index":
            events[2]["event_index"] = 99
        elif corruption == "nfe":
            events[2]["sampling"]["backbone_evaluations"] = 127
        elif corruption == "runtime":
            events[2]["elapsed_seconds"] = -1
        elif corruption == "after_budget":
            events.append(deepcopy(events[-1]))
        data = jsonl(events)
    result = audit.prefix_counts(data, cfg, population())
    assert result["status"] == "unvalidated_prefix"
    assert result["durable_unique_charges"] is None and "error" in result


@pytest.fixture
def incident_case(tmp_path, monkeypatch):
    root = tmp_path
    vocab = write(
        root,
        "vocab.csv",
        ("frag,score\n" + "".join(f"f{i},1\n" for i in range(100))).encode(),
    )
    source_file = write(root, "synthetic_source.py", b"# synthetic\n")
    source = {
        "head": audit.historical.GENERATION_SOURCE,
        "upstream": audit.historical.GENERATION_SOURCE,
    }
    entries = []
    for seed in audit.historical.SEEDS:
        for arm in ("mdlm", "s_ct", "mask_ce"):
            yaml = write(root, f"{arm}_{seed}.yaml", b"# synthetic configuration\n")
            entries.append(
                {
                    "id": f"{arm}_{seed}",
                    "seed": seed,
                    "max_oracle_calls": 2000,
                    "checkpoint": str(root / f"NEVER_READ_{arm}.ckpt"),
                    "checkpoint_sha256": audit.historical.CHECKPOINTS[arm],
                    "sampling_config": yaml["path"],
                    "sampling_config_sha256": yaml["sha256"],
                    "vocabulary": vocab["path"],
                    "vocabulary_sha256": vocab["sha256"],
                }
            )
    panel = {
        "oracle": "fexofenadine_mpo",
        "output_root": "campaign",
        "entries": entries,
    }
    panel_record = write(root, "panel.json", panel)
    monkeypatch.setattr(audit, "PANEL_SHA256", panel_record["sha256"])
    jobs, outcomes = [], []
    paths = {}
    for index, entry in enumerate(entries):
        arm, seed = audit.historical.slot(entry)
        cfg = config(arm, seed)
        cfg.update(experiment_id=entry["id"], matrix_sha256=panel_record["sha256"])
        cp = {
            "path": entry["checkpoint"],
            "sha256": entry["checkpoint_sha256"],
            "size_bytes": 999999,
        }
        cfg["pmo_sampling"].update(
            source={
                "path": entry["sampling_config"],
                "sha256": entry["sampling_config_sha256"],
            },
            checkpoint_sha256=cp["sha256"],
        )
        cfg["pmo_sampling"]["configuration_sha256"] = audit.historical.digest(
            audit.historical.encoded(cfg["pmo_sampling"]["configuration"])
        )
        cfg_sha = audit.historical.digest(audit.historical.encoded(cfg))
        directory = f"campaign/runs/{entry['id']}/fexofenadine_mpo/released/seed_{seed}"
        command = ["synthetic-python", "synthetic-runner", entry["id"]]
        jobs.append(
            {
                "entry": entry,
                "config": cfg,
                "config_sha256": cfg_sha,
                "checkpoint": cp,
                "run_relative": directory,
                "command": command,
            }
        )
        if index >= 2:
            continue
        gpu = {"uuid": f"GPU-SYNTHETIC-{index}"}
        environment = {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": gpu["uuid"],
        }
        launch = {
            "entry_id": entry["id"],
            "command": command,
            "gpu": gpu,
            "environment": environment,
        }
        launch_record = write(root, f"campaign/{entry['id']}.launch.json", launch)
        outcome = {
            **launch,
            "pid": 100 + index,
            "return_code": 0 if index == 0 else -15,
            "status": "completed" if index == 0 else "failed",
            "subprocess_seconds": 200 if index == 0 else 3600,
            "timed_out": index == 1,
            "launch_sha256": launch_record["sha256"],
        }
        manifest = {
            "config": cfg,
            "config_sha256": cfg_sha,
            "resume_count": 0,
            "run_id": f"{entry['id']}:fexofenadine_mpo:released:seed{seed}",
            "model": cp,
            "extra": {
                "git": {"commit": source["head"], "dirty": False},
                "sampling": {
                    "checkpoint": cp,
                    "contract": cfg["pmo_sampling"],
                    "oracle_call_protocol": audit.historical.ORACLE_PROTOCOL,
                    "implementation_inputs": {"synthetic": source_file},
                },
                "vocabulary": vocab,
                "runtime": {
                    "python_hash_seed": "0",
                    "cuda_visible_devices": gpu["uuid"],
                },
            },
        }
        events, _scores = make_events(cfg)
        if index == 1:
            events = events[:1004]
        artifacts = {
            "manifest.json": write(root, f"{directory}/manifest.json", manifest),
            "events.jsonl": write(root, f"{directory}/events.jsonl", jsonl(events)),
        }
        paths[entry["id"]] = root / directory
        if index == 0:
            outcome["acceptance"] = {"artifacts": artifacts, "oracle_calls": 2000}
        outcomes.append(outcome)
    request = {
        "source": source,
        "plan": {"panel": panel, "panel_input": panel_record, "jobs": jobs},
    }
    request_record = write(root, "campaign/request_manifest.json", request)
    terminal = {
        "status": "failed",
        "source": source,
        "panel_sha256": panel_record["sha256"],
        "request_sha256": request_record["sha256"],
        "jobs": outcomes,
        "lease_release_authorized": True,
        "final_input_validation": "unchanged",
        "unlaunched_entry_ids": [e["id"] for e in entries[2:]],
        "error": "synthetic timeout",
    }
    terminal_record = write(root, "campaign/terminal_manifest.json", terminal)
    pipeline = write(
        root, "pipeline.json", {"source": source["head"], "return_code": 1}
    )
    return {
        "root": root,
        "panel": panel_record,
        "terminal": terminal_record,
        "pipeline": pipeline,
        "paths": paths,
    }


def build(case):
    return audit.build_audit(
        case["root"],
        "panel.json",
        case["panel"]["sha256"],
        case["terminal"]["sha256"],
        "pipeline.json",
        case["pipeline"]["sha256"],
    )


def test_all_six_slots_durable_lower_bound_and_exclusive_publication(
    incident_case, tmp_path, monkeypatch
):
    original_import = builtins.__import__

    def forbidden_heavy(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "genmol", "rdkit", "tdc", "pickle"}:
            pytest.fail("heavy import " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_heavy)
    result, inputs = build(incident_case)
    assert result["status"] == "diagnostic_only_unrankable" and len(result["runs"]) == 6
    assert result["accounting"]["durable_unique_charge_lower_bound"] == 3004
    assert result["accounting"]["controller_claimed_accepted_calls"] == 2000
    assert result["accounting"]["unlaunched_slots"] == 4
    assert all(row["durable_unique_charges"] == 0 for row in result["runs"][2:])
    partial = result["runs"][1]
    assert partial["controller_status"] == "failed" and partial["timed_out"] is True
    assert partial["ledger"]["accepted_event_sampling"]["backbone_evaluations"] == 384
    assert partial["ledger"]["last_event_elapsed_seconds"] == 1003
    assert partial["subprocess_seconds"] == 3600
    assert all(not row["ranking_eligible"] for row in result["runs"])
    assert not any(
        Path(record["path"]).suffix in {".pkl", ".ckpt"}
        for record in inputs.records.values()
    )
    output = tmp_path / "addendum"
    manifest = audit.publish(result, inputs, output)
    for name, record in manifest["outputs"].items():
        data = (output / name).read_bytes()
        assert (
            audit.historical.digest(data) == record["sha256"]
            and len(data) == record["size_bytes"]
        )
    with pytest.raises(FileExistsError):
        audit.publish(result, inputs, output)


@pytest.mark.parametrize(
    "gate",
    [
        "lease",
        "dangling_lease",
        "pipeline_missing",
        "pipeline_source",
        "terminal_running",
        "cleanup",
        "terminal_hash",
    ],
)
def test_terminal_and_lease_gates_reject_before_child_ledger_reads(
    incident_case, monkeypatch, gate
):
    root = incident_case["root"]
    if gate in {"lease", "dangling_lease"}:
        path = root / audit.LEASE
        path.parent.mkdir(parents=True)
        if gate == "lease":
            path.write_text("active")
        else:
            path.symlink_to(root / "nonexistent")
    elif gate == "pipeline_missing":
        (root / "pipeline.json").unlink()
    elif gate == "pipeline_source":
        incident_case["pipeline"] = write(
            root, "pipeline.json", {"source": "wrong", "return_code": 1}
        )
    elif gate in {"terminal_running", "cleanup"}:
        terminal = json.loads((root / "campaign/terminal_manifest.json").read_bytes())
        terminal[
            "status" if gate == "terminal_running" else "lease_release_authorized"
        ] = "running" if gate == "terminal_running" else False
        incident_case["terminal"] = write(
            root, "campaign/terminal_manifest.json", terminal
        )
    elif gate == "terminal_hash":
        incident_case["terminal"]["sha256"] = "0" * 64
    read = audit.Inputs.read

    def guarded(self, path, expected=None):
        if str(path).endswith("events.jsonl"):
            pytest.fail("ledger read before terminal gate")
        return read(self, path, expected)

    monkeypatch.setattr(audit.Inputs, "read", guarded)
    with pytest.raises((ValueError, FileNotFoundError)):
        build(incident_case)


def test_partial_corruption_preserves_slot_without_invented_charge_count(incident_case):
    path = incident_case["paths"]["s_ct_2300"] / "events.jsonl"
    path.write_bytes(path.read_bytes() + b"{invalid complete record}\n")
    result, _inputs = build(incident_case)
    assert result["runs"][1]["ledger"]["status"] == "unvalidated_prefix"
    assert result["runs"][1]["ledger"]["durable_unique_charges"] is None
    assert result["accounting"]["durable_unique_charge_lower_bound"] == 2000


def test_post_read_changes_fail_publication(incident_case, tmp_path):
    result, inputs = build(incident_case)
    path = incident_case["paths"]["s_ct_2300"] / "events.jsonl"
    path.write_bytes(path.read_bytes() + b"tail")
    with pytest.raises(ValueError, match="input hash changed"):
        audit.publish(result, inputs, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
