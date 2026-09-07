"""Synthetic-only saved-evidence tests: no molecules, property oracle, or model."""

from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import pickle

import pytest
from pypdf import PdfReader

from scripts.udlm import report_pmo_pilot as report
from scripts.exps.pmo.main.genmol.experiment_io import summarize_scores


def write(root, path, value):
    data = value if isinstance(value, bytes) else report.encoded(value, pretty=True)
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"path": str(target), "sha256": report.digest(data), "size_bytes": len(data)}


def config(arm="s_ct", seed=2300, budget=2000, warmup=1000):
    law = {
        "diffusion_type": "mdlm" if arm == "mdlm" else "udlm",
        "softmax_temp": 1.2 if arm == "mdlm" else 0.5,
        "randomness": 2.0 if arm == "mdlm" else 0.0,
        "min_add_len": 18,
    }
    if arm != "mdlm":
        law.update(
            num_steps=128,
            raw_loo_top_p=1.0,
            inference_eps=1e-5,
            exclude_special_tokens=False,
            prior_variant=(
                "mask_rich_empirical" if arm == "mask_ce" else "schedule_uniform"
            ),
        )
    if arm == "mask_ce":
        law["parameterization"] = "x0_denoiser"
    return {
        "experiment_id": "SYNTHETIC_PANEL",
        "oracle": "fexofenadine_mpo",
        "seed": seed,
        "variant": "released",
        "policy_mode": "released",
        "parent_control": False,
        "gamma": 0.0,
        "warmup": warmup,
        "legacy_warmup_off_by_one": True,
        "population_size": 100,
        "reporting_frequency": 100,
        "checkpoint_every": 100,
        "guidance_scale": 2.0,
        "max_oracle_calls": budget,
        "max_iterations": 5000,
        "min_mol_size": 20,
        "max_mol_size": 40,
        "pmo_sampling": {"configuration": law},
    }


def make_events(cfg, *, post_score=0.4, fallback=False, duplicate=False):
    scores, events = [], []
    nfe = (
        11 if cfg["pmo_sampling"]["configuration"]["diffusion_type"] == "mdlm" else 128
    )
    for index in range(cfg["max_oracle_calls"] + int(duplicate)):
        remask = index > cfg["warmup"]
        cached = duplicate and index == cfg["max_oracle_calls"] - 1
        if cached:
            molecule, score, call = events[0]["child_smiles"], scores[0], 1
        else:
            score = post_score if remask else 0.2
            scores.append(score)
            call = len(scores)
            molecule = f"SYNTHETIC_seed{cfg['seed']}_call{call}"
        counters = {
            "generation_calls": int(remask and not fallback),
            "backbone_evaluations": nfe if remask and not fallback else 0,
            "pre_generation_fallbacks": int(remask and fallback),
        }
        events.append(
            {
                "event_index": index,
                "iteration": index,
                "selected_fragments": ["f0", "f1"],
                "parent_smiles": "SYNTHETIC_PARENT",
                "child_smiles": molecule,
                "atom_count": 25,
                "parent_atom_count": 25,
                "child_atom_count": 25,
                "proposal_attempts": 1,
                "remask_enabled": remask,
                "sampling": counters,
                "parent_oracle": None,
                "child_oracle": {
                    "raw_smiles": molecule,
                    "canonical_smiles": molecule,
                    "valid": True,
                    "score": score,
                    "charged": not cached,
                    "call_index": call,
                    "reason": "cache_hit" if cached else "scored",
                },
                "attribution": None,
                "population_update": {
                    "updated": False,
                    "reason": "score_below_cutoff",
                    "observed_fragments": [],
                    "admitted": [],
                    "displaced": [],
                },
                "fragment_statistics_after": {},
                "population_cutoff_after": 1.0,
                "population_size_after": 100,
                "oracle_calls": len(scores),
                **{
                    f"top_{k}": sum(sorted(scores, reverse=True)[:k])
                    / min(k, len(scores))
                    for k in (1, 10, 100)
                },
                "elapsed_seconds": float(index),
            }
        )
    return events, scores


def population():
    return [(1.0, f"f{i}") for i in range(100)]


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_pmo")
    vocab = write(
        root,
        "vocab.csv",
        ("frag,score\n" + "".join(f"f{i},1\n" for i in range(100))).encode(),
    )
    source_file = write(root, "source.py", b"# SYNTHETIC SOURCE ONLY\n")
    source = {"head": report.GENERATION_SOURCE, "upstream": report.GENERATION_SOURCE}
    panel = {"oracle": "fexofenadine_mpo", "output_root": "campaign", "entries": []}
    jobs, outcomes, run_data = [], [], {}
    input_records = [vocab, source_file]
    for arm in ("mdlm", "mask_ce", "s_ct"):
        yaml = write(root, f"{arm}.yaml", f"# SYNTHETIC {arm}\n".encode())
        input_records.append(yaml)
        cp = {
            "path": str(root / f"NEVER_READ_{arm}.ckpt"),
            "sha256": report.CHECKPOINTS[arm],
            "size_bytes": 999999999,
        }
        input_records.append(cp)
        for seed in report.SEEDS:
            entry = {
                "id": f"{arm}-{seed}",
                "seed": seed,
                "max_oracle_calls": 2000,
                "checkpoint": cp["path"],
                "checkpoint_sha256": cp["sha256"],
                "sampling_config": yaml["path"],
                "sampling_config_sha256": yaml["sha256"],
                "vocabulary": vocab["path"],
                "vocabulary_sha256": vocab["sha256"],
            }
            panel["entries"].append(entry)
            cfg = config(arm, seed)
            cfg["experiment_id"] = entry["id"]
            cfg["matrix_sha256"] = "PANEL_HASH_PLACEHOLDER"
            cfg["pmo_sampling"].update(
                source=yaml,
                checkpoint_sha256=cp["sha256"],
                configuration_sha256=report.digest(
                    report.encoded(cfg["pmo_sampling"]["configuration"])
                ),
            )
            weights = {
                "source": "ema",
                "ema_applied": True,
                "ema": {"num_updates": 1000},
            }
            sampling = {
                "contract": cfg["pmo_sampling"],
                "checkpoint": cp,
                "oracle_call_protocol": report.ORACLE_PROTOCOL,
                "inference_weights": weights,
                "implementation_inputs": {"source": source_file},
            }
            post = {"mdlm": 0.4, "mask_ce": 0.5, "s_ct": 0.3}[arm]
            events, scores = make_events(cfg, post_score=post, duplicate=True)
            counters = {
                key: sum(e["sampling"][key] for e in events) for key in report.COUNTERS
            }
            # Include discarded proposals to exercise the distinct total counter denominator.
            counters["generation_calls"] += 2
            counters["backbone_evaluations"] += 2 * (11 if arm == "mdlm" else 128)
            counters["pre_generation_fallbacks"] += 3
            counters["modification_attempts"] = (
                counters["generation_calls"] + counters["pre_generation_fallbacks"]
            )
            run_id = f"{entry['id']}:fexofenadine_mpo:released:seed{seed}"
            manifest = {
                "resume_count": 0,
                "status": "completed",
                "config": cfg,
                "config_sha256": report.digest(report.encoded(cfg)),
                "model": cp,
                "run_id": run_id,
                "oracle_calls": 2000,
                "extra": {
                    "git": {"commit": source["head"], "dirty": False},
                    "sampling": sampling,
                    "vocabulary": vocab,
                    "runtime": {
                        "python_hash_seed": "0",
                        "cuda_visible_devices": "GPU-SYNTHETIC",
                    },
                },
                "task": "fexofenadine_mpo",
                "seed": seed,
                "variant": "released",
                "oracle_budget": 2000,
            }
            summary = {
                "status": "completed",
                "checkpoint_consistent": True,
                "config_sha256": manifest["config_sha256"],
                "run_id": run_id,
                "model_sha256": cp["sha256"],
                "scores": {
                    "all_charged_molecules": summarize_scores(scores, budget=2000)
                },
                "events": len(events),
                "iterations_completed": len(events),
                "recoverable_events": len(events),
                "recoverable_oracle_calls": 2000,
                "population": {
                    "size": 100,
                    "active_rows": [list(v) for v in population()],
                },
                "sampling": {"identity": sampling, "observed": counters},
                "elapsed_seconds": 2100.0,
            }
            directory = (
                f"campaign/runs/{entry['id']}/fexofenadine_mpo/released/seed_{seed}"
            )
            artifacts = {
                "manifest.json": write(root, f"{directory}/manifest.json", manifest),
                "summary.json": write(root, f"{directory}/summary.json", summary),
                "events.jsonl": write(
                    root,
                    f"{directory}/events.jsonl",
                    b"\n".join(report.encoded(e) for e in events) + b"\n",
                ),
                "state/latest.pkl": write(
                    root, f"{directory}/state/latest.pkl", b"THIS IS NOT A PICKLE\n"
                ),
            }
            jobs.append(
                {
                    "entry": entry,
                    "config": cfg,
                    "config_sha256": manifest["config_sha256"],
                    "checkpoint": cp,
                    "run_relative": directory,
                }
            )
            outcomes.append(
                {
                    "entry_id": entry["id"],
                    "status": "completed",
                    "pid": 123,
                    "return_code": 0,
                    "subprocess_seconds": 2120.0,
                    "gpu": {"uuid": "GPU-SYNTHETIC"},
                    "environment": {
                        "PYTHONHASHSEED": "0",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "CUDA_VISIBLE_DEVICES": "GPU-SYNTHETIC",
                    },
                    "acceptance": {
                        "artifacts": artifacts,
                        "scores": summary["scores"],
                        "observed_sampling": counters,
                        "oracle_calls": 2000,
                    },
                }
            )
            run_data[entry["id"]] = (cfg, events, manifest, artifacts)
    panel_ref = write(root, "panel.json", panel)
    # Resolve the panel digest after entries exist, matching producer config binding.
    for job, outcome in zip(jobs, outcomes):
        entry = job["entry"]
        cfg, events, manifest, artifacts = run_data[entry["id"]]
        cfg["matrix_sha256"] = panel_ref["sha256"]
        config_sha = report.digest(report.encoded(cfg))
        job["config_sha256"] = manifest["config_sha256"] = config_sha
        summary_path = Path(artifacts["summary.json"]["path"])
        summary = json.loads(summary_path.read_bytes())
        summary["config_sha256"] = config_sha
        summary["sampling"]["identity"]["contract"] = cfg["pmo_sampling"]
        artifacts["manifest.json"] = write(
            root, artifacts["manifest.json"]["path"], manifest
        )
        artifacts["summary.json"] = write(
            root, artifacts["summary.json"]["path"], summary
        )
    request = {
        "source": source,
        "plan": {
            "panel": panel,
            "panel_input": panel_ref,
            "inputs": input_records,
            "jobs": jobs,
        },
    }
    request_ref = write(root, "campaign/request_manifest.json", request)
    terminal = {
        "status": "completed",
        "source": source,
        "panel_sha256": panel_ref["sha256"],
        "request_sha256": request_ref["sha256"],
        "jobs": outcomes,
        "final_input_validation": "unchanged",
        "lease_release_authorized": True,
    }
    terminal_ref = write(root, "campaign/terminal_manifest.json", terminal)
    verification_refs = {}
    for entry in panel["entries"]:
        cfg, events, manifest, artifacts = run_data[entry["id"]]
        charged = [
            {
                "call_index": e["child_oracle"]["call_index"],
                "canonical_smiles": e["child_smiles"],
                "atom_count": 25,
                "original_score": e["child_oracle"]["score"],
                "rescored_score": e["child_oracle"]["score"],
                "absolute_error": 0.0,
            }
            for e in events
            if e["child_oracle"]["charged"]
        ]
        receipt = {
            "status": "verified",
            "evaluation_mode": "tdc_fresh_oracle_singleton_list",
            "test_data_notice": "FABRICATED RECEIPT; NO REAL EVALUATION WAS PERFORMED",
            "run": {
                "entry_id": entry["id"],
                "run_id": manifest["run_id"],
                "oracle": "fexofenadine_mpo",
                "seed": entry["seed"],
                "optimization_status": "completed",
                "online_budget": 2000,
            },
            "inputs": {
                "protocol": panel_ref,
                "controller_request": request_ref,
                "controller_terminal": terminal_ref,
                "manifest": artifacts["manifest.json"],
                "summary": artifacts["summary.json"],
                "events": artifacts["events.jsonl"],
                "sampling_config": cfg["pmo_sampling"]["source"],
                "vocabulary": vocab,
            },
            "verification": {
                "unique_charged_count": 2000,
                "verification_calls": 2000,
                "completed_verification_calls": 2000,
                "duplicates_checked": 1,
                "mismatch_count": 0,
                "tolerance": 1e-12,
                "max_abs_error": 0.0,
                "rows": charged,
            },
        }
        verification_refs[entry["id"]] = write(
            root, f"verification/{entry['id']}.json", receipt
        )
    return root, panel_ref, terminal_ref, verification_refs


def build(campaign, *, verified=False):
    root, panel, terminal, receipts = campaign
    return report.build_report(
        root,
        panel["path"],
        panel["sha256"],
        terminal["sha256"],
        receipts if verified else None,
    )


def test_analytical_auc_uses_origin_and_reports_partial_without_padding():
    partial = report.score_metrics([0.2, 0.4], 4, 1)
    assert partial["top_10"] == pytest.approx(0.3)
    assert partial["auc_top_10"] == pytest.approx(0.0875)  # (.1+.25)/4
    padded = report.score_metrics([0.2, 0.4], 4, 1, pad=True)
    assert padded["auc_top_10"] == pytest.approx(0.2375)
    assert partial["trajectory_top_10"][-1]["oracle_calls"] == 2
    assert padded["trajectory_top_10"][-1]["oracle_calls"] == 4


def test_full_saved_replay_pending_without_receipts(campaign, monkeypatch):
    monkeypatch.setattr(pickle, "loads", lambda *_: pytest.fail("pickle deserialized"))
    value, inputs = build(campaign)
    assert value["status"] == "pending_verification"
    assert value["accepted_online_calls"] == 12000
    assert value["outside_budget_verification_calls"] == 0
    assert all(
        w["identical"] and w["events_per_arm"] == 1001
        for w in value["warmup_prefix_identity"]
    )
    assert all(not c["engineering_gate_qualified"] for c in value["contrasts"])
    primary, secondary = value["contrasts"]
    assert primary["metrics"]["auc_top_10"]["mean"] > 0
    assert secondary["metrics"]["auc_top_10"]["mean"] < 0
    assert primary["metrics"]["auc_top_10"]["sample_sd"] == 0
    assert all(run["duplicate_events"] == 1 for run in value["runs"])
    assert all(
        run["unaccepted_proposal_sampling"]["generation_calls"] == 2
        for run in value["runs"]
    )
    assert not any("NEVER_READ" in row["path"] for row in inputs)


def test_verified_receipts_only_change_gate_not_curves(campaign):
    initial, _ = build(campaign)
    verified, _ = build(campaign, verified=True)
    assert verified["status"] == "verified_complete"
    assert verified["outside_budget_verification_calls"] == 12000
    assert verified["contrasts"][0]["engineering_gate_qualified"]
    assert not verified["contrasts"][1]["engineering_gate_qualified"]
    assert [r["metrics"] for r in initial["runs"]] == [
        r["metrics"] for r in verified["runs"]
    ]
    assert [c["metrics"] for c in initial["contrasts"]] == [
        c["metrics"] for c in verified["contrasts"]
    ]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda e: e[0]["child_oracle"].update(call_index=2), "call index"),
        (lambda e: e[0]["child_oracle"].update(score=True), "score"),
        (lambda e: e[0]["child_oracle"].update(score=float("nan")), "score"),
        (lambda e: e[0]["child_oracle"].update(charged=False), "uncharged"),
        (
            lambda e: e[0]["sampling"].update(
                generation_calls=1, backbone_evaluations=128
            ),
            "warmup",
        ),
        (lambda e: e[2]["sampling"].update(backbone_evaluations=127), "NFE"),
        (lambda e: e[0].update(top_10=0.99), "top mean"),
        (lambda e: e[0].update(selected_fragments=["MISSING", "f1"]), "fragments"),
        (lambda e: e[0]["population_update"].update(admitted=["fake"]), "population"),
        (lambda e: e[0].update(child_atom_count=41, atom_count=41), "size"),
        (lambda e: e[0].update(proposal_attempts=1001), "proposal"),
        (lambda e: e[0].update(event_index=True), "event index"),
        (lambda e: e[1].update(remask_enabled=True), "warmup schedule"),
    ],
)
def test_replay_rejects_malformed_evidence(mutation, match):
    cfg = config(budget=3, warmup=1)
    events, _ = make_events(cfg)
    mutation(events)
    with pytest.raises(ValueError, match=match):
        report.replay(events, cfg, population(), complete=True)


@pytest.mark.parametrize(
    "change", [{"charged": True}, {"score": 0.7}, {"call_index": 2}]
)
def test_duplicate_must_reuse_original_score_and_index(change):
    cfg = config(budget=3, warmup=0)
    events, _ = make_events(cfg, duplicate=True)
    events[2]["child_oracle"].update(change)
    with pytest.raises(ValueError, match="duplicate"):
        report.replay(events, cfg, population(), complete=True)


def test_population_replay_uses_saved_fragments_and_released_score_order():
    cfg = config(budget=3, warmup=1)
    event = make_events(cfg)[0][0]
    pop = [(0.3, "f0"), (0.2, "f1")]
    event.update(population_size_after=2, population_cutoff_after=0.3)
    event["population_update"] = {
        "updated": True,
        "reason": "updated",
        "observed_fragments": ["new"],
        "admitted": ["new"],
        "displaced": ["f1"],
    }
    assert report.population_step(pop, 0.5, event) == [(0.5, "new"), (0.3, "f0")]


def test_pure_fallback_remasking_does_not_satisfy_scientific_gate():
    cfg = config(budget=3, warmup=1)
    events, _ = make_events(cfg, fallback=True)
    value = report.replay(events, cfg, population(), complete=True)
    assert value["charged_remasking_events"] == 1
    assert value["charged_remasking_with_nfe"] == 0
    assert value["charged_remasking_with_nfe_fraction"] == 0


def test_missing_or_pending_terminal_cannot_publish(campaign, tmp_path):
    root, panel, _, _ = campaign
    with pytest.raises(FileNotFoundError):
        report.build_report(tmp_path, "panel.json", panel["sha256"], "0" * 64)
    fake = {"status": "running"}
    reference = write(tmp_path, "campaign/terminal_manifest.json", fake)
    write(tmp_path, "panel.json", Path(panel["path"]).read_bytes())
    with pytest.raises(ValueError, match="not terminal"):
        report.build_report(
            tmp_path, "panel.json", panel["sha256"], reference["sha256"]
        )


def test_failed_campaign_keeps_all_scheduled_rows_unrankable(campaign, tmp_path):
    _, panel, terminal, _ = campaign
    panel_data = json.loads(Path(panel["path"]).read_bytes())
    terminal_data = json.loads(Path(terminal["path"]).read_bytes())
    request_path = Path(terminal["path"]).with_name("request_manifest.json")
    request = json.loads(request_path.read_bytes())
    # Request source artifacts remain at original absolute synthetic locations.
    request["plan"]["inputs"] = [r for r in request["plan"]["inputs"]]
    req = write(tmp_path, "campaign/request_manifest.json", request)
    terminal_data.update(status="failed", jobs=[], request_sha256=req["sha256"])
    term = write(tmp_path, "campaign/terminal_manifest.json", terminal_data)
    write(tmp_path, "panel.json", panel_data)
    value, _ = report.build_report(
        tmp_path, "panel.json", panel["sha256"], term["sha256"]
    )
    assert value["completed_runs"] == 0 and len(value["runs"]) == 6
    assert all(
        r["status"] == "not_launched" and r["metrics"] is None for r in value["runs"]
    )
    assert all(c["metrics"]["auc_top_10"]["mean"] is None for c in value["contrasts"])


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r.update(evaluation_mode="synthetic_test"), "accepted"),
        (lambda r: r["inputs"]["events"].update(sha256="0" * 64), "binding"),
        (
            lambda r: r["verification"].update(verification_calls=1999),
            "verification_calls",
        ),
        (lambda r: r["verification"].update(duplicates_checked=2), "duplicate"),
        (
            lambda r: r["verification"]["rows"][0].update(rescored_score=0.9),
            "score mismatch",
        ),
        (
            lambda r: r["verification"]["rows"][0].update(canonical_smiles="wrong"),
            "canonical",
        ),
    ],
)
def test_verification_receipt_relabel_rejected(campaign, tmp_path, mutation, match):
    root, panel, terminal, receipts = campaign
    key = next(iter(receipts))
    changed = json.loads(Path(receipts[key]["path"]).read_bytes())
    mutation(changed)
    selected = dict(receipts)
    selected[key] = write(tmp_path, "changed.json", changed)
    with pytest.raises(ValueError, match=match):
        report.build_report(
            root, panel["path"], panel["sha256"], terminal["sha256"], selected
        )


def test_pdf_csv_and_manifest_reproduce_and_publish_exclusively(campaign, tmp_path):
    value, inputs = build(campaign)
    assert report.render_pdf(value) == report.render_pdf(value)
    reader = PdfReader(io.BytesIO(report.render_pdf(value)))
    assert 2 <= len(reader.pages) <= 5
    text = "\n".join(page.extract_text() for page in reader.pages)
    for phrase in (
        "s_ct_minus_mdlm",
        "mask_ce_minus_mdlm",
        "pending_verification",
        "not a confidence",
        "Partial-run padded",
    ):
        assert phrase in text
    rows = list(csv.DictReader(io.StringIO(report.report_csv(value).decode())))
    assert len(rows) == 6 and rows[0]["verification_status"] == "pending_verification"
    one = report.publish(value, inputs, tmp_path / "one")
    two = report.publish(value, inputs, tmp_path / "two")
    assert one == two
    for name in ("report.pdf", "report.json", "runs.csv", "manifest.json"):
        assert (tmp_path / "one" / name).read_bytes() == (
            tmp_path / "two" / name
        ).read_bytes()
    with pytest.raises(FileExistsError):
        report.publish(value, inputs, tmp_path / "one")


def test_actual_cached_oracle_dataclass_producer_schema(monkeypatch):
    """Use the real producer with a synthetic constant evaluator, never TDC."""
    from scripts.exps.pmo import run_ablation as producer

    monkeypatch.setattr(
        producer, "TDCOracle", lambda *a, **k: pytest.fail("real oracle factory")
    )
    monkeypatch.setattr(
        producer, "Sampler", lambda *a, **k: pytest.fail("model factory")
    )
    cfg = config(budget=3, warmup=1)
    events, _ = make_events(cfg)
    scores = iter((0.2, 0.2, 0.4))
    oracle = producer.CachedOracle(lambda _: next(scores), budget=3)
    for index, event in enumerate(events):
        molecule = "C" * (20 + index)  # synthetic valid parse only, no property scoring
        outcome = oracle.score(molecule)
        event.update(
            child_smiles=outcome.canonical_smiles,
            child_atom_count=20 + index,
            atom_count=20 + index,
            child_oracle=producer._score_row(outcome),
            **producer._top_means(oracle),
        )
    assert "raw_smiles" in events[0]["child_oracle"]
    assert "input_smiles" not in events[0]["child_oracle"]
    result = report.replay(events, cfg, population(), complete=True)
    assert result["metrics"]["oracle_calls"] == 3
    assert result["metrics"]["top_10"] == pytest.approx(0.8 / 3)


def test_failed_verification_retained_with_no_contrasts(campaign, tmp_path):
    root, panel, terminal, receipts = campaign
    entry = next(iter(receipts))
    failed = json.loads(Path(receipts[entry]["path"]).read_bytes())
    failed.update(status="failed", error="Synthetic evaluator failure")
    failed["verification"].update(
        verification_calls=3,
        completed_verification_calls=2,
        rows=failed["verification"]["rows"][:2],
    )
    chosen = dict(receipts)
    chosen[entry] = write(tmp_path, "failed_verification.json", failed)
    value, _ = report.build_report(
        root, panel["path"], panel["sha256"], terminal["sha256"], chosen
    )
    assert value["status"] == "incomplete_or_unrankable"
    assert value["outside_budget_verification_calls"] == 10003
    assert all(
        c["metrics"]["auc_top_10"]["mean"] is None
        and not c["engineering_gate_qualified"]
        for c in value["contrasts"]
    )
    assert value["runs"][0]["verification"]["error"] == "Synthetic evaluator failure"


def test_same_seed_prefix_checks_population_and_child_but_ignores_time():
    cfg = config(budget=3, warmup=1)
    one, _ = make_events(cfg)
    two = deepcopy(one)
    two[0]["elapsed_seconds"] += 0.01
    first = report.replay(one, cfg, population(), complete=True)
    second = report.replay(two, cfg, population(), complete=True)
    assert first["warmup_sha256"] == second["warmup_sha256"]
    two[0]["parent_smiles"] = "DIFFERENT_PARENT"
    assert (
        first["warmup_sha256"]
        != report.replay(two, cfg, population(), complete=True)["warmup_sha256"]
    )


def test_cache_events_after_full_budget_are_not_a_possible_producer_history():
    cfg = config(budget=3, warmup=1)
    events, _ = make_events(cfg)
    events.append(deepcopy(events[0]))
    with pytest.raises(ValueError, match="continue after full budget"):
        report.replay(events, cfg, population(), complete=True)


def test_actual_runner_emits_replayable_cache_population_and_nfe(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    import yaml
    from scripts.exps.pmo import run_ablation as runner
    from scripts.exps.pmo import udlm_sampling as sampling
    from test_pmo_udlm_adapter_runner import SyntheticSampler, arguments

    model = tmp_path / "synthetic.ckpt"
    model.write_bytes(b"not a checkpoint: synthetic preparation only")
    vocab = tmp_path / "vocab.csv"
    vocab.write_text("frag,score,size\n[1*]CC,0.9,2\n[1*]CN,0.8,2\n[1*]CO,0.7,2\n")
    path = tmp_path / "sampling.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                checkpoint_sha256=report.digest(model.read_bytes()),
                diffusion_type="udlm",
                softmax_temp=0.5,
                randomness=0.0,
                min_add_len=18,
                num_steps=128,
                inference_eps=1e-5,
                exclude_special_tokens=False,
                prior_variant="schedule_uniform",
                prior_metadata_sha256="a" * 64,
                raw_loo_top_p=1.0,
            )
        )
    )
    sampler = SyntheticSampler(outputs=tuple("C" * n for n in (20, 20, 21, 22)))

    def prepare(contract, **kwargs):
        return sampling.SamplingAdapter(
            sampler,
            dict(
                contract=contract,
                sampler_kwargs=sampling.modification_kwargs(
                    contract, gamma=0.0, guidance_scale=2.0
                ),
                configured_nfe_per_generation=128,
                implementation_inputs={},
                oracle_call_protocol=report.ORACLE_PROTOCOL,
            ),
        )

    calls = []

    def fake_oracle(**kwargs):
        assert kwargs == {"name": "fexofenadine_mpo"}

        def score(value):
            molecule = value[0] if type(value) is list else value
            calls.append(molecule)
            result = len(molecule) / 100
            return [result] if type(value) is list else result

        return score

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        runner,
        "_git_metadata",
        lambda: {"commit": "synthetic", "tracked_diff_sha256": "b" * 64},
    )
    monkeypatch.setattr(runner, "_runtime_metadata", lambda _: {"synthetic": True})
    monkeypatch.setattr(runner, "_attach_fragments", lambda *_: "C" * 20)
    monkeypatch.setattr(runner, "TDCOracle", fake_oracle)
    monkeypatch.setattr(sampling, "prepare", prepare)
    args = arguments(
        SimpleNamespace(root=tmp_path, model=model, vocab=vocab, config=path),
        oracle="fexofenadine_mpo",
        min_mol_size=20,
        max_mol_size=40,
        max_oracle_calls=3,
    )
    output = runner.run(args)
    manifest = json.loads((output / "manifest.json").read_bytes())
    summary = json.loads((output / "summary.json").read_bytes())
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_bytes().splitlines()
    ]
    ledger = report.replay(
        events,
        manifest["config"],
        report.initial_population(vocab.read_bytes(), 3),
        complete=True,
    )
    report.same(
        summary["scores"]["all_charged_molecules"],
        ledger["runner_padded_metrics"],
        "actual runner summary",
    )
    assert calls == ["C" * n for n in (20, 21, 22)]
    assert ledger["duplicate_events"] == 1
    assert ledger["accepted_event_sampling"]["backbone_evaluations"] == 4 * 128
    assert ledger["charged_remasking_with_nfe"] == 3
