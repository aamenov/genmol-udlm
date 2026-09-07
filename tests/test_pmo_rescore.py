"""Saved-ledger CPU verification with synthetic evaluators; no real oracle calls."""

import fcntl
import json
from pathlib import Path

import pytest

from scripts.udlm import rescore_pmo_run as audit


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fingerprint(path, root):
    payload = path.read_bytes()
    return dict(
        path=str(path.relative_to(root)),
        sha256=audit.sha(payload),
        size_bytes=len(payload),
    )


@pytest.fixture
def saved(tmp_path):
    """The production controller/runner schemas, with a three-charge CPU budget."""
    root = tmp_path / "inputs"
    root.mkdir()
    sampling = root / "sampling.yaml"
    sampling.write_text("diffusion_type: mdlm\n")
    vocabulary = root / "vocabulary.csv"
    vocabulary.write_text("score,fragment\n0.1,C\n")
    oracle_source = root / "oracle_source.py"
    oracle_source.write_text("# synthetic pinned dependency\n")
    entry = dict(
        id="synthetic-2300",
        seed=2300,
        max_oracle_calls=3,
        checkpoint="NEVER_OPEN.ckpt",
        checkpoint_sha256="c" * 64,
        sampling_config="sampling.yaml",
        sampling_config_sha256=audit.sha(sampling.read_bytes()),
        vocabulary="vocabulary.csv",
        vocabulary_sha256=audit.sha(vocabulary.read_bytes()),
    )
    options = dict(
        checkpoint_every=100,
        guidance_scale=2.0,
        legacy_warmup_off_by_one=True,
        max_iterations=5000,
        min_mol_size=20,
        max_mol_size=40,
        population_size=100,
        reporting_frequency=100,
        warmup=1000,
    )
    protocol = dict(
        schema_version=1,
        oracle=audit.ORACLE,
        output_root="output/campaign",
        runner=options,
        entries=[entry],
        input_files=[fingerprint(oracle_source, root)],
    )
    protocol_path = root / "protocol.json"
    dump(protocol_path, protocol)
    protocol_sha = audit.sha(protocol_path.read_bytes())
    run = (
        root
        / protocol["output_root"]
        / "runs"
        / entry["id"]
        / audit.ORACLE
        / "released/seed_2300"
    )
    run.mkdir(parents=True)
    (run / ".run.lock").touch()
    source = dict(head="a" * 40, upstream="a" * 40)
    config = dict(
        experiment_id=entry["id"],
        oracle=audit.ORACLE,
        variant="released",
        policy_mode="released",
        parent_control=False,
        gamma=0.0,
        seed=2300,
        max_oracle_calls=3,
        matrix_sha256=protocol_sha,
        **options,
        pmo_sampling=dict(
            source=dict(sha256=entry["sampling_config_sha256"]),
            checkpoint_sha256=entry["checkpoint_sha256"],
        ),
    )
    config_sha = audit.sha(audit.canonical_bytes(config))
    run_id = f"{entry['id']}:{audit.ORACLE}:released:seed2300"
    oracle_policy = {
        "input": "singleton list containing the CachedOracle canonical SMILES",
        "output": "exactly one finite real scalar; booleans rejected",
        "exception_policy": "propagate ordinary non-docking TDC list-path evaluator failures",
        "budget": "CachedOracle charges only after a finite successful return",
    }
    identity = dict(oracle_call_protocol=oracle_policy)
    manifest = dict(
        status="completed",
        run_id=run_id,
        config=config,
        config_sha256=config_sha,
        model=dict(sha256=entry["checkpoint_sha256"]),
        extra=dict(git=dict(commit=source["head"]), sampling=identity),
        task=audit.ORACLE,
        variant="released",
        seed=2300,
        oracle_budget=3,
        oracle_calls=3,
        resume_count=0,
    )
    summary = dict(
        status="completed",
        run_id=run_id,
        checkpoint_consistent=True,
        config_sha256=config_sha,
        model_sha256=entry["checkpoint_sha256"],
        scores=dict(all_charged_molecules=dict(oracle_calls=3, oracle_budget=3)),
        events=4,
        iterations_completed=4,
        sampling=dict(identity=identity),
    )
    events = []
    for index, (atoms, call, charged, calls) in enumerate(
        [(20, 1, True, 1), (21, 2, True, 2), (20, 1, False, 2), (22, 3, True, 3)]
    ):
        smiles = "C" * atoms
        events.append(
            dict(
                event_index=index,
                iteration=index,
                parent_oracle=None,
                child_smiles=smiles,
                atom_count=atoms,
                child_atom_count=atoms,
                oracle_calls=calls,
                child_oracle=dict(
                    valid=True,
                    charged=charged,
                    raw_smiles=smiles,
                    canonical_smiles=smiles,
                    score=call / 10,
                    call_index=call,
                    reason="scored" if charged else "cache_hit",
                ),
            )
        )
    request = dict(
        source=source,
        plan=dict(
            panel=protocol,
            panel_input=dict(sha256=protocol_sha),
            jobs=[
                dict(
                    entry=entry,
                    config=config,
                    config_sha256=config_sha,
                    run_relative=str(run.relative_to(root)),
                )
            ],
        ),
    )
    terminal = dict(
        status="completed",
        final_input_validation="unchanged",
        lease_release_authorized=True,
        panel_sha256=protocol_sha,
        source=source,
        jobs=[
            dict(
                entry_id=entry["id"],
                status="completed",
                return_code=0,
                acceptance=dict(artifacts={}, oracle_calls=3),
            )
        ],
    )
    case = dict(
        root=root,
        protocol_path=protocol_path,
        entry=entry,
        run=run,
        protocol=protocol,
        manifest=manifest,
        summary=summary,
        events=events,
        request=request,
        terminal=terminal,
        oracle_source=oracle_source,
    )
    save(case)
    return case


def save(case):
    """Rebind intentionally modified synthetic artifacts to test semantic guards."""
    root, run = case["root"], case["run"]
    dump(run / "manifest.json", case["manifest"])
    dump(run / "summary.json", case["summary"])
    (run / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in case["events"])
    )
    artifacts = {
        name: fingerprint(run / name, root)
        for name in ("manifest.json", "summary.json", "events.jsonl")
    }
    # A deliberately nonexistent pickle reference must never be read by this audit.
    artifacts["state/latest.pkl"] = dict(
        path=str(run.relative_to(root) / "state/latest.pkl"),
        sha256="d" * 64,
        size_bytes=123,
    )
    case["terminal"]["jobs"][0]["acceptance"]["artifacts"] = artifacts
    controller = root / case["protocol"]["output_root"]
    dump(controller / "request_manifest.json", case["request"])
    case["terminal"]["request_sha256"] = audit.sha(
        (controller / "request_manifest.json").read_bytes()
    )
    dump(controller / "terminal_manifest.json", case["terminal"])


def verify(case, evaluator=None):
    calls = []
    factories = []

    def factory():
        factories.append(True)

        def score(values):
            calls.append(values)
            return (
                evaluator(values)
                if evaluator is not None
                else [(len(values[0]) - 19) / 10]
            )

        return score

    receipt = audit.verify_run(
        case["root"],
        case["protocol_path"],
        case["entry"]["id"],
        case["run"],
        evaluator_factory=factory,
    )
    return receipt, calls, factories


def test_complete_replay_exact_charge_order_once_per_unique_no_original_writes(saved):
    original = {str(p): p.read_bytes() for p in saved["root"].rglob("*") if p.is_file()}
    receipt, calls, factories = verify(saved)
    assert receipt["status"] == "verified", receipt["error"]
    assert receipt["evaluation_mode"] == "synthetic_test"
    assert factories == [True] and calls == [["C" * n] for n in (20, 21, 22)]
    assert receipt["verification"] == dict(
        unique_charged_count=3,
        verification_calls=3,
        completed_verification_calls=3,
        duplicates_checked=1,
        mismatch_count=0,
        tolerance=1e-12,
        max_abs_error=0.0,
        rows=[
            dict(
                call_index=i,
                canonical_smiles="C" * (19 + i),
                atom_count=19 + i,
                original_score=i / 10,
                rescored_score=i / 10,
                absolute_error=0.0,
            )
            for i in (1, 2, 3)
        ],
    )
    assert {
        "controller_terminal",
        "controller_request",
        "protocol",
        "manifest",
        "summary",
        "events",
        "sampling_config",
        "vocabulary",
        "verifier_source",
    } <= receipt["inputs"].keys()
    assert receipt["run"]["online_budget"] == 3
    assert all(Path(path).read_bytes() == content for path, content in original.items())
    assert not any(name.endswith((".pkl", ".ckpt")) for name in receipt["inputs"])


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda s: s["terminal"].update(status="running"), "Campaign"),
        (lambda s: s["terminal"]["jobs"][0].update(return_code=7), "did not complete"),
        (lambda s: s["terminal"].update(lease_release_authorized=False), "Campaign"),
        (lambda s: s["request"]["source"].update(upstream="b" * 40), "source differs"),
        (lambda s: s["manifest"].update(resume_count=1), "fresh runs"),
        (lambda s: s["summary"].update(events=3), "event count"),
        (lambda s: s["summary"].update(checkpoint_consistent=False), "not complete"),
        (lambda s: s["manifest"].update(oracle_calls=2), "budget was not completed"),
        (
            lambda s: s["events"][2]["child_oracle"].update(call_index=2),
            "exact first charged",
        ),
        (
            lambda s: s["events"][2]["child_oracle"].update(score=0.2),
            "exact first charged",
        ),
        (
            lambda s: s["events"][2]["child_oracle"].update(
                charged=True, reason="scored"
            ),
            "order/uniqueness",
        ),
        (
            lambda s: s["events"][0]["child_oracle"].update(
                charged=False, reason="cache_hit"
            ),
            "no earlier charge",
        ),
        (
            lambda s: s["events"][0]["child_oracle"].update(call_index=True),
            "Invalid integer",
        ),
        (lambda s: s["events"][0].update(oracle_calls=2), "cumulative"),
        (lambda s: s["events"][0].update(iteration=1), "not contiguous"),
        (lambda s: s["events"][0].update(parent_oracle={}), "must not score parents"),
        (lambda s: s["events"][0].update(atom_count=21), "atom counts"),
        (
            lambda s: s["events"][0]["child_oracle"].update(
                canonical_smiles="not_a_smiles"
            ),
            "invalid or noncanonical",
        ),
        (
            lambda s: s["events"][0]["child_oracle"].update(
                canonical_smiles="C(C)" + "C" * 18
            ),
            "invalid or noncanonical",
        ),
        (
            lambda s: s["events"][0]["child_oracle"].update(canonical_smiles="CCO"),
            "outside 20..40",
        ),
    ],
)
def test_invalid_saved_evidence_never_constructs_evaluator(saved, change, match):
    change(saved)
    save(saved)
    receipt, calls, factories = verify(saved)
    assert receipt["status"] == "failed" and match in receipt["error"]
    assert calls == factories == []
    assert receipt["verification"]["verification_calls"] == 0


def test_original_hash_mismatch_never_scores(saved):
    (saved["run"] / "events.jsonl").write_text("changed\n")
    receipt, calls, factories = verify(saved)
    assert receipt["status"] == "failed" and "hash differs" in receipt["error"]
    assert calls == factories == []


def test_active_optimizer_lock_refuses_before_oracle(saved):
    with (saved["run"] / ".run.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt, calls, factories = verify(saved)
    assert receipt["status"] == "failed" and "BlockingIOError" in receipt["error"]
    assert calls == factories == []


@pytest.mark.parametrize(
    "result",
    [
        0.1,
        [],
        [0.1, 0.2],
        [True],
        [None],
        [float("nan")],
        [float("inf")],
        [-0.1],
        [1.1],
    ],
)
def test_bad_oracle_results_record_one_attempt_and_no_retry(saved, result):
    receipt, calls, factories = verify(saved, lambda _: result)
    assert receipt["status"] == "failed" and len(calls) == len(factories) == 1
    assert receipt["verification"]["verification_calls"] == 1
    assert receipt["verification"]["completed_verification_calls"] == 0
    assert receipt["verification"]["rows"] == []


@pytest.mark.parametrize(
    "exception",
    [ValueError("synthetic evaluator error"), KeyboardInterrupt(), SystemExit(7)],
)
def test_oracle_exceptions_remain_failure_no_retry(saved, exception):
    def fail(_):
        raise exception

    receipt, calls, _ = verify(saved, fail)
    assert (
        receipt["status"] == "failed" and type(exception).__name__ in receipt["error"]
    )
    assert len(calls) == receipt["verification"]["verification_calls"] == 1
    assert receipt["verification"]["completed_verification_calls"] == 0


def test_legitimate_zero_and_tolerance_all_unique_rows_preserved(saved):
    saved["events"][0]["child_oracle"]["score"] = 0.0
    saved["events"][2]["child_oracle"]["score"] = 0.0
    save(saved)
    receipt, calls, _ = verify(
        saved, lambda x: [0.0 if len(x[0]) == 20 else (len(x[0]) - 19) / 10 + 5e-13]
    )
    assert receipt["status"] == "verified" and len(calls) == 3
    assert 0 < receipt["verification"]["max_abs_error"] <= 1e-12


def test_finite_mismatch_evaluates_all_unique_once_then_fails(saved):
    receipt, calls, _ = verify(saved, lambda x: [(len(x[0]) - 19) / 10 + 2e-12])
    assert receipt["status"] == "failed" and "tolerance" in receipt["error"]
    assert len(calls) == len(receipt["verification"]["rows"]) == 3
    assert receipt["verification"]["mismatch_count"] == 3


def test_post_scoring_input_mutation_fails_without_changing_original_scores(saved):
    def score(values):
        saved["oracle_source"].write_text("# changed dependency\n")
        return [(len(values[0]) - 19) / 10]

    receipt, calls, _ = verify(saved, score)
    assert receipt["status"] == "failed" and "hash differs" in receipt["error"]
    assert len(calls) == receipt["verification"]["completed_verification_calls"] == 3


def test_cli_existing_output_refuses_before_verification(saved, monkeypatch):
    output = saved["root"] / "receipt.json"
    output.write_text("preserved")
    monkeypatch.setattr(
        audit, "verify_run", lambda *a, **k: pytest.fail("Must not invoke verifier")
    )
    with pytest.raises(ValueError, match="fresh file"):
        audit.main(
            [
                "--input-root",
                str(saved["root"]),
                "--protocol",
                str(saved["protocol_path"]),
                "--entry-id",
                saved["entry"]["id"],
                "--run-directory",
                str(saved["run"]),
                "--output",
                str(output),
            ]
        )
    assert output.read_text() == "preserved"


def test_cli_output_inside_run_refuses_before_verification(saved, monkeypatch):
    monkeypatch.setattr(
        audit, "verify_run", lambda *a, **k: pytest.fail("Must not invoke verifier")
    )
    with pytest.raises(ValueError, match="original run directory"):
        audit.main(
            [
                "--input-root",
                str(saved["root"]),
                "--protocol",
                str(saved["protocol_path"]),
                "--entry-id",
                saved["entry"]["id"],
                "--run-directory",
                str(saved["run"]),
                "--output",
                str(saved["run"] / "new.json"),
            ]
        )


def test_actual_runner_and_cached_oracle_artifacts_verify_with_synthetic_scoring(
    saved, monkeypatch
):
    """Exercise the producer, including real config types and sampling receipts."""
    from types import SimpleNamespace
    import torch
    import yaml
    from scripts.exps.pmo import run_ablation as runner
    from scripts.exps.pmo import udlm_sampling as sampling
    from test_pmo_udlm_adapter_runner import SyntheticSampler, arguments

    root, entry, protocol = saved["root"], saved["entry"], saved["protocol"]
    model = root / "synthetic_model.ckpt"
    model.write_bytes(b"synthetic; never unpickled")
    entry["checkpoint"] = str(model.relative_to(root))
    entry["checkpoint_sha256"] = audit.sha(model.read_bytes())
    sampling_path = root / entry["sampling_config"]
    sampling_path.write_text(
        yaml.safe_dump(
            dict(
                checkpoint_sha256=entry["checkpoint_sha256"],
                diffusion_type="udlm",
                parameterization="x0_denoiser",
                temperature_space="x0_denoiser",
                softmax_temp=0.5,
                randomness=0,
                min_add_len=18,
                num_steps=2,
                inference_eps=1e-5,
                exclude_special_tokens=False,
                prior_variant="schedule_uniform",
                prior_metadata_sha256="a" * 64,
                raw_loo_top_p=1.0,
            )
        )
    )
    entry["sampling_config_sha256"] = audit.sha(sampling_path.read_bytes())
    vocab = root / entry["vocabulary"]
    vocab.write_text("frag,score,size\n[1*]CC,0.9,2\n[1*]CN,0.8,2\n[1*]CO,0.7,2\n")
    entry["vocabulary_sha256"] = audit.sha(vocab.read_bytes())
    protocol["runner"].update(
        warmup=0,
        legacy_warmup_off_by_one=False,
        checkpoint_every=1,
        reporting_frequency=1,
        population_size=3,
        max_iterations=5,
    )
    dump(saved["protocol_path"], protocol)
    protocol_sha = audit.sha(saved["protocol_path"].read_bytes())
    # Remove only the pre-existing handwritten fixture files in this test's
    # exclusive temporary run directory; the real producer requires a fresh run.
    for path in saved["run"].iterdir():
        path.unlink()
    saved["run"].rmdir()
    sampler = SyntheticSampler(outputs=tuple("C" * n for n in (20, 20, 21, 22)))
    policy = saved["manifest"]["extra"]["sampling"]["oracle_call_protocol"]

    def prepare(contract, **kwargs):
        return sampling.SamplingAdapter(
            sampler,
            dict(
                contract=contract,
                sampler_kwargs=sampling.modification_kwargs(
                    contract,
                    gamma=kwargs["gamma"],
                    guidance_scale=kwargs["guidance_scale"],
                ),
                configured_nfe_per_generation=2,
                implementation_inputs={},
                oracle_call_protocol=policy,
                test_only="synthetic preparation",
            ),
        )

    online_calls = []

    def synthetic_oracle(**kwargs):
        assert kwargs == {"name": audit.ORACLE}

        def score(values):
            assert type(values) is list and len(values) == 1
            online_calls.append(values[0])
            return [len(values[0]) / 100]

        return score

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        runner,
        "_git_metadata",
        lambda: dict(commit="a" * 40, tracked_diff_sha256="b" * 64),
    )
    monkeypatch.setattr(runner, "_runtime_metadata", lambda _: {"synthetic": True})
    monkeypatch.setattr(runner, "_attach_fragments", lambda *_: "C" * 20)
    monkeypatch.setattr(runner, "TDCOracle", synthetic_oracle)
    monkeypatch.setattr(sampling, "prepare", prepare)
    args = arguments(
        SimpleNamespace(model=model, vocab=vocab, config=sampling_path, root=root),
        oracle=audit.ORACLE,
        experiment_id=entry["id"],
        seed=entry["seed"],
        max_oracle_calls=3,
        output_root=root / protocol["output_root"] / "runs",
        matrix_path=saved["protocol_path"],
        matrix_sha256=protocol_sha,
        **protocol["runner"],
    )
    actual_run = runner.run(args)
    assert actual_run == saved["run"]
    assert online_calls == ["C" * n for n in (20, 21, 22)]
    manifest = json.loads((actual_run / "manifest.json").read_text())
    summary = json.loads((actual_run / "summary.json").read_text())
    events = [
        json.loads(line)
        for line in (actual_run / "events.jsonl").read_text().splitlines()
    ]
    assert [row["child_oracle"]["charged"] for row in events] == [
        True,
        False,
        True,
        True,
    ]
    saved["manifest"], saved["summary"], saved["events"] = manifest, summary, events
    saved["request"]["plan"]["panel_input"]["sha256"] = protocol_sha
    saved["request"]["plan"]["jobs"][0].update(
        config=manifest["config"], config_sha256=manifest["config_sha256"]
    )
    saved["terminal"]["panel_sha256"] = protocol_sha
    # Bind the original producer bytes directly; never rewrite its JSON/JSONL.
    before = {
        name: (actual_run / name).read_bytes()
        for name in ("manifest.json", "summary.json", "events.jsonl")
    }
    artifacts = {name: fingerprint(actual_run / name, root) for name in before}
    saved["terminal"]["jobs"][0]["acceptance"]["artifacts"] = artifacts
    controller = root / protocol["output_root"]
    dump(controller / "request_manifest.json", saved["request"])
    saved["terminal"]["request_sha256"] = audit.sha(
        (controller / "request_manifest.json").read_bytes()
    )
    dump(controller / "terminal_manifest.json", saved["terminal"])
    receipt, offline_calls, _ = verify(saved, lambda values: [len(values[0]) / 100])
    assert receipt["status"] == "verified", receipt["error"]
    assert offline_calls == [[smiles] for smiles in online_calls]
    assert receipt["verification"]["duplicates_checked"] == 1
    assert all(
        (actual_run / name).read_bytes() == payload for name, payload in before.items()
    )
