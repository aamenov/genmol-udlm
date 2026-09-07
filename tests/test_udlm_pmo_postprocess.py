"""Synthetic orchestration only: no actual PMO artifacts, oracle or model calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.udlm import postprocess_v14_pmo as post


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def study(tmp_path, monkeypatch):
    sources = tmp_path / "run_sources"
    names = (*post.PINS, "udlm_pmo_overview_worktree")
    directories = {name: sources / name for name in names}
    for directory in directories.values():
        directory.mkdir(parents=True)
    root = directories["udlm_genmol_worktree"]
    own = directories["udlm_pmo_overview_worktree"]
    wrapper = own / "scripts/udlm/postprocess_v14_pmo.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("# Synthetic wrapper identity; never executed\n")
    campaign = root / "output/udlm/engineering_v14_pmo"
    output = root / "output/udlm/synthetic_postprocessing"
    entries = [
        {"id": f"v14-{arm}-{seed}", "seed": seed}
        for seed in (2300, 2301)
        for arm in ("mdlm", "s-ct", "mask-ce")
    ]
    write_json(root / post.PANEL, {"entries": entries})
    pipeline_path = root / "output/logs/engineering-v14-pmo-pipeline-status.json"
    pipeline = {"source": post.PINS["udlm_genmol_worktree"], "return_code": 0}
    terminal_path = campaign / "terminal_manifest.json"
    terminal = {
        "status": "completed",
        "final_input_validation": "unchanged",
        "lease_release_authorized": True,
    }
    write_json(pipeline_path, pipeline)
    write_json(terminal_path, terminal)
    pins = {**post.PINS, own.name: "a" * 40}
    observed_git = []

    def fake_git(directory, *args):
        observed_git.append((Path(directory).name, args))
        if args[0] == "status":
            return ""
        assert args in (("rev-parse", "HEAD"), ("rev-parse", "@{upstream}"))
        return pins[Path(directory).name]

    for name, value in {
        "SOURCES": sources,
        "ROOT": root,
        "CAMPAIGN": campaign,
        "OUTPUT": output,
        "PANEL_SHA": post.sha(root / post.PANEL),
        "__file__": str(wrapper),
    }.items():
        monkeypatch.setattr(post, name, value)
    monkeypatch.setattr(post, "git", fake_git)
    monkeypatch.setattr(post.time, "sleep", lambda _: None)
    return SimpleNamespace(
        root=root,
        output=output,
        campaign=campaign,
        pipeline=pipeline,
        pipeline_path=pipeline_path,
        terminal=terminal,
        terminal_path=terminal_path,
        entries=entries,
        pins=pins,
        git=fake_git,
        git_calls=observed_git,
        directories=directories,
    )


def fake_steps(study, monkeypatch, *, failed=None, malformed=None, reporter_error=0):
    calls = []

    def execute(label, worktree, arguments, record, pins):
        argv = [str(value) for value in arguments]
        calls.append((label, argv))
        record["steps"].append({"label": label, "command": argv})
        target = Path(argv[argv.index("--output") + 1])
        if label.startswith("v14-"):
            identifier = argv[argv.index("--entry-id") + 1]
            target.parent.mkdir(parents=True, exist_ok=True)
            if identifier == malformed:
                target.write_text("{interrupted synthetic receipt")
                return 1
            write_json(
                target,
                {
                    "status": "failed" if identifier == failed else "verified",
                    "run": {"entry_id": identifier},
                    "inputs": {
                        "controller_terminal": {"sha256": post.sha(study.terminal_path)}
                    },
                    "verification": {
                        "verification_calls": 3 if identifier == failed else 2000
                    },
                },
            )
            return int(identifier == failed)
        if label == "report":
            if reporter_error:
                return reporter_error
            receipt_args = [
                argv[i + 1 : i + 4]
                for i, arg in enumerate(argv)
                if arg == "--verification"
            ]
            receipts = []
            for identifier, path, sha in receipt_args:
                assert post.sha(path) == sha
                receipt = json.loads(Path(path).read_bytes())
                assert receipt["run"]["entry_id"] == identifier
                receipts.append(receipt)
            verified = len(receipts) == 6 and all(
                row["status"] == "verified" for row in receipts
            )
            write_json(
                target / "report.json",
                {
                    "status": "verified_complete"
                    if verified
                    else "incomplete_or_unrankable",
                    "outside_budget_verification_calls": sum(
                        row["verification"]["verification_calls"] for row in receipts
                    ),
                },
            )
            write_json(target / "manifest.json", {"synthetic": True})
            (target / "report.pdf").write_bytes(b"SYNTHETIC PDF, NO EXPERIMENT")
            return 0
        assert label == "combine"
        target.mkdir()
        (target / "study_overview.pdf").write_bytes(b"SYNTHETIC COMBINED PDF")
        return 0

    monkeypatch.setattr(post, "execute", execute)
    return calls


def terminal_record(study):
    return json.loads((study.output / "terminal.json").read_bytes())


def test_six_verifications_report_and_combiner_use_exact_receipt_paths(
    study, monkeypatch
):
    calls = fake_steps(study, monkeypatch)
    assert post.main() == 0
    ids = [entry["id"] for entry in study.entries]
    assert [label for label, _ in calls] == ids + ["report", "combine"]
    for (label, argv), entry in zip(calls, study.entries):
        expected = (
            study.campaign
            / "runs"
            / label
            / "fexofenadine_mpo/released"
            / f"seed_{entry['seed']}"
        )
        assert argv[argv.index("--run-directory") + 1] == str(expected)
        assert argv[argv.index("--output") + 1] == str(
            study.output / "verification" / f"{label}.json"
        )
    report_args = calls[-2][1]
    assert report_args.count("--verification") == 6
    assert report_args[report_args.index("--terminal-sha256") + 1] == post.sha(
        study.terminal_path
    )
    combine_args = calls[-1][1]
    assert combine_args[combine_args.index("--pmo-manifest-sha256") + 1] == post.sha(
        study.output / "report/manifest.json"
    )
    result = terminal_record(study)
    assert result["status"] == "published_locally"
    assert result["scientific_report_status"] == "verified_complete"
    assert result["outside_budget_verification_calls"] == 12000


@pytest.mark.parametrize(
    "reason", ["failed", "bad_return", "inputs_changed", "lease_not_authorized"]
)
def test_unaccepted_campaign_runs_no_oracles_but_retains_unrankable_pdf(
    study, monkeypatch, reason
):
    if reason == "failed":
        study.terminal["status"] = "failed"
        study.pipeline["return_code"] = 1
    elif reason == "bad_return":
        study.pipeline["return_code"] = 1
    elif reason == "inputs_changed":
        study.terminal["final_input_validation"] = "changed"
    else:
        study.terminal["lease_release_authorized"] = False
    write_json(study.terminal_path, study.terminal)
    write_json(study.pipeline_path, study.pipeline)
    calls = fake_steps(study, monkeypatch)
    assert post.main() == 2
    assert [label for label, _ in calls] == ["report", "combine"]
    assert "--verification" not in calls[0][1]
    result = terminal_record(study)
    assert "no verification oracle calls" in result["verification_skipped"]
    assert result["scientific_report_status"] == "incomplete_or_unrankable"
    assert result["outside_budget_verification_calls"] == 0
    assert (study.output / "combined/study_overview.pdf").is_file()


def test_failed_valid_receipt_retained_and_remaining_checks_run_once(
    study, monkeypatch
):
    failed = study.entries[1]["id"]
    calls = fake_steps(study, monkeypatch, failed=failed)
    assert post.main() == 2
    assert [label for label, _ in calls] == [entry["id"] for entry in study.entries] + [
        "report",
        "combine",
    ]
    assert calls[-2][1].count("--verification") == 6
    assert (
        json.loads((study.output / "verification" / f"{failed}.json").read_bytes())[
            "status"
        ]
        == "failed"
    )
    assert terminal_record(study)["outside_budget_verification_calls"] == 10003


def test_reporter_error_retains_failure_and_never_combines(study, monkeypatch):
    calls = fake_steps(study, monkeypatch, reporter_error=17)
    assert post.main() == 1
    assert [label for label, _ in calls][-1] == "report"
    assert not (study.output / "combined").exists()
    assert "return code 17" in terminal_record(study)["error"]


def test_malformed_receipt_stops_without_retry_or_fabrication(study, monkeypatch):
    first = study.entries[0]["id"]
    calls = fake_steps(study, monkeypatch, malformed=first)
    assert post.main() == 1
    assert [label for label, _ in calls] == [first]
    assert "JSONDecodeError" in terminal_record(study)["error"]
    assert (
        study.output / "verification" / f"{first}.json"
    ).read_text() == "{interrupted synthetic receipt"


def test_existing_namespace_prevents_any_steps(study, monkeypatch):
    study.output.mkdir(parents=True)
    sentinel = study.output / "original.json"
    sentinel.write_text("PRESERVE")
    calls = fake_steps(study, monkeypatch)
    with pytest.raises(FileExistsError):
        post.main()
    assert calls == [] and sentinel.read_text() == "PRESERVE"


def test_source_guard_prevents_namespace_and_calls(study, monkeypatch):
    def dirty(directory, *args):
        return " M source.py" if args[0] == "status" else study.git(directory, *args)

    monkeypatch.setattr(post, "git", dirty)
    calls = fake_steps(study, monkeypatch)
    with pytest.raises(ValueError, match="Source changed"):
        post.main()
    assert not study.output.exists() and calls == []


@pytest.mark.parametrize("kind", ["file", "dangling_symlink"])
def test_retained_generation_lease_prevents_every_postprocessing_step(
    study, monkeypatch, kind
):
    lock = study.root / "output/.single_generation_job.lock"
    if kind == "file":
        lock.write_text("PRESERVE FOREIGN OR RETAINED LEASE")
    else:
        lock.symlink_to(lock.parent / "missing-owner")
    calls = fake_steps(study, monkeypatch)
    assert post.main() == 1
    assert calls == []
    result = terminal_record(study)
    assert result["status"] == "failed"
    assert "lease" in result["error"].lower()
    assert (
        lock.is_symlink()
        if kind == "dangling_symlink"
        else lock.read_text() == "PRESERVE FOREIGN OR RETAINED LEASE"
    )


def execute_fixture(study):
    study.output.mkdir(parents=True)
    return {"controller_terminal_sha256": post.sha(study.terminal_path), "steps": []}


def test_execute_cpu_environment_and_hashed_terminal_before_after(study, monkeypatch):
    record = execute_fixture(study)
    invocations = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-MUST-NOT-BE-USED")
    monkeypatch.setenv("PYTHONHASHSEED", "123")

    def run(command, **kwargs):
        invocations.append((command, kwargs))
        kwargs["stdout"].write("Synthetic subprocess only\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(post.subprocess, "run", run)
    worktree = study.directories["udlm_pmo_rescore_worktree"]
    assert post.execute("cpu", worktree, ["synthetic.py"], record, study.pins) == 0
    assert len(invocations) == 1
    command, args = invocations[0]
    assert command == [str(post.PYTHON), "-u", "synthetic.py"]
    assert args["cwd"] == worktree and args["timeout"] == 3600
    assert args["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert args["env"]["PYTHONHASHSEED"] == "0"
    assert all(
        args["env"][key] == "1"
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    assert args["env"]["TOKENIZERS_PARALLELISM"] == "false"
    assert args["env"]["PYTHONPATH"] == f"{worktree}/src:{worktree}"
    assert json.loads((study.output / "cpu.exit.json").read_bytes())["return_code"] == 0
    assert len(study.git_calls) == 2 * 3 * len(study.pins)


@pytest.mark.parametrize("phase", ["before", "during"])
def test_terminal_drift_blocks_or_invalidates_step(study, monkeypatch, phase):
    record = execute_fixture(study)
    invocations = []

    def mutate():
        write_json(study.terminal_path, {**study.terminal, "tampered": True})

    def run(*args, **kwargs):
        invocations.append(args)
        mutate()
        return SimpleNamespace(returncode=0)

    if phase == "before":
        mutate()
    monkeypatch.setattr(post.subprocess, "run", run)
    with pytest.raises(ValueError, match="terminal changed"):
        post.execute("drift", study.root, ["synthetic.py"], record, study.pins)
    assert len(invocations) == int(phase == "during")


def test_execute_source_guard_prevents_subprocess(study, monkeypatch):
    record = execute_fixture(study)
    monkeypatch.setattr(post, "git", lambda *_: "changed")
    monkeypatch.setattr(
        post.subprocess, "run", lambda *a, **k: pytest.fail("subprocess launched")
    )
    with pytest.raises(ValueError, match="Source changed"):
        post.execute("drift", study.root, ["synthetic.py"], record, study.pins)
