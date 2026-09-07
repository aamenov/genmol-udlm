"""CPU-only guards for the separate V8b CE training decision."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts.udlm import launch_ce_followup_training as launcher


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = launcher.ROOT
    shutil.copytree(root / "configs", tmp_path / "configs")
    for relative in [
        "scripts/train.py",
        *[str(p.relative_to(root)) for p in (root / "src").rglob("*.py")],
    ]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    original_path = tmp_path / launcher.original.PROTOCOL
    original_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / launcher.original.PROTOCOL, original_path)
    ct = launcher.original.build_plans(2, root=tmp_path)[0]
    protocol = json.loads((root / launcher.PROTOCOL).read_text())
    source = {"head": launcher.ORIGINAL_SOURCE, "upstream": launcher.ORIGINAL_SOURCE}
    documents = {
        "original_ct_terminal": {
            "status": "failed",
            "source": source,
            "training_return_code": 0,
            "completed_example_exposures": None,
            "plan": ct,
        },
        "original_campaign_terminal": {
            "status": "failed",
            "source": source,
            "arms": [
                {"arm_id": "ct", "status": "failed"},
                {"arm_id": "ce", "status": "not_completed_after_campaign_stop"},
            ],
        },
    }
    references = copy.deepcopy(launcher.EVIDENCE)
    for name, document in documents.items():
        references[name]["sha256"] = write_json(
            tmp_path / references[name]["relative_path"], document
        )
    documents["post_exit_audit"] = {
        "checkpoint_status": "separately_validated_after_controller_failure",
        "original_campaign_status": "failed",
        "original_training_return_code": 0,
        "source": source,
        "leases_released": True,
        "original_terminal_sha256": references["original_ct_terminal"]["sha256"],
        "original_campaign_terminal_sha256": references["original_campaign_terminal"][
            "sha256"
        ],
        "checkpoint": {
            "relative_path": f"{launcher.ORIGINAL_OUTPUT}/ct/checkpoints/1000.ckpt",
            "sha256": protocol["separately_validated_ct_checkpoint_sha256"],
            "size_bytes": 1636492424,
            "global_step": 1000,
        },
        "configured_example_exposures_supported_by_step_and_batch": 128000,
        "training_implementation_sha256": {
            relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
            for relative in [
                "scripts/train.py",
                *[
                    str(p.relative_to(tmp_path))
                    for p in (tmp_path / "src").rglob("*.py")
                ],
            ]
        },
    }
    references["post_exit_audit"]["sha256"] = write_json(
        tmp_path / references["post_exit_audit"]["relative_path"],
        documents["post_exit_audit"],
    )
    protocol["evidence"] = references
    write_json(tmp_path / launcher.PROTOCOL, protocol)
    monkeypatch.setattr(launcher, "EVIDENCE", references)
    snapshot = launcher.artifact_io.snapshot_file
    checkpoint = documents["post_exit_audit"]["checkpoint"]

    def snapshot_without_large_checkpoint(root, relative, **kwargs):
        if str(relative) == checkpoint["relative_path"]:
            return SimpleNamespace(
                sha256=checkpoint["sha256"], size_bytes=checkpoint["size_bytes"]
            ), None
        return snapshot(root, relative, **kwargs)

    monkeypatch.setattr(
        launcher.artifact_io, "snapshot_file", snapshot_without_large_checkpoint
    )
    return tmp_path, protocol, documents


def test_followup_is_exact_original_ce_except_output_with_fresh_mdlm_start(study):
    root, protocol, _ = study
    before = launcher.original.build_plans(2, root=root)[1]
    plan = launcher.build_plan(2, root=root)
    expected = copy.deepcopy(before["config"])
    expected["callback"]["dirpath"] = str(root / launcher.OUTPUT / "checkpoints")
    assert plan["config"] == expected
    assert plan["config"]["trainer"]["accumulate_grad_batches"] == 4
    assert plan["config"]["training"]["init_from_mdlm_ema"] is True
    assert plan["checkpoint_path"] == before["checkpoint_path"]
    assert plan["checkpoint_sha256"] == before["checkpoint_sha256"]
    assert plan["output_relative"] == launcher.OUTPUT
    assert "campaign_relative" not in plan
    assert plan["protocol"] == protocol
    assert plan["protocol"]["claim"].endswith("original_v8_campaign_remains_failed")
    assert plan["training_argv"][4] == Path(launcher.CONFIG).stem
    assert f"hydra.run.dir={root / launcher.OUTPUT / 'hydra'}" in plan["training_argv"]
    assert (
        launcher.engine.canonical_digest(plan["training_argv"]) == plan["argv_sha256"]
    )
    assert plan["example_exposures"] == 128000


@pytest.mark.parametrize("count", [1, 3, True, False, 2.0, "2", None])
def test_followup_requires_same_two_gpu_training_count(count):
    with pytest.raises(ValueError, match="same two-GPU"):
        launcher.build_plan(count)


@pytest.mark.parametrize("reference", list(launcher.EVIDENCE))
def test_incident_evidence_byte_changes_rejected_before_launch(study, reference):
    root, protocol, _ = study
    path = root / protocol["evidence"][reference]["relative_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        launcher.build_plan(2, root=root)


@pytest.mark.parametrize("mutation", ["missing", "changed", "wrong_size"])
def test_actual_ct_checkpoint_must_still_match_separate_audit(
    study, monkeypatch, mutation
):
    root, _, documents = study
    checkpoint = documents["post_exit_audit"]["checkpoint"]
    snapshot = launcher.artifact_io.snapshot_file

    def changed_checkpoint(root, relative, **kwargs):
        if str(relative) == checkpoint["relative_path"]:
            if mutation == "missing":
                raise FileNotFoundError("CT checkpoint missing")
            return SimpleNamespace(
                sha256="f" * 64 if mutation == "changed" else checkpoint["sha256"],
                size_bytes=0 if mutation == "wrong_size" else checkpoint["size_bytes"],
            ), None
        return snapshot(root, relative, **kwargs)

    monkeypatch.setattr(launcher.artifact_io, "snapshot_file", changed_checkpoint)
    with pytest.raises((FileNotFoundError, ValueError), match="CT checkpoint"):
        launcher.build_plan(2, root=root)


@pytest.mark.parametrize("mutation", ["source_change", "extra_source"])
def test_training_code_must_remain_identical_to_audited_ct(study, mutation):
    root, _, _ = study
    path = root / (
        "src/genmol/model.py" if mutation == "source_change" else "src/genmol/extra.py"
    )
    with path.open("a") as stream:
        stream.write("\n# Changed training implementation\n")
    with pytest.raises(ValueError, match="training implementation differs"):
        launcher.build_plan(2, root=root)


def test_changed_followup_training_setting_rejected_even_with_new_config_hash(study):
    root, protocol, _ = study
    path = root / launcher.CONFIG
    with path.open("a") as stream:
        stream.write("\nseed: 1501\n")
    protocol["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root / launcher.PROTOCOL, protocol)
    with pytest.raises(ValueError, match="beyond its output directory"):
        launcher.build_plan(2, root=root)


def test_original_protocol_source_cannot_be_relabelled(study):
    root, protocol, _ = study
    protocol["original_source_revision"] = "f" * 40
    write_json(root / launcher.PROTOCOL, protocol)
    with pytest.raises(ValueError, match="training design changed"):
        launcher.build_plan(2, root=root)


def test_live_followup_requires_exact_reviewed_cleanup_grace(monkeypatch):
    launcher.require_cleanup_grace()
    monkeypatch.setattr(launcher.engine, "PROCESS_GROUP_EXIT_GRACE_SECONDS", 0)
    with pytest.raises(RuntimeError, match="reviewed bounded"):
        launcher.require_cleanup_grace()


def test_dry_run_never_queries_gpus_or_acquires_leases(study, monkeypatch, capsys):
    root, _, _ = study
    plan = launcher.build_plan(2, root=root)
    source = {"head": "a" * 40, "upstream": "a" * 40}
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "build_plan", lambda count: plan)
    monkeypatch.setattr(
        launcher.engine.benchmark, "_require_clean_pushed_source", lambda: source
    )
    monkeypatch.setattr(launcher.engine, "verify_checkpoint_input", lambda plan: None)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run touched resources or launched training")

    monkeypatch.setattr(launcher.engine, "execute", forbidden)
    monkeypatch.setattr(launcher.engine, "acquire_leases", forbidden)
    monkeypatch.setattr(launcher.engine.audited, "probe_all_gpus", forbidden)
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert launcher.main(["--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["gpu_queries"] == result["artifact_mutations"] == 0
    assert result["plan"] == plan
    assert before == sorted(str(path.relative_to(root)) for path in root.rglob("*"))
