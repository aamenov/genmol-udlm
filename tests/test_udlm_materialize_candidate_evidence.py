from __future__ import annotations

import copy
import hashlib
import json
import stat
from types import SimpleNamespace

import pytest

from scripts import artifact_io
from scripts.udlm import materialize_candidate_evidence as materialize


def _digest(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(payload).hexdigest()


def _claim(
    path: str, payload: bytes, *, digest: str | None = None
) -> artifact_io.FileClaim:
    return artifact_io.FileClaim(
        relative_path=path,
        device=1,
        inode=abs(hash(path)) + 1,
        mode=stat.S_IFREG | 0o444,
        link_count=1,
        size_bytes=len(payload),
        mtime_ns=1,
        ctime_ns=1,
        sha256=digest or _digest(payload),
    )


def _stage_configs(stage_id: str) -> list[str]:
    if stage_id == "D":
        return ["e_t100_p100"]
    if stage_id == "A":
        return sorted(
            f"{arm}_t{temperature}_p100"
            for arm in "rse"
            for temperature in ("050", "070", "085", "100")
        )
    if stage_id == "B":
        return sorted(
            f"{arm}_t{temperature}_p{top_p}"
            for arm in "rse"
            for temperature in ("050", "070")
            for top_p in ("095", "098", "100")
        )
    if stage_id == "C":
        return sorted(
            f"{arm}_t{temperature}_p095"
            for arm in "rse"
            for temperature in ("050", "070")
        )
    assert stage_id == "eligible"
    return sorted(f"{arm}_t050_p095" for arm in "rse")


def _valid_manifest() -> dict:
    stage_decisions = []
    children = []
    for stage_id in materialize.campaign.STAGE_IDS:
        stage_path = materialize.campaign._decision_paths(stage_id)[1]
        stage_decisions.append(
            {
                "stage_id": stage_id,
                "relative_path": stage_path,
                "sha256": _digest(stage_path),
                "schema_version": 1,
            }
        )
        for config_id in _stage_configs(stage_id):
            attempt_id = f"stage-{stage_id.lower()}-{config_id}"
            candidate_id = materialize.campaign.CANDIDATE_IDS[config_id[0].upper()]
            for seed in materialize.campaign.STAGE_CONTRACT[stage_id]["seeds"]:
                live_path = (
                    "output/udlm/de_novo_candidate_campaign_v1/evidence/"
                    f"{attempt_id}/seed_{seed}.json"
                )
                tracked_path = f"experiments/udlm/pilots/{attempt_id}/seed_{seed}.json"
                envelope_sha = _digest(f"envelope:{attempt_id}:{seed}")
                support = [
                    {
                        "artifact_role": "benchmark_raw_samples_csv",
                        "relative_path": (
                            "output/udlm/de_novo_candidate_campaign_v1/attempts/"
                            f"{attempt_id}/seed_{seed}/raw_samples.csv"
                        ),
                        "sha256": _digest(f"raw:{attempt_id}:{seed}"),
                        "size_bytes": 17,
                        "schema_version": None,
                    },
                    {
                        "artifact_role": "benchmark_summary_json",
                        "relative_path": (
                            "output/udlm/de_novo_candidate_campaign_v1/attempts/"
                            f"{attempt_id}/seed_{seed}/summary.json"
                        ),
                        "sha256": _digest(f"summary:{attempt_id}:{seed}"),
                        "size_bytes": 19,
                        "schema_version": 8,
                    },
                    {
                        "artifact_role": "training_exit_receipt",
                        "relative_path": (
                            f"output/udlm/scaleup-w1-{config_id[0]}-dcb271453411/"
                            "pilot_exit_status.json"
                        ),
                        "sha256": _digest(f"receipt:{config_id[0]}"),
                        "size_bytes": 23,
                        "schema_version": 5,
                    },
                ]
                children.append(
                    {
                        "stage_id": stage_id,
                        "config_id": config_id,
                        "attempt_id": attempt_id,
                        "candidate_id": candidate_id,
                        "pilot_seed": seed,
                        "artifact_kind": "pilot_evaluation",
                        "live_envelope": {
                            "relative_path": live_path,
                            "sha256": envelope_sha,
                            "size_bytes": 101,
                            "schema_version": 2,
                        },
                        "tracked_envelope": {
                            "relative_path": tracked_path,
                            "sha256": envelope_sha,
                            "size_bytes": 101,
                            "schema_version": 2,
                        },
                        "supporting_artifacts": support,
                    }
                )
    required = sorted(
        {
            *(row["relative_path"] for row in stage_decisions),
            *(row["tracked_envelope"]["relative_path"] for row in children),
            *(
                support["relative_path"]
                for row in children
                for support in row["supporting_artifacts"]
            ),
        }
    )
    return {
        "schema_version": 1,
        "protocol_id": materialize.PROTOCOL_ID,
        "status": materialize.MANIFEST_STATUS,
        "registry": {
            "relative_path": materialize.campaign.REGISTRY_RELATIVE_PATH,
            "sha256": "1" * 64,
            "canonical_sha256": "2" * 64,
            "size_bytes": 123,
            "schema_version": 1,
        },
        "source_revision": {"head": "3" * 40, "upstream": "3" * 40},
        "counts": {
            "executed_entry_count": 40,
            "child_outcome_count": 43,
            "requested_molecule_count": 3680,
            "stage_decision_count": 5,
            "tracked_envelope_count": 43,
            "required_git_path_count": len(required),
        },
        "stage_decisions": stage_decisions,
        "children": children,
        "required_git_paths": required,
    }


def test_manifest_exact_schema_and_reference_derived_closure() -> None:
    manifest = _valid_manifest()
    assert materialize.validate_manifest(manifest) == manifest

    for mutation in (
        lambda value: value.__setitem__("evidence_revision", "4" * 40),
        lambda value: value["children"][0]["tracked_envelope"].__setitem__(
            "sha256", "5" * 64
        ),
        lambda value: value["children"].reverse(),
        lambda value: value["required_git_paths"].pop(),
        lambda value: value["required_git_paths"].append(
            materialize.MANIFEST_RELATIVE_PATH
        ),
    ):
        tampered = copy.deepcopy(manifest)
        mutation(tampered)
        with pytest.raises(materialize.CandidateEvidenceError):
            materialize.validate_manifest(tampered)


def _plan_inputs(manifest: dict) -> tuple[dict, dict]:
    entries: dict[str, dict[str, dict]] = {
        stage: {} for stage in materialize.campaign.STAGE_IDS
    }
    for child in manifest["children"]:
        stage_entries = entries[child["stage_id"]]
        entry = stage_entries.setdefault(
            child["attempt_id"],
            {
                "attempt_id": child["attempt_id"],
                "candidate_id": child["candidate_id"],
                "config_id": child["config_id"],
                "child_outcomes": [],
            },
        )
        entry["child_outcomes"].append(
            {
                "artifact_kind": child["artifact_kind"],
                "pilot_seed": child["pilot_seed"],
                "relative_path": child["live_envelope"]["relative_path"],
                "sha256": child["live_envelope"]["sha256"],
                "schema_version": 2,
            }
        )
    decisions = {
        stage: {"status": "completed", "entries": list(entries[stage].values())}
        for stage in materialize.campaign.STAGE_IDS
    }
    completions = {
        row["stage_id"]: {
            key: row[key] for key in ("relative_path", "sha256", "schema_version")
        }
        for row in manifest["stage_decisions"]
    }
    return decisions, completions


def test_plan_visits_all_43_children_before_returning_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _valid_manifest()
    decisions, completions = _plan_inputs(manifest)
    registry = SimpleNamespace(
        reference=manifest["registry"], raw_sha256="1" * 64, size_bytes=123, configs=()
    )
    expected_children = iter(manifest["children"])
    calls = []
    monkeypatch.setattr(
        materialize, "_assert_preexisting_sources_at_g", lambda *_: None
    )
    monkeypatch.setattr(materialize, "_git_blob_absent", lambda *_: True)

    def snapshot(path: str, _label: str, *, capture_bytes: bool = True):
        payload = b"stage"
        stage = next(
            row for row in manifest["stage_decisions"] if row["relative_path"] == path
        )
        return _claim(path, payload, digest=stage["sha256"]), (
            payload if capture_bytes else None
        )

    def build_child(**_kwargs):
        child = next(expected_children)
        calls.append((child["attempt_id"], child["pilot_seed"]))
        return child, artifact_io.PublishItem(
            child["tracked_envelope"]["relative_path"], b"formal-envelope"
        )

    monkeypatch.setattr(materialize, "_snapshot", snapshot)
    monkeypatch.setattr(materialize, "_build_tracked_child", build_child)
    plan = materialize.build_evidence_plan(
        registry,
        source_revision="3" * 40,
        decisions=decisions,
        completions=completions,
    )
    assert len(calls) == 43
    assert len(plan.members) == 43
    assert plan.manifest == manifest


def test_regenerated_envelope_must_be_byte_identical_to_live_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_id = "stage-a-e_t050_p100"
    seed = 1101
    summary_path = f"output/runs/{attempt_id}/seed_{seed}/summary.json"
    envelope = {
        "schema_version": 2,
        "artifact_kind": "pilot_evaluation",
        "status": "completed",
        "attempt_id": attempt_id,
        "candidate_id": "e-w1-1000u-dcb271453411",
        "pilot_seed": seed,
        "final_seed_results_included": False,
        "training_exit_receipt": {
            "relative_path": "output/udlm/scaleup-w1-e/pilot_exit_status.json",
            "sha256": "1" * 64,
            "schema_version": 5,
        },
        "benchmark_artifacts": {
            "summary_json": {
                "relative_path": summary_path,
                "sha256": "2" * 64,
                "schema_version": 8,
            },
            "raw_samples_csv": {
                "relative_path": f"output/runs/{attempt_id}/seed_{seed}/raw_samples.csv",
                "sha256": "3" * 64,
            },
        },
    }
    live_payload = materialize.canonical_json_bytes(envelope)
    live_path = (
        "output/udlm/de_novo_candidate_campaign_v1/evidence/"
        f"{attempt_id}/seed_{seed}.json"
    )
    live_claim = _claim(live_path, live_payload)
    monkeypatch.setattr(
        materialize,
        "_source_envelope",
        lambda *_args, **_kwargs: (envelope, live_claim, live_payload),
    )
    built = []

    def builder(**_kwargs):
        built.append(True)
        return envelope

    monkeypatch.setattr(
        materialize.write_pilot_evidence,
        "build_completed_pilot_evidence_payload",
        builder,
    )
    summary_payload = json.dumps({"num_samples": 32}).encode()
    monkeypatch.setattr(
        materialize,
        "_snapshot",
        lambda path, _label, *, capture_bytes=True: (
            _claim(path, summary_payload),
            summary_payload if capture_bytes else None,
        ),
    )
    monkeypatch.setattr(materialize, "_completed_support", lambda *_: [])
    monkeypatch.setattr(materialize.os.path, "lexists", lambda _path: False)
    source_child = {
        "artifact_kind": "pilot_evaluation",
        "pilot_seed": seed,
        "relative_path": live_path,
        "sha256": live_claim.sha256,
        "schema_version": 2,
    }
    child, member = materialize._build_tracked_child(
        stage_id="A",
        entry={
            "attempt_id": attempt_id,
            "candidate_id": envelope["candidate_id"],
            "config_id": "e_t050_p100",
        },
        source_child=source_child,
        requested_samples=32,
        retained={},
    )
    assert built == [True]
    assert member.payload == live_payload
    assert child["live_envelope"]["sha256"] == child["tracked_envelope"]["sha256"]

    altered = copy.deepcopy(envelope)
    altered["candidate_id"] = "r-w1-1000u-dcb271453411"
    monkeypatch.setattr(
        materialize.write_pilot_evidence,
        "build_completed_pilot_evidence_payload",
        lambda **_kwargs: altered,
    )
    with pytest.raises(materialize.CandidateEvidenceError, match="regenerated"):
        materialize._build_tracked_child(
            stage_id="A",
            entry={
                "attempt_id": attempt_id,
                "candidate_id": envelope["candidate_id"],
                "config_id": "e_t050_p100",
            },
            source_child=source_child,
            requested_samples=32,
            retained={},
        )


def test_materializer_publishes_43_members_with_manifest_completion_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _valid_manifest()
    plan = materialize.EvidencePlan(
        members=tuple(
            artifact_io.PublishItem(
                child["tracked_envelope"]["relative_path"], b"envelope"
            )
            for child in manifest["children"]
        ),
        manifest=manifest,
        retained_inputs=(),
    )
    state = {"head": "3" * 40, "upstream": "3" * 40}
    clean_checks = []
    monkeypatch.setattr(
        materialize,
        "_require_clean_pushed_source",
        lambda _revision: clean_checks.append(True) or state,
    )
    monkeypatch.setattr(materialize.os.path, "lexists", lambda _path: False)
    registry = object()
    monkeypatch.setattr(
        materialize.campaign, "load_registry", lambda *_args, **_kwargs: registry
    )
    monkeypatch.setattr(
        materialize.campaign, "replay_decisions", lambda _registry: ({}, {})
    )
    monkeypatch.setattr(
        materialize, "build_evidence_plan", lambda *_args, **_kwargs: plan
    )
    monkeypatch.setattr(materialize, "_revalidate_inputs", lambda _claims: None)
    monkeypatch.setattr(materialize, "_create_target_parents", lambda _plan: None)
    published = []

    def publish(_root, members, *, completion):
        published.append((members, completion))
        return SimpleNamespace(
            completion=_claim(completion.relative_path, completion.payload)
        )

    monkeypatch.setattr(materialize.artifact_io, "publish_bundle_exclusive", publish)
    result = materialize.materialize_candidate_evidence(
        expected_source_revision="3" * 40,
        expected_registry_sha256="1" * 64,
        expected_registry_canonical_sha256="2" * 64,
    )
    assert clean_checks == [True, True]
    assert len(published) == 1 and len(published[0][0]) == 43
    assert published[0][1].relative_path == materialize.MANIFEST_RELATIVE_PATH
    assert result.completion.relative_path == materialize.MANIFEST_RELATIVE_PATH


def test_stage_helper_force_adds_only_manifest_closure_without_committing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _valid_manifest()
    manifest_payload = materialize.canonical_json_bytes(manifest)
    expected = materialize._manifest_git_file_claims(manifest, manifest_payload)
    registry = SimpleNamespace(reference=manifest["registry"])
    monkeypatch.setattr(
        materialize.campaign, "load_registry", lambda *_args, **_kwargs: registry
    )
    calls = []

    def git_capture(*arguments: str) -> bytes:
        calls.append(arguments)
        if arguments[:2] == ("rev-parse", "HEAD") or arguments[:2] == (
            "rev-parse",
            "@{upstream}",
        ):
            return ("3" * 40 + "\n").encode()
        if arguments[0] == "add":
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr(materialize, "_git_capture", git_capture)
    monkeypatch.setattr(materialize, "_git_name_status", lambda *_args: ())
    monkeypatch.setattr(materialize, "_git_blob_absent", lambda *_args: True)

    def snapshot(path: str, _label: str, *, capture_bytes: bool = True):
        if path == materialize.MANIFEST_RELATIVE_PATH:
            payload = manifest_payload
            digest, size_bytes = expected[path]
        else:
            payload = path.encode()
            digest, size_bytes = expected[path]
        claim = SimpleNamespace(
            sha256=digest,
            size_bytes=len(payload) if size_bytes is None else size_bytes,
            mode=stat.S_IFREG | 0o644,
        )
        return claim, payload if capture_bytes else None

    monkeypatch.setattr(materialize, "_snapshot", snapshot)
    verified = []
    monkeypatch.setattr(
        materialize,
        "_verify_index_additions",
        lambda **kwargs: verified.append(kwargs),
    )
    result = materialize.stage_candidate_evidence(
        expected_source_revision="3" * 40,
        expected_registry_sha256="1" * 64,
        expected_registry_canonical_sha256="2" * 64,
    )
    add_call = next(call for call in calls if call[0] == "add")
    assert add_call[:3] == ("add", "-f", "--")
    assert list(add_call[3:]) == sorted(expected, key=str.encode)
    assert verified == [{"source_revision": "3" * 40, "expected": expected}]
    assert result["status"] == "staged_without_commit"
    assert result["addition_count"] == len(expected)


def test_stage_helper_rejects_executable_evidence_before_git_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _valid_manifest()
    manifest_payload = materialize.canonical_json_bytes(manifest)
    expected = materialize._manifest_git_file_claims(manifest, manifest_payload)
    registry = SimpleNamespace(reference=manifest["registry"])
    monkeypatch.setattr(
        materialize.campaign, "load_registry", lambda *_args, **_kwargs: registry
    )
    add_called = []

    def git_capture(*arguments: str) -> bytes:
        if arguments[0] == "rev-parse":
            return ("3" * 40 + "\n").encode()
        if arguments[0] == "add":
            add_called.append(True)
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr(materialize, "_git_capture", git_capture)
    monkeypatch.setattr(materialize, "_git_name_status", lambda *_args: ())
    monkeypatch.setattr(materialize, "_git_blob_absent", lambda *_args: True)
    executable_path = next(iter(expected))

    def snapshot(path: str, _label: str, *, capture_bytes: bool = True):
        payload = (
            manifest_payload if path == materialize.MANIFEST_RELATIVE_PATH else b"x"
        )
        digest, size_bytes = expected[path]
        return (
            SimpleNamespace(
                sha256=digest,
                size_bytes=len(payload) if size_bytes is None else size_bytes,
                mode=stat.S_IFREG | (0o755 if path == executable_path else 0o644),
            ),
            payload if capture_bytes else None,
        )

    monkeypatch.setattr(materialize, "_snapshot", snapshot)
    with pytest.raises(materialize.CandidateEvidenceError, match="not be executable"):
        materialize.stage_candidate_evidence(
            expected_source_revision="3" * 40,
            expected_registry_sha256="1" * 64,
            expected_registry_canonical_sha256="2" * 64,
        )
    assert add_called == []


def test_index_verifier_rejects_non_100644_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        materialize, "_git_name_status", lambda *_args: (("A", "evidence.json"),)
    )
    monkeypatch.setattr(
        materialize,
        "_git_capture",
        lambda *arguments: (
            b"100755 deadbeef 0\tevidence.json\0" if arguments[0] == "ls-files" else b""
        ),
    )
    with pytest.raises(materialize.CandidateEvidenceError, match="mode-100644"):
        materialize._verify_index_additions(
            source_revision="3" * 40,
            expected={"evidence.json": (hashlib.sha256(b"x").hexdigest(), 1)},
        )
