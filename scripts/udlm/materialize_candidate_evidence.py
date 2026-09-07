"""Materialize the complete pre-final campaign evidence into tracked paths.

This CPU-only bridge starts from the exact clean pushed registry revision G and
the ignored, immutable campaign outputs.  It deterministically replays all five
pre-final stages, independently validates every source envelope (including a
fresh rescore for each completed child), and constructs all formal schema-2
envelopes in memory.  Only after all 43 children pass does it publish the 43
tracked envelopes plus one completion-last campaign manifest as an exclusive
``artifact_io`` bundle.  It never generates molecules, loads a model, commits,
or pushes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import benchmark  # noqa: E402
from scripts.udlm import launch_candidate_campaign as campaign  # noqa: E402
from scripts.udlm import write_pilot_evidence  # noqa: E402


MANIFEST_RELATIVE_PATH = "experiments/udlm/pilots/campaign_evidence_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
PROTOCOL_ID = "genmol_udlm_de_novo_superiority_v4"
MANIFEST_STATUS = "complete_before_candidate_decision"
AUTHORITY_RELATIVE_PATHS = (
    "experiments/udlm/candidates/candidate_decision.json",
    "experiments/udlm/candidates/candidate_ledger.json",
    "experiments/udlm/candidates/candidate_lock.json",
)
LAUNCHER_RELATIVE_PATHS = (
    "scripts/exps/denovo/launch_benchmark.py",
    "scripts/udlm/launch_candidate_campaign.py",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
CONFIG_ID = re.compile(r"[rse]_t(?:050|070|085|100)_p(?:095|098|100)\Z")
ATTEMPT_ID = re.compile(
    r"stage-(?:d|a|b|c|eligible)-[rse]_t(?:050|070|085|100)_p(?:095|098|100)\Z"
)
CANDIDATE_ID = re.compile(r"[rse]-w1-1000u-dcb271453411\Z")
COMPLETED_SUPPORT_ROLES = {
    "training_exit_receipt",
    "benchmark_summary_json",
    "benchmark_raw_samples_csv",
}
FAILED_REQUIRED_SUPPORT_ROLES = {"failure_receipt", "failure_log"}
FAILED_OPTIONAL_SUPPORT_ROLES = {
    "partial_summary_json",
    "partial_raw_samples_csv",
}


class CandidateEvidenceError(ValueError):
    """Raised before publication when the complete evidence bridge is invalid."""


@dataclass(frozen=True)
class EvidencePlan:
    """All in-memory target payloads plus retained immutable input claims."""

    members: tuple[artifact_io.PublishItem, ...]
    manifest: Mapping[str, Any]
    retained_inputs: tuple[artifact_io.FileClaim, ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Match the canonical pretty JSON used by the source envelope writer."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateEvidenceError(f"{label} must be an object")
    return value


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    record = _mapping(value, label)
    if set(record) != fields:
        raise CandidateEvidenceError(
            f"{label} fields differ: {sorted(record)!r} != {sorted(fields)!r}"
        )
    return record


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise CandidateEvidenceError(f"{label} must be 64 lowercase hex")
    return value


def _revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise CandidateEvidenceError(f"{label} must be 40 lowercase hex")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CandidateEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CandidateEvidenceError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise CandidateEvidenceError(f"{label} must be a canonical relative path")
    return value


def _strict_json(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = write_pilot_evidence.strict_json_loads(payload, label=label)
    except (UnicodeDecodeError, ValueError) as error:
        raise CandidateEvidenceError(f"{label} is not strict JSON") from error
    return _mapping(value, label)


def _snapshot(
    relative_path: str, label: str, *, capture_bytes: bool = True
) -> tuple[artifact_io.FileClaim, bytes | None]:
    _relative(relative_path, f"{label} path")
    try:
        return artifact_io.snapshot_file(
            REPOSITORY_ROOT, relative_path, capture_bytes=capture_bytes
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise CandidateEvidenceError(f"cannot read stable {label}") from error


def _git_blob(revision: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CandidateEvidenceError(
            f"Git revision {revision} lacks required blob {relative_path}"
        )
    return result.stdout


def _git_blob_absent(revision: str, relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}:{relative_path}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return False
    if result.returncode in {1, 128}:
        return True
    raise CandidateEvidenceError(
        f"Git absence check was indeterminate for {relative_path}"
    )


def _require_clean_pushed_source(expected_revision: str) -> Mapping[str, str]:
    expected_revision = _revision(expected_revision, "expected source revision G")
    try:
        state = benchmark.require_clean_pushed_source(expected_revision)
    except (OSError, RuntimeError, ValueError) as error:
        raise CandidateEvidenceError(
            "materialization requires the exact clean pushed registry revision G"
        ) from error
    if state != {"head": expected_revision, "upstream": expected_revision}:
        raise CandidateEvidenceError("clean pushed source identity differs from G")
    return state


def _git_capture(*arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (
            (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        )
        raise CandidateEvidenceError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _git_name_status(*arguments: str) -> tuple[tuple[str, str], ...]:
    payload = _git_capture(*arguments)
    records = payload.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    if len(records) % 2:
        raise CandidateEvidenceError("Git name-status output is malformed")
    entries: list[tuple[str, str]] = []
    for index in range(0, len(records), 2):
        try:
            status = records[index].decode("ascii")
            path = records[index + 1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise CandidateEvidenceError(
                "Git name-status output contains a noncanonical path"
            ) from error
        entries.append((status, path))
    return tuple(entries)


def _manifest_git_file_claims(
    manifest: Mapping[str, Any], manifest_payload: bytes
) -> dict[str, tuple[str, int | None]]:
    expected: dict[str, tuple[str, int | None]] = {}

    def add(path: str, digest: str, size_bytes: int | None) -> None:
        record = (_sha256(digest, f"manifest claim {path}"), size_bytes)
        prior = expected.get(path)
        if prior is not None and prior != record:
            raise CandidateEvidenceError(
                f"manifest repeats a Git path inconsistently: {path}"
            )
        expected[path] = record

    for reference in manifest["stage_decisions"]:
        add(reference["relative_path"], reference["sha256"], None)
    for child in manifest["children"]:
        tracked = child["tracked_envelope"]
        add(tracked["relative_path"], tracked["sha256"], tracked["size_bytes"])
        for support in child["supporting_artifacts"]:
            add(
                support["relative_path"],
                support["sha256"],
                support["size_bytes"],
            )
    if set(expected) != set(manifest["required_git_paths"]):
        raise CandidateEvidenceError("manifest Git claims do not derive its closure")
    expected[MANIFEST_RELATIVE_PATH] = (
        hashlib.sha256(manifest_payload).hexdigest(),
        len(manifest_payload),
    )
    return expected


def _verify_index_additions(
    *, source_revision: str, expected: Mapping[str, tuple[str, int | None]]
) -> None:
    changed = _git_name_status(
        "diff",
        "--cached",
        "--name-status",
        "--no-renames",
        "-z",
        source_revision,
        "--",
    )
    if (
        len(changed) != len(expected)
        or {path for status, path in changed if status == "A"} != set(expected)
        or any(status != "A" for status, _path in changed)
    ):
        raise CandidateEvidenceError(
            "index must contain exactly the manifest-derived additions"
        )
    raw_entries = _git_capture("ls-files", "--stage", "-z", "--", *sorted(expected))
    records = raw_entries.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    observed: dict[str, tuple[str, str]] = {}
    for record in records:
        if b"\t" not in record:
            raise CandidateEvidenceError("Git index entry is malformed")
        metadata, raw_path = record.split(b"\t", 1)
        try:
            mode, _object_id, stage = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise CandidateEvidenceError("Git index entry is malformed") from error
        if path in observed:
            raise CandidateEvidenceError(f"Git index repeats evidence path: {path}")
        observed[path] = (mode, stage)
    if set(observed) != set(expected) or any(
        mode != "100644" or stage != "0" for mode, stage in observed.values()
    ):
        raise CandidateEvidenceError(
            "every evidence addition must be a stage-0 mode-100644 blob"
        )
    for path, (digest, expected_size) in expected.items():
        indexed = _git_capture("show", f":{path}")
        claim, live = _snapshot(path, f"staged evidence {path}")
        if (
            hashlib.sha256(indexed).hexdigest() != digest
            or claim.sha256 != digest
            or live != indexed
            or (expected_size is not None and len(indexed) != expected_size)
            or (expected_size is not None and claim.size_bytes != expected_size)
        ):
            raise CandidateEvidenceError(f"indexed evidence bytes differ: {path}")
    if _git_capture("diff", "--cached", "--check"):
        raise CandidateEvidenceError("staged evidence fails git diff --check")


def stage_candidate_evidence(
    *,
    expected_source_revision: str,
    expected_registry_sha256: str,
    expected_registry_canonical_sha256: str,
) -> dict[str, Any]:
    """Force-add exactly the completed manifest closure, without committing it."""

    source_revision = _revision(
        expected_source_revision, "expected registry revision G"
    )
    head = _git_capture("rev-parse", "HEAD").decode("ascii").strip()
    upstream = _git_capture("rev-parse", "@{upstream}").decode("ascii").strip()
    if head != source_revision or upstream != source_revision:
        raise CandidateEvidenceError(
            "evidence staging requires exact pushed registry revision G"
        )
    if _git_name_status(
        "diff", "--cached", "--name-status", "--no-renames", "-z", source_revision, "--"
    ):
        raise CandidateEvidenceError("evidence staging requires an empty index")
    if _git_name_status(
        "diff", "--name-status", "--no-renames", "-z", source_revision, "--"
    ):
        raise CandidateEvidenceError(
            "tracked worktree changes exist outside unpublished evidence"
        )
    try:
        registry = campaign.load_registry(
            REPOSITORY_ROOT / campaign.REGISTRY_RELATIVE_PATH,
            expected_raw_sha256=expected_registry_sha256,
            expected_canonical_sha256=expected_registry_canonical_sha256,
        )
    except (OSError, ValueError) as error:
        raise CandidateEvidenceError("candidate registry is invalid") from error
    manifest_claim, payload = _snapshot(
        MANIFEST_RELATIVE_PATH, "campaign evidence manifest"
    )
    if payload is None:  # pragma: no cover - capture contract
        raise AssertionError("manifest payload was not captured")
    manifest = validate_manifest(_strict_json(payload, "campaign evidence manifest"))
    if manifest["registry"] != registry.reference or manifest["source_revision"] != {
        "head": source_revision,
        "upstream": source_revision,
    }:
        raise CandidateEvidenceError("campaign evidence manifest does not bind G")
    expected = _manifest_git_file_claims(manifest, payload)
    if manifest_claim.sha256 != expected[MANIFEST_RELATIVE_PATH][0]:
        raise CandidateEvidenceError("campaign evidence manifest bytes differ")
    for path, (digest, expected_size) in expected.items():
        if not _git_blob_absent(source_revision, path):
            raise CandidateEvidenceError(f"evidence path already existed at G: {path}")
        claim, _contents = _snapshot(path, f"evidence addition {path}")
        if claim.sha256 != digest or (
            expected_size is not None and claim.size_bytes != expected_size
        ):
            raise CandidateEvidenceError(f"evidence addition bytes differ: {path}")
        if claim.mode & 0o111:
            raise CandidateEvidenceError(
                f"evidence addition must not be executable: {path}"
            )
    ordered_paths = sorted(expected, key=lambda path: path.encode("ascii"))
    _git_capture("add", "-f", "--", *ordered_paths)
    _verify_index_additions(source_revision=source_revision, expected=expected)
    return {
        "status": "staged_without_commit",
        "source_revision": source_revision,
        "addition_count": len(expected),
        "manifest_relative_path": MANIFEST_RELATIVE_PATH,
        "manifest_sha256": manifest_claim.sha256,
    }


def _retain_claim(
    retained: dict[str, artifact_io.FileClaim], claim: artifact_io.FileClaim
) -> None:
    existing = retained.get(claim.relative_path)
    if existing is not None and existing != claim:
        raise CandidateEvidenceError(
            f"input changed between reads: {claim.relative_path}"
        )
    retained[claim.relative_path] = claim


def _reference(
    *,
    relative_path: str,
    sha256: str,
    size_bytes: int,
    schema_version: int,
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "schema_version": schema_version,
    }


def _supporting_reference(
    *,
    role: str,
    relative_path: str,
    expected_sha256: str,
    schema_version: int | None,
    retained: dict[str, artifact_io.FileClaim],
) -> dict[str, Any]:
    relative_path = _relative(relative_path, f"{role} supporting artifact")
    expected_sha256 = _sha256(expected_sha256, f"{role} digest")
    claim, _payload = _snapshot(relative_path, role, capture_bytes=False)
    if claim.sha256 != expected_sha256:
        raise CandidateEvidenceError(f"{role} bytes differ from the envelope")
    _retain_claim(retained, claim)
    return {
        "artifact_role": role,
        "relative_path": relative_path,
        "sha256": claim.sha256,
        "size_bytes": claim.size_bytes,
        "schema_version": schema_version,
    }


def _relative_from_absolute(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CandidateEvidenceError(f"{label} must be an absolute path")
    path = Path(value)
    root = REPOSITORY_ROOT.resolve(strict=True)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise CandidateEvidenceError(f"{label} must be normalized and absolute")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise CandidateEvidenceError(f"{label} escapes the repository") from error
    return _relative(relative, label)


def _completed_support(
    envelope: Mapping[str, Any],
    retained: dict[str, artifact_io.FileClaim],
) -> list[dict[str, Any]]:
    receipt = _exact(
        envelope["training_exit_receipt"],
        {"relative_path", "sha256", "schema_version"},
        "completed training receipt reference",
    )
    artifacts = _exact(
        envelope["benchmark_artifacts"],
        {"summary_json", "raw_samples_csv"},
        "completed benchmark artifacts",
    )
    summary = _exact(
        artifacts["summary_json"],
        {"relative_path", "sha256", "schema_version"},
        "completed summary reference",
    )
    raw = _exact(
        artifacts["raw_samples_csv"],
        {"relative_path", "sha256"},
        "completed raw CSV reference",
    )
    if receipt["schema_version"] != 5 or summary["schema_version"] != 8:
        raise CandidateEvidenceError("completed supporting schemas differ")
    values = [
        _supporting_reference(
            role="training_exit_receipt",
            relative_path=receipt["relative_path"],
            expected_sha256=receipt["sha256"],
            schema_version=5,
            retained=retained,
        ),
        _supporting_reference(
            role="benchmark_summary_json",
            relative_path=summary["relative_path"],
            expected_sha256=summary["sha256"],
            schema_version=8,
            retained=retained,
        ),
        _supporting_reference(
            role="benchmark_raw_samples_csv",
            relative_path=raw["relative_path"],
            expected_sha256=raw["sha256"],
            schema_version=None,
            retained=retained,
        ),
    ]
    return sorted(values, key=lambda row: (row["artifact_role"], row["relative_path"]))


def _failed_support(
    envelope: Mapping[str, Any],
    retained: dict[str, artifact_io.FileClaim],
) -> list[dict[str, Any]]:
    failure = _exact(
        envelope["failure_receipt"],
        {"relative_path", "sha256", "schema_version"},
        "failure receipt reference",
    )
    if failure["schema_version"] != 1:
        raise CandidateEvidenceError("failure receipt schema differs")
    failure_ref = _supporting_reference(
        role="failure_receipt",
        relative_path=failure["relative_path"],
        expected_sha256=failure["sha256"],
        schema_version=1,
        retained=retained,
    )
    receipt_claim, payload = _snapshot(
        failure_ref["relative_path"], "failure receipt", capture_bytes=True
    )
    if payload is None:  # pragma: no cover - capture contract
        raise AssertionError("failure receipt payload was not captured")
    _retain_claim(retained, receipt_claim)
    receipt = _strict_json(payload, "failure receipt")
    log = _exact(receipt.get("log"), {"path", "sha256", "size_bytes"}, "failure log")
    values = [
        failure_ref,
        _supporting_reference(
            role="failure_log",
            relative_path=_relative_from_absolute(log["path"], "failure log path"),
            expected_sha256=log["sha256"],
            schema_version=None,
            retained=retained,
        ),
    ]
    partials = _exact(
        receipt.get("partial_artifacts"),
        {"summary_json", "raw_samples_csv"},
        "failure partial artifacts",
    )
    for field, role in (
        ("summary_json", "partial_summary_json"),
        ("raw_samples_csv", "partial_raw_samples_csv"),
    ):
        raw_reference = partials[field]
        if raw_reference is None:
            continue
        reference = _exact(
            raw_reference, {"path", "sha256", "size_bytes"}, f"failure {role}"
        )
        values.append(
            _supporting_reference(
                role=role,
                relative_path=_relative_from_absolute(
                    reference["path"], f"failure {role} path"
                ),
                expected_sha256=reference["sha256"],
                schema_version=None,
                retained=retained,
            )
        )
    return sorted(values, key=lambda row: (row["artifact_role"], row["relative_path"]))


def _source_envelope(
    value: Any,
    *,
    attempt_id: str,
    candidate_id: str,
    pilot_seed: int,
) -> tuple[Mapping[str, Any], artifact_io.FileClaim, bytes]:
    reference = _exact(
        value,
        {"artifact_kind", "pilot_seed", "relative_path", "sha256", "schema_version"},
        "source child reference",
    )
    expected_live_path = (
        "output/udlm/de_novo_candidate_campaign_v1/evidence/"
        f"{attempt_id}/seed_{pilot_seed}.json"
    )
    if (
        reference["artifact_kind"] not in {"pilot_evaluation", "pilot_failure"}
        or reference["pilot_seed"] != pilot_seed
        or reference["relative_path"] != expected_live_path
        or reference["schema_version"] != 2
    ):
        raise CandidateEvidenceError("source child reference identity differs")
    expected_digest = _sha256(reference["sha256"], "source envelope digest")
    claim, payload = _snapshot(
        expected_live_path, "source envelope", capture_bytes=True
    )
    if payload is None:  # pragma: no cover - capture contract
        raise AssertionError("source envelope payload was not captured")
    if claim.sha256 != expected_digest:
        raise CandidateEvidenceError("source envelope bytes differ from stage decision")
    envelope = _strict_json(payload, "source envelope")
    common = {
        "schema_version",
        "artifact_kind",
        "status",
        "attempt_id",
        "candidate_id",
        "pilot_seed",
        "final_seed_results_included",
    }
    kind = reference["artifact_kind"]
    fields = common | (
        {"training_exit_receipt", "benchmark_artifacts"}
        if kind == "pilot_evaluation"
        else {"failure_receipt"}
    )
    _exact(envelope, fields, "source envelope")
    expected = {
        "schema_version": 2,
        "artifact_kind": kind,
        "status": "completed" if kind == "pilot_evaluation" else "failed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": pilot_seed,
        "final_seed_results_included": False,
    }
    for field, expected_value in expected.items():
        if envelope[field] != expected_value:
            raise CandidateEvidenceError(f"source envelope {field} differs")
    return envelope, claim, payload


def _build_tracked_child(
    *,
    stage_id: str,
    entry: Mapping[str, Any],
    source_child: Mapping[str, Any],
    requested_samples: int,
    retained: dict[str, artifact_io.FileClaim],
) -> tuple[dict[str, Any], artifact_io.PublishItem]:
    attempt_id = str(entry["attempt_id"])
    candidate_id = str(entry["candidate_id"])
    pilot_seed = int(source_child["pilot_seed"])
    envelope, live_claim, live_payload = _source_envelope(
        source_child,
        attempt_id=attempt_id,
        candidate_id=candidate_id,
        pilot_seed=pilot_seed,
    )
    _retain_claim(retained, live_claim)
    tracked_path = f"experiments/udlm/pilots/{attempt_id}/seed_{pilot_seed}.json"
    if os.path.lexists(REPOSITORY_ROOT / tracked_path):
        raise CandidateEvidenceError(f"tracked envelope is not fresh: {tracked_path}")
    try:
        if envelope["artifact_kind"] == "pilot_evaluation":
            benchmark_artifacts = _mapping(
                envelope["benchmark_artifacts"], "benchmark artifacts"
            )
            summary_ref = _mapping(
                benchmark_artifacts["summary_json"], "benchmark summary"
            )
            receipt_ref = _mapping(
                envelope["training_exit_receipt"], "training receipt"
            )
            run_dir = (REPOSITORY_ROOT / str(summary_ref["relative_path"])).parent
            regenerated = write_pilot_evidence.build_completed_pilot_evidence_payload(
                attempt_id=attempt_id,
                candidate_id=candidate_id,
                pilot_seed=pilot_seed,
                run_dir=run_dir,
                training_exit_receipt=(
                    REPOSITORY_ROOT / str(receipt_ref["relative_path"])
                ),
                output=REPOSITORY_ROOT / tracked_path,
            )
            summary_claim, summary_payload = _snapshot(
                str(summary_ref["relative_path"]),
                "completed child summary",
                capture_bytes=True,
            )
            if summary_payload is None:  # pragma: no cover - capture contract
                raise AssertionError("summary payload was not captured")
            summary = _strict_json(summary_payload, "completed child summary")
            if summary.get("num_samples") != requested_samples:
                raise CandidateEvidenceError(
                    "completed child requested sample count differs from its stage"
                )
            _retain_claim(retained, summary_claim)
            support = _completed_support(envelope, retained)
        else:
            failure_ref = _mapping(envelope["failure_receipt"], "failure receipt")
            regenerated = write_pilot_evidence.build_failed_pilot_evidence_payload(
                attempt_id=attempt_id,
                candidate_id=candidate_id,
                pilot_seed=pilot_seed,
                failure_receipt=REPOSITORY_ROOT / str(failure_ref["relative_path"]),
                output=REPOSITORY_ROOT / tracked_path,
            )
            failure_claim, failure_payload = _snapshot(
                str(failure_ref["relative_path"]),
                "failure receipt",
                capture_bytes=True,
            )
            if failure_payload is None:  # pragma: no cover - capture contract
                raise AssertionError("failure payload was not captured")
            failure_receipt = _strict_json(failure_payload, "failure receipt")
            if failure_receipt.get("requested_samples") != requested_samples:
                raise CandidateEvidenceError(
                    "failed child requested sample count differs from its stage"
                )
            _retain_claim(retained, failure_claim)
            support = _failed_support(envelope, retained)
    except (OSError, ValueError) as error:
        raise CandidateEvidenceError(
            f"independent validation failed for {attempt_id} seed {pilot_seed}"
        ) from error
    tracked_payload = canonical_json_bytes(regenerated)
    if tracked_payload != live_payload or hashlib.sha256(
        tracked_payload
    ).hexdigest() != (live_claim.sha256):
        raise CandidateEvidenceError(
            "regenerated tracked envelope differs from immutable source envelope"
        )
    child = {
        "stage_id": stage_id,
        "config_id": entry["config_id"],
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": pilot_seed,
        "artifact_kind": envelope["artifact_kind"],
        "live_envelope": _reference(
            relative_path=live_claim.relative_path,
            sha256=live_claim.sha256,
            size_bytes=live_claim.size_bytes,
            schema_version=2,
        ),
        "tracked_envelope": _reference(
            relative_path=tracked_path,
            sha256=live_claim.sha256,
            size_bytes=len(tracked_payload),
            schema_version=2,
        ),
        "supporting_artifacts": support,
    }
    return child, artifact_io.PublishItem(tracked_path, tracked_payload)


def _assert_preexisting_sources_at_g(
    registry: campaign.ValidatedRegistry,
    source_revision: str,
    retained: dict[str, artifact_io.FileClaim],
) -> None:
    registry_claim, registry_payload = _snapshot(
        campaign.REGISTRY_RELATIVE_PATH, "candidate registry", capture_bytes=True
    )
    if (
        registry_payload is None
        or registry_claim.sha256 != registry.raw_sha256
        or registry_claim.size_bytes != registry.size_bytes
        or _git_blob(source_revision, campaign.REGISTRY_RELATIVE_PATH)
        != registry_payload
    ):
        raise CandidateEvidenceError("candidate registry bytes differ at G")
    _retain_claim(retained, registry_claim)
    for config in registry.configs:
        claim, payload = _snapshot(
            config.relative_path,
            f"candidate config {config.config_id}",
            capture_bytes=True,
        )
        if (
            payload is None
            or claim.sha256 != config.sha256
            or claim.size_bytes != config.size_bytes
            or _git_blob(source_revision, config.relative_path) != payload
        ):
            raise CandidateEvidenceError(
                f"candidate config {config.config_id} differs from G"
            )
        _retain_claim(retained, claim)
    for launcher_path in LAUNCHER_RELATIVE_PATHS:
        claim, payload = _snapshot(
            launcher_path, "campaign launcher", capture_bytes=True
        )
        if payload is None or _git_blob(source_revision, launcher_path) != payload:
            raise CandidateEvidenceError(f"launcher differs from G: {launcher_path}")
        _retain_claim(retained, claim)


def validate_manifest(value: Any) -> dict[str, Any]:
    """Validate the exact schema-1 completion manifest without trusting its closure."""

    manifest = _exact(
        value,
        {
            "schema_version",
            "protocol_id",
            "status",
            "registry",
            "source_revision",
            "counts",
            "stage_decisions",
            "children",
            "required_git_paths",
        },
        "campaign evidence manifest",
    )
    if (
        manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["status"] != MANIFEST_STATUS
    ):
        raise CandidateEvidenceError("campaign evidence manifest identity differs")
    registry = _exact(
        manifest["registry"],
        {"relative_path", "sha256", "canonical_sha256", "size_bytes", "schema_version"},
        "manifest registry",
    )
    if (
        registry["relative_path"] != campaign.REGISTRY_RELATIVE_PATH
        or registry["schema_version"] != campaign.REGISTRY_SCHEMA_VERSION
    ):
        raise CandidateEvidenceError("manifest registry identity differs")
    _sha256(registry["sha256"], "manifest registry raw digest")
    _sha256(registry["canonical_sha256"], "manifest registry canonical digest")
    _integer(registry["size_bytes"], "manifest registry size", minimum=1)
    source = _exact(
        manifest["source_revision"], {"head", "upstream"}, "manifest source revision"
    )
    revision = _revision(source["head"], "manifest source revision G")
    if source["upstream"] != revision:
        raise CandidateEvidenceError("manifest source was not clean and pushed")

    stage_decisions = manifest["stage_decisions"]
    if not isinstance(stage_decisions, list) or len(stage_decisions) != 5:
        raise CandidateEvidenceError("manifest must contain five stage decisions")
    normalized_stage_refs: list[dict[str, Any]] = []
    for stage_id, raw_reference in zip(
        campaign.STAGE_IDS, stage_decisions, strict=True
    ):
        reference = _exact(
            raw_reference,
            {"stage_id", "relative_path", "sha256", "schema_version"},
            f"manifest stage {stage_id}",
        )
        expected_path = campaign._decision_paths(stage_id)[1]  # noqa: SLF001
        if (
            reference["stage_id"] != stage_id
            or reference["relative_path"] != expected_path
            or reference["schema_version"] != 1
        ):
            raise CandidateEvidenceError(f"manifest stage {stage_id} identity differs")
        _sha256(reference["sha256"], f"manifest stage {stage_id} digest")
        normalized_stage_refs.append(dict(reference))

    children = manifest["children"]
    if not isinstance(children, list) or len(children) != 43:
        raise CandidateEvidenceError("manifest must contain exactly 43 children")
    normalized_children: list[dict[str, Any]] = []
    observed_attempts: dict[str, set[str]] = {
        stage: set() for stage in campaign.STAGE_IDS
    }
    requested_molecules = 0
    prior_order: tuple[int, bytes, int] | None = None
    for index, raw_child in enumerate(children):
        child = _exact(
            raw_child,
            {
                "stage_id",
                "config_id",
                "attempt_id",
                "candidate_id",
                "pilot_seed",
                "artifact_kind",
                "live_envelope",
                "tracked_envelope",
                "supporting_artifacts",
            },
            f"manifest child {index}",
        )
        stage_id = child["stage_id"]
        if stage_id not in campaign.STAGE_IDS:
            raise CandidateEvidenceError(f"manifest child {index} stage differs")
        config_id = child["config_id"]
        attempt_id = child["attempt_id"]
        candidate_id = child["candidate_id"]
        pilot_seed = child["pilot_seed"]
        if (
            not isinstance(config_id, str)
            or CONFIG_ID.fullmatch(config_id) is None
            or attempt_id != f"stage-{stage_id.lower()}-{config_id}"
            or ATTEMPT_ID.fullmatch(str(attempt_id)) is None
            or candidate_id != campaign.CANDIDATE_IDS[config_id[0].upper()]
            or CANDIDATE_ID.fullmatch(str(candidate_id)) is None
            or type(pilot_seed) is not int
            or pilot_seed not in campaign.STAGE_CONTRACT[stage_id]["seeds"]
            or child["artifact_kind"] not in {"pilot_evaluation", "pilot_failure"}
        ):
            raise CandidateEvidenceError(f"manifest child {index} identity differs")
        order = (
            campaign.STAGE_IDS.index(stage_id),
            config_id.encode("ascii"),
            pilot_seed,
        )
        if prior_order is not None and order <= prior_order:
            raise CandidateEvidenceError("manifest children are not in canonical order")
        prior_order = order
        observed_attempts[stage_id].add(attempt_id)
        requested_molecules += campaign.STAGE_CONTRACT[stage_id]["samples"]

        references = {}
        for name in ("live_envelope", "tracked_envelope"):
            reference = _exact(
                child[name],
                {"relative_path", "sha256", "size_bytes", "schema_version"},
                f"manifest child {index} {name}",
            )
            references[name] = {
                "relative_path": _relative(
                    reference["relative_path"],
                    f"manifest child {index} {name} path",
                ),
                "sha256": _sha256(
                    reference["sha256"], f"manifest child {index} {name} digest"
                ),
                "size_bytes": _integer(
                    reference["size_bytes"],
                    f"manifest child {index} {name} size",
                    minimum=1,
                ),
                "schema_version": reference["schema_version"],
            }
            if reference["schema_version"] != 2:
                raise CandidateEvidenceError("manifest envelope schema differs")
        expected_live = (
            "output/udlm/de_novo_candidate_campaign_v1/evidence/"
            f"{attempt_id}/seed_{pilot_seed}.json"
        )
        expected_tracked = (
            f"experiments/udlm/pilots/{attempt_id}/seed_{pilot_seed}.json"
        )
        if (
            references["live_envelope"]["relative_path"] != expected_live
            or references["tracked_envelope"]["relative_path"] != expected_tracked
            or {
                key: references["live_envelope"][key]
                for key in ("sha256", "size_bytes", "schema_version")
            }
            != {
                key: references["tracked_envelope"][key]
                for key in ("sha256", "size_bytes", "schema_version")
            }
        ):
            raise CandidateEvidenceError(
                "manifest live/tracked envelope bridge differs"
            )

        supports = child["supporting_artifacts"]
        if not isinstance(supports, list):
            raise CandidateEvidenceError("manifest supporting artifacts must be a list")
        normalized_supports = []
        support_order: tuple[bytes, bytes] | None = None
        for support_index, raw_support in enumerate(supports):
            support = _exact(
                raw_support,
                {
                    "artifact_role",
                    "relative_path",
                    "sha256",
                    "size_bytes",
                    "schema_version",
                },
                f"manifest child {index} support {support_index}",
            )
            role = support["artifact_role"]
            if not isinstance(role, str):
                raise CandidateEvidenceError("manifest support role must be a string")
            path = _relative(
                support["relative_path"], f"manifest child {index} support path"
            )
            current_order = (role.encode("ascii"), path.encode("ascii"))
            if support_order is not None and current_order <= support_order:
                raise CandidateEvidenceError(
                    "manifest supporting artifacts are not canonically ordered"
                )
            support_order = current_order
            _sha256(support["sha256"], f"manifest child {index} support digest")
            _integer(
                support["size_bytes"],
                f"manifest child {index} support size",
                minimum=0,
            )
            normalized_supports.append(dict(support))
        roles = {support["artifact_role"] for support in normalized_supports}
        if len(roles) != len(normalized_supports):
            raise CandidateEvidenceError("manifest child repeats a support role")
        if child["artifact_kind"] == "pilot_evaluation":
            if roles != COMPLETED_SUPPORT_ROLES or any(
                support["schema_version"]
                != {
                    "training_exit_receipt": 5,
                    "benchmark_summary_json": 8,
                    "benchmark_raw_samples_csv": None,
                }[support["artifact_role"]]
                for support in normalized_supports
            ):
                raise CandidateEvidenceError("completed supporting artifacts differ")
        elif not (
            FAILED_REQUIRED_SUPPORT_ROLES.issubset(roles)
            and roles.issubset(
                FAILED_REQUIRED_SUPPORT_ROLES | FAILED_OPTIONAL_SUPPORT_ROLES
            )
            and all(
                support["schema_version"]
                == (1 if support["artifact_role"] == "failure_receipt" else None)
                for support in normalized_supports
            )
        ):
            raise CandidateEvidenceError("failed supporting artifacts differ")
        normalized_children.append(dict(child))

    expected_attempt_counts = {
        stage_id: campaign.STAGE_CONTRACT[stage_id]["entries"]
        for stage_id in campaign.STAGE_IDS
    }
    if {
        stage_id: len(attempts) for stage_id, attempts in observed_attempts.items()
    } != expected_attempt_counts or requested_molecules != 3680:
        raise CandidateEvidenceError("manifest scientific accounting differs")
    derived_required = sorted(
        {
            *(reference["relative_path"] for reference in normalized_stage_refs),
            *(
                child["tracked_envelope"]["relative_path"]
                for child in normalized_children
            ),
            *(
                support["relative_path"]
                for child in normalized_children
                for support in child["supporting_artifacts"]
            ),
        },
        key=lambda path: path.encode("ascii"),
    )
    required = manifest["required_git_paths"]
    if (
        not isinstance(required, list)
        or required != derived_required
        or MANIFEST_RELATIVE_PATH in required
        or any(path in required for path in AUTHORITY_RELATIVE_PATHS)
    ):
        raise CandidateEvidenceError("manifest required Git closure differs")
    counts = _exact(
        manifest["counts"],
        {
            "executed_entry_count",
            "child_outcome_count",
            "requested_molecule_count",
            "stage_decision_count",
            "tracked_envelope_count",
            "required_git_path_count",
        },
        "manifest counts",
    )
    if dict(counts) != {
        "executed_entry_count": 40,
        "child_outcome_count": 43,
        "requested_molecule_count": 3680,
        "stage_decision_count": 5,
        "tracked_envelope_count": 43,
        "required_git_path_count": len(derived_required),
    }:
        raise CandidateEvidenceError("manifest counts differ")
    return dict(manifest)


def build_evidence_plan(
    registry: campaign.ValidatedRegistry,
    *,
    source_revision: str,
    decisions: Mapping[str, Mapping[str, Any]],
    completions: Mapping[str, Mapping[str, Any]],
) -> EvidencePlan:
    """Validate every source first and return one all-in-memory publication plan."""

    source_revision = _revision(source_revision, "registry revision G")
    if (
        tuple(decisions) != campaign.STAGE_IDS
        or tuple(completions) != campaign.STAGE_IDS
    ):
        raise CandidateEvidenceError("campaign decision chain is incomplete")
    if any(decisions[stage]["status"] != "completed" for stage in campaign.STAGE_IDS):
        raise CandidateEvidenceError("campaign contains an incomplete stage")
    retained: dict[str, artifact_io.FileClaim] = {}
    _assert_preexisting_sources_at_g(registry, source_revision, retained)
    stage_refs: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    members: list[artifact_io.PublishItem] = []
    for stage_id in campaign.STAGE_IDS:
        completion = _exact(
            completions[stage_id],
            {"relative_path", "sha256", "schema_version"},
            f"stage {stage_id} completion reference",
        )
        expected_stage_path = campaign._decision_paths(stage_id)[1]  # noqa: SLF001
        stage_claim, stage_payload = _snapshot(
            expected_stage_path, f"stage {stage_id} decision", capture_bytes=True
        )
        if (
            stage_payload is None
            or completion["relative_path"] != expected_stage_path
            or completion["sha256"] != stage_claim.sha256
            or completion["schema_version"] != 1
        ):
            raise CandidateEvidenceError(f"stage {stage_id} completion bytes differ")
        _retain_claim(retained, stage_claim)
        stage_refs.append({"stage_id": stage_id, **dict(completion)})
        decision = decisions[stage_id]
        contract = campaign.STAGE_CONTRACT[stage_id]
        entries = decision.get("entries")
        if not isinstance(entries, list) or len(entries) != contract["entries"]:
            raise CandidateEvidenceError(f"stage {stage_id} entry count differs")
        for entry in entries:
            entry = _mapping(entry, f"stage {stage_id} entry")
            source_children = entry.get("child_outcomes")
            if not isinstance(source_children, list) or len(source_children) != len(
                contract["seeds"]
            ):
                raise CandidateEvidenceError(f"stage {stage_id} child count differs")
            for source_child in source_children:
                child, member = _build_tracked_child(
                    stage_id=stage_id,
                    entry=entry,
                    source_child=_mapping(source_child, "source child reference"),
                    requested_samples=contract["samples"],
                    retained=retained,
                )
                children.append(child)
                members.append(member)
    if len(children) != 43 or len(members) != 43:
        raise CandidateEvidenceError("campaign child accounting differs")
    required_paths = sorted(
        {
            *(reference["relative_path"] for reference in stage_refs),
            *(member.relative_path for member in members),
            *(
                support["relative_path"]
                for child in children
                for support in child["supporting_artifacts"]
            ),
        },
        key=lambda path: path.encode("ascii"),
    )
    for path in (*required_paths, MANIFEST_RELATIVE_PATH, *AUTHORITY_RELATIVE_PATHS):
        if not _git_blob_absent(source_revision, path):
            raise CandidateEvidenceError(f"evidence path already existed at G: {path}")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": MANIFEST_STATUS,
        "registry": registry.reference,
        "source_revision": {"head": source_revision, "upstream": source_revision},
        "counts": {
            "executed_entry_count": 40,
            "child_outcome_count": 43,
            "requested_molecule_count": 3680,
            "stage_decision_count": 5,
            "tracked_envelope_count": 43,
            "required_git_path_count": len(required_paths),
        },
        "stage_decisions": stage_refs,
        "children": children,
        "required_git_paths": required_paths,
    }
    validate_manifest(manifest)
    return EvidencePlan(
        members=tuple(members),
        manifest=manifest,
        retained_inputs=tuple(retained.values()),
    )


def _revalidate_inputs(claims: Sequence[artifact_io.FileClaim]) -> None:
    for original in claims:
        current, _payload = _snapshot(
            original.relative_path,
            f"final input {original.relative_path}",
            capture_bytes=False,
        )
        if current != original:
            raise CandidateEvidenceError(
                f"input changed before bundle publication: {original.relative_path}"
            )


def _create_target_parents(plan: EvidencePlan) -> None:
    root = REPOSITORY_ROOT.resolve(strict=True)
    paths = [member.relative_path for member in plan.members]
    paths.append(MANIFEST_RELATIVE_PATH)
    for relative_path in paths:
        parent = root / PurePosixPath(relative_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent.resolve(strict=True) != parent or not parent.is_dir():
            raise CandidateEvidenceError(f"unsafe evidence target parent: {parent}")


def materialize_candidate_evidence(
    *,
    expected_source_revision: str,
    expected_registry_sha256: str,
    expected_registry_canonical_sha256: str,
) -> artifact_io.PublishedBundle:
    """Validate the complete campaign and publish its formal evidence bundle."""

    source_state = _require_clean_pushed_source(expected_source_revision)
    for path in (MANIFEST_RELATIVE_PATH, *AUTHORITY_RELATIVE_PATHS):
        if os.path.lexists(REPOSITORY_ROOT / path):
            raise CandidateEvidenceError(f"materialization path is not fresh: {path}")
    try:
        registry = campaign.load_registry(
            REPOSITORY_ROOT / campaign.REGISTRY_RELATIVE_PATH,
            expected_raw_sha256=expected_registry_sha256,
            expected_canonical_sha256=expected_registry_canonical_sha256,
        )
        decisions, completions = campaign.replay_decisions(registry)
    except (OSError, ValueError) as error:
        raise CandidateEvidenceError(
            "candidate campaign is not a complete replay-valid chain"
        ) from error
    plan = build_evidence_plan(
        registry,
        source_revision=source_state["head"],
        decisions=decisions,
        completions=completions,
    )
    _revalidate_inputs(plan.retained_inputs)
    if _require_clean_pushed_source(expected_source_revision) != source_state:
        raise CandidateEvidenceError("source revision changed before publication")
    _create_target_parents(plan)
    try:
        return artifact_io.publish_bundle_exclusive(
            REPOSITORY_ROOT,
            plan.members,
            completion=artifact_io.PublishItem(
                MANIFEST_RELATIVE_PATH, canonical_json_bytes(plan.manifest)
            ),
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise CandidateEvidenceError(
            "campaign evidence bundle publication failed"
        ) from error


def _canonical_hash(value: str) -> str:
    if HEX64.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hex")
    return value


def _canonical_revision(value: str) -> str:
    if HEX40.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("revision must be 40 lowercase hex")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-published-evidence",
        action="store_true",
        help=(
            "force-add exactly the already materialized manifest closure and verify "
            "the index; never commit or push"
        ),
    )
    parser.add_argument(
        "--expected-source-revision", required=True, type=_canonical_revision
    )
    parser.add_argument(
        "--expected-registry-sha256", required=True, type=_canonical_hash
    )
    parser.add_argument(
        "--expected-registry-canonical-sha256", required=True, type=_canonical_hash
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage_published_evidence:
        staged = stage_candidate_evidence(
            expected_source_revision=args.expected_source_revision,
            expected_registry_sha256=args.expected_registry_sha256,
            expected_registry_canonical_sha256=(
                args.expected_registry_canonical_sha256
            ),
        )
        print(json.dumps(staged, sort_keys=True, separators=(",", ":")))
        return 0
    bundle = materialize_candidate_evidence(
        expected_source_revision=args.expected_source_revision,
        expected_registry_sha256=args.expected_registry_sha256,
        expected_registry_canonical_sha256=(args.expected_registry_canonical_sha256),
    )
    print(
        "Candidate evidence manifest: "
        f"{bundle.completion.relative_path} {bundle.completion.sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
