"""Prepare immutable candidate decision, ledger, and lock publications.

Each invocation publishes exactly one fixed-path authority artifact.  The
three phases are intentionally separate so their Git commits can be
decision-only, then ledger-only, then lock-only.  Final-seed artifacts are
never read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import benchmark  # noqa: E402
from scripts.udlm import launch_candidate_campaign as campaign  # noqa: E402
from scripts.udlm import materialize_candidate_evidence  # noqa: E402


DECISION_RELATIVE_PATH = "experiments/udlm/candidates/candidate_decision.json"
LEDGER_RELATIVE_PATH = "experiments/udlm/candidates/candidate_ledger.json"
LOCK_RELATIVE_PATH = "experiments/udlm/candidates/candidate_lock.json"
STAGE_DECISION_TEMPLATE = (
    "output/udlm/de_novo_candidate_campaign_v1/stages/{stage}/stage_decision.json"
)
DECISION_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 2
LOCK_SCHEMA_VERSION = 2
PILOT_EVIDENCE_SCHEMA_VERSION = 2
PROTOCOL_ID = "genmol_udlm_de_novo_superiority_v4"
SELECTION_RULE = (
    "maximize_mean_released_quality_then_mean_released_diversity_then_"
    "lexicographically_smallest_attempt_id"
)
CHECKPOINT_SELECTION_RULE = "last_completed_optimizer_step"
RANKING_ORDER = [
    "released_quality_descending",
    "released_diversity_descending",
    "config_id_ascii_ascending",
    "attempt_id_ascii_ascending",
]
STAGE_ROLES = {
    "D": "nonranking_decode_diagnostic",
    "A": "raw_loo_temperature_screen",
    "B": "joint_temperature_nucleus_screen",
    "C": "held_out_operating_point_confirmation",
    "eligible": "registered_candidate_selection",
}
PROMOTION_QUOTA = {"D": None, "A": 2, "B": 2, "C": 1, "eligible": None}
FINAL_SEEDS = (0, 1, 2)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
CONFIG_ID = re.compile(r"[rse]_t(?:050|070|085|100)_p(?:095|098|100)\Z")
ATTEMPT_ID = re.compile(
    r"stage-(?:d|a|b|c|eligible)-[rse]_t(?:050|070|085|100)_p(?:095|098|100)\Z"
)
CANDIDATE_ID = re.compile(r"[rse]-w1-1000u-dcb271453411\Z")


class CandidateAuthorityError(ValueError):
    """Raised before any candidate-authority publication on invalid evidence."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAuthorityError(f"{label} must be an object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        found = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise CandidateAuthorityError(
            f"{label} fields differ: {found!r} != {sorted(fields)!r}"
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise CandidateAuthorityError(f"{label} must be 64 lowercase hex")
    return value


def _revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise CandidateAuthorityError(f"{label} must be 40 lowercase hex")
    return value


def _relative(value: Any, label: str, *, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise CandidateAuthorityError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or (suffix is not None and path.suffix != suffix)
    ):
        raise CandidateAuthorityError(f"{label} is not a canonical relative path")
    return value


def _strict_json(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise CandidateAuthorityError(f"{label} contains non-finite {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CandidateAuthorityError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateAuthorityError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, Mapping):
        raise CandidateAuthorityError(f"{label} root must be an object")
    return parsed


def _snapshot(relative_path: str, label: str) -> tuple[artifact_io.FileClaim, bytes]:
    try:
        claim, payload = artifact_io.snapshot_file(
            REPOSITORY_ROOT, relative_path, capture_bytes=True
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise CandidateAuthorityError(f"cannot read stable {label}") from error
    if payload is None:  # pragma: no cover - capture contract
        raise AssertionError("stable payload was not retained")
    return claim, payload


def _repository_artifact_loader(path: Path) -> bytes:
    _claim, payload = _snapshot(path.as_posix(), f"artifact {path.as_posix()}")
    return payload


def _validate_completed_envelope_live(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run the gate's independent schema-8 structure/decode/rescore adapter."""

    from scripts.udlm import superiority_gate

    try:
        return superiority_gate._validate_completed_pilot_evidence_live(envelope)
    except (OSError, ValueError) as error:
        raise CandidateAuthorityError(
            "completed pilot envelope failed independent schema-8 validation"
        ) from error


def _validate_failed_envelope_live(
    envelope: Mapping[str, Any], *, expected_samples: int
) -> Mapping[str, Any]:
    """Validate the nested schema-1 producer receipt of a failed envelope."""

    from scripts.udlm import superiority_gate

    failure = _exact(
        envelope.get("failure_receipt"),
        {"relative_path", "sha256", "schema_version"},
        "pilot failure receipt reference",
    )
    path = _relative(
        failure["relative_path"], "pilot failure receipt path", suffix=".json"
    )
    digest = _sha256(failure["sha256"], "pilot failure receipt digest")
    if failure["schema_version"] != 1:
        raise CandidateAuthorityError("pilot failure receipt schema differs")
    claim, payload = _snapshot(path, "pilot failure receipt")
    if claim.sha256 != digest:
        raise CandidateAuthorityError("pilot failure receipt bytes differ")
    receipt = _strict_json(payload, "pilot failure receipt")
    try:
        validated = superiority_gate._validate_pilot_failure_receipt(
            receipt,
            receipt_path=Path(path),
            attempt_id=str(envelope["attempt_id"]),
            candidate_id=str(envelope["candidate_id"]),
            pilot_seed=int(envelope["pilot_seed"]),
            artifact_loader=_repository_artifact_loader,
        )
    except (OSError, ValueError) as error:
        raise CandidateAuthorityError(
            "failed pilot envelope failed producer-receipt validation"
        ) from error
    if validated.get("requested_samples") != expected_samples:
        raise CandidateAuthorityError("failed pilot requested sample count differs")
    return validated


def _artifact_reference(
    value: Any,
    label: str,
    *,
    expected_attempt_id: str,
    expected_candidate_id: str,
    expected_seed: int,
    expected_samples: int,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    ref = _exact(
        value,
        {"artifact_kind", "pilot_seed", "relative_path", "sha256", "schema_version"},
        label,
    )
    kind = ref["artifact_kind"]
    if kind not in {"pilot_evaluation", "pilot_failure"}:
        raise CandidateAuthorityError(f"{label} artifact kind differs")
    seed = ref["pilot_seed"]
    if type(seed) is not int or seed != expected_seed:
        raise CandidateAuthorityError(f"{label} seed differs")
    live_path = _relative(ref["relative_path"], f"{label} path", suffix=".json")
    expected_live_path = (
        "output/udlm/de_novo_candidate_campaign_v1/evidence/"
        f"{expected_attempt_id}/seed_{expected_seed}.json"
    )
    if live_path != expected_live_path:
        raise CandidateAuthorityError(f"{label} live path differs from its attempt")
    digest = _sha256(ref["sha256"], f"{label} digest")
    if ref["schema_version"] != PILOT_EVIDENCE_SCHEMA_VERSION:
        raise CandidateAuthorityError(f"{label} schema differs")
    live_claim, payload = _snapshot(live_path, f"{label} live envelope")
    if live_claim.sha256 != digest or hashlib.sha256(payload).hexdigest() != digest:
        raise CandidateAuthorityError(f"{label} live bytes differ")
    tracked_path = (
        f"experiments/udlm/pilots/{expected_attempt_id}/seed_{expected_seed}.json"
    )
    tracked_claim, tracked_payload = _snapshot(
        tracked_path, f"{label} tracked envelope"
    )
    if (
        tracked_claim.sha256 != digest
        or tracked_claim.size_bytes != live_claim.size_bytes
        or tracked_payload != payload
    ):
        raise CandidateAuthorityError(f"{label} live/tracked bytes differ")
    tracked_ref = {**dict(ref), "relative_path": tracked_path}
    envelope = _strict_json(payload, label)
    expected_status = "completed" if kind == "pilot_evaluation" else "failed"
    envelope_fields = {
        "schema_version",
        "artifact_kind",
        "status",
        "attempt_id",
        "candidate_id",
        "pilot_seed",
        "final_seed_results_included",
    } | (
        {"training_exit_receipt", "benchmark_artifacts"}
        if kind == "pilot_evaluation"
        else {"failure_receipt"}
    )
    _exact(envelope, envelope_fields, f"{label} envelope")
    for field, expected in {
        "schema_version": PILOT_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": kind,
        "status": expected_status,
        "attempt_id": expected_attempt_id,
        "candidate_id": expected_candidate_id,
        "pilot_seed": seed,
        "final_seed_results_included": False,
    }.items():
        if envelope.get(field) != expected:
            raise CandidateAuthorityError(f"{label} envelope {field} differs")
    if kind == "pilot_evaluation":
        validated = _validate_completed_envelope_live(envelope)
        for field, expected in {
            "attempt_id": expected_attempt_id,
            "candidate_id": expected_candidate_id,
            "pilot_seed": expected_seed,
            "requested_samples": expected_samples,
            "nfe": 128,
            "metric_branch": "released_comparable",
        }.items():
            if validated.get(field) != expected:
                raise CandidateAuthorityError(
                    f"{label} independently validated {field} differs"
                )
        return tracked_ref, validated
    validated = _validate_failed_envelope_live(
        envelope, expected_samples=expected_samples
    )
    return tracked_ref, validated


def _finite_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateAuthorityError(f"{label} must be finite")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise CandidateAuthorityError(f"{label} must lie in [0,1]")
    return score


def _timestamp(value: Any, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise CandidateAuthorityError(f"{label} is not ISO-8601") from error
    else:
        raise CandidateAuthorityError(f"{label} is not an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise CandidateAuthorityError(f"{label} lacks a timezone")
    return parsed


def _stage_path(stage_id: str) -> str:
    return STAGE_DECISION_TEMPLATE.format(stage=stage_id.lower())


def _rank(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        (entry for entry in entries if entry["rankable"]),
        key=lambda entry: (
            -entry["selection_score"]["released_quality"],
            -entry["selection_score"]["released_diversity"],
            entry["config_id"].encode("ascii"),
            entry["attempt_id"].encode("ascii"),
        ),
    )


def _expected_stage_configs(
    stage_id: str,
    *,
    configs: Mapping[str, campaign.CandidateConfig],
    previous_promotions: Mapping[str, list[str]],
) -> list[str]:
    if stage_id == "D":
        return ["e_t100_p100"]
    if stage_id == "A":
        selected = [
            config.config_id
            for config in configs.values()
            if config.raw_loo_top_p == 1.0
        ]
        return sorted(selected, key=lambda config_id: config_id.encode("ascii"))
    if stage_id == "B":
        retained_temperatures = {
            config_id.rsplit("_p", 1)[0] for config_id in previous_promotions["A"]
        }
        selected = [
            config.config_id
            for config in configs.values()
            if config.config_id.rsplit("_p", 1)[0] in retained_temperatures
        ]
        return sorted(selected, key=lambda config_id: config_id.encode("ascii"))
    if stage_id in {"C", "eligible"}:
        predecessor = "B" if stage_id == "C" else "C"
        return sorted(
            previous_promotions[predecessor],
            key=lambda config_id: config_id.encode("ascii"),
        )
    raise AssertionError(stage_id)


def _promotions(
    stage_id: str, entries: Sequence[Mapping[str, Any]]
) -> tuple[list[str], str | None]:
    if stage_id == "D":
        return [], None
    ranked = _rank(entries)
    if stage_id == "eligible":
        if not ranked:
            raise CandidateAuthorityError("eligible stage has no rankable winner")
        return [ranked[0]["config_id"]], ranked[0]["config_id"]
    quota = PROMOTION_QUOTA[stage_id]
    assert quota is not None
    promoted: list[str] = []
    for arm_id in campaign.ARM_IDS:
        arm_ranked = [entry for entry in ranked if entry["arm_id"] == arm_id]
        if len(arm_ranked) < quota:
            raise CandidateAuthorityError(
                f"stage {stage_id} has insufficient rankable quota for {arm_id}"
            )
        promoted.extend(entry["config_id"] for entry in arm_ranked[:quota])
    return promoted, None


def build_candidate_decision(
    registry: campaign.ValidatedRegistry,
    *,
    registry_revision: str,
) -> dict[str, Any]:
    """Reconstruct the exact 40-entry/43-child pre-final campaign."""

    registry_revision = _revision(registry_revision, "registry revision")
    configs = {config.config_id: config for config in registry.configs}
    normalized_stages: list[dict[str, Any]] = []
    previous_promotions: dict[str, list[str]] = {}
    previous_stage_ref: dict[str, Any] | None = None
    previous_stage_completed_at: datetime | None = None
    total_entries = total_children = total_molecules = 0
    for stage_id in campaign.STAGE_IDS:
        path = _stage_path(stage_id)
        claim, payload = _snapshot(path, f"stage {stage_id} decision")
        source = _strict_json(payload, f"stage {stage_id} decision")
        source = _exact(
            source,
            {
                "schema_version",
                "stage_id",
                "status",
                "registry",
                "entries",
                "advancement",
                "completed_at_utc",
            },
            f"stage {stage_id} decision",
        )
        if (
            source["schema_version"] != 1
            or source["stage_id"] != stage_id
            or source["status"] != "completed"
            or source["registry"] != registry.reference
        ):
            raise CandidateAuthorityError(f"stage {stage_id} identity differs")
        completed_at = _timestamp(
            source["completed_at_utc"], f"stage {stage_id} completion timestamp"
        )

        contract = campaign.STAGE_CONTRACT[stage_id]
        expected_config_ids = _expected_stage_configs(
            stage_id,
            configs=configs,
            previous_promotions=previous_promotions,
        )
        if len(expected_config_ids) != contract["entries"]:
            raise CandidateAuthorityError(f"stage {stage_id} schedule count differs")
        raw_entries = source["entries"]
        if not isinstance(raw_entries, list) or len(raw_entries) != contract["entries"]:
            raise CandidateAuthorityError(f"stage {stage_id} entry count differs")
        entries: list[dict[str, Any]] = []
        for index, raw_entry in enumerate(raw_entries):
            entry = _exact(
                raw_entry,
                {
                    "attempt_id",
                    "candidate_id",
                    "config_id",
                    "arm_id",
                    "seeds",
                    "requested_samples_per_seed",
                    "status",
                    "rankable",
                    "released_quality",
                    "released_diversity",
                    "diagnostic_structural_passed",
                    "child_outcomes",
                },
                f"stage {stage_id} entry {index}",
            )
            config_id = entry["config_id"]
            if (
                config_id != expected_config_ids[index]
                or CONFIG_ID.fullmatch(str(config_id)) is None
            ):
                raise CandidateAuthorityError(
                    f"stage {stage_id} entry config/order differs"
                )
            config = configs[config_id]
            expected_attempt = f"stage-{stage_id.lower()}-{config_id}"
            if (
                entry["attempt_id"] != expected_attempt
                or ATTEMPT_ID.fullmatch(expected_attempt) is None
                or entry["candidate_id"] != campaign.CANDIDATE_IDS[config.arm_id]
                or CANDIDATE_ID.fullmatch(str(entry["candidate_id"])) is None
                or entry["arm_id"] != config.arm_id
                or entry["seeds"] != list(contract["seeds"])
                or entry["requested_samples_per_seed"] != contract["samples"]
            ):
                raise CandidateAuthorityError(
                    f"stage {stage_id} entry identity/operating point differs"
                )
            children_raw = entry["child_outcomes"]
            if not isinstance(children_raw, list) or len(children_raw) != len(
                contract["seeds"]
            ):
                raise CandidateAuthorityError(f"{expected_attempt} child count differs")
            validated_children = [
                _artifact_reference(
                    child,
                    f"{expected_attempt} child {child_index}",
                    expected_attempt_id=expected_attempt,
                    expected_candidate_id=str(entry["candidate_id"]),
                    expected_seed=int(contract["seeds"][child_index]),
                    expected_samples=int(contract["samples"]),
                )
                for child_index, child in enumerate(children_raw)
            ]
            children = [reference for reference, _result in validated_children]
            child_results = [result for _reference, result in validated_children]
            if [child["pilot_seed"] for child in children] != list(contract["seeds"]):
                raise CandidateAuthorityError(
                    f"{expected_attempt} child seed order differs"
                )
            for child, result in zip(children, child_results, strict=True):
                if result is None:  # pragma: no cover - validator contract
                    raise CandidateAuthorityError(
                        f"{expected_attempt} child chronology is missing"
                    )
                if child["artifact_kind"] == "pilot_evaluation":
                    child_started = _timestamp(
                        result.get("started_at_utc"),
                        f"{expected_attempt} completed child start",
                    )
                    child_terminal = _timestamp(
                        result.get("completed_at_utc"),
                        f"{expected_attempt} completed child terminal",
                    )
                else:
                    child_started = _timestamp(
                        result.get("started_at"),
                        f"{expected_attempt} failed child start",
                    )
                    child_terminal = _timestamp(
                        result.get("failed_at"),
                        f"{expected_attempt} failed child terminal",
                    )
                if not child_started < child_terminal < completed_at:
                    raise CandidateAuthorityError(
                        f"{expected_attempt} child must terminate before its stage "
                        "decision"
                    )
                if (
                    previous_stage_completed_at is not None
                    and not previous_stage_completed_at < child_started
                ):
                    raise CandidateAuthorityError(
                        f"{expected_attempt} child must start after its predecessor "
                        "stage decision"
                    )
            any_failed = any(
                child["artifact_kind"] == "pilot_failure" for child in children
            )
            if stage_id == "D":
                derived_status = "failed" if any_failed else "completed"
                if entry["status"] != derived_status:
                    raise CandidateAuthorityError(f"{expected_attempt} status differs")
                if (
                    entry["rankable"] is not False
                    or entry["released_quality"] is not None
                    or entry["released_diversity"] is not None
                    or entry["diagnostic_structural_passed"] is not True
                    or derived_status != "completed"
                ):
                    raise CandidateAuthorityError(
                        "diagnostic D must pass structurally and never rank chemistry"
                    )
                score = None
                rankable = False
            else:
                all_metrics_defined = not any_failed and all(
                    result is not None
                    and result["quality"] is not None
                    and result["diversity"] is not None
                    for result in child_results
                )
                derived_status = (
                    "failed"
                    if any_failed
                    else ("completed" if all_metrics_defined else "undefined")
                )
                if entry["status"] != derived_status:
                    raise CandidateAuthorityError(f"{expected_attempt} status differs")
                derived_quality = (
                    statistics.fmean(
                        float(result["quality"])
                        for result in child_results
                        if result is not None
                    )
                    if all_metrics_defined
                    else None
                )
                derived_diversity = (
                    statistics.fmean(
                        float(result["diversity"])
                        for result in child_results
                        if result is not None
                    )
                    if all_metrics_defined
                    else None
                )
                rankable = derived_status == "completed"
                if rankable:
                    score = {
                        "released_quality": _finite_score(
                            derived_quality, f"{expected_attempt} released quality"
                        ),
                        "released_diversity": _finite_score(
                            derived_diversity, f"{expected_attempt} released diversity"
                        ),
                    }
                    if (
                        entry["released_quality"] != score["released_quality"]
                        or entry["released_diversity"] != score["released_diversity"]
                    ):
                        raise CandidateAuthorityError(
                            f"{expected_attempt} released score differs from raw evidence"
                        )
                else:
                    if (
                        entry["released_quality"] is not None
                        or entry["released_diversity"] is not None
                    ):
                        raise CandidateAuthorityError(
                            f"{expected_attempt} unrankable score must be null"
                        )
                    score = None
                if entry["rankable"] is not rankable:
                    raise CandidateAuthorityError(
                        f"{expected_attempt} rankable flag differs"
                    )
                if entry["diagnostic_structural_passed"] is not None:
                    raise CandidateAuthorityError(
                        f"{expected_attempt} diagnostic flag must be null"
                    )
            entries.append(
                {
                    "config_id": config_id,
                    "attempt_id": expected_attempt,
                    "candidate_id": entry["candidate_id"],
                    "arm_id": config.arm_id,
                    "softmax_temp": config.softmax_temp,
                    "raw_loo_top_p": config.raw_loo_top_p,
                    "status": derived_status,
                    "rankable": rankable,
                    "selection_score": score,
                    "child_outcomes": children,
                }
            )

        promoted, global_winner = _promotions(stage_id, entries)
        source_advancement = _exact(
            source["advancement"],
            {
                "predecessor_stage_decision",
                "ranking_order",
                "promoted_config_ids",
                "global_winner_config_id",
                "required_promotions_per_arm",
                "no_retry_or_substitution",
                "on_failed_or_undefined_child",
                "on_insufficient_rankable_quota",
                "accounting",
            },
            f"stage {stage_id} advancement",
        )
        expected_accounting = {
            "scheduled_entry_count": contract["entries"],
            "terminal_entry_count": len(entries),
            "scheduled_child_count": contract["children"],
            "terminal_child_count": sum(
                len(entry["child_outcomes"]) for entry in entries
            ),
            "requested_molecule_count": (contract["children"] * contract["samples"]),
        }
        expected_advancement = {
            "predecessor_stage_decision": previous_stage_ref,
            "ranking_order": [] if stage_id == "D" else RANKING_ORDER,
            "promoted_config_ids": promoted,
            "global_winner_config_id": global_winner,
            "required_promotions_per_arm": PROMOTION_QUOTA[stage_id],
            "no_retry_or_substitution": True,
            "on_failed_or_undefined_child": "retain_unrankable",
            "on_insufficient_rankable_quota": ("campaign_incomplete_without_promotion"),
            "accounting": expected_accounting,
        }
        if dict(source_advancement) != expected_advancement:
            raise CandidateAuthorityError(f"stage {stage_id} advancement differs")
        source_ref = {
            "relative_path": path,
            "sha256": claim.sha256,
            "schema_version": 1,
        }
        normalized_stages.append(
            {
                "stage_id": stage_id,
                "role": STAGE_ROLES[stage_id],
                "seed_values": list(contract["seeds"]),
                "requested_samples_per_child": contract["samples"],
                "scheduled_entry_count": contract["entries"],
                "scheduled_child_count": contract["children"],
                "entries": entries,
                "advancement": {
                    **expected_advancement,
                    "source_stage_decision": source_ref,
                },
            }
        )
        previous_promotions[stage_id] = promoted
        previous_stage_ref = source_ref
        previous_stage_completed_at = completed_at
        total_entries += len(entries)
        total_children += expected_accounting["terminal_child_count"]
        total_molecules += expected_accounting["requested_molecule_count"]

    if (
        total_entries != campaign.PREFINAL_ENTRY_COUNT
        or total_children != campaign.PREFINAL_CHILD_COUNT
        or total_molecules != campaign.PREFINAL_MOLECULE_COUNT
    ):
        raise CandidateAuthorityError("prefinal campaign accounting differs")
    winner_config_id = previous_promotions["eligible"][0]
    winner_entry = next(
        entry
        for entry in normalized_stages[-1]["entries"]
        if entry["config_id"] == winner_config_id
    )
    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "closed_before_candidate_ledger",
        "final_seed_results_included": False,
        "registry": {
            "relative_path": campaign.REGISTRY_RELATIVE_PATH,
            "sha256": registry.raw_sha256,
            "schema_version": campaign.REGISTRY_SCHEMA_VERSION,
            "registry_id": registry.data["registry_id"],
            "registry_revision": registry_revision,
        },
        "campaign": {
            "grid_universe_entry_count": 36,
            "executed_entry_count": 40,
            "child_outcome_count": 43,
            "requested_molecule_count": 3680,
            "nfe": 128,
            "no_cross_stage_pooling": True,
            "no_retries_or_substitutions": True,
            "failed_or_undefined_children_retained_unrankable": True,
            "quota_failure_policy": "campaign_incomplete_and_candidate_lock_forbidden",
            "shared_seed_inference": (
                "blocking_or_common_random_number_control_only_not_paired_inference"
            ),
            "ranking": list(RANKING_ORDER),
        },
        "stages": normalized_stages,
        "selection": {
            "candidate_id": winner_entry["candidate_id"],
            "selected_attempt_id": winner_entry["attempt_id"],
            "selected_config_id": winner_config_id,
            "rule": SELECTION_RULE,
            "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
            "selected_without_final_seed_results": True,
        },
    }
    validate_candidate_decision(decision)
    return decision


def validate_candidate_decision(value: Any) -> dict[str, Any]:
    """Validate the closed decision schema and its non-cyclic identities."""

    decision = _exact(
        value,
        {
            "schema_version",
            "protocol_id",
            "status",
            "final_seed_results_included",
            "registry",
            "campaign",
            "stages",
            "selection",
        },
        "candidate decision",
    )
    if (
        decision["schema_version"] != 1
        or decision["protocol_id"] != PROTOCOL_ID
        or decision["status"] != "closed_before_candidate_ledger"
        or decision["final_seed_results_included"] is not False
    ):
        raise CandidateAuthorityError("candidate decision identity differs")

    def reject_cyclic_keys(node: Any) -> None:
        if isinstance(node, Mapping):
            forbidden = {"candidate_decision_sha256", "decision_revision"}.intersection(
                node
            )
            if forbidden:
                raise CandidateAuthorityError(
                    "candidate decision contains a self-hash cycle"
                )
            for child in node.values():
                reject_cyclic_keys(child)
        elif isinstance(node, list):
            for child in node:
                reject_cyclic_keys(child)

    reject_cyclic_keys(decision)
    registry = _exact(
        decision["registry"],
        {
            "relative_path",
            "sha256",
            "schema_version",
            "registry_id",
            "registry_revision",
        },
        "candidate decision registry",
    )
    if (
        registry["relative_path"] != campaign.REGISTRY_RELATIVE_PATH
        or registry["schema_version"] != campaign.REGISTRY_SCHEMA_VERSION
        or not isinstance(registry["registry_id"], str)
        or not registry["registry_id"]
    ):
        raise CandidateAuthorityError("candidate decision registry identity differs")
    _sha256(registry["sha256"], "candidate decision registry digest")
    _revision(registry["registry_revision"], "candidate decision registry revision")
    expected_campaign = {
        "grid_universe_entry_count": 36,
        "executed_entry_count": 40,
        "child_outcome_count": 43,
        "requested_molecule_count": 3680,
        "nfe": 128,
        "no_cross_stage_pooling": True,
        "no_retries_or_substitutions": True,
        "failed_or_undefined_children_retained_unrankable": True,
        "quota_failure_policy": "campaign_incomplete_and_candidate_lock_forbidden",
        "shared_seed_inference": (
            "blocking_or_common_random_number_control_only_not_paired_inference"
        ),
        "ranking": RANKING_ORDER,
    }
    if decision["campaign"] != expected_campaign:
        raise CandidateAuthorityError("candidate decision campaign contract differs")
    stages = decision["stages"]
    if not isinstance(stages, list) or [
        stage.get("stage_id") for stage in stages
    ] != list(campaign.STAGE_IDS):
        raise CandidateAuthorityError("candidate decision stage order differs")
    previous_promotions: dict[str, list[str]] = {}
    previous_source: dict[str, Any] | None = None
    artifact_paths: set[str] = set()
    normalized_entries = 0
    normalized_children = 0
    normalized_molecules = 0
    for stage_id, raw_stage in zip(campaign.STAGE_IDS, stages, strict=True):
        stage = _exact(
            raw_stage,
            {
                "stage_id",
                "role",
                "seed_values",
                "requested_samples_per_child",
                "scheduled_entry_count",
                "scheduled_child_count",
                "entries",
                "advancement",
            },
            f"candidate decision stage {stage_id}",
        )
        contract = campaign.STAGE_CONTRACT[stage_id]
        expected_stage_identity = {
            "stage_id": stage_id,
            "role": STAGE_ROLES[stage_id],
            "seed_values": list(contract["seeds"]),
            "requested_samples_per_child": contract["samples"],
            "scheduled_entry_count": contract["entries"],
            "scheduled_child_count": contract["children"],
        }
        for field, expected in expected_stage_identity.items():
            if stage[field] != expected:
                raise CandidateAuthorityError(
                    f"candidate decision stage {stage_id} {field} differs"
                )
        if stage_id == "D":
            expected_config_ids = ["e_t100_p100"]
        elif stage_id == "A":
            expected_config_ids = sorted(
                f"{arm.lower()}_t{temperature}_p100"
                for arm in campaign.ARM_IDS
                for temperature in ("050", "070", "085", "100")
            )
        elif stage_id == "B":
            expected_config_ids = sorted(
                f"{config_id.rsplit('_p', 1)[0]}_p{top_p}"
                for config_id in previous_promotions["A"]
                for top_p in ("095", "098", "100")
            )
        else:
            predecessor = "B" if stage_id == "C" else "C"
            expected_config_ids = sorted(previous_promotions[predecessor])
        entries = stage["entries"]
        if (
            not isinstance(entries, list)
            or len(entries) != contract["entries"]
            or [entry.get("config_id") for entry in entries] != expected_config_ids
        ):
            raise CandidateAuthorityError(
                f"candidate decision stage {stage_id} entries/order differ"
            )
        normalized_stage_entries: list[dict[str, Any]] = []
        for entry_index, raw_entry in enumerate(entries):
            label = f"candidate decision stage {stage_id} entry {entry_index}"
            entry = _exact(
                raw_entry,
                {
                    "config_id",
                    "attempt_id",
                    "candidate_id",
                    "arm_id",
                    "softmax_temp",
                    "raw_loo_top_p",
                    "status",
                    "rankable",
                    "selection_score",
                    "child_outcomes",
                },
                label,
            )
            config_id = str(entry["config_id"])
            match = CONFIG_ID.fullmatch(config_id)
            if match is None:
                raise CandidateAuthorityError(f"{label} config ID differs")
            arm_id = config_id[0].upper()
            attempt_id = f"stage-{stage_id.lower()}-{config_id}"
            temperature_code = config_id.split("_t", 1)[1].split("_p", 1)[0]
            top_p_code = config_id.rsplit("_p", 1)[1]
            expected_temperature = int(temperature_code) / 100
            expected_top_p = int(top_p_code) / 100
            expected_candidate = campaign.CANDIDATE_IDS[arm_id]
            if (
                entry["attempt_id"] != attempt_id
                or ATTEMPT_ID.fullmatch(attempt_id) is None
                or entry["candidate_id"] != expected_candidate
                or entry["arm_id"] != arm_id
                or entry["softmax_temp"] != expected_temperature
                or entry["raw_loo_top_p"] != expected_top_p
            ):
                raise CandidateAuthorityError(f"{label} identity differs")
            children = entry["child_outcomes"]
            if not isinstance(children, list) or len(children) != len(
                contract["seeds"]
            ):
                raise CandidateAuthorityError(f"{label} child count differs")
            normalized_child_refs: list[dict[str, Any]] = []
            for child_index, (raw_child, seed) in enumerate(
                zip(children, contract["seeds"], strict=True)
            ):
                child = _exact(
                    raw_child,
                    {
                        "artifact_kind",
                        "pilot_seed",
                        "relative_path",
                        "sha256",
                        "schema_version",
                    },
                    f"{label} child {child_index}",
                )
                expected_child_path = (
                    f"experiments/udlm/pilots/{attempt_id}/seed_{seed}.json"
                )
                if (
                    child["artifact_kind"] not in {"pilot_evaluation", "pilot_failure"}
                    or child["pilot_seed"] != seed
                    or child["relative_path"] != expected_child_path
                    or child["schema_version"] != PILOT_EVIDENCE_SCHEMA_VERSION
                    or expected_child_path in artifact_paths
                ):
                    raise CandidateAuthorityError(f"{label} child identity differs")
                _sha256(child["sha256"], f"{label} child digest")
                artifact_paths.add(expected_child_path)
                normalized_child_refs.append(dict(child))
            any_failed = any(
                child["artifact_kind"] == "pilot_failure"
                for child in normalized_child_refs
            )
            if type(entry["rankable"]) is not bool:
                raise CandidateAuthorityError(f"{label} status/rankability differs")
            score = entry["selection_score"]
            if stage_id == "D":
                derived_status = "failed" if any_failed else "completed"
                if derived_status != "completed":
                    raise CandidateAuthorityError(
                        "diagnostic stage must complete before campaign advancement"
                    )
                if (
                    entry["status"] != derived_status
                    or entry["rankable"] is not False
                    or score is not None
                ):
                    raise CandidateAuthorityError(f"{label} must be unrankable")
            elif any_failed:
                derived_status = "failed"
                if (
                    entry["status"] != derived_status
                    or entry["rankable"] is not False
                    or score is not None
                ):
                    raise CandidateAuthorityError(f"{label} failed semantics differ")
            elif entry["status"] == "undefined":
                derived_status = "undefined"
                if entry["rankable"] is not False or score is not None:
                    raise CandidateAuthorityError(f"{label} undefined semantics differ")
            else:
                derived_status = "completed"
                if entry["status"] != derived_status or entry["rankable"] is not True:
                    raise CandidateAuthorityError(f"{label} completed semantics differ")
                score = _exact(
                    score,
                    {"released_quality", "released_diversity"},
                    f"{label} selection score",
                )
                _finite_score(score["released_quality"], f"{label} quality")
                _finite_score(score["released_diversity"], f"{label} diversity")
            normalized_stage_entries.append(dict(entry))

        promoted, global_winner = _promotions(stage_id, normalized_stage_entries)
        source_path = _stage_path(stage_id)
        advancement = _exact(
            stage["advancement"],
            {
                "predecessor_stage_decision",
                "ranking_order",
                "promoted_config_ids",
                "global_winner_config_id",
                "required_promotions_per_arm",
                "no_retry_or_substitution",
                "on_failed_or_undefined_child",
                "on_insufficient_rankable_quota",
                "accounting",
                "source_stage_decision",
            },
            f"candidate decision stage {stage_id} advancement",
        )
        source_record = _exact(
            advancement["source_stage_decision"],
            {"relative_path", "sha256", "schema_version"},
            f"candidate decision stage {stage_id} source reference",
        )
        source_ref = {
            "relative_path": source_path,
            "sha256": _sha256(
                source_record["sha256"], f"stage {stage_id} source digest"
            ),
            "schema_version": 1,
        }
        if advancement["source_stage_decision"] != source_ref:
            raise CandidateAuthorityError(f"stage {stage_id} source reference differs")
        expected_accounting = {
            "scheduled_entry_count": contract["entries"],
            "terminal_entry_count": contract["entries"],
            "scheduled_child_count": contract["children"],
            "terminal_child_count": contract["children"],
            "requested_molecule_count": contract["children"] * contract["samples"],
        }
        expected_advancement = {
            "predecessor_stage_decision": previous_source,
            "ranking_order": [] if stage_id == "D" else RANKING_ORDER,
            "promoted_config_ids": promoted,
            "global_winner_config_id": global_winner,
            "required_promotions_per_arm": PROMOTION_QUOTA[stage_id],
            "no_retry_or_substitution": True,
            "on_failed_or_undefined_child": "retain_unrankable",
            "on_insufficient_rankable_quota": "campaign_incomplete_without_promotion",
            "accounting": expected_accounting,
            "source_stage_decision": source_ref,
        }
        if dict(advancement) != expected_advancement:
            raise CandidateAuthorityError(f"stage {stage_id} advancement differs")
        previous_promotions[stage_id] = promoted
        previous_source = source_ref
        normalized_entries += len(entries)
        normalized_children += contract["children"]
        normalized_molecules += contract["children"] * contract["samples"]
    if (
        normalized_entries != 40
        or normalized_children != 43
        or normalized_molecules != 3680
    ):
        raise CandidateAuthorityError("candidate decision accounting differs")
    selection = _exact(
        decision["selection"],
        {
            "candidate_id",
            "selected_attempt_id",
            "selected_config_id",
            "rule",
            "checkpoint_selection_rule",
            "selected_without_final_seed_results",
        },
        "candidate decision selection",
    )
    winner_config_id = previous_promotions["eligible"][0]
    winner_entry = next(
        entry
        for entry in stages[-1]["entries"]
        if entry["config_id"] == winner_config_id
    )
    if (
        CANDIDATE_ID.fullmatch(str(selection["candidate_id"])) is None
        or selection["candidate_id"] != winner_entry["candidate_id"]
        or selection["selected_config_id"] != winner_config_id
        or selection["selected_attempt_id"] != winner_entry["attempt_id"]
        or selection["rule"] != SELECTION_RULE
        or selection["checkpoint_selection_rule"] != CHECKPOINT_SELECTION_RULE
        or selection["selected_without_final_seed_results"] is not True
    ):
        raise CandidateAuthorityError("candidate decision selection differs")
    return dict(decision)


def project_candidate_ledger(decision_value: Any) -> dict[str, Any]:
    """Project all and only executed attempts into the unchanged ledger schema 2."""

    decision = validate_candidate_decision(decision_value)
    attempts: list[dict[str, Any]] = []
    for stage in decision["stages"]:
        stage_id = stage["stage_id"]
        for entry in stage["entries"]:
            if entry["status"] == "failed":
                reason = "pilot_failed"
            elif stage_id == "eligible" and not entry["rankable"]:
                reason = "undefined_released_diversity_no_unique_molecules"
            elif stage_id != "eligible":
                reason = "engineering_or_nonregistered_operating_point"
            else:
                reason = None
            eligible = stage_id == "eligible" and entry["rankable"]
            attempts.append(
                {
                    "attempt_id": entry["attempt_id"],
                    "candidate_id": entry["candidate_id"],
                    "status": (
                        "completed"
                        if entry["status"] == "undefined"
                        else entry["status"]
                    ),
                    "eligible_for_selection": eligible,
                    "ineligibility_reason": reason,
                    "pilot_seeds": [
                        child["pilot_seed"] for child in entry["child_outcomes"]
                    ],
                    "selection_score": (
                        {
                            "mean_released_quality": entry["selection_score"][
                                "released_quality"
                            ],
                            "mean_released_diversity": entry["selection_score"][
                                "released_diversity"
                            ],
                        }
                        if eligible
                        else None
                    ),
                    "artifact_refs": list(entry["child_outcomes"]),
                }
            )
    if len(attempts) != 40 or sum(len(row["artifact_refs"]) for row in attempts) != 43:
        raise CandidateAuthorityError("ledger projection accounting differs")
    selection = decision["selection"]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "closed_before_final_evaluation",
        "final_seed_results_included": False,
        "attempts": attempts,
        "selection": {
            "candidate_id": selection["candidate_id"],
            "selected_attempt_id": selection["selected_attempt_id"],
            "rule": SELECTION_RULE,
            "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
            "selected_without_final_seed_results": True,
        },
    }


def _reference_subset(value: Any, label: str) -> dict[str, Any]:
    record = _mapping(value, label)
    try:
        return {
            key: record[key] for key in ("relative_path", "sha256", "schema_version")
        }
    except KeyError as error:
        raise CandidateAuthorityError(f"{label} is incomplete") from error


def _selected_pilot_authority(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate both selected eligible children and return their shared identity."""

    selected = _mapping(decision.get("selection"), "candidate decision selection")
    eligible_stage = _mapping(decision["stages"][-1], "eligible decision stage")
    if eligible_stage.get("stage_id") != "eligible":
        raise CandidateAuthorityError("candidate decision lacks its eligible stage")
    selected_entry = next(
        (
            entry
            for entry in eligible_stage["entries"]
            if entry.get("attempt_id") == selected["selected_attempt_id"]
        ),
        None,
    )
    if (
        not isinstance(selected_entry, Mapping)
        or selected_entry.get("candidate_id") != selected["candidate_id"]
        or selected_entry.get("config_id") != selected["selected_config_id"]
        or selected_entry.get("status") != "completed"
        or selected_entry.get("rankable") is not True
    ):
        raise CandidateAuthorityError("selected eligible entry identity differs")

    source_ref = _mapping(
        eligible_stage["advancement"]["source_stage_decision"],
        "eligible-stage source decision reference",
    )
    source_claim, source_payload = _snapshot(
        str(source_ref["relative_path"]), "eligible-stage source decision"
    )
    if source_claim.sha256 != source_ref.get("sha256"):
        raise CandidateAuthorityError("eligible-stage source decision bytes differ")
    source = _strict_json(source_payload, "eligible-stage source decision")
    if (
        source.get("schema_version") != 1
        or source.get("stage_id") != "eligible"
        or source.get("status") != "completed"
    ):
        raise CandidateAuthorityError("eligible-stage source decision is incomplete")
    completed_at = _timestamp(
        source.get("completed_at_utc"), "eligible-stage decision completion"
    )
    source_entries = source.get("entries")
    if not isinstance(source_entries, list):
        raise CandidateAuthorityError("eligible-stage source entries differ")
    source_entry = next(
        (
            entry
            for entry in source_entries
            if isinstance(entry, Mapping)
            and entry.get("attempt_id") == selected["selected_attempt_id"]
        ),
        None,
    )
    if (
        not isinstance(source_entry, Mapping)
        or source_entry.get("candidate_id") != selected["candidate_id"]
        or source_entry.get("config_id") != selected["selected_config_id"]
        or source_entry.get("status") != "completed"
        or source_entry.get("rankable") is not True
    ):
        raise CandidateAuthorityError("selected source entry identity differs")
    source_children = source_entry.get("child_outcomes")
    tracked_children = selected_entry.get("child_outcomes")
    if (
        not isinstance(source_children, list)
        or not isinstance(tracked_children, list)
        or len(source_children) != 2
        or len(tracked_children) != 2
    ):
        raise CandidateAuthorityError("selected eligible child accounting differs")

    results: list[Mapping[str, Any]] = []
    seeds = campaign.STAGE_CONTRACT["eligible"]["seeds"]
    samples = campaign.STAGE_CONTRACT["eligible"]["samples"]
    for index, (source_child, tracked_child, seed) in enumerate(
        zip(source_children, tracked_children, seeds, strict=True)
    ):
        tracked_ref, result = _artifact_reference(
            source_child,
            f"selected eligible child {index}",
            expected_attempt_id=str(selected["selected_attempt_id"]),
            expected_candidate_id=str(selected["candidate_id"]),
            expected_seed=int(seed),
            expected_samples=int(samples),
        )
        if (
            tracked_ref != tracked_child
            or tracked_ref["artifact_kind"] != "pilot_evaluation"
        ):
            raise CandidateAuthorityError(
                "selected eligible tracked/source evidence differs"
            )
        if result is None:  # pragma: no cover - completed-validator contract
            raise CandidateAuthorityError("selected eligible validation is missing")
        child_completed_at = _timestamp(
            result.get("completed_at_utc"), "selected eligible child completion"
        )
        if not child_completed_at < completed_at:
            raise CandidateAuthorityError(
                "selected eligible child must predate its stage decision"
            )
        results.append(result)

    first = results[0]
    shared_fields = (
        "candidate_id",
        "requested_samples",
        "nfe",
        "metric_branch",
        "checkpoint",
        "evaluation_config",
        "sampling",
        "inference_weights",
        "runner_sha256",
        "sampler_source_sha256",
        "implementation_inputs_sha256",
        "metric_inputs_sha256",
        "benchmark_revision",
    )
    first_receipt = _reference_subset(
        first.get("training_exit_receipt"), "selected pilot training receipt"
    )
    for result in results[1:]:
        result_receipt = _reference_subset(
            result.get("training_exit_receipt"), "selected pilot training receipt"
        )
        if (
            any(result.get(field) != first.get(field) for field in shared_fields)
            or result_receipt != first_receipt
        ):
            raise CandidateAuthorityError(
                "selected eligible children do not share one inference identity"
            )
    quality = statistics.fmean(
        _finite_score(result.get("quality"), "selected pilot quality")
        for result in results
    )
    diversity = statistics.fmean(
        _finite_score(result.get("diversity"), "selected pilot diversity")
        for result in results
    )
    if selected_entry.get("selection_score") != {
        "released_quality": quality,
        "released_diversity": diversity,
    } or (
        source_entry.get("released_quality") != quality
        or source_entry.get("released_diversity") != diversity
    ):
        raise CandidateAuthorityError(
            "selected eligible score differs from independent rescoring"
        )
    return {
        "identity": dict(first),
        "completed_at_utc": max(
            _timestamp(result["completed_at_utc"], "selected pilot completion")
            for result in results
        ).isoformat(),
        "eligible_stage_completed_at_utc": completed_at.isoformat(),
    }


def _load_registry_for_decision(
    decision: Mapping[str, Any],
) -> campaign.ValidatedRegistry:
    reference = _mapping(decision.get("registry"), "candidate decision registry")
    path = str(reference["relative_path"])
    claim, payload = _snapshot(path, "candidate registry")
    if claim.sha256 != reference.get("sha256"):
        raise CandidateAuthorityError("candidate registry bytes differ from decision")
    revision = _revision(reference.get("registry_revision"), "registry revision G")
    if _git_blob(revision, path) != payload:
        raise CandidateAuthorityError("candidate registry bytes differ at G")
    parsed = _strict_json(payload, "candidate registry")
    try:
        registry = campaign.load_registry(
            REPOSITORY_ROOT / path,
            expected_raw_sha256=claim.sha256,
            expected_canonical_sha256=canonical_json_sha256(parsed),
        )
    except (OSError, ValueError) as error:
        raise CandidateAuthorityError("candidate registry is invalid") from error
    if registry.data.get("registry_id") != reference.get(
        "registry_id"
    ) or registry.raw_sha256 != reference.get("sha256"):
        raise CandidateAuthorityError("candidate registry identity differs")
    return registry


def _terminal_training_authority(
    protocol: Mapping[str, Any], *, candidate_id: str
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    """Derive lock training fields from one validated terminal-scale member."""

    terminal = _mapping(
        protocol.get("terminal_scale_up_authority"), "terminal authority"
    )
    members = terminal.get("members")
    if not isinstance(members, list) or len(members) != 3:
        raise CandidateAuthorityError("terminal authority R/S/E members differ")
    selected_member = next(
        (member for member in members if member.get("candidate_id") == candidate_id),
        None,
    )
    terminal_e = next(
        (member for member in members if member.get("arm_id") == "E"), None
    )
    if not isinstance(selected_member, Mapping) or not isinstance(terminal_e, Mapping):
        raise CandidateAuthorityError("candidate is absent from terminal authority")

    parsed_artifacts: dict[str, Mapping[str, Any]] = {}
    for field in (
        "training_summary",
        "successful_exit_receipt",
        "runtime_config",
        "launch_manifest",
    ):
        reference = _mapping(selected_member.get(field), f"selected {field}")
        claim, payload = _snapshot(str(reference["relative_path"]), f"selected {field}")
        if claim.sha256 != reference.get("sha256") or claim.size_bytes != reference.get(
            "size_bytes"
        ):
            raise CandidateAuthorityError(f"selected {field} bytes differ")
        parsed_artifacts[field] = _strict_json(payload, f"selected {field}")
    summary = parsed_artifacts["training_summary"]
    accounting = _mapping(summary.get("training_accounting"), "training accounting")
    counts = _mapping(
        accounting.get("trainable_parameter_counts"), "training parameter counts"
    )
    startup = _mapping(summary.get("startup"), "training startup")
    startup_mode = startup.get("mode")
    if startup_mode == "warm_start":
        warm_report = _mapping(
            startup.get("verified_mdlm_warm_start_report"), "warm-start report"
        )
        initialization_sha256 = warm_report.get("source_sha256")
    elif startup_mode == "scratch":
        initialization_sha256 = None
    else:
        raise CandidateAuthorityError("selected training startup mode differs")
    final_checkpoint = _mapping(summary.get("final_checkpoint"), "training checkpoint")
    semantic = _mapping(
        final_checkpoint.get("semantic_audit"), "training checkpoint semantic audit"
    )
    ema_metadata = dict(_mapping(semantic.get("ema_metadata"), "training EMA metadata"))
    parameter_counts = {
        "base_model_trainable": counts.get("base_backbone"),
        "time_conditioner_trainable": counts.get("time_conditioner"),
        "total_trainable": counts.get("total"),
    }
    if "film_modulation" in counts:
        parameter_counts["film_modulation_trainable"] = counts.get("film_modulation")
    optimizer_updates = accounting.get("optimizer_updates")
    global_examples = accounting.get("effective_global_examples_per_optimizer_step")
    training = {
        "source_revision": terminal.get("source_revision"),
        "training_summary": _reference_subset(
            selected_member["training_summary"], "selected training summary"
        ),
        "exit_receipt": _reference_subset(
            selected_member["successful_exit_receipt"], "selected exit receipt"
        ),
        "runtime_config": _reference_subset(
            selected_member["runtime_config"], "selected runtime config"
        ),
        "launch_manifest": _reference_subset(
            selected_member["launch_manifest"], "selected launch manifest"
        ),
        "resolved_training_config_sha256": summary.get(
            "resolved_training_config_sha256"
        ),
        "training_argv_sha256": summary.get("training_argv_sha256"),
        "checkpoint": {
            **{
                key: selected_member["checkpoint"][key]
                for key in ("relative_path", "sha256", "size_bytes", "global_step")
            },
            "weights": "ema",
        },
        "startup": {
            "mode": startup_mode,
            "initialization_checkpoint_sha256": initialization_sha256,
        },
        "training_seed": accounting.get("training_seed"),
        "optimizer_updates": optimizer_updates,
        "world_size": accounting.get("world_size"),
        "data_exposure": {
            "global_examples_per_optimizer_step": global_examples,
            "optimizer_updates": optimizer_updates,
            "total_requested_examples": (
                global_examples * optimizer_updates
                if type(global_examples) is int and type(optimizer_updates) is int
                else None
            ),
            "stream_partition_policy": accounting.get(
                "hosted_stream_rank_partition_policy"
            ),
        },
        "parameter_counts": parameter_counts,
    }
    inference_weights = {
        "source": "ema",
        "ema_applied": True,
        "ema": ema_metadata,
    }
    return training, selected_member, terminal_e, inference_weights


def _source_digest_at_revision(revision: str, relative_path: str) -> str:
    committed = _git_blob(revision, relative_path)
    _claim, live = _snapshot(relative_path, f"locked source {relative_path}")
    if live != committed:
        raise CandidateAuthorityError(
            f"locked source differs from selected benchmark revision: {relative_path}"
        )
    return hashlib.sha256(committed).hexdigest()


def build_candidate_lock(
    *,
    decision: Mapping[str, Any],
    ledger_claim: artifact_io.FileClaim,
    registry: campaign.ValidatedRegistry,
    locked_at_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build the lock from immutable pre-final evidence; never inspect final runs."""

    from scripts.udlm import superiority_gate

    decision = validate_candidate_decision(decision)
    selected = decision["selection"]
    if (
        registry.raw_sha256 != decision["registry"]["sha256"]
        or registry.data.get("registry_id") != decision["registry"]["registry_id"]
    ):
        raise CandidateAuthorityError("lock registry differs from decision")
    selected_pilot = _selected_pilot_authority(decision)
    identity = _mapping(selected_pilot["identity"], "selected pilot identity")

    protocol_claim, protocol_payload = _snapshot(
        superiority_gate.PROTOCOL_RELATIVE_PATH.as_posix(), "superiority protocol"
    )
    if protocol_claim.sha256 != superiority_gate.PROTOCOL_SHA256:
        raise CandidateAuthorityError("candidate lock protocol source digest differs")
    protocol = _strict_json(protocol_payload, "superiority protocol")
    try:
        superiority_gate.validate_protocol(protocol)
    except (OSError, ValueError) as error:
        raise CandidateAuthorityError("candidate lock protocol is invalid") from error
    training, selected_member, terminal_e, training_inference_weights = (
        _terminal_training_authority(
            protocol, candidate_id=str(selected["candidate_id"])
        )
    )

    configs = {config.config_id: config for config in registry.configs}
    selected_config = configs.get(str(selected["selected_config_id"]))
    if selected_config is None or selected_config.arm_id != selected_member.get(
        "arm_id"
    ):
        raise CandidateAuthorityError("selected config differs from terminal member")
    sampling_record = _mapping(identity.get("sampling"), "selected pilot sampling")
    try:
        sampling = benchmark.validate_sampling_config(
            _mapping(sampling_record.get("config"), "selected sampling config")
        )
    except ValueError as error:
        raise CandidateAuthorityError("selected sampling config is invalid") from error
    expected_evaluation_config = {
        "relative_path": selected_config.relative_path,
        "sha256": selected_config.sha256,
    }
    if (
        identity.get("evaluation_config") != expected_evaluation_config
        or sampling_record.get("sha256") != selected_config.normalized_sampling_sha256
        or canonical_json_sha256(sampling) != selected_config.normalized_sampling_sha256
        or identity.get("benchmark_revision")
        != decision["registry"]["registry_revision"]
    ):
        raise CandidateAuthorityError(
            "selected pilot config/source identity differs from registry G"
        )
    expected_checkpoint_identity = {
        key: training["checkpoint"][key]
        for key in ("sha256", "size_bytes", "global_step")
    }
    if (
        identity.get("checkpoint") != expected_checkpoint_identity
        or identity.get("inference_weights") != training_inference_weights
        or _reference_subset(
            identity.get("training_exit_receipt"), "selected pilot training receipt"
        )
        != training["exit_receipt"]
    ):
        raise CandidateAuthorityError(
            "selected pilot checkpoint/training identity differs"
        )

    benchmark_revision = _revision(
        identity.get("benchmark_revision"), "selected benchmark revision"
    )
    source_paths = {
        "gate_source_sha256": "scripts/udlm/superiority_gate.py",
        "report_source_sha256": "scripts/exps/denovo/report.py",
        "rescore_source_sha256": "scripts/udlm/rescore_denovo_run.py",
        "rescore_dependency_sha256": "scripts/udlm/rescore_mdlm_baseline.py",
        "benchmark_launcher_source_sha256": "scripts/exps/denovo/launch_benchmark.py",
        "pilot_evidence_writer_source_sha256": "scripts/udlm/write_pilot_evidence.py",
    }
    analysis = {
        field: _source_digest_at_revision(benchmark_revision, path)
        for field, path in source_paths.items()
    }
    if (
        _source_digest_at_revision(benchmark_revision, selected_config.relative_path)
        != selected_config.sha256
        or _source_digest_at_revision(
            benchmark_revision, "scripts/exps/denovo/benchmark.py"
        )
        != identity.get("runner_sha256")
        or _source_digest_at_revision(benchmark_revision, "src/genmol/sampler.py")
        != identity.get("sampler_source_sha256")
    ):
        raise CandidateAuthorityError("selected pilot Git source bytes differ")
    import scipy

    analysis["scipy_version"] = scipy.__version__

    locked_at = (
        datetime.now(timezone.utc)
        if locked_at_utc is None
        else _timestamp(locked_at_utc, "candidate lock timestamp")
    )
    if (
        not _timestamp(
            selected_pilot["eligible_stage_completed_at_utc"],
            "eligible-stage decision completion",
        )
        < locked_at
        or not _timestamp(
            selected_pilot["completed_at_utc"], "selected pilot completion"
        )
        < locked_at
    ):
        raise CandidateAuthorityError(
            "candidate lock timestamp must follow selected eligible evidence"
        )
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "candidate_id": selected["candidate_id"],
        "status": "locked_before_final_evaluation",
        "locked_at_utc": locked_at.isoformat(),
        "protocol": {"id": PROTOCOL_ID, "sha256": protocol_claim.sha256},
        "selection": {
            "candidate_ledger": {
                "relative_path": LEDGER_RELATIVE_PATH,
                "sha256": ledger_claim.sha256,
                "schema_version": LEDGER_SCHEMA_VERSION,
            },
            "terminal_e_exit_receipt": _reference_subset(
                terminal_e["successful_exit_receipt"], "terminal E receipt"
            ),
            "selection_rule": SELECTION_RULE,
            "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
            "all_pilot_attempts_disclosed": True,
            "selected_without_final_seed_results": True,
            "final_seeds_used_during_selection": [],
        },
        "training": training,
        "inference": {
            "evaluation_config_relative_path": selected_config.relative_path,
            "evaluation_config_sha256": selected_config.sha256,
            "sampling_config": sampling,
            "sampling_sha256": selected_config.normalized_sampling_sha256,
            "checkpoint_sha256": training["checkpoint"]["sha256"],
            "weights": "ema",
            "inference_weights": identity["inference_weights"],
            "nfe": 128,
            "final_seeds": list(FINAL_SEEDS),
            "samples_per_seed": 1000,
            "sampler_source_sha256": identity["sampler_source_sha256"],
            "benchmark_runner_sha256": identity["runner_sha256"],
            "implementation_inputs_sha256": identity["implementation_inputs_sha256"],
            "metric_inputs_sha256": identity["metric_inputs_sha256"],
            "final_run_directories_by_seed": [
                {
                    "seed": seed,
                    "relative_path": (
                        f"output/udlm/final/{selected['candidate_id']}/"
                        f"{selected['selected_config_id']}/seed_{seed}"
                    ),
                }
                for seed in FINAL_SEEDS
            ],
        },
        "analysis": analysis,
        "claim_scope": superiority_gate.CLAIM_SCOPE_BY_STARTUP[
            training["startup"]["mode"]
        ],
    }
    return validate_lock_draft(
        lock,
        decision=decision,
        ledger_claim=ledger_claim,
        selected_pilot_authority=selected_pilot,
    )


def validate_lock_draft(
    value: Any,
    *,
    decision: Mapping[str, Any],
    ledger_claim: artifact_io.FileClaim,
    selected_pilot_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a fully derived lock draft without changing schema 2."""

    from scripts.udlm import superiority_gate

    lock = _exact(
        value,
        {
            "schema_version",
            "candidate_id",
            "status",
            "locked_at_utc",
            "protocol",
            "selection",
            "training",
            "inference",
            "analysis",
            "claim_scope",
        },
        "candidate lock draft",
    )
    selected = decision["selection"]
    if (
        lock["schema_version"] != LOCK_SCHEMA_VERSION
        or lock["candidate_id"] != selected["candidate_id"]
        or CANDIDATE_ID.fullmatch(str(lock["candidate_id"])) is None
        or lock["status"] != "locked_before_final_evaluation"
    ):
        raise CandidateAuthorityError("candidate lock identity differs")
    locked_at = _timestamp(lock["locked_at_utc"], "candidate lock timestamp")
    eligible_source_ref = _mapping(
        decision["stages"][-1]["advancement"]["source_stage_decision"],
        "eligible-stage source decision",
    )
    eligible_claim, eligible_payload = _snapshot(
        str(eligible_source_ref["relative_path"]), "eligible-stage source decision"
    )
    if eligible_claim.sha256 != eligible_source_ref["sha256"]:
        raise CandidateAuthorityError("eligible-stage source decision bytes differ")
    eligible_source = _strict_json(eligible_payload, "eligible-stage source decision")
    eligible_completed_at = _timestamp(
        eligible_source.get("completed_at_utc"),
        "eligible-stage decision completion",
    )
    if not eligible_completed_at < locked_at:
        raise CandidateAuthorityError(
            "eligible-stage decision must strictly predate the candidate lock"
        )
    selection = _exact(
        lock["selection"],
        {
            "candidate_ledger",
            "terminal_e_exit_receipt",
            "selection_rule",
            "checkpoint_selection_rule",
            "all_pilot_attempts_disclosed",
            "selected_without_final_seed_results",
            "final_seeds_used_during_selection",
        },
        "candidate lock selection",
    )
    ledger_ref = selection["candidate_ledger"]
    if ledger_ref != {
        "relative_path": LEDGER_RELATIVE_PATH,
        "sha256": ledger_claim.sha256,
        "schema_version": LEDGER_SCHEMA_VERSION,
    }:
        raise CandidateAuthorityError("candidate lock ledger binding differs")
    if (
        selection["selection_rule"] != SELECTION_RULE
        or selection["checkpoint_selection_rule"] != CHECKPOINT_SELECTION_RULE
        or selection["all_pilot_attempts_disclosed"] is not True
        or selection["selected_without_final_seed_results"] is not True
        or selection["final_seeds_used_during_selection"] != []
    ):
        raise CandidateAuthorityError("candidate lock selection firewall differs")
    inference = _exact(
        lock["inference"],
        {
            "evaluation_config_relative_path",
            "evaluation_config_sha256",
            "sampling_config",
            "sampling_sha256",
            "checkpoint_sha256",
            "weights",
            "inference_weights",
            "nfe",
            "final_seeds",
            "samples_per_seed",
            "sampler_source_sha256",
            "benchmark_runner_sha256",
            "implementation_inputs_sha256",
            "metric_inputs_sha256",
            "final_run_directories_by_seed",
        },
        "candidate lock inference",
    )
    sampling = _exact(
        inference["sampling_config"],
        set(benchmark.validate_sampling_config(inference["sampling_config"])),
        "candidate lock sampling",
    )
    normalized = benchmark.validate_sampling_config(sampling)
    winner_config = selected["selected_config_id"]
    temperature = int(winner_config.split("_t", 1)[1].split("_p", 1)[0]) / 100
    top_p = int(winner_config.rsplit("_p", 1)[1]) / 100
    if (
        normalized.get("diffusion_type") != "udlm"
        or normalized.get("num_steps") != 128
        or normalized.get("softmax_temp") != temperature
        or normalized.get("raw_loo_top_p") != top_p
        or inference["sampling_sha256"] != canonical_json_sha256(normalized)
        or inference["nfe"] != 128
        or inference["final_seeds"] != list(FINAL_SEEDS)
        or inference["samples_per_seed"] != 1000
        or inference["weights"] != "ema"
    ):
        raise CandidateAuthorityError(
            "candidate lock inference operating point differs"
        )
    directories = inference["final_run_directories_by_seed"]
    if not isinstance(directories, list) or [
        row.get("seed") for row in directories
    ] != list(FINAL_SEEDS):
        raise CandidateAuthorityError("candidate lock final directories differ")
    protocol_claim, protocol_payload = _snapshot(
        superiority_gate.PROTOCOL_RELATIVE_PATH.as_posix(), "superiority protocol"
    )
    if protocol_claim.sha256 != superiority_gate.PROTOCOL_SHA256:
        raise CandidateAuthorityError("candidate lock protocol source digest differs")
    protocol = _strict_json(protocol_payload, "superiority protocol")
    try:
        superiority_gate.validate_protocol(protocol)
        normalized_lock = superiority_gate.validate_candidate_lock(lock, protocol)
    except (OSError, ValueError) as error:
        raise CandidateAuthorityError(
            "candidate lock draft fails the full schema-2 gate contract"
        ) from error
    if normalized_lock.get("candidate_id") != selected["candidate_id"]:
        raise CandidateAuthorityError("candidate lock winner differs from decision")

    pilot_identity = _mapping(
        selected_pilot_authority.get("identity"), "selected pilot authority identity"
    )
    pilot_receipt = _reference_subset(
        pilot_identity.get("training_exit_receipt"),
        "selected pilot training receipt",
    )
    pilot_checks = {
        "candidate_id": selected["candidate_id"],
        "requested_samples": campaign.STAGE_CONTRACT["eligible"]["samples"],
        "nfe": 128,
        "metric_branch": "released_comparable",
        "evaluation_config": {
            "relative_path": inference["evaluation_config_relative_path"],
            "sha256": inference["evaluation_config_sha256"],
        },
        "sampling": {
            "config": inference["sampling_config"],
            "sha256": inference["sampling_sha256"],
        },
        "inference_weights": inference["inference_weights"],
        "runner_sha256": inference["benchmark_runner_sha256"],
        "sampler_source_sha256": inference["sampler_source_sha256"],
        "implementation_inputs_sha256": inference["implementation_inputs_sha256"],
        "metric_inputs_sha256": inference["metric_inputs_sha256"],
        "benchmark_revision": decision["registry"]["registry_revision"],
    }
    if any(
        pilot_identity.get(field) != expected
        for field, expected in pilot_checks.items()
    ):
        raise CandidateAuthorityError(
            "candidate lock differs from independently validated selected evidence"
        )
    if (
        pilot_identity.get("checkpoint")
        != {
            key: lock["training"]["checkpoint"][key]
            for key in ("sha256", "size_bytes", "global_step")
        }
        or pilot_receipt != lock["training"]["exit_receipt"]
    ):
        raise CandidateAuthorityError(
            "candidate lock training differs from selected pilot evidence"
        )
    if (
        not _timestamp(
            selected_pilot_authority.get("completed_at_utc"),
            "selected pilot completion",
        )
        < locked_at
    ):
        raise CandidateAuthorityError(
            "selected pilot evidence must strictly predate the candidate lock"
        )

    terminal = _mapping(
        protocol.get("terminal_scale_up_authority"), "terminal authority"
    )
    members = terminal.get("members")
    if not isinstance(members, list) or len(members) != 3:
        raise CandidateAuthorityError("terminal authority R/S/E members differ")
    terminal_e = next(
        (member for member in members if member.get("arm_id") == "E"), None
    )
    selected_member = next(
        (
            member
            for member in members
            if member.get("candidate_id") == selected["candidate_id"]
        ),
        None,
    )
    if not isinstance(terminal_e, Mapping) or not isinstance(selected_member, Mapping):
        raise CandidateAuthorityError("candidate is absent from terminal authority")

    terminal_receipt = _reference_subset(
        terminal_e["successful_exit_receipt"],
        "terminal authority artifact reference",
    )
    if selection["terminal_e_exit_receipt"] != terminal_receipt:
        raise CandidateAuthorityError("candidate lock terminal E receipt differs")
    terminal_receipt_claim, terminal_receipt_payload = _snapshot(
        terminal_receipt["relative_path"], "terminal E receipt"
    )
    if terminal_receipt_claim.sha256 != terminal_receipt["sha256"]:
        raise CandidateAuthorityError("terminal E receipt bytes differ")
    parsed_terminal_receipt = _strict_json(
        terminal_receipt_payload, "terminal E receipt"
    )
    if (
        parsed_terminal_receipt.get("schema_version")
        != terminal_receipt["schema_version"]
        or parsed_terminal_receipt.get("status") != "completed"
        or parsed_terminal_receipt.get("overall_status") != "completed"
    ):
        raise CandidateAuthorityError("terminal E receipt is not successful")
    training = lock["training"]
    expected_training_refs = {
        "training_summary": "training_summary",
        "exit_receipt": "successful_exit_receipt",
        "runtime_config": "runtime_config",
        "launch_manifest": "launch_manifest",
    }
    for lock_field, member_field in expected_training_refs.items():
        if training[lock_field] != _reference_subset(
            selected_member[member_field], "terminal authority artifact reference"
        ):
            raise CandidateAuthorityError(
                f"candidate lock {lock_field} differs from terminal authority"
            )
    member_checkpoint = _mapping(selected_member["checkpoint"], "selected checkpoint")
    for field in ("relative_path", "sha256", "size_bytes", "global_step"):
        if training["checkpoint"].get(field) != member_checkpoint.get(field):
            raise CandidateAuthorityError(
                "candidate lock checkpoint differs from terminal authority"
            )
    if training.get("source_revision") != terminal.get("source_revision"):
        raise CandidateAuthorityError(
            "candidate lock training revision differs from terminal authority"
        )
    if "candidate_decision" in json.dumps(lock, sort_keys=True):
        raise CandidateAuthorityError(
            "candidate lock must not create a decision hash cycle"
        )
    try:
        training_evidence = superiority_gate.validate_training_evidence(normalized_lock)
        superiority_gate.validate_completed_matched_panel(
            normalized_lock, training_evidence
        )
        superiority_gate.validate_analysis_runtime(normalized_lock)
    except (OSError, ValueError) as error:
        raise CandidateAuthorityError(
            "candidate lock fails independent training/source validation"
        ) from error
    return dict(lock)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CandidateAuthorityError(
            f"git {' '.join(arguments)} failed: {(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def _git_blob(revision: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CandidateAuthorityError(
            f"Git revision {revision} lacks {relative_path}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
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
    raise CandidateAuthorityError(
        f"cannot prove {relative_path} absent at revision {revision}"
    )


def _git_blob_mode(revision: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "ls-tree", "-z", revision, "--", relative_path],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CandidateAuthorityError(
            f"cannot inspect Git mode for {relative_path} at {revision}"
        )
    records = result.stdout.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    if len(records) != 1 or b"\t" not in records[0]:
        raise CandidateAuthorityError(
            f"Git tree has no unique entry for {relative_path} at {revision}"
        )
    metadata, raw_path = records[0].split(b"\t", 1)
    try:
        mode, object_type, _object_id = metadata.decode("ascii").split()
        observed_path = raw_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as error:
        raise CandidateAuthorityError(
            f"Git tree entry is malformed for {relative_path}"
        ) from error
    if object_type != "blob" or observed_path != relative_path:
        raise CandidateAuthorityError(
            f"Git tree entry identity differs for {relative_path}"
        )
    return mode


def _require_git_blob_absent(revision: str, relative_path: str) -> None:
    if not _git_blob_absent(revision, relative_path):
        raise CandidateAuthorityError(
            f"{relative_path} already exists at pre-publication revision {revision}"
        )


def _require_exact_publication_commit(revision: str, relative_path: str) -> str:
    revision = _revision(revision, "publication revision")
    parent_line = _git("rev-list", "--parents", "-n", "1", revision).split()
    if len(parent_line) != 2:
        raise CandidateAuthorityError(
            "authority publication commit must have one parent"
        )
    changed = set(
        filter(
            None,
            _git(
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                revision,
            ).splitlines(),
        )
    )
    if changed != {relative_path}:
        raise CandidateAuthorityError(
            f"authority publication commit must change only {relative_path}"
        )
    return parent_line[1]


def _git_changed_entries(parent: str, revision: str) -> tuple[tuple[str, str], ...]:
    result = subprocess.run(
        [
            "git",
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--no-renames",
            "-r",
            "-z",
            parent,
            revision,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CandidateAuthorityError("cannot inspect evidence publication changes")
    records = result.stdout.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    if len(records) % 2:
        raise CandidateAuthorityError("evidence publication change record is malformed")
    entries: list[tuple[str, str]] = []
    for index in range(0, len(records), 2):
        try:
            status = records[index].decode("ascii")
            path = records[index + 1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise CandidateAuthorityError(
                "evidence publication contains a noncanonical path"
            ) from error
        entries.append((status, path))
    return tuple(entries)


def _validate_evidence_publication(
    evidence_revision: str,
    *,
    registry_revision: str,
    registry: campaign.ValidatedRegistry,
) -> dict[str, Any]:
    """Prove E is the exact evidence-only child of G before decision publication."""

    evidence_revision = _revision(evidence_revision, "evidence revision")
    registry_revision = _revision(registry_revision, "registry revision G")
    parent_line = _git("rev-list", "--parents", "-n", "1", evidence_revision).split()
    if len(parent_line) != 2 or parent_line[1] != registry_revision:
        raise CandidateAuthorityError(
            "evidence revision must be the exact sole child of registry revision G"
        )
    manifest_path = materialize_candidate_evidence.MANIFEST_RELATIVE_PATH
    if not _git_blob_absent(registry_revision, manifest_path):
        raise CandidateAuthorityError("campaign evidence manifest existed at G")
    manifest_claim, manifest_payload = _snapshot(
        manifest_path, "campaign evidence manifest"
    )
    if _git_blob(evidence_revision, manifest_path) != manifest_payload:
        raise CandidateAuthorityError(
            "evidence revision lacks the exact live campaign manifest"
        )
    try:
        manifest = materialize_candidate_evidence.validate_manifest(
            _strict_json(manifest_payload, "campaign evidence manifest")
        )
    except ValueError as error:
        raise CandidateAuthorityError(
            "campaign evidence manifest is invalid"
        ) from error
    if manifest["registry"] != registry.reference or manifest["source_revision"] != {
        "head": registry_revision,
        "upstream": registry_revision,
    }:
        raise CandidateAuthorityError("campaign evidence manifest does not bind G")

    expected_additions = set(manifest["required_git_paths"])
    expected_additions.add(manifest_path)
    changed = _git_changed_entries(registry_revision, evidence_revision)
    if (
        len(changed) != len(expected_additions)
        or {path for status, path in changed if status == "A"} != expected_additions
        or any(status != "A" for status, _path in changed)
    ):
        raise CandidateAuthorityError(
            "evidence revision must add only the exact manifest-derived closure"
        )
    for path in expected_additions:
        if _git_blob_mode(evidence_revision, path) != "100644":
            raise CandidateAuthorityError(
                f"evidence addition must be a non-executable regular blob: {path}"
            )

    expected_refs: dict[str, tuple[str, int | None]] = {}

    def retain_expected(
        path: str, digest: str, size_bytes: int | None, *, label: str
    ) -> None:
        expected = (_sha256(digest, f"{label} digest"), size_bytes)
        prior = expected_refs.get(path)
        if prior is not None and prior != expected:
            raise CandidateAuthorityError(f"{label} repeats a path inconsistently")
        expected_refs[path] = expected

    for reference in manifest["stage_decisions"]:
        retain_expected(
            reference["relative_path"],
            reference["sha256"],
            None,
            label=f"stage {reference['stage_id']} decision",
        )
    for child in manifest["children"]:
        tracked = child["tracked_envelope"]
        live = child["live_envelope"]
        retain_expected(
            tracked["relative_path"],
            tracked["sha256"],
            tracked["size_bytes"],
            label="tracked envelope",
        )
        live_claim, live_payload = _snapshot(
            live["relative_path"], "live campaign envelope"
        )
        if (
            live_claim.sha256 != live["sha256"]
            or live_claim.size_bytes != live["size_bytes"]
            or hashlib.sha256(live_payload).hexdigest() != tracked["sha256"]
        ):
            raise CandidateAuthorityError(
                "live campaign envelope differs from its manifest bridge"
            )
        for support in child["supporting_artifacts"]:
            retain_expected(
                support["relative_path"],
                support["sha256"],
                support["size_bytes"],
                label=f"supporting artifact {support['artifact_role']}",
            )
    if set(expected_refs) != set(manifest["required_git_paths"]):
        raise CandidateAuthorityError("manifest Git closure is not reference-derived")
    for path, (digest, expected_size) in expected_refs.items():
        if not _git_blob_absent(registry_revision, path):
            raise CandidateAuthorityError(f"evidence path already existed at G: {path}")
        committed = _git_blob(evidence_revision, path)
        claim, live_payload = _snapshot(path, f"committed evidence {path}")
        if (
            hashlib.sha256(committed).hexdigest() != digest
            or (expected_size is not None and len(committed) != expected_size)
            or claim.sha256 != digest
            or (expected_size is not None and claim.size_bytes != expected_size)
            or live_payload != committed
        ):
            raise CandidateAuthorityError(f"committed evidence blob differs: {path}")
    if hashlib.sha256(manifest_payload).hexdigest() != manifest_claim.sha256:
        raise CandidateAuthorityError("campaign evidence manifest hash differs")

    registry_blob = _git_blob(registry_revision, campaign.REGISTRY_RELATIVE_PATH)
    if (
        _git_blob(evidence_revision, campaign.REGISTRY_RELATIVE_PATH) != registry_blob
        or hashlib.sha256(registry_blob).hexdigest() != registry.raw_sha256
        or len(registry_blob) != registry.size_bytes
    ):
        raise CandidateAuthorityError("candidate registry changed after G")
    for config in registry.configs:
        config_blob = _git_blob(registry_revision, config.relative_path)
        if (
            _git_blob(evidence_revision, config.relative_path) != config_blob
            or hashlib.sha256(config_blob).hexdigest() != config.sha256
            or len(config_blob) != config.size_bytes
        ):
            raise CandidateAuthorityError(
                f"candidate config changed after G: {config.config_id}"
            )
    for launcher_path in materialize_candidate_evidence.LAUNCHER_RELATIVE_PATHS:
        if _git_blob(evidence_revision, launcher_path) != _git_blob(
            registry_revision, launcher_path
        ):
            raise CandidateAuthorityError(f"launcher changed after G: {launcher_path}")
    for path in (DECISION_RELATIVE_PATH, LEDGER_RELATIVE_PATH, LOCK_RELATIVE_PATH):
        if not _git_blob_absent(registry_revision, path) or not _git_blob_absent(
            evidence_revision, path
        ):
            raise CandidateAuthorityError(
                "candidate authority existed before decision publication"
            )
    return manifest


def _require_clean_pushed_source(expected_revision: str) -> str:
    expected_revision = _revision(expected_revision, "expected source revision")
    try:
        state = benchmark.require_clean_pushed_source(expected_revision)
    except (RuntimeError, ValueError) as error:
        raise CandidateAuthorityError(
            "source is not the exact clean pushed revision"
        ) from error
    if (
        state.get("head") != expected_revision
        or state.get("upstream") != expected_revision
    ):
        raise CandidateAuthorityError("clean pushed source identity differs")
    return expected_revision


def _publish(relative_path: str, value: Mapping[str, Any]) -> artifact_io.FileClaim:
    output_parent = REPOSITORY_ROOT / PurePosixPath(relative_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    try:
        return artifact_io.publish_bytes_exclusive(
            REPOSITORY_ROOT, relative_path, canonical_json_bytes(value)
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise CandidateAuthorityError(f"cannot publish {relative_path}") from error


def publish_decision(
    *,
    expected_source_revision: str,
    expected_registry_sha256: str,
    expected_registry_canonical_sha256: str,
    registry_revision: str,
) -> artifact_io.FileClaim:
    _require_clean_pushed_source(expected_source_revision)
    for path in (DECISION_RELATIVE_PATH, LEDGER_RELATIVE_PATH, LOCK_RELATIVE_PATH):
        _require_git_blob_absent(expected_source_revision, path)
        if os.path.lexists(REPOSITORY_ROOT / path):
            raise CandidateAuthorityError(
                "decision phase requires fresh authority paths"
            )
    registry = campaign.load_registry(
        REPOSITORY_ROOT / campaign.REGISTRY_RELATIVE_PATH,
        expected_raw_sha256=expected_registry_sha256,
        expected_canonical_sha256=expected_registry_canonical_sha256,
    )
    registry_revision = _revision(registry_revision, "registry revision")
    if (
        _git_blob(registry_revision, campaign.REGISTRY_RELATIVE_PATH)
        != _snapshot(campaign.REGISTRY_RELATIVE_PATH, "candidate registry")[1]
    ):
        raise CandidateAuthorityError(
            "registry revision does not contain live registry"
        )
    _validate_evidence_publication(
        expected_source_revision,
        registry_revision=registry_revision,
        registry=registry,
    )
    decision = build_candidate_decision(registry, registry_revision=registry_revision)
    return _publish(DECISION_RELATIVE_PATH, decision)


def publish_ledger(*, expected_source_revision: str) -> artifact_io.FileClaim:
    _require_clean_pushed_source(expected_source_revision)
    for path in (LEDGER_RELATIVE_PATH, LOCK_RELATIVE_PATH):
        _require_git_blob_absent(expected_source_revision, path)
    if os.path.lexists(REPOSITORY_ROOT / LOCK_RELATIVE_PATH):
        raise CandidateAuthorityError("ledger phase must precede lock")
    _require_exact_publication_commit(expected_source_revision, DECISION_RELATIVE_PATH)
    _claim, payload = _snapshot(DECISION_RELATIVE_PATH, "candidate decision")
    if _git_blob(expected_source_revision, DECISION_RELATIVE_PATH) != payload:
        raise CandidateAuthorityError("clean pushed revision lacks the exact decision")
    decision = validate_candidate_decision(_strict_json(payload, "candidate decision"))
    return _publish(LEDGER_RELATIVE_PATH, project_candidate_ledger(decision))


def publish_lock(*, expected_source_revision: str) -> artifact_io.FileClaim:
    _require_clean_pushed_source(expected_source_revision)
    _require_git_blob_absent(expected_source_revision, LOCK_RELATIVE_PATH)
    parent_revision = _require_exact_publication_commit(
        expected_source_revision, LEDGER_RELATIVE_PATH
    )
    evidence_revision = _require_exact_publication_commit(
        parent_revision, DECISION_RELATIVE_PATH
    )
    decision_claim, decision_payload = _snapshot(
        DECISION_RELATIVE_PATH, "candidate decision"
    )
    del decision_claim
    decision = validate_candidate_decision(
        _strict_json(decision_payload, "candidate decision")
    )
    if _git_blob(parent_revision, DECISION_RELATIVE_PATH) != decision_payload:
        raise CandidateAuthorityError(
            "decision publication revision lacks the exact decision"
        )
    ledger_claim, ledger_payload = _snapshot(LEDGER_RELATIVE_PATH, "candidate ledger")
    expected_ledger = canonical_json_bytes(project_candidate_ledger(decision))
    if ledger_payload != expected_ledger:
        raise CandidateAuthorityError(
            "ledger is not the deterministic decision projection"
        )
    committed_ledger = _git_blob(expected_source_revision, LEDGER_RELATIVE_PATH)
    if committed_ledger != ledger_payload:
        raise CandidateAuthorityError(
            "clean pushed revision does not contain exact ledger"
        )
    registry = _load_registry_for_decision(decision)
    _validate_evidence_publication(
        evidence_revision,
        registry_revision=decision["registry"]["registry_revision"],
        registry=registry,
    )
    lock = build_candidate_lock(
        decision=decision,
        ledger_claim=ledger_claim,
        registry=registry,
    )
    return _publish(LOCK_RELATIVE_PATH, lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=("decision", "ledger", "lock")
    )
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-registry-sha256")
    parser.add_argument("--expected-registry-canonical-sha256")
    parser.add_argument("--registry-revision")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "decision":
        if not all(
            (
                args.expected_registry_sha256,
                args.expected_registry_canonical_sha256,
                args.registry_revision,
            )
        ):
            raise CandidateAuthorityError(
                "decision phase requires both registry digests and registry revision"
            )
        claim = publish_decision(
            expected_source_revision=args.expected_source_revision,
            expected_registry_sha256=args.expected_registry_sha256,
            expected_registry_canonical_sha256=(
                args.expected_registry_canonical_sha256
            ),
            registry_revision=args.registry_revision,
        )
    elif args.phase == "ledger":
        if any(
            value is not None
            for value in (
                args.expected_registry_sha256,
                args.expected_registry_canonical_sha256,
                args.registry_revision,
            )
        ):
            raise CandidateAuthorityError("ledger phase rejects decision-only inputs")
        claim = publish_ledger(expected_source_revision=args.expected_source_revision)
    else:
        if any(
            value is not None
            for value in (
                args.expected_registry_sha256,
                args.expected_registry_canonical_sha256,
                args.registry_revision,
            )
        ):
            raise CandidateAuthorityError("lock phase rejects decision-only inputs")
        claim = publish_lock(expected_source_revision=args.expected_source_revision)
    print(
        json.dumps(
            {
                "status": "published",
                "phase": args.phase,
                "relative_path": claim.relative_path,
                "sha256": claim.sha256,
                "size_bytes": claim.size_bytes,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
