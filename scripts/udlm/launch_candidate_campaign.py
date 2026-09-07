"""Run a caller-bounded prefix of the frozen pre-final R/S/E campaign.

Real execution requires an explicit ``--through-stage`` authorization.  A
later invocation resumes from the immutable contiguous decision chain; it
never reruns an already terminal stage.  Selecting ``eligible`` explicitly
authorizes the full remaining pre-final campaign, but never the final stage.
A dry run is CPU/read-only: it never acquires the generation lease, queries
NVIDIA telemetry, creates output, or touches tmux.  Real execution holds the
repository-global generation lease across every handed-off child and each
completion-last stage decision in the selected prefix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    while str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import launch_benchmark as benchmark_launcher  # noqa: E402
from scripts.udlm import (  # noqa: E402
    verify_denovo_candidate_config_registry as registry_verifier,
)


REGISTRY_RELATIVE_PATH = (
    "experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json"
)
CONFIG_DIRECTORY_RELATIVE_PATH = (
    "experiments/udlm/protocols/de_novo_candidate_configs_v1"
)
CAMPAIGN_RELATIVE_ROOT = "output/udlm/de_novo_candidate_campaign_v1"
CAMPAIGN_LOG_RELATIVE_ROOT = "output/logs/de_novo_candidate_campaign_v1"
DECISION_SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 1
ARM_IDS = ("R", "S", "E")
TEMPERATURES = (0.5, 0.7, 0.85, 1.0)
RAW_TOP_P_VALUES = (1.0, 0.98, 0.95)
STAGE_IDS = ("D", "A", "B", "C", "eligible")
REGISTRY_STAGE_IDS = (*STAGE_IDS, "final")
CANDIDATE_IDS = {
    "R": "r-w1-1000u-dcb271453411",
    "S": "s-w1-1000u-dcb271453411",
    "E": "e-w1-1000u-dcb271453411",
}
STAGE_CONTRACT = {
    "D": {"entries": 1, "children": 1, "seeds": (1100,), "samples": 32},
    "A": {"entries": 12, "children": 12, "seeds": (1101,), "samples": 32},
    "B": {"entries": 18, "children": 18, "seeds": (1102,), "samples": 64},
    "C": {"entries": 6, "children": 6, "seeds": (1103,), "samples": 96},
    "eligible": {
        "entries": 3,
        "children": 6,
        "seeds": (1000, 1001),
        "samples": 256,
    },
}
MAX_CONCURRENT_CHILDREN = 3
MAX_UTILIZATION_PERCENT = 10
MIN_FREE_MEMORY_MIB = 30_000
PREFINAL_ENTRY_COUNT = 40
PREFINAL_CHILD_COUNT = 43
PREFINAL_MOLECULE_COUNT = 3_680
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class CampaignValidationError(ValueError):
    """The registry, replay state, or one terminal child is invalid."""


@dataclass(frozen=True)
class CandidateConfig:
    config_id: str
    arm_id: str
    scale_up_member_slug: str
    softmax_temp: float
    raw_loo_top_p: float
    is_reused_identity: bool
    relative_path: str
    sha256: str
    size_bytes: int
    storage: str
    normalized_sampling_sha256: str


@dataclass(frozen=True)
class ValidatedRegistry:
    path: Path
    raw_sha256: str
    canonical_sha256: str
    size_bytes: int
    data: Mapping[str, Any]
    configs: tuple[CandidateConfig, ...]
    checkpoint_by_arm: Mapping[str, Mapping[str, Any]]
    training_receipt_by_arm: Mapping[str, Mapping[str, Any]]

    @property
    def reference(self) -> dict[str, Any]:
        return {
            "relative_path": REGISTRY_RELATIVE_PATH,
            "sha256": self.raw_sha256,
            "canonical_sha256": self.canonical_sha256,
            "size_bytes": self.size_bytes,
            "schema_version": REGISTRY_SCHEMA_VERSION,
        }


@dataclass(frozen=True)
class AttemptSpec:
    stage_id: str
    config_id: str
    candidate_id: str
    arm_id: str
    attempt_id: str
    seeds: tuple[int, ...]
    requested_samples_per_seed: int


@dataclass(frozen=True)
class AttemptOutcome:
    spec: AttemptSpec
    status: str
    child_outcomes: tuple[Mapping[str, Any], ...]
    released_quality: float | None
    released_diversity: float | None
    diagnostic_structural_passed: bool | None = None

    @property
    def rankable(self) -> bool:
        return (
            self.status == "completed"
            and self.released_quality is not None
            and self.released_diversity is not None
            and math.isfinite(self.released_quality)
            and math.isfinite(self.released_diversity)
        )


@dataclass(frozen=True)
class CampaignRunningJob:
    spec: AttemptSpec
    expected: benchmark_launcher.ExpectedRunIdentity
    job: benchmark_launcher.RunningJob


@dataclass(frozen=True)
class TerminalChild:
    """One handed-off child after process and completion receipt finalization."""

    spec: AttemptSpec
    seed: int
    expected: benchmark_launcher.ExpectedRunIdentity
    failed: bool
    command: tuple[str, ...]


@dataclass(frozen=True)
class ChildEvidence:
    terminal: TerminalChild
    reference: Mapping[str, Any]
    released_quality: float | None
    released_diversity: float | None
    diagnostic_structural_passed: bool | None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-canonical-sha256", required=True)
    parser.add_argument(
        "--through-stage",
        choices=STAGE_IDS,
        help=(
            "last pre-final stage authorized by this invocation; required for "
            "real execution"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and args.through_stage is None:
        parser.error("--through-stage is required unless --dry-run is used")
    return args


def _exact_keys(value: object, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        found = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise CampaignValidationError(
            f"{label} fields differ: found {found!r}, expected {sorted(expected)!r}"
        )
    return value


def _reject_constant(value: str) -> None:
    raise CampaignValidationError(f"non-finite JSON constant is forbidden: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignValidationError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, Mapping):
        raise CampaignValidationError(f"{label} root must be an object")
    return parsed


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CampaignValidationError("value is not canonicalizable JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise CampaignValidationError(f"{label} must be 64 lowercase hex")
    return value


def _revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise CampaignValidationError(f"{label} must be 40 lowercase hex")
    return value


def _utc_timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise CampaignValidationError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CampaignValidationError(
            f"{label} must be an ISO-8601 UTC timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise CampaignValidationError(f"{label} must carry an explicit UTC offset")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CampaignValidationError(f"{label} must be a nonempty path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise CampaignValidationError(f"{label} must be canonical root-relative")
    return value


def _config_id(arm: str, temperature: float, top_p: float) -> str:
    temperature_token = {0.5: "050", 0.7: "070", 0.85: "085", 1.0: "100"}[temperature]
    top_p_token = {1.0: "100", 0.98: "098", 0.95: "095"}[top_p]
    return f"{arm.lower()}_t{temperature_token}_p{top_p_token}"


def _attempt_id(stage_id: str, config_id: str, seeds: tuple[int, ...]) -> str:
    if stage_id not in STAGE_IDS:
        raise CampaignValidationError("attempt stage is unknown")
    if seeds != STAGE_CONTRACT[stage_id]["seeds"]:
        raise CampaignValidationError("attempt seed tuple differs from its stage")
    attempt_id = f"stage-{stage_id.lower()}-{config_id}"
    if benchmark_launcher.ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None:
        raise CampaignValidationError("derived attempt ID violates launcher grammar")
    return attempt_id


def _stable_repository_file(
    relative_path: str, *, label: str
) -> tuple[artifact_io.FileClaim, bytes]:
    try:
        claim, payload = artifact_io.snapshot_file(
            REPOSITORY_ROOT, relative_path, capture_bytes=True
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise CampaignValidationError(f"cannot read stable {label}") from error
    if payload is None:  # pragma: no cover - capture_bytes invariant
        raise AssertionError("stable artifact payload was not retained")
    return claim, payload


def _reference(
    value: object, *, label: str, expected_schema_version: int | None = None
) -> tuple[str, str, int]:
    reference = _exact_keys(
        value,
        {"relative_path", "sha256", "schema_version"},
        label=label,
    )
    relative_path = _relative_path(reference["relative_path"], label=f"{label} path")
    digest = _sha256(reference["sha256"], label=f"{label} hash")
    schema_version = reference["schema_version"]
    if type(schema_version) is not int or schema_version <= 0:
        raise CampaignValidationError(f"{label} schema version is invalid")
    if (
        expected_schema_version is not None
        and schema_version != expected_schema_version
    ):
        raise CampaignValidationError(f"{label} schema version differs")
    claim, payload = _stable_repository_file(relative_path, label=label)
    if claim.sha256 != digest or hashlib.sha256(payload).hexdigest() != digest:
        raise CampaignValidationError(f"{label} bytes differ from registry")
    parsed = _strict_json(payload, label=label)
    if parsed.get("schema_version") != schema_version:
        raise CampaignValidationError(f"{label} embedded schema version differs")
    return relative_path, digest, schema_version


def _validate_config_record(value: object) -> CandidateConfig:
    record = _exact_keys(
        value,
        {
            "config_id",
            "arm_id",
            "scale_up_member_slug",
            "softmax_temp",
            "raw_loo_top_p",
            "is_reused_identity",
            "config",
            "normalized_sampling_sha256",
        },
        label="candidate config",
    )
    arm_id = record["arm_id"]
    temperature = record["softmax_temp"]
    top_p = record["raw_loo_top_p"]
    if arm_id not in ARM_IDS:
        raise CampaignValidationError("candidate config arm is invalid")
    if type(temperature) not in (int, float) or float(temperature) not in TEMPERATURES:
        raise CampaignValidationError("candidate config temperature is invalid")
    if type(top_p) not in (int, float) or float(top_p) not in RAW_TOP_P_VALUES:
        raise CampaignValidationError("candidate config raw top-p is invalid")
    temperature = float(temperature)
    top_p = float(top_p)
    config_id = _config_id(str(arm_id), temperature, top_p)
    if record["config_id"] != config_id:
        raise CampaignValidationError("candidate config ID is noncanonical")
    if record["scale_up_member_slug"] != str(arm_id).lower():
        raise CampaignValidationError("candidate config scale-up slug differs")
    reused = record["is_reused_identity"]
    if type(reused) is not bool or reused is not (temperature == 1.0 and top_p == 1.0):
        raise CampaignValidationError("candidate config identity-reuse flag differs")
    reference = _exact_keys(
        record["config"],
        {"relative_path", "sha256", "size_bytes", "storage"},
        label=f"{config_id} config reference",
    )
    relative_path = _relative_path(
        reference["relative_path"], label=f"{config_id} config path"
    )
    digest = _sha256(reference["sha256"], label=f"{config_id} config hash")
    size_bytes = reference["size_bytes"]
    storage = reference["storage"]
    if type(size_bytes) is not int or size_bytes <= 0:
        raise CampaignValidationError("candidate config size must be positive")
    if not isinstance(storage, str) or not storage:
        raise CampaignValidationError("candidate config storage must be nonempty")
    if not reused and not relative_path.startswith(
        CONFIG_DIRECTORY_RELATIVE_PATH + "/"
    ):
        raise CampaignValidationError(
            "generated candidate config escapes its directory"
        )
    claim, payload = _stable_repository_file(relative_path, label=f"{config_id} config")
    if claim.sha256 != digest or claim.size_bytes != size_bytes:
        raise CampaignValidationError("candidate config byte identity differs")
    path = REPOSITORY_ROOT.joinpath(*PurePosixPath(relative_path).parts)
    source = benchmark_launcher.benchmark_runner.load_yaml_config(path)
    normalized = benchmark_launcher.benchmark_runner.validate_sampling_config(source)
    normalized_sha256 = _sha256(
        record["normalized_sampling_sha256"],
        label=f"{config_id} normalized sampling hash",
    )
    if (
        _canonical_sha256(normalized) != normalized_sha256
        or normalized.get("diffusion_type") != "udlm"
        or normalized.get("num_steps") != 128
        or normalized.get("softmax_temp") != temperature
        or normalized.get("raw_loo_top_p") != top_p
    ):
        raise CampaignValidationError(
            f"{config_id} normalized sampling settings differ from registry"
        )
    return CandidateConfig(
        config_id=config_id,
        arm_id=str(arm_id),
        scale_up_member_slug=str(record["scale_up_member_slug"]),
        softmax_temp=temperature,
        raw_loo_top_p=top_p,
        is_reused_identity=reused,
        relative_path=relative_path,
        sha256=digest,
        size_bytes=size_bytes,
        storage=storage,
        normalized_sampling_sha256=normalized_sha256,
    )


def _validate_stages(value: object) -> None:
    expected = registry_verifier.expected_stages()
    if value != expected:
        raise CampaignValidationError("registry stage contract differs")


def _checkpoint_authority(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    terminal = protocol.get("terminal_scale_up_authority")
    if not isinstance(terminal, Mapping) or terminal.get("status") != "validated":
        raise CampaignValidationError("protocol lacks validated scale-up authority")
    members = terminal.get("members")
    if not isinstance(members, list) or len(members) != 3:
        raise CampaignValidationError("protocol scale-up authority must contain R/S/E")
    result: dict[str, Mapping[str, Any]] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    for member in members:
        if not isinstance(member, Mapping):
            raise CampaignValidationError("scale-up member authority must be an object")
        arm_id = member.get("arm_id")
        if arm_id not in ARM_IDS or arm_id in result:
            raise CampaignValidationError("scale-up member arm identity is invalid")
        if member.get("candidate_id") != CANDIDATE_IDS[str(arm_id)]:
            raise CampaignValidationError("scale-up candidate ID differs")
        checkpoint = member.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise CampaignValidationError("scale-up member checkpoint is missing")
        relative_path = _relative_path(
            checkpoint.get("relative_path"), label=f"{arm_id} checkpoint"
        )
        digest = _sha256(checkpoint.get("sha256"), label=f"{arm_id} checkpoint")
        size_bytes = checkpoint.get("size_bytes")
        if (
            type(size_bytes) is not int
            or size_bytes <= 0
            or checkpoint.get("global_step") != 1000
        ):
            raise CampaignValidationError("scale-up checkpoint metadata differs")
        claim, _payload = artifact_io.snapshot_file(
            REPOSITORY_ROOT, relative_path, capture_bytes=False
        )
        if claim.sha256 != digest or claim.size_bytes != size_bytes:
            raise CampaignValidationError("live scale-up checkpoint bytes differ")
        result[str(arm_id)] = dict(checkpoint)
        receipt = member.get("successful_exit_receipt")
        if not isinstance(receipt, Mapping):
            raise CampaignValidationError("scale-up training receipt is missing")
        receipt_path = _relative_path(
            receipt.get("relative_path"), label=f"{arm_id} training receipt"
        )
        receipt_digest = _sha256(
            receipt.get("sha256"), label=f"{arm_id} training receipt"
        )
        receipt_size = receipt.get("size_bytes")
        if (
            type(receipt_size) is not int
            or receipt_size <= 0
            or receipt.get("schema_version") != 5
        ):
            raise CampaignValidationError("scale-up training receipt metadata differs")
        receipt_claim, _receipt_payload = artifact_io.snapshot_file(
            REPOSITORY_ROOT, receipt_path, capture_bytes=False
        )
        if (
            receipt_claim.sha256 != receipt_digest
            or receipt_claim.size_bytes != receipt_size
        ):
            raise CampaignValidationError("live scale-up training receipt bytes differ")
        receipts[str(arm_id)] = {
            "relative_path": receipt_path,
            "sha256": receipt_digest,
            "size_bytes": receipt_size,
            "schema_version": 5,
        }
    if tuple(result) != ARM_IDS or tuple(receipts) != ARM_IDS:
        raise CampaignValidationError("scale-up checkpoint arm order differs")
    return result, receipts


def load_registry(
    path: Path, *, expected_raw_sha256: str, expected_canonical_sha256: str
) -> ValidatedRegistry:
    """Load all frozen CPU authority without trusting pathname resolution."""

    expected_raw_sha256 = _sha256(expected_raw_sha256, label="expected registry hash")
    expected_canonical_sha256 = _sha256(
        expected_canonical_sha256, label="expected canonical registry hash"
    )
    expected_path = REPOSITORY_ROOT / REGISTRY_RELATIVE_PATH
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    if Path(os.path.abspath(os.fspath(candidate))) != Path(
        os.path.abspath(os.fspath(expected_path))
    ):
        raise CampaignValidationError("campaign registry path is not the fixed path")
    claim, payload = _stable_repository_file(REGISTRY_RELATIVE_PATH, label="registry")
    data = _strict_json(payload, label="registry")
    if claim.sha256 != expected_raw_sha256:
        raise CampaignValidationError("registry raw hash differs from caller pin")
    canonical = _canonical_sha256(data)
    if canonical != expected_canonical_sha256:
        raise CampaignValidationError("registry canonical hash differs from caller pin")
    try:
        registry_verifier.validate_registry_document(data)
        registry_verifier.validate_publication_history(data)
    except (OSError, ValueError) as error:
        raise CampaignValidationError(
            f"frozen registry verifier rejected campaign authority: {error}"
        ) from error
    _exact_keys(
        data,
        {
            "schema_version",
            "registry_id",
            "status",
            "claim_scope",
            "publication",
            "authority",
            "grid",
            "configs",
            "stages",
        },
        label="registry",
    )
    if (
        data["schema_version"] != REGISTRY_SCHEMA_VERSION
        or data["status"] != "frozen_before_candidate_generation"
    ):
        raise CampaignValidationError("registry identity or status is invalid")
    if not isinstance(data["registry_id"], str) or not data["registry_id"]:
        raise CampaignValidationError("registry ID is invalid")
    if not isinstance(data["claim_scope"], str) or not data["claim_scope"]:
        raise CampaignValidationError("registry claim scope is invalid")
    publication = _exact_keys(
        data["publication"],
        {
            "framework_revision",
            "config_revision",
            "config_directory",
            "registry_relative_path",
            "separate_config_and_registry_publications_required",
        },
        label="registry publication",
    )
    _revision(publication["framework_revision"], label="framework revision")
    _revision(publication["config_revision"], label="config revision")
    if (
        publication["config_directory"] != CONFIG_DIRECTORY_RELATIVE_PATH
        or publication["registry_relative_path"] != REGISTRY_RELATIVE_PATH
        or publication["separate_config_and_registry_publications_required"] is not True
    ):
        raise CampaignValidationError("registry publication paths or firewall differ")
    authority = _exact_keys(
        data["authority"],
        {
            "superiority_protocol",
            "scale_up_registry",
            "benchmark_schema_version",
            "inference_weights",
            "nfe",
            "metric_branch",
        },
        label="registry authority",
    )
    if (
        authority["benchmark_schema_version"] != 8
        or authority["inference_weights"] != "ema"
        or authority["nfe"] != 128
        or authority["metric_branch"] != "released_comparable"
    ):
        raise CampaignValidationError("registry benchmark authority differs")
    protocol_path, _digest, protocol_schema = _reference(
        authority["superiority_protocol"], label="superiority protocol"
    )
    if protocol_schema != 4:
        raise CampaignValidationError("campaign requires protocol schema 4")
    _reference(
        authority["scale_up_registry"],
        label="scale-up registry",
        expected_schema_version=1,
    )
    _protocol_claim, protocol_payload = _stable_repository_file(
        protocol_path, label="protocol"
    )
    protocol = _strict_json(protocol_payload, label="protocol")
    checkpoints, training_receipts = _checkpoint_authority(protocol)
    grid = _exact_keys(
        data["grid"],
        {
            "arm_ids",
            "softmax_temperatures",
            "raw_loo_top_p_values",
            "cartesian_count",
            "generated_config_count",
            "reused_identity_count",
        },
        label="registry grid",
    )
    if grid != {
        "arm_ids": list(ARM_IDS),
        "softmax_temperatures": list(TEMPERATURES),
        "raw_loo_top_p_values": list(RAW_TOP_P_VALUES),
        "cartesian_count": 36,
        "generated_config_count": 33,
        "reused_identity_count": 3,
    }:
        raise CampaignValidationError("registry grid differs from the frozen 36 points")
    configs_value = data["configs"]
    if not isinstance(configs_value, list) or len(configs_value) != 36:
        raise CampaignValidationError("registry must contain 36 configs")
    configs = tuple(_validate_config_record(value) for value in configs_value)
    expected_ids = tuple(
        _config_id(arm, temperature, top_p)
        for arm in ARM_IDS
        for temperature in TEMPERATURES
        for top_p in RAW_TOP_P_VALUES
    )
    if tuple(config.config_id for config in configs) != expected_ids:
        raise CampaignValidationError(
            "registry configs are not in canonical grid order"
        )
    _validate_stages(data["stages"])
    return ValidatedRegistry(
        path=expected_path,
        raw_sha256=expected_raw_sha256,
        canonical_sha256=expected_canonical_sha256,
        size_bytes=claim.size_bytes,
        data=data,
        configs=configs,
        checkpoint_by_arm=checkpoints,
        training_receipt_by_arm=training_receipts,
    )


def _stage_record(registry: ValidatedRegistry, stage_id: str) -> Mapping[str, Any]:
    try:
        index = STAGE_IDS.index(stage_id)
    except ValueError as error:
        raise CampaignValidationError(f"unknown stage: {stage_id}") from error
    stage = registry.data["stages"][index]
    if not isinstance(stage, Mapping) or stage.get("stage_id") != stage_id:
        raise CampaignValidationError("registry stage order changed")
    return stage


def _config_map(registry: ValidatedRegistry) -> dict[str, CandidateConfig]:
    return {config.config_id: config for config in registry.configs}


def _promoted_ids(
    decision: Mapping[str, Any], *, expected_stage: str
) -> tuple[str, ...]:
    if (
        decision.get("stage_id") != expected_stage
        or decision.get("status") != "completed"
    ):
        raise CampaignValidationError(
            f"stage {expected_stage} did not complete its promotion quota"
        )
    advancement = decision.get("advancement")
    if not isinstance(advancement, Mapping):
        raise CampaignValidationError("prior decision advancement is missing")
    values = advancement.get("promoted_config_ids")
    if not isinstance(values, list) or any(
        not isinstance(value, str) for value in values
    ):
        raise CampaignValidationError("prior promoted config IDs are invalid")
    if len(values) != len(set(values)):
        raise CampaignValidationError("prior promoted config IDs are duplicated")
    return tuple(values)


def derive_attempts(
    registry: ValidatedRegistry,
    stage_id: str,
    decisions: Mapping[str, Mapping[str, Any]],
) -> tuple[AttemptSpec, ...]:
    """Derive conditional slots only from the registry and prior decisions."""

    stage = _stage_record(registry, stage_id)
    configs = _config_map(registry)
    selected: list[CandidateConfig]
    if stage_id == "D":
        selected = [configs[_config_id("E", 1.0, 1.0)]]
    elif stage_id == "A":
        _promoted_ids(decisions["D"], expected_stage="D")
        selected = [
            config for config in registry.configs if config.raw_loo_top_p == 1.0
        ]
    elif stage_id == "B":
        retained_temperatures = _promoted_ids(decisions["A"], expected_stage="A")
        selected = []
        for config_id in retained_temperatures:
            source = configs.get(config_id)
            if source is None or source.raw_loo_top_p != 1.0:
                raise CampaignValidationError("stage A promoted a non-temperature slot")
            for top_p in RAW_TOP_P_VALUES:
                selected.append(
                    configs[_config_id(source.arm_id, source.softmax_temp, top_p)]
                )
    elif stage_id == "C":
        retained = _promoted_ids(decisions["B"], expected_stage="B")
        selected = [configs[config_id] for config_id in retained]
    elif stage_id == "eligible":
        retained = _promoted_ids(decisions["C"], expected_stage="C")
        selected = [configs[config_id] for config_id in retained]
    else:  # pragma: no cover - _stage_record rejects it
        raise AssertionError(stage_id)
    selected.sort(key=lambda config: config.config_id.encode("ascii"))
    seeds = tuple(stage["seeds"])
    attempts = tuple(
        AttemptSpec(
            stage_id=stage_id,
            config_id=config.config_id,
            candidate_id=CANDIDATE_IDS[config.arm_id],
            arm_id=config.arm_id,
            attempt_id=_attempt_id(stage_id, config.config_id, seeds),
            seeds=seeds,
            requested_samples_per_seed=int(stage["requested_samples_per_seed"]),
        )
        for config in selected
    )
    contract = STAGE_CONTRACT[stage_id]
    if (
        len(attempts) != contract["entries"]
        or sum(len(attempt.seeds) for attempt in attempts) != contract["children"]
    ):
        raise CampaignValidationError(
            f"stage {stage_id} conditional population does not match its quota"
        )
    return attempts


def _rank_key(outcome: AttemptOutcome) -> tuple[float, float, bytes, bytes]:
    if not outcome.rankable:  # pragma: no cover - callers filter
        raise CampaignValidationError("unrankable outcome reached rank key")
    assert outcome.released_quality is not None
    assert outcome.released_diversity is not None
    return (
        -outcome.released_quality,
        -outcome.released_diversity,
        outcome.spec.config_id.encode("ascii"),
        outcome.spec.attempt_id.encode("ascii"),
    )


def _select_per_arm(
    outcomes: Sequence[AttemptOutcome], *, retained_per_arm: int
) -> tuple[list[str], bool]:
    promoted: list[str] = []
    complete = True
    for arm_id in ARM_IDS:
        ranked = sorted(
            (
                outcome
                for outcome in outcomes
                if outcome.spec.arm_id == arm_id and outcome.rankable
            ),
            key=_rank_key,
        )
        if len(ranked) < retained_per_arm:
            complete = False
            continue
        promoted.extend(outcome.spec.config_id for outcome in ranked[:retained_per_arm])
    return (promoted if complete else []), complete


def _outcome_record(outcome: AttemptOutcome) -> dict[str, Any]:
    child_outcomes = [
        _validated_child_outcome_reference(value, expected_seed=seed)
        for value, seed in zip(outcome.child_outcomes, outcome.spec.seeds, strict=True)
    ]
    return {
        "attempt_id": outcome.spec.attempt_id,
        "candidate_id": outcome.spec.candidate_id,
        "config_id": outcome.spec.config_id,
        "arm_id": outcome.spec.arm_id,
        "seeds": list(outcome.spec.seeds),
        "requested_samples_per_seed": outcome.spec.requested_samples_per_seed,
        "status": outcome.status,
        "rankable": outcome.rankable,
        "released_quality": outcome.released_quality,
        "released_diversity": outcome.released_diversity,
        "diagnostic_structural_passed": outcome.diagnostic_structural_passed,
        "child_outcomes": child_outcomes,
    }


def _validated_child_outcome_reference(
    value: object, *, expected_seed: int
) -> dict[str, Any]:
    record = _exact_keys(
        value,
        {
            "artifact_kind",
            "pilot_seed",
            "relative_path",
            "sha256",
            "schema_version",
        },
        label="child outcome reference",
    )
    if record["artifact_kind"] not in {"pilot_evaluation", "pilot_failure"}:
        raise CampaignValidationError("child outcome artifact kind is invalid")
    if record["pilot_seed"] != expected_seed:
        raise CampaignValidationError("child outcome seed differs")
    relative_path = _relative_path(record["relative_path"], label="child outcome path")
    digest = _sha256(record["sha256"], label="child outcome hash")
    schema_version = record["schema_version"]
    if schema_version != 2:
        raise CampaignValidationError("child outcome schema version differs")
    return {
        "artifact_kind": record["artifact_kind"],
        "pilot_seed": expected_seed,
        "relative_path": relative_path,
        "sha256": digest,
        "schema_version": schema_version,
    }


def _validate_live_child_outcome_reference(
    value: object, *, expected_seed: int
) -> dict[str, Any]:
    reference = _validated_child_outcome_reference(value, expected_seed=expected_seed)
    claim, payload = _snapshot_exact(
        reference["relative_path"], label="decision child evidence"
    )
    envelope = _strict_json(payload, label="decision child evidence")
    if (
        claim.sha256 != reference["sha256"]
        or envelope.get("schema_version") != 2
        or envelope.get("artifact_kind") != reference["artifact_kind"]
        or envelope.get("pilot_seed") != reference["pilot_seed"]
    ):
        raise CampaignValidationError("decision child evidence binding differs")
    return reference


def _validate_outcome_semantics(outcome: AttemptOutcome) -> None:
    references = [
        _validated_child_outcome_reference(value, expected_seed=seed)
        for value, seed in zip(outcome.child_outcomes, outcome.spec.seeds, strict=True)
    ]
    kinds = [value["artifact_kind"] for value in references]
    failed_child = "pilot_failure" in kinds
    if any(kind not in {"pilot_evaluation", "pilot_failure"} for kind in kinds):
        raise CampaignValidationError("outcome contains an invalid child kind")
    if outcome.spec.stage_id == "D":
        if (
            outcome.released_quality is not None
            or outcome.released_diversity is not None
            or outcome.rankable
            or outcome.status not in {"completed", "failed"}
            or (outcome.status == "completed") is not (not failed_child)
            or outcome.diagnostic_structural_passed
            is not (outcome.status == "completed")
        ):
            raise CampaignValidationError("diagnostic outcome semantics differ")
        return
    if outcome.diagnostic_structural_passed is not None:
        raise CampaignValidationError("ranked outcome carries a diagnostic flag")
    if (outcome.released_quality is None) is not (outcome.released_diversity is None):
        raise CampaignValidationError("outcome metrics must be jointly defined")
    expected_status = (
        "failed" if failed_child else ("completed" if outcome.rankable else "undefined")
    )
    if outcome.status != expected_status:
        raise CampaignValidationError("outcome status differs from child evidence")


def derive_decision(
    *,
    registry: ValidatedRegistry,
    stage_id: str,
    attempts: Sequence[AttemptSpec],
    outcomes: Sequence[AttemptOutcome],
    prior_decisions: Mapping[str, Mapping[str, Any]],
    predecessor: Mapping[str, Any] | None,
    source_revision: str,
    decided_at_utc: str,
) -> dict[str, Any]:
    """Deterministically decide one complete stage without cross-stage pooling."""

    _revision(source_revision, label="decision source revision")
    decided_at_utc = _utc_timestamp(decided_at_utc, label="decision completion time")
    expected_attempts = derive_attempts(registry, stage_id, prior_decisions)
    if tuple(attempts) != expected_attempts:
        raise CampaignValidationError("stage attempts differ from deterministic slots")
    if len(attempts) != STAGE_CONTRACT[stage_id]["entries"]:
        raise CampaignValidationError("stage attempt count differs from contract")
    by_attempt = {outcome.spec.attempt_id: outcome for outcome in outcomes}
    if len(by_attempt) != len(outcomes) or set(by_attempt) != {
        attempt.attempt_id for attempt in attempts
    }:
        raise CampaignValidationError("stage outcomes do not exactly cover attempts")
    ordered_outcomes = tuple(by_attempt[attempt.attempt_id] for attempt in attempts)
    if any(
        outcome.spec != attempt
        or outcome.status not in {"completed", "failed", "undefined"}
        or len(outcome.child_outcomes) != len(attempt.seeds)
        for attempt, outcome in zip(attempts, ordered_outcomes, strict=True)
    ):
        raise CampaignValidationError("stage contains nonterminal or misbound outcomes")
    for outcome in ordered_outcomes:
        _validate_outcome_semantics(outcome)

    promoted: list[str] = []
    winner: str | None = None
    quota_met = False
    if stage_id == "D":
        diagnostic = ordered_outcomes[0]
        quota_met = (
            diagnostic.status == "completed"
            and diagnostic.diagnostic_structural_passed is True
        )
        if (
            diagnostic.released_quality is not None
            or diagnostic.released_diversity is not None
        ):
            raise CampaignValidationError(
                "diagnostic outcome must not read chemistry metrics"
            )
    elif stage_id in {"A", "B"}:
        promoted, quota_met = _select_per_arm(ordered_outcomes, retained_per_arm=2)
    elif stage_id == "C":
        promoted, quota_met = _select_per_arm(ordered_outcomes, retained_per_arm=1)
    elif stage_id == "eligible":
        ranked = sorted(
            (outcome for outcome in ordered_outcomes if outcome.rankable), key=_rank_key
        )
        quota_met = bool(ranked)
        if ranked:
            winner = ranked[0].spec.config_id
            promoted = [winner]
    else:
        raise CampaignValidationError("decision stage is unknown")

    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "stage_id": stage_id,
        "status": "completed" if quota_met else "campaign_incomplete",
        "registry": registry.reference,
        "entries": [_outcome_record(outcome) for outcome in ordered_outcomes],
        "advancement": {
            "predecessor_stage_decision": (
                None if predecessor is None else dict(predecessor)
            ),
            "ranking_order": (
                []
                if stage_id == "D"
                else [
                    "released_quality_descending",
                    "released_diversity_descending",
                    "config_id_ascii_ascending",
                    "attempt_id_ascii_ascending",
                ]
            ),
            "promoted_config_ids": promoted,
            "global_winner_config_id": winner,
            "required_promotions_per_arm": (
                2 if stage_id in {"A", "B"} else (1 if stage_id == "C" else None)
            ),
            "no_retry_or_substitution": True,
            "on_failed_or_undefined_child": "retain_unrankable",
            "on_insufficient_rankable_quota": ("campaign_incomplete_without_promotion"),
            "accounting": {
                "scheduled_entry_count": len(attempts),
                "terminal_entry_count": len(ordered_outcomes),
                "scheduled_child_count": sum(
                    len(attempt.seeds) for attempt in attempts
                ),
                "terminal_child_count": sum(
                    len(outcome.child_outcomes) for outcome in ordered_outcomes
                ),
                "requested_molecule_count": sum(
                    len(attempt.seeds) * attempt.requested_samples_per_seed
                    for attempt in attempts
                ),
            },
        },
        "completed_at_utc": decided_at_utc,
    }


DECISION_FIELDS = {
    "schema_version",
    "stage_id",
    "status",
    "registry",
    "entries",
    "advancement",
    "completed_at_utc",
}


def _decision_paths(stage_id: str) -> tuple[str, str]:
    if stage_id not in STAGE_IDS:
        raise CampaignValidationError("decision stage is unknown")
    stem = f"{CAMPAIGN_RELATIVE_ROOT}/stages/{stage_id.lower()}"
    return f"{stem}/stage_entries.json", f"{stem}/stage_decision.json"


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _decision_reference(*, relative_path: str, payload: bytes) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema_version": DECISION_SCHEMA_VERSION,
    }


def publish_decision(
    *,
    registry: ValidatedRegistry,
    decision: Mapping[str, Any],
    predecessor_completion: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Publish one immutable decision and its completion marker last."""

    stage_id = decision.get("stage_id")
    if stage_id not in STAGE_IDS or set(decision) != DECISION_FIELDS:
        raise CampaignValidationError("decision payload has an invalid schema")
    entries_path, decision_path = _decision_paths(str(stage_id))
    benchmark_launcher._ensure_repository_directory(  # noqa: SLF001
        (REPOSITORY_ROOT / decision_path).parent,
        label="campaign stage decision parent",
    )
    advancement = decision.get("advancement")
    if (
        not isinstance(advancement, Mapping)
        or advancement.get("predecessor_stage_decision") != predecessor_completion
    ):
        raise CampaignValidationError("decision predecessor binding differs")
    for entry in decision["entries"]:
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("child_outcomes"), list
        ):
            raise CampaignValidationError("decision child outcomes are invalid")
        seeds = entry.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != len(entry["child_outcomes"]):
            raise CampaignValidationError("decision child seed coverage is invalid")
        for child, seed in zip(entry["child_outcomes"], seeds, strict=True):
            _validate_live_child_outcome_reference(child, expected_seed=seed)
    entries_payload = _json_bytes(
        {
            "schema_version": DECISION_SCHEMA_VERSION,
            "stage_id": stage_id,
            "registry": registry.reference,
            "entries": decision["entries"],
        }
    )
    decision_payload = _json_bytes(decision)
    artifact_io.publish_bundle_exclusive(
        REPOSITORY_ROOT,
        [artifact_io.PublishItem(entries_path, entries_payload)],
        completion=artifact_io.PublishItem(decision_path, decision_payload),
    )
    return _decision_reference(
        relative_path=decision_path,
        payload=decision_payload,
    )


def _attempt_from_record(value: object, *, stage_id: str) -> AttemptSpec:
    record = _exact_keys(
        value,
        {
            "attempt_id",
            "candidate_id",
            "config_id",
            "arm_id",
            "seeds",
            "requested_samples_per_seed",
        },
        label="authorized attempt",
    )
    seeds = record["seeds"]
    if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds):
        raise CampaignValidationError("authorized attempt seeds are invalid")
    spec = AttemptSpec(
        stage_id=stage_id,
        config_id=str(record["config_id"]),
        candidate_id=str(record["candidate_id"]),
        arm_id=str(record["arm_id"]),
        attempt_id=str(record["attempt_id"]),
        seeds=tuple(seeds),
        requested_samples_per_seed=int(record["requested_samples_per_seed"]),
    )
    if spec.attempt_id != _attempt_id(stage_id, spec.config_id, spec.seeds):
        raise CampaignValidationError("authorized attempt ID is noncanonical")
    return spec


def _outcome_from_record(value: object, spec: AttemptSpec) -> AttemptOutcome:
    record = _exact_keys(
        value,
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
        label="attempt outcome",
    )
    for field, expected in {
        "attempt_id": spec.attempt_id,
        "candidate_id": spec.candidate_id,
        "config_id": spec.config_id,
        "arm_id": spec.arm_id,
        "seeds": list(spec.seeds),
        "requested_samples_per_seed": spec.requested_samples_per_seed,
    }.items():
        if record[field] != expected:
            raise CampaignValidationError(f"outcome {field} differs from attempt")
    child_outcomes = record["child_outcomes"]
    if not isinstance(child_outcomes, list) or len(child_outcomes) != len(spec.seeds):
        raise CampaignValidationError("child outcomes are invalid")
    validated_children = tuple(
        _validated_child_outcome_reference(child, expected_seed=seed)
        for child, seed in zip(child_outcomes, spec.seeds, strict=True)
    )
    quality = record["released_quality"]
    diversity = record["released_diversity"]
    for label, metric in (("quality", quality), ("diversity", diversity)):
        if metric is not None and (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise CampaignValidationError(f"outcome {label} is non-finite")
    outcome = AttemptOutcome(
        spec=spec,
        status=str(record["status"]),
        child_outcomes=validated_children,
        released_quality=None if quality is None else float(quality),
        released_diversity=None if diversity is None else float(diversity),
        diagnostic_structural_passed=record["diagnostic_structural_passed"],
    )
    if record["rankable"] is not outcome.rankable:
        raise CampaignValidationError("recorded rankability differs from metrics")
    return outcome


def _attempt_from_outcome_record(value: object, *, stage_id: str) -> AttemptSpec:
    if not isinstance(value, Mapping):
        raise CampaignValidationError("attempt outcome must be an object")
    required = {
        "attempt_id",
        "candidate_id",
        "config_id",
        "arm_id",
        "seeds",
        "requested_samples_per_seed",
    }
    if not required.issubset(value):
        raise CampaignValidationError("attempt outcome lacks its attempt identity")
    return _attempt_from_record(
        {key: value[key] for key in required}, stage_id=stage_id
    )


def replay_decisions(
    registry: ValidatedRegistry,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Replay the contiguous immutable decision chain; gaps fail closed."""

    decisions: dict[str, Mapping[str, Any]] = {}
    completions: dict[str, Mapping[str, Any]] = {}
    previous_completion: Mapping[str, Any] | None = None
    gap_seen = False
    for stage_id in STAGE_IDS:
        entries_path, decision_path = _decision_paths(stage_id)
        absolute_entries = REPOSITORY_ROOT / entries_path
        absolute_decision = REPOSITORY_ROOT / decision_path
        entries_exist = os.path.lexists(absolute_entries)
        decision_exists = os.path.lexists(absolute_decision)
        if entries_exist != decision_exists:
            raise CampaignValidationError(
                f"stage {stage_id} decision bundle is partial"
            )
        if not decision_exists:
            gap_seen = True
            continue
        if gap_seen:
            raise CampaignValidationError("decision bundles contain a stage gap")
        entries_claim, entries_payload = _stable_repository_file(
            entries_path, label=f"stage {stage_id} entries"
        )
        decision_claim, decision_payload = _stable_repository_file(
            decision_path, label=f"stage {stage_id} decision"
        )
        entries = _strict_json(entries_payload, label=f"stage {stage_id} entries")
        decision = _strict_json(decision_payload, label=f"stage {stage_id} decision")
        _exact_keys(
            entries,
            {"schema_version", "stage_id", "registry", "entries"},
            label=f"stage {stage_id} entries",
        )
        _exact_keys(decision, DECISION_FIELDS, label=f"stage {stage_id} decision")
        expected_decision_reference = _decision_reference(
            relative_path=decision_path,
            payload=decision_payload,
        )
        advancement = decision.get("advancement")
        if (
            entries_claim.sha256 != hashlib.sha256(entries_payload).hexdigest()
            or decision_claim.sha256 != expected_decision_reference["sha256"]
            or entries.get("schema_version") != DECISION_SCHEMA_VERSION
            or entries.get("stage_id") != stage_id
            or entries.get("registry") != registry.reference
            or entries.get("entries") != decision.get("entries")
            or decision.get("stage_id") != stage_id
            or decision.get("registry") != registry.reference
            or not isinstance(advancement, Mapping)
            or advancement.get("predecessor_stage_decision") != previous_completion
        ):
            raise CampaignValidationError(
                f"stage {stage_id} decision chain binding differs"
            )
        outcome_records = decision.get("entries")
        if not isinstance(outcome_records, list):
            raise CampaignValidationError("decision entries are invalid")
        for record in outcome_records:
            if not isinstance(record, Mapping):
                raise CampaignValidationError("decision entry is invalid")
            seeds = record.get("seeds")
            children = record.get("child_outcomes")
            if not isinstance(seeds, list) or not isinstance(children, list):
                raise CampaignValidationError("decision child coverage is invalid")
            for child, seed in zip(children, seeds, strict=True):
                _validate_live_child_outcome_reference(child, expected_seed=seed)
        attempts = tuple(
            _attempt_from_outcome_record(value, stage_id=stage_id)
            for value in outcome_records
        )
        outcomes = tuple(
            _outcome_from_record(value, spec)
            for value, spec in zip(outcome_records, attempts, strict=True)
        )
        recomputed = derive_decision(
            registry=registry,
            stage_id=stage_id,
            attempts=attempts,
            outcomes=outcomes,
            prior_decisions=decisions,
            predecessor=previous_completion,
            source_revision="0" * 40,
            decided_at_utc=str(decision["completed_at_utc"]),
        )
        if recomputed != decision:
            raise CampaignValidationError(
                f"stage {stage_id} decision differs from deterministic replay"
            )
        decisions[stage_id] = dict(decision)
        completions[stage_id] = expected_decision_reference
        previous_completion = expected_decision_reference
    return decisions, completions


def _git_bytes(*arguments: str, check: bool = True) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=check,
        capture_output=True,
    )
    return completed.stdout


def _require_exact_registry_revision(
    registry: ValidatedRegistry,
) -> dict[str, str]:
    """Require clean pushed G whose only change from C is the frozen registry."""

    source = benchmark_launcher._require_clean_pushed_source()  # noqa: SLF001
    revision = _revision(source["head"], label="campaign source revision G")
    config_revision = _revision(
        registry.data["publication"]["config_revision"], label="config revision C"
    )
    parents = _git_bytes("rev-list", "--parents", "-n", "1", revision).split()
    if len(parents) != 2 or parents[0].decode("ascii") != revision:
        raise CampaignValidationError(
            "registry revision G must have exactly one parent"
        )
    if parents[1].decode("ascii") != config_revision:
        raise CampaignValidationError("registry revision G must have exact parent C")
    changed = _git_bytes("diff", "--name-only", "-z", config_revision, revision, "--")
    if changed and not changed.endswith(b"\0"):
        raise CampaignValidationError("G changed-path result is truncated")
    paths = (
        tuple(os.fsdecode(value) for value in changed[:-1].split(b"\0"))
        if changed
        else ()
    )
    if paths != (REGISTRY_RELATIVE_PATH,):
        raise CampaignValidationError("G must change only the candidate registry JSON")
    try:
        blob = _git_bytes("cat-file", "blob", f"{revision}:{REGISTRY_RELATIVE_PATH}")
    except subprocess.CalledProcessError as error:
        raise CampaignValidationError("registry is absent from revision G") from error
    if hashlib.sha256(blob).hexdigest() != registry.raw_sha256:
        raise CampaignValidationError("live registry bytes differ from the G Git blob")
    return {"head": revision, "upstream": revision}


def _next_stage_id(decisions: Mapping[str, Mapping[str, Any]]) -> str | None:
    for stage_id in STAGE_IDS:
        decision = decisions.get(stage_id)
        if decision is None:
            return stage_id
        if decision.get("status") != "completed":
            return None
    return None


def _authorized_stage_ids(
    decisions: Mapping[str, Mapping[str, Any]], through_stage: str
) -> tuple[str, ...]:
    """Return the exact unexecuted contiguous prefix authorized by the caller.

    Replay is the sole source of the resume frontier.  A caller may extend that
    frontier, but cannot name an already-terminal stage, choose ``final``, or
    provide a state with a gap.  An incomplete decision is terminal and yields
    no authorization, preserving the no-retry/no-substitution policy.
    """

    if through_stage not in STAGE_IDS:
        raise CampaignValidationError(
            f"through-stage must be one of {STAGE_IDS!r}; final is not authorized"
        )
    unknown = set(decisions).difference(STAGE_IDS)
    present = tuple(stage_id for stage_id in STAGE_IDS if stage_id in decisions)
    if unknown or present != STAGE_IDS[: len(decisions)]:
        raise CampaignValidationError(
            "replayed decisions are not an exact contiguous pre-final prefix"
        )
    statuses = [decisions[stage_id].get("status") for stage_id in present]
    if any(status not in {"completed", "campaign_incomplete"} for status in statuses):
        raise CampaignValidationError("replayed decision has an invalid status")
    incomplete_indexes = [
        index
        for index, status in enumerate(statuses)
        if status == "campaign_incomplete"
    ]
    if incomplete_indexes:
        if incomplete_indexes != [len(present) - 1]:
            raise CampaignValidationError(
                "only the last replayed decision may be terminal-incomplete"
            )
        return ()
    next_index = len(present)
    through_index = STAGE_IDS.index(through_stage)
    if next_index == len(STAGE_IDS) and through_index == len(STAGE_IDS) - 1:
        return ()
    if through_index < next_index:
        raise CampaignValidationError(
            f"through-stage {through_stage} is already terminal; next stage is "
            f"{STAGE_IDS[next_index] if next_index < len(STAGE_IDS) else 'none'}"
        )
    return STAGE_IDS[next_index : through_index + 1]


def _attempt_root(spec: AttemptSpec) -> Path:
    return REPOSITORY_ROOT / CAMPAIGN_RELATIVE_ROOT / "attempts" / spec.attempt_id


def _attempt_log_root(spec: AttemptSpec) -> Path:
    return REPOSITORY_ROOT / CAMPAIGN_LOG_RELATIVE_ROOT / spec.attempt_id


def _evidence_relative_path(spec: AttemptSpec, seed: int) -> str:
    return f"{CAMPAIGN_RELATIVE_ROOT}/evidence/{spec.attempt_id}/seed_{seed}.json"


def _fresh_stage_paths(attempts: Sequence[AttemptSpec]) -> None:
    """Reject any prior claim of an authorized slot; retries are forbidden."""

    seen: set[Path] = set()
    for spec in attempts:
        paths = [_attempt_root(spec), _attempt_log_root(spec)]
        paths.extend(
            REPOSITORY_ROOT / _evidence_relative_path(spec, seed) for seed in spec.seeds
        )
        for path in paths:
            if path in seen or os.path.lexists(path):
                raise FileExistsError(
                    f"campaign slot is not fresh; retry/substitution forbidden: {path}"
                )
            seen.add(path)


def _reserve_attempt_roots(
    attempts: Sequence[AttemptSpec],
) -> tuple[tuple[artifact_io.OwnedDirectory, artifact_io.OwnedDirectory], ...]:
    """Exclusively reserve every attempt output/log root before GPU inventory."""

    benchmark_launcher._ensure_repository_directory(  # noqa: SLF001
        REPOSITORY_ROOT / CAMPAIGN_RELATIVE_ROOT / "attempts",
        label="campaign attempt parent",
    )
    benchmark_launcher._ensure_repository_directory(  # noqa: SLF001
        REPOSITORY_ROOT / CAMPAIGN_LOG_RELATIVE_ROOT,
        label="campaign log parent",
    )
    owners: list[tuple[artifact_io.OwnedDirectory, artifact_io.OwnedDirectory]] = []
    try:
        for spec in attempts:
            output_owner = artifact_io.create_directory_exclusive(
                REPOSITORY_ROOT,
                benchmark_launcher._repository_relative(  # noqa: SLF001
                    _attempt_root(spec), label="campaign attempt root"
                ),
            )
            try:
                log_owner = artifact_io.create_directory_exclusive(
                    REPOSITORY_ROOT,
                    benchmark_launcher._repository_relative(  # noqa: SLF001
                        _attempt_log_root(spec), label="campaign attempt log root"
                    ),
                )
            except BaseException as error:
                try:
                    artifact_io.remove_empty_directory_exact(
                        REPOSITORY_ROOT, output_owner
                    )
                except BaseException as rollback_error:
                    raise artifact_io.RollbackError(error, [rollback_error]) from error
                raise
            owners.append((output_owner, log_owner))
    except BaseException as error:
        rollback_errors: list[BaseException] = []
        for output_owner, log_owner in reversed(owners):
            for owner in (log_owner, output_owner):
                try:
                    artifact_io.remove_empty_directory_exact(REPOSITORY_ROOT, owner)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
        if rollback_errors:
            raise artifact_io.RollbackError(error, rollback_errors) from error
        raise
    return tuple(owners)


class _IdentityFactory:
    """Cache shared CPU fingerprints while retaining per-config/sample identities."""

    def __init__(self, registry: ValidatedRegistry, source_revision: str) -> None:
        self.registry = registry
        self.source_revision = _revision(
            source_revision, label="identity source revision"
        )
        self.metric_inputs = (
            benchmark_launcher.benchmark_runner.metric_input_provenance()
        )
        self.implementation_inputs = (
            benchmark_launcher.benchmark_runner.implementation_input_provenance()
        )
        self.runner_sha256 = benchmark_launcher._sha256_file(  # noqa: SLF001
            Path(benchmark_launcher.benchmark_runner.__file__).resolve()
        )
        self.checkpoints: dict[str, Mapping[str, Any]] = {}
        self.identities: dict[
            tuple[str, int], benchmark_launcher.ExpectedRunIdentity
        ] = {}
        self.configs = _config_map(registry)

    def expected(self, spec: AttemptSpec) -> benchmark_launcher.ExpectedRunIdentity:
        key = (spec.config_id, spec.requested_samples_per_seed)
        cached = self.identities.get(key)
        if cached is not None:
            return cached
        config = self.configs[spec.config_id]
        checkpoint_record = self.registry.checkpoint_by_arm[spec.arm_id]
        checkpoint_path = REPOSITORY_ROOT / str(checkpoint_record["relative_path"])
        checkpoint_info = self.checkpoints.get(spec.arm_id)
        if checkpoint_info is None:
            checkpoint_info = benchmark_launcher.benchmark_runner.checkpoint_metadata(
                checkpoint_path
            )
            if (
                checkpoint_info.get("sha256") != checkpoint_record["sha256"]
                or checkpoint_info.get("size_bytes") != checkpoint_record["size_bytes"]
                or checkpoint_info.get("global_step") != 1_000
            ):
                raise CampaignValidationError(
                    f"{spec.arm_id} checkpoint inspection differs from registry"
                )
            self.checkpoints[spec.arm_id] = checkpoint_info
        expected = benchmark_launcher._build_expected_run_identity(  # noqa: SLF001
            checkpoint_path,
            REPOSITORY_ROOT / config.relative_path,
            spec.requested_samples_per_seed,
            source_revision=self.source_revision,
            checkpoint_info=checkpoint_info,
            metric_inputs=self.metric_inputs,
            implementation_inputs=self.implementation_inputs,
            benchmark_runner_sha256=self.runner_sha256,
        )
        if (
            expected.source_config_sha256 != config.sha256
            or expected.sampling_config_sha256 != config.normalized_sampling_sha256
            or expected.checkpoint_sha256 != checkpoint_record["sha256"]
        ):
            raise CampaignValidationError(
                "derived child identity differs from registry"
            )
        self.identities[key] = expected
        return expected


def _launchable_gpus(
    states: Sequence[benchmark_launcher.GPUState],
    *,
    running_uuids: set[str],
    free_slots: int,
) -> list[benchmark_launcher.GPUState]:
    """Select only policy-eligible distinct UUIDs, never physical fixed IDs."""

    if type(free_slots) is not int or not 0 <= free_slots <= MAX_CONCURRENT_CHILDREN:
        raise CampaignValidationError("free GPU slot count is invalid")
    if len({state.uuid for state in states}) != len(states):
        raise CampaignValidationError("GPU inventory contains duplicate UUIDs")
    if len({state.index for state in states}) != len(states):
        raise CampaignValidationError("GPU inventory contains duplicate indices")
    eligible = [
        state
        for state in states
        if state.uuid not in running_uuids
        and benchmark_launcher._eligible(  # noqa: SLF001
            state,
            max_utilization_percent=MAX_UTILIZATION_PERCENT,
            min_free_memory_mib=MIN_FREE_MEMORY_MIB,
        )
    ]
    return sorted(
        eligible,
        key=lambda state: (
            -(state.memory_total_mib - state.memory_used_mib),
            state.utilization_percent,
            state.uuid,
        ),
    )[:free_slots]


def _launch_child(
    *,
    lease: benchmark_launcher.GenerationLease,
    spec: AttemptSpec,
    seed: int,
    expected: benchmark_launcher.ExpectedRunIdentity,
    gpu: benchmark_launcher.GPUState,
    inventory: Sequence[benchmark_launcher.GPUState],
    running_uuids: set[str],
    source_revision: Mapping[str, str],
    policy: Mapping[str, Any],
    inventory_completed_at_utc: str,
) -> CampaignRunningJob:
    output_dir = Path(os.path.abspath(_attempt_root(spec) / f"seed_{seed}"))
    run_label = benchmark_launcher.benchmark_runner.benchmark_run_label(
        expected.checkpoint_global_step, expected.checkpoint_sha256, seed
    )
    log_path = Path(os.path.abspath(_attempt_log_root(spec) / f"{run_label}.log"))
    output_owner, log_owner = benchmark_launcher._reserve_child_paths(  # noqa: SLF001
        output_dir=output_dir, log_path=log_path
    )
    command = benchmark_launcher._command(  # noqa: SLF001
        checkpoint=expected.checkpoint_path,
        expected_checkpoint_sha256=expected.checkpoint_sha256,
        expected_source_revision=source_revision["head"],
        config=expected.config_path,
        expected_config_sha256=expected.source_config_sha256,
        num_samples=spec.requested_samples_per_seed,
        seed=seed,
        output_dir=output_dir,
        expected_output_directory_device=output_owner.device,
        expected_output_directory_inode=output_owner.inode,
    )
    if len(command) != 24:
        raise AssertionError("campaign child command is not exact-24")
    authority = benchmark_launcher._launch_authority(  # noqa: SLF001
        lease=lease, output_directory=output_owner, command=command
    )
    final_probe_at = datetime.now(timezone.utc).isoformat()
    selection = {
        "event": "launch",
        "gpu_selection_schema_version": 3,
        "timestamp_utc": final_probe_at,
        "inventory_snapshot_completed_at_utc": inventory_completed_at_utc,
        "final_uuid_probe_completed_at_utc": final_probe_at,
        "source_revision": dict(source_revision),
        "gpu_inventory_at_selection": [state.as_dict() for state in inventory],
        "running_gpu_uuids_at_selection": sorted(running_uuids),
        "physical_gpu_at_final_uuid_probe": gpu.as_dict(),
        "policy": dict(policy),
        "launch_authority": authority,
        "command": command,
    }
    log_context, log_handle = benchmark_launcher._open_job_log(  # noqa: SLF001
        log_owner, selection
    )
    try:
        environment = benchmark_launcher._child_environment(  # noqa: SLF001
            seed=seed,
            gpu=gpu,
            selection=selection,
            run_label=run_label,
            generation_lease=lease,
            launch_authority=authority,
        )
        benchmark_launcher._revalidate_generation_lease(lease)  # noqa: SLF001
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except BaseException:
        log_context.__exit__(*sys.exc_info())
        raise
    job = benchmark_launcher.RunningJob(
        seed=seed,
        gpu=gpu,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
        command=tuple(command),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        log_owner=log_owner,
        log_context=log_context,
        output_directory_owner=output_owner,
    )
    return CampaignRunningJob(spec=spec, expected=expected, job=job)


def _finish_child(
    *,
    running: CampaignRunningJob,
    return_code: int,
    source_revision: Mapping[str, str],
) -> TerminalChild:
    benchmark_launcher._close_job_log(running.job)  # noqa: SLF001
    pilot_mode = (
        "registered_selection" if running.spec.stage_id == "eligible" else "engineering"
    )
    failed, details = benchmark_launcher._finalize_finished_job(  # noqa: SLF001
        job=running.job,
        return_code=return_code,
        output_root=_attempt_root(running.spec),
        expected=running.expected,
        pilot_mode=pilot_mode,
        attempt_id=running.spec.attempt_id,
        candidate_id=running.spec.candidate_id,
        source_revision=source_revision,
    )
    if any("failure receipt could not be published" in detail for detail in details):
        raise CampaignValidationError("terminal child lacks durable failure evidence")
    return TerminalChild(
        spec=running.spec,
        seed=running.job.seed,
        expected=running.expected,
        failed=failed,
        command=running.job.command,
    )


def _execute_stage_children(
    *,
    lease: benchmark_launcher.GenerationLease,
    attempts: Sequence[AttemptSpec],
    identities: _IdentityFactory,
    source_revision: Mapping[str, str],
) -> tuple[TerminalChild, ...]:
    """Run every authorized child once; failures never cancel sibling slots."""

    pending = [
        (spec, seed, identities.expected(spec))
        for spec in attempts
        for seed in spec.seeds
    ]
    if len(pending) != STAGE_CONTRACT[attempts[0].stage_id]["children"]:
        raise CampaignValidationError("stage child schedule differs from registry")
    concurrency = 1 if attempts[0].stage_id == "D" else MAX_CONCURRENT_CHILDREN
    policy = benchmark_launcher._selection_policy(  # noqa: SLF001
        requested_gpu_count=concurrency,
        max_utilization_percent=MAX_UTILIZATION_PERCENT,
        min_free_memory_mib=MIN_FREE_MEMORY_MIB,
    )
    running_by_uuid: dict[str, CampaignRunningJob] = {}
    terminal: list[TerminalChild] = []
    fatal_errors: list[BaseException] = []
    unchanged_polls = 0
    previous_signature: str | None = None
    while pending or running_by_uuid:
        for gpu_uuid, running in list(running_by_uuid.items()):
            return_code = running.job.process.poll()
            if return_code is None:
                continue
            del running_by_uuid[gpu_uuid]
            try:
                terminal.append(
                    _finish_child(
                        running=running,
                        return_code=return_code,
                        source_revision=source_revision,
                    )
                )
            except BaseException as error:
                fatal_errors.append(error)
            try:
                benchmark_launcher._revalidate_generation_lease(lease)  # noqa: SLF001
            except BaseException as error:
                fatal_errors.append(error)
            print(
                json.dumps(
                    {
                        "event": "campaign_child_terminal",
                        "stage_id": running.spec.stage_id,
                        "attempt_id": running.spec.attempt_id,
                        "seed": running.job.seed,
                        "gpu_uuid": gpu_uuid,
                        "return_code": return_code,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if fatal_errors:
            if not running_by_uuid:
                break
            time.sleep(1)
            continue

        if pending and len(running_by_uuid) < concurrency:
            try:
                benchmark_launcher._revalidate_generation_lease(lease)  # noqa: SLF001
                states = benchmark_launcher._snapshot()  # noqa: SLF001
                inventory_completed_at = datetime.now(timezone.utc).isoformat()
                signature = benchmark_launcher._snapshot_signature(  # noqa: SLF001
                    list(states)
                )
                unchanged_polls = (
                    unchanged_polls + 1 if signature == previous_signature else 0
                )
                previous_signature = signature
                candidates = _launchable_gpus(
                    states,
                    running_uuids=set(running_by_uuid),
                    free_slots=concurrency - len(running_by_uuid),
                )
                if unchanged_polls == 0 or unchanged_polls % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "campaign_gpu_poll",
                                "stage_id": attempts[0].stage_id,
                                "eligible_gpu_uuids": [
                                    state.uuid for state in candidates
                                ],
                                "active_processes_recorded": {
                                    state.uuid: list(state.compute_processes)
                                    for state in states
                                },
                                "pending_child_count": len(pending),
                                "running_gpu_uuids": sorted(running_by_uuid),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if candidates:
                    candidate = candidates[0]
                    benchmark_launcher._revalidate_generation_lease(  # noqa: SLF001
                        lease
                    )
                    rechecked, reasons = (
                        benchmark_launcher._recheck_gpu_for_launch(  # noqa: SLF001
                            candidate,
                            max_utilization_percent=MAX_UTILIZATION_PERCENT,
                            min_free_memory_mib=MIN_FREE_MEMORY_MIB,
                        )
                    )
                    if rechecked is None:
                        print(
                            json.dumps(
                                {
                                    "event": "campaign_gpu_final_probe_rejected",
                                    "gpu_uuid": candidate.uuid,
                                    "reasons": reasons,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    else:
                        spec, seed, expected = pending[0]
                        benchmark_launcher._revalidate_generation_lease(  # noqa: SLF001
                            lease
                        )
                        launched = _launch_child(
                            lease=lease,
                            spec=spec,
                            seed=seed,
                            expected=expected,
                            gpu=rechecked,
                            inventory=states,
                            running_uuids=set(running_by_uuid),
                            source_revision=source_revision,
                            policy=policy,
                            inventory_completed_at_utc=inventory_completed_at,
                        )
                        pending.pop(0)
                        if rechecked.uuid in running_by_uuid:
                            raise CampaignValidationError(
                                "one UUID was assigned to concurrent children"
                            )
                        running_by_uuid[rechecked.uuid] = launched
                        print(
                            json.dumps(
                                {
                                    "event": "campaign_child_launched",
                                    "stage_id": spec.stage_id,
                                    "attempt_id": spec.attempt_id,
                                    "seed": seed,
                                    "pid": launched.job.process.pid,
                                    "gpu_uuid": rechecked.uuid,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        continue
            except BaseException as error:
                fatal_errors.append(error)

        if pending or running_by_uuid:
            time.sleep(30 if not running_by_uuid else 1)

    if fatal_errors:
        detail = "; ".join(str(error) for error in fatal_errors)
        raise CampaignValidationError(
            "campaign stopped after all handed-off siblings became terminal: " + detail
        ) from fatal_errors[0]
    if pending or len(terminal) != sum(len(spec.seeds) for spec in attempts):
        raise CampaignValidationError(
            "stage ended without all authorized children terminal"
        )
    return tuple(terminal)


def _snapshot_exact(
    relative_path: str, *, label: str
) -> tuple[artifact_io.FileClaim, bytes]:
    try:
        claim, payload = artifact_io.snapshot_file(
            REPOSITORY_ROOT, relative_path, capture_bytes=True
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise CampaignValidationError(f"cannot snapshot {label}") from error
    if payload is None:  # pragma: no cover - capture contract
        raise AssertionError("snapshot bytes were not retained")
    return claim, payload


def _raw_model_texts(payload: bytes, *, expected_rows: int) -> list[str]:
    try:
        text = payload.decode("utf-8")
        reader = csv.DictReader(text.splitlines(), strict=True)
        if reader.fieldnames != list(
            benchmark_launcher.benchmark_runner.RAW_SAMPLE_FIELDS
        ):
            raise CampaignValidationError("diagnostic raw CSV fields differ")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise CampaignValidationError("diagnostic raw CSV is invalid") from error
    if len(rows) != expected_rows:
        raise CampaignValidationError("diagnostic raw CSV row count differs")
    result: list[str] = []
    for index, row in enumerate(rows):
        if row.get("sample_index") != str(index):
            raise CampaignValidationError("diagnostic raw CSV order differs")
        value = row.get("raw_model_text")
        if not isinstance(value, str):
            raise CampaignValidationError("diagnostic raw model text is invalid")
        result.append(value)
    return result


def _diagnostic_structural_validation(
    terminal: TerminalChild,
    *,
    summary_payload: bytes,
    raw_payload: bytes,
) -> None:
    """Validate schema/audit/decode/instrumentation without reading chemistry."""

    from scripts.udlm import rescore_denovo_run

    if len(summary_payload) > rescore_denovo_run.MAXIMUM_SUMMARY_SIZE_BYTES:
        raise CampaignValidationError("diagnostic summary exceeds 2 MiB")
    summary = _strict_json(summary_payload, label="diagnostic summary")
    if summary.get("schema_version") != 8:
        raise CampaignValidationError("diagnostic summary is not schema 8")
    raw_texts = _raw_model_texts(
        raw_payload, expected_rows=terminal.spec.requested_samples_per_seed
    )
    implementation = terminal.expected.implementation_inputs
    sampler = implementation.get("sampler_source")
    ema = implementation.get("ema_source")
    if not isinstance(sampler, Mapping) or not isinstance(ema, Mapping):
        raise CampaignValidationError("diagnostic implementation provenance is absent")
    try:
        identity = rescore_denovo_run._validate_summary_identity(  # noqa: SLF001
            summary,
            expected_seed=terminal.seed,
            expected_sample_count=terminal.spec.requested_samples_per_seed,
            raw_sha256=hashlib.sha256(raw_payload).hexdigest(),
            expected_checkpoint_sha256=terminal.expected.checkpoint_sha256,
            expected_config_sha256=terminal.expected.source_config_sha256,
            expected_source_revision=terminal.expected.source_revision,
            expected_runner_sha256=terminal.expected.benchmark_runner_sha256,
            expected_sampler_source_sha256=sampler.get("sha256"),
            expected_ema_source_sha256=ema.get("sha256"),
            expected_implementation_inputs_sha256=_canonical_sha256(
                terminal.expected.implementation_inputs
            ),
            expected_metric_inputs_sha256=_canonical_sha256(
                terminal.expected.metric_inputs
            ),
        )
        if (
            identity.get("summary_schema_version") != 8
            or identity.get("generation", {}).get("nfe") != 128
        ):
            raise CampaignValidationError("diagnostic schema/NFE identity differs")
        sampling = identity.get("config", {}).get("sampling")
        if (
            not isinstance(sampling, Mapping)
            or type(sampling.get("exclude_special_tokens")) is not bool
        ):
            raise CampaignValidationError("diagnostic sampling identity differs")
        rescore_denovo_run.validate_sampled_token_control_audit(
            summary.get("sampled_token_control_audit"),
            expected_rows=terminal.spec.requested_samples_per_seed,
            exclude_special_tokens=sampling["exclude_special_tokens"],
            raw_model_texts=raw_texts,
            tokenizer_batch_decode=(
                rescore_denovo_run._load_pinned_tokenizer_batch_decode()  # noqa: SLF001
            ),
        )
    except (OSError, ValueError) as error:
        raise CampaignValidationError(
            f"diagnostic structural validation failed: {error}"
        ) from error


def _metric_value(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignValidationError(f"{label} must be finite or null")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CampaignValidationError(f"{label} must lie in [0,1]")
    return result


def _validate_ranked_child_evidence_live(
    envelope: Mapping[str, Any],
    *,
    terminal: TerminalChild,
    registry: ValidatedRegistry,
    summary_claim: artifact_io.FileClaim,
    raw_claim: artifact_io.FileClaim,
    training_claim: artifact_io.FileClaim,
) -> tuple[float | None, float | None]:
    """Return only metrics established by the gate's fresh CPU rescore.

    The import stays local because candidate-authority preparation imports this
    campaign module.  In addition to the gate's complete structural and fresh
    rescore checks, bind its result back to the exact registry-authorized child
    before a score can influence a stage decision.
    """

    if terminal.spec.stage_id == "D":  # pragma: no cover - caller contract
        raise CampaignValidationError("diagnostic must not enter ranked validation")
    from scripts.udlm import superiority_gate

    try:
        validated = (
            superiority_gate._validate_completed_pilot_evidence_live(  # noqa: SLF001
                envelope
            )
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise CampaignValidationError(
            "campaign child failed independent live gate validation"
        ) from error
    validated = _exact_keys(
        validated,
        set(superiority_gate._PILOT_RUN_VALIDATION_KEYS),  # noqa: SLF001
        label="independently validated campaign child",
    )

    spec = terminal.spec
    expected = terminal.expected
    config = _config_map(registry).get(spec.config_id)
    if config is None or config.arm_id != spec.arm_id:
        raise CampaignValidationError("ranked child config is absent from registry")
    if (
        expected.source_config_sha256 != config.sha256
        or expected.sampling_config_sha256 != config.normalized_sampling_sha256
    ):
        raise CampaignValidationError(
            "ranked child config identity differs from registry"
        )
    sampler_source = expected.implementation_inputs.get("sampler_source")
    if not isinstance(sampler_source, Mapping):
        raise CampaignValidationError("ranked child sampler provenance is absent")
    sampler_source_sha256 = _sha256(
        sampler_source.get("sha256"), label="ranked child sampler source digest"
    )
    source_revision = _revision(
        expected.source_revision, label="ranked child source revision"
    )
    expected_checkpoint = {
        "sha256": expected.checkpoint_sha256,
        "size_bytes": expected.checkpoint_size_bytes,
        "global_step": expected.checkpoint_global_step,
    }
    expected_evaluation_config = {
        "relative_path": config.relative_path,
        "sha256": expected.source_config_sha256,
    }
    expected_sampling = {
        "config": expected.sampling_config,
        "sha256": expected.sampling_config_sha256,
    }
    for field, expected_value in {
        "attempt_id": spec.attempt_id,
        "candidate_id": spec.candidate_id,
        "pilot_seed": terminal.seed,
        "requested_samples": spec.requested_samples_per_seed,
        "nfe": 128,
        "metric_branch": "released_comparable",
        "checkpoint": expected_checkpoint,
        "evaluation_config": expected_evaluation_config,
        "sampling": expected_sampling,
        "inference_weights": "ema",
        "runner_sha256": expected.benchmark_runner_sha256,
        "sampler_source_sha256": sampler_source_sha256,
        "implementation_inputs_sha256": _canonical_sha256(
            expected.implementation_inputs
        ),
        "metric_inputs_sha256": _canonical_sha256(expected.metric_inputs),
        "benchmark_revision": source_revision,
        "summary_json_sha256": summary_claim.sha256,
        "raw_samples_csv_sha256": raw_claim.sha256,
    }.items():
        if validated.get(field) != expected_value:
            raise CampaignValidationError(
                f"independently validated campaign child {field} differs"
            )
    training = validated.get("training_exit_receipt")
    if not isinstance(training, Mapping) or {
        key: training.get(key) for key in ("relative_path", "sha256", "schema_version")
    } != {
        "relative_path": training_claim.relative_path,
        "sha256": training_claim.sha256,
        "schema_version": 5,
    }:
        raise CampaignValidationError(
            "independently validated campaign training receipt differs"
        )
    if validated.get("independent_rescore") != {
        "all_21_fields_match": True,
        "both_metric_branches_match": True,
        "failure_counts_match": True,
        "raw_model_text_redecoded": True,
    }:
        raise CampaignValidationError("independent campaign rescore proof differs")
    return (
        _metric_value(validated.get("quality"), label="independent released quality"),
        _metric_value(
            validated.get("diversity"), label="independent released diversity"
        ),
    )


def _revalidate_child_inputs(
    inputs: Sequence[artifact_io.FileClaim], *, phase: str
) -> None:
    """Require every retained descriptor-bound input claim to remain exact."""

    for original in inputs:
        try:
            current, _payload = artifact_io.snapshot_file(
                REPOSITORY_ROOT, original.relative_path, capture_bytes=False
            )
        except (OSError, artifact_io.ArtifactIOError) as error:
            raise CampaignValidationError(
                f"cannot revalidate child evidence input {phase}"
            ) from error
        if current != original:
            raise CampaignValidationError(f"child evidence input changed {phase}")


def _publish_child_evidence(
    terminal: TerminalChild,
    *,
    registry: ValidatedRegistry,
) -> ChildEvidence:
    """Publish an ignored immutable schema-2 outer envelope for one child."""

    spec = terminal.spec
    run_relative = benchmark_launcher._repository_relative(  # noqa: SLF001
        _attempt_root(spec) / f"seed_{terminal.seed}", label="campaign run directory"
    )
    evidence_relative = _evidence_relative_path(spec, terminal.seed)
    benchmark_launcher._ensure_repository_directory(  # noqa: SLF001
        (REPOSITORY_ROOT / evidence_relative).parent,
        label="campaign evidence parent",
    )
    quality: float | None = None
    diversity: float | None = None
    diagnostic_passed: bool | None = None
    inputs: list[artifact_io.FileClaim] = []
    if terminal.failed:
        receipt_relative = f"{run_relative}/failure_receipt.json"
        receipt_claim, receipt_payload = _snapshot_exact(
            receipt_relative, label="child failure receipt"
        )
        receipt = _strict_json(receipt_payload, label="child failure receipt")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("artifact_kind") != "pilot_failure"
            or receipt.get("status") != "failed"
            or receipt.get("attempt_id") != spec.attempt_id
            or receipt.get("candidate_id") != spec.candidate_id
            or receipt.get("pilot_seed") != terminal.seed
            or receipt.get("requested_samples") != spec.requested_samples_per_seed
            or receipt.get("command") != list(terminal.command)
        ):
            raise CampaignValidationError("schema-1 failure receipt binding differs")
        envelope = {
            "schema_version": 2,
            "artifact_kind": "pilot_failure",
            "status": "failed",
            "attempt_id": spec.attempt_id,
            "candidate_id": spec.candidate_id,
            "pilot_seed": terminal.seed,
            "final_seed_results_included": False,
            "failure_receipt": {
                "relative_path": receipt_relative,
                "sha256": receipt_claim.sha256,
                "schema_version": 1,
            },
        }
        inputs.append(receipt_claim)
    else:
        summary_relative = f"{run_relative}/summary.json"
        raw_relative = f"{run_relative}/raw_samples.csv"
        summary_claim, summary_payload = _snapshot_exact(
            summary_relative, label="child summary"
        )
        raw_claim, raw_payload = _snapshot_exact(raw_relative, label="child raw CSV")
        inputs.extend((summary_claim, raw_claim))
        if spec.stage_id == "D":
            _diagnostic_structural_validation(
                terminal,
                summary_payload=summary_payload,
                raw_payload=raw_payload,
            )
            diagnostic_passed = True
        training = registry.training_receipt_by_arm[spec.arm_id]
        training_relative = str(training["relative_path"])
        training_claim, _training_payload = _snapshot_exact(
            training_relative, label="training exit receipt"
        )
        if (
            training_claim.sha256 != training["sha256"]
            or training_claim.size_bytes != training["size_bytes"]
        ):
            raise CampaignValidationError("training exit receipt changed")
        inputs.append(training_claim)
        envelope = {
            "schema_version": 2,
            "artifact_kind": "pilot_evaluation",
            "status": "completed",
            "attempt_id": spec.attempt_id,
            "candidate_id": spec.candidate_id,
            "pilot_seed": terminal.seed,
            "final_seed_results_included": False,
            "training_exit_receipt": {
                "relative_path": training_relative,
                "sha256": training_claim.sha256,
                "schema_version": 5,
            },
            "benchmark_artifacts": {
                "summary_json": {
                    "relative_path": summary_relative,
                    "sha256": summary_claim.sha256,
                    "schema_version": 8,
                },
                "raw_samples_csv": {
                    "relative_path": raw_relative,
                    "sha256": raw_claim.sha256,
                },
            },
        }
        if spec.stage_id != "D":
            quality, diversity = _validate_ranked_child_evidence_live(
                envelope,
                terminal=terminal,
                registry=registry,
                summary_claim=summary_claim,
                raw_claim=raw_claim,
                training_claim=training_claim,
            )
    _revalidate_child_inputs(inputs, phase="before envelope publication")
    payload = _json_bytes(envelope)
    try:
        claim = artifact_io.publish_bytes_exclusive(
            REPOSITORY_ROOT, evidence_relative, payload
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise CampaignValidationError(
            "cannot publish child evidence envelope"
        ) from error
    _revalidate_child_inputs(inputs, phase="during envelope publication")
    reference = {
        "artifact_kind": envelope["artifact_kind"],
        "pilot_seed": terminal.seed,
        "relative_path": evidence_relative,
        "sha256": claim.sha256,
        "schema_version": 2,
    }
    return ChildEvidence(
        terminal=terminal,
        reference=reference,
        released_quality=quality,
        released_diversity=diversity,
        diagnostic_structural_passed=diagnostic_passed,
    )


def _attempt_outcomes(
    attempts: Sequence[AttemptSpec], evidence: Sequence[ChildEvidence]
) -> tuple[AttemptOutcome, ...]:
    by_child = {
        (item.terminal.spec.attempt_id, item.terminal.seed): item for item in evidence
    }
    if len(by_child) != len(evidence):
        raise CampaignValidationError("terminal child evidence is duplicated")
    outcomes: list[AttemptOutcome] = []
    for spec in attempts:
        children = [by_child.get((spec.attempt_id, seed)) for seed in spec.seeds]
        if any(child is None for child in children):
            raise CampaignValidationError("attempt lacks exact child evidence coverage")
        present = [child for child in children if child is not None]
        child_refs = tuple(child.reference for child in present)
        if any(child.terminal.failed for child in present):
            status = "failed"
            quality = diversity = None
            diagnostic = False if spec.stage_id == "D" else None
        elif spec.stage_id == "D":
            status = "completed"
            quality = diversity = None
            diagnostic = all(
                child.diagnostic_structural_passed is True for child in present
            )
        elif any(
            child.released_quality is None or child.released_diversity is None
            for child in present
        ):
            status = "undefined"
            quality = diversity = None
            diagnostic = None
        else:
            status = "completed"
            quality = statistics.fmean(
                float(child.released_quality) for child in present
            )
            diversity = statistics.fmean(
                float(child.released_diversity) for child in present
            )
            diagnostic = None
        outcomes.append(
            AttemptOutcome(
                spec=spec,
                status=status,
                child_outcomes=child_refs,
                released_quality=quality,
                released_diversity=diversity,
                diagnostic_structural_passed=diagnostic,
            )
        )
    return tuple(outcomes)


def _dry_run_record(
    *,
    registry: ValidatedRegistry,
    source_revision: Mapping[str, str],
    decisions: Mapping[str, Mapping[str, Any]],
    through_stage: str | None,
) -> dict[str, Any]:
    stage_id = _next_stage_id(decisions)
    authorized_stage_ids = (
        () if through_stage is None else _authorized_stage_ids(decisions, through_stage)
    )
    attempts = (
        () if stage_id is None else derive_attempts(registry, stage_id, decisions)
    )
    return {
        "event": "candidate_campaign_dry_run",
        "registry": registry.reference,
        "source_revision": dict(source_revision),
        "generation_lease_acquired": False,
        "gpu_query_performed": False,
        "artifact_mutation_performed": False,
        "tmux_operation_performed": False,
        "completed_stage_ids": list(decisions),
        "next_stage_id": stage_id,
        "requested_through_stage": through_stage,
        "authorized_stage_ids": list(authorized_stage_ids),
        "next_stage_attempt_ids": [spec.attempt_id for spec in attempts],
        "next_stage_child_count": sum(len(spec.seeds) for spec in attempts),
        "prefinal_contract": {
            "entry_count": PREFINAL_ENTRY_COUNT,
            "child_count": PREFINAL_CHILD_COUNT,
            "requested_molecule_count": PREFINAL_MOLECULE_COUNT,
            "nfe": 128,
            "maximum_concurrent_gpu_uuids": MAX_CONCURRENT_CHILDREN,
            "diagnostic_gpu_count": 1,
            "utilization_percent_must_be_strictly_less_than": (MAX_UTILIZATION_PERCENT),
            "minimum_free_memory_mib": MIN_FREE_MEMORY_MIB,
        },
        "final_stage_not_authorized_or_launched": True,
    }


def run_campaign(
    *,
    registry: ValidatedRegistry,
    source_revision: Mapping[str, str],
    through_stage: str,
) -> int:
    """Execute only the authorized remaining prefix under one exact lease."""

    decisions, completions = replay_decisions(registry)
    authorized_stage_ids = _authorized_stage_ids(decisions, through_stage)
    if not authorized_stage_ids:
        incomplete = next(
            (item for item in decisions.values() if item.get("status") != "completed"),
            None,
        )
        print(
            json.dumps(
                {
                    "event": "candidate_campaign_already_terminal",
                    "status": (
                        "campaign_incomplete"
                        if incomplete is not None
                        else "prefinal_completed"
                    ),
                    "final_stage_launched": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2 if incomplete is not None else 0

    stage_id = authorized_stage_ids[0]
    factory = _IdentityFactory(registry, source_revision["head"])
    first_attempts = derive_attempts(registry, stage_id, decisions)
    _fresh_stage_paths(first_attempts)
    for spec in first_attempts:
        factory.expected(spec)

    lease = benchmark_launcher._acquire_generation_lease(  # noqa: SLF001
        source_revision=source_revision["head"]
    )
    release_authorized = False
    exit_status = 0
    try:
        for stage_id in authorized_stage_ids:
            attempts = derive_attempts(registry, stage_id, decisions)
            if stage_id != first_attempts[0].stage_id:
                _fresh_stage_paths(attempts)
                for spec in attempts:
                    factory.expected(spec)
            benchmark_launcher._revalidate_generation_lease(lease)  # noqa: SLF001
            _reserve_attempt_roots(attempts)
            terminal = _execute_stage_children(
                lease=lease,
                attempts=attempts,
                identities=factory,
                source_revision=source_revision,
            )
            evidence = tuple(
                _publish_child_evidence(item, registry=registry) for item in terminal
            )
            outcomes = _attempt_outcomes(attempts, evidence)
            predecessor = (
                None
                if not completions
                else completions[STAGE_IDS[STAGE_IDS.index(stage_id) - 1]]
            )
            decision = derive_decision(
                registry=registry,
                stage_id=stage_id,
                attempts=attempts,
                outcomes=outcomes,
                prior_decisions=decisions,
                predecessor=predecessor,
                source_revision=source_revision["head"],
                decided_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            benchmark_launcher._revalidate_generation_lease(lease)  # noqa: SLF001
            completion = publish_decision(
                registry=registry,
                decision=decision,
                predecessor_completion=predecessor,
            )
            benchmark_launcher._revalidate_generation_lease(lease)  # noqa: SLF001
            decisions[stage_id] = decision
            completions[stage_id] = completion
            print(
                json.dumps(
                    {
                        "event": "candidate_campaign_stage_terminal",
                        "stage_id": stage_id,
                        "status": decision["status"],
                        "completion": completion,
                        "promoted_config_ids": decision["advancement"][
                            "promoted_config_ids"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if decision["status"] != "completed":
                exit_status = 2
                release_authorized = True
                break
            current_source = _require_exact_registry_revision(registry)
            if current_source != dict(source_revision):
                raise CampaignValidationError(
                    "source revision changed during campaign execution"
                )
        else:
            release_authorized = True
        if release_authorized and exit_status == 0:
            print(
                json.dumps(
                    {
                        "event": "candidate_campaign_stage_limit_reached",
                        "through_stage": through_stage,
                        "next_stage_id": _next_stage_id(decisions),
                        "final_stage_launched": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if release_authorized:
            benchmark_launcher._release_generation_lease_exact(lease)  # noqa: SLF001
    return exit_status


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise RuntimeError(f"run from repository root: {REPOSITORY_ROOT}")
    benchmark_launcher._require_project_virtual_environment()  # noqa: SLF001
    registry = load_registry(
        args.registry,
        expected_raw_sha256=args.expected_registry_sha256,
        expected_canonical_sha256=args.expected_registry_canonical_sha256,
    )
    source_revision = _require_exact_registry_revision(registry)
    decisions, _completions = replay_decisions(registry)
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_record(
                    registry=registry,
                    source_revision=source_revision,
                    decisions=decisions,
                    through_stage=args.through_stage,
                ),
                sort_keys=True,
            )
        )
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            "campaign controller inherited CUDA_VISIBLE_DEVICES; dynamic UUID "
            "selection requires an unmasked shell"
        )
    benchmark_launcher._require_tmux_for_execution()  # noqa: SLF001
    return run_campaign(
        registry=registry,
        source_revision=source_revision,
        through_stage=args.through_stage,
    )


if __name__ == "__main__":
    raise SystemExit(main())
