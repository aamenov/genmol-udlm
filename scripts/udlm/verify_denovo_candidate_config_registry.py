"""Verify the frozen de-novo candidate config registry without using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    while str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import benchmark  # noqa: E402


SCHEMA_VERSION = 1
REGISTRY_ID = "genmol_udlm_de_novo_candidate_config_v1"
REGISTRY_STATUS = "frozen_before_candidate_generation"
CLAIM_SCOPE = "prospective_greedy_sampler_operating_point_selection"
CONFIG_DIRECTORY = "experiments/udlm/protocols/de_novo_candidate_configs_v1"
REGISTRY_RELATIVE_PATH = (
    "experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json"
)
SUPERIORITY_PROTOCOL_RELATIVE_PATH = (
    "experiments/udlm/protocols/de_novo_superiority_v4.json"
)
SCALE_UP_REGISTRY_RELATIVE_PATH = (
    "experiments/udlm/protocols/selection_bound_scale_up_registry_gpu1.json"
)
ARM_IDS = ("R", "S", "E")
SOFTMAX_TEMPERATURES = (0.5, 0.7, 0.85, 1.0)
RAW_LOO_TOP_P_VALUES = (1.0, 0.98, 0.95)
RANKING_ORDER = (
    "released_quality_descending",
    "released_diversity_descending",
    "config_id_ascii_ascending",
    "attempt_id_ascii_ascending",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")

IDENTITY_CONFIGS = {
    "R": {
        "relative_path": "scripts/exps/denovo/hparams_udlm.yaml",
        "sha256": "bc28eb26297702624e009eed77d13b2a600aeba341b21fbc938d7d01d4c0d84d",
        "size_bytes": 613,
    },
    "S": {
        "relative_path": "scripts/exps/denovo/hparams_udlm_schedule_uniform.yaml",
        "sha256": "8f07751d3504eabc938dbccf91b90217df809945cb06b2fc8f96671cde59f739",
        "size_bytes": 518,
    },
    "E": {
        "relative_path": "scripts/exps/denovo/hparams_udlm_categorical_floor0002.yaml",
        "sha256": "309bbed7b67ad2090def3ba5429f00f4e56b4fb7af2f6be245b9e79619aa3c06",
        "size_bytes": 602,
    },
}


class RegistryValidationError(ValueError):
    """The prospective registry or one of its immutable inputs is invalid."""


def _exact_mapping(value: object, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        found = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise RegistryValidationError(
            f"{label} fields differ: found={found!r}, expected={sorted(fields)!r}"
        )
    return value


def _reject_constant(value: str) -> None:
    raise RegistryValidationError(f"non-finite JSON constant is forbidden: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistryValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        result = json.loads(
            payload,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RegistryValidationError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(result, Mapping):
        raise RegistryValidationError(f"{label} root must be an object")
    return result


def json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RegistryValidationError("value is not finite JSON") from error


def canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RegistryValidationError("value is not canonicalizable JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise RegistryValidationError(f"{label} must be 64 lowercase hex")
    return value


def _revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise RegistryValidationError(f"{label} must be 40 lowercase hex")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RegistryValidationError(f"{label} must be a nonempty POSIX path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise RegistryValidationError(f"{label} must be canonical and root-relative")
    return value


def config_id(arm_id: str, softmax_temp: float, raw_loo_top_p: float) -> str:
    try:
        temperature = {0.5: "050", 0.7: "070", 0.85: "085", 1.0: "100"}[
            float(softmax_temp)
        ]
        top_p = {1.0: "100", 0.98: "098", 0.95: "095"}[float(raw_loo_top_p)]
    except (KeyError, TypeError, ValueError) as error:
        raise RegistryValidationError(
            "sampling point is outside the frozen grid"
        ) from error
    if arm_id not in ARM_IDS:
        raise RegistryValidationError("arm is outside the frozen R/S/E set")
    return f"{arm_id.lower()}_t{temperature}_p{top_p}"


def all_config_ids() -> tuple[str, ...]:
    return tuple(
        config_id(arm, temperature, top_p)
        for arm in ARM_IDS
        for temperature in SOFTMAX_TEMPERATURES
        for top_p in RAW_LOO_TOP_P_VALUES
    )


def generated_config_relative_paths() -> tuple[str, ...]:
    return tuple(
        f"{CONFIG_DIRECTORY}/{config_id(arm, temperature, top_p)}.yaml"
        for arm in ARM_IDS
        for temperature in SOFTMAX_TEMPERATURES
        for top_p in RAW_LOO_TOP_P_VALUES
        if (temperature, top_p) != (1.0, 1.0)
    )


def _sampling_scalar_text(value: float) -> str:
    return {
        0.5: "0.5",
        0.7: "0.7",
        0.85: "0.85",
        1.0: "1.0",
        0.98: "0.98",
        0.95: "0.95",
    }[float(value)]


def generated_config_payload(
    identity_payload: bytes,
    *,
    softmax_temp: float,
    raw_loo_top_p: float,
) -> bytes:
    """Change only the two registered inference controls in an identity YAML."""

    if (float(softmax_temp), float(raw_loo_top_p)) == (1.0, 1.0):
        raise RegistryValidationError("the identity point must reuse historical bytes")
    try:
        source = identity_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RegistryValidationError("identity config is not UTF-8") from error
    if "raw_loo_top_p:" in source:
        raise RegistryValidationError("historical identity unexpectedly declares top-p")
    lines = source.splitlines(keepends=True)
    positions = [
        index for index, line in enumerate(lines) if line.startswith("softmax_temp:")
    ]
    if len(positions) != 1:
        raise RegistryValidationError(
            "identity config must declare softmax_temp exactly once"
        )
    position = positions[0]
    newline = "\r\n" if lines[position].endswith("\r\n") else "\n"
    lines[position] = f"softmax_temp: {_sampling_scalar_text(softmax_temp)}{newline}"
    lines.insert(
        position + 1,
        f"raw_loo_top_p: {_sampling_scalar_text(raw_loo_top_p)}{newline}",
    )
    payload = "".join(lines).encode("utf-8")

    import yaml

    base = yaml.safe_load(identity_payload)
    generated = yaml.safe_load(payload)
    expected = dict(base)
    expected["softmax_temp"] = float(softmax_temp)
    expected["raw_loo_top_p"] = float(raw_loo_top_p)
    if generated != expected:
        raise RegistryValidationError(
            "generated YAML changed a field outside the two sampling controls"
        )
    normalized = benchmark.validate_sampling_config(generated)
    if (
        normalized["diffusion_type"] != "udlm"
        or normalized["num_steps"] != 128
        or normalized["softmax_temp"] != float(softmax_temp)
        or normalized["raw_loo_top_p"] != float(raw_loo_top_p)
    ):
        raise RegistryValidationError(
            "generated YAML does not normalize to its grid point"
        )
    return payload


def _ranking(*, aggregation: str) -> dict[str, Any]:
    return {
        "metric_branch": "released_comparable",
        "aggregation": aggregation,
        "order": list(RANKING_ORDER),
        "all_scheduled_slots_must_be_terminal": True,
        "on_failed_or_undefined_child": "retain_unrankable",
        "on_insufficient_rankable_quota": ("campaign_incomplete_without_promotion"),
    }


def expected_stages() -> list[dict[str, Any]]:
    p100_ids = [
        config_id(arm, temperature, 1.0)
        for arm in ARM_IDS
        for temperature in SOFTMAX_TEMPERATURES
    ]
    return [
        {
            "stage_id": "D",
            "role": "nonranking_decode_diagnostic",
            "candidate_source": {
                "kind": "fixed_config_ids",
                "config_ids": ["e_t100_p100"],
            },
            "expected_config_count": 1,
            "expected_child_count": 1,
            "seeds": [1100],
            "requested_samples_per_seed": 32,
            "nfe": 128,
            "ranking": None,
            "promotion": None,
            "next_stage_id": "A",
        },
        {
            "stage_id": "A",
            "role": "raw_loo_temperature_screen",
            "candidate_source": {
                "kind": "fixed_config_ids",
                "config_ids": p100_ids,
            },
            "expected_config_count": 12,
            "expected_child_count": 12,
            "seeds": [1101],
            "requested_samples_per_seed": 32,
            "nfe": 128,
            "ranking": _ranking(aggregation="current_stage_only"),
            "promotion": {
                "scope": "per_arm",
                "count": 2,
                "selection_unit": "softmax_temp",
            },
            "next_stage_id": "B",
        },
        {
            "stage_id": "B",
            "role": "joint_temperature_nucleus_screen",
            "candidate_source": {
                "kind": "cross_product_of_promoted_temperatures",
                "from_stage_id": "A",
                "raw_loo_top_p_values": list(RAW_LOO_TOP_P_VALUES),
                "expected_per_arm": 6,
            },
            "expected_config_count": 18,
            "expected_child_count": 18,
            "seeds": [1102],
            "requested_samples_per_seed": 64,
            "nfe": 128,
            "ranking": _ranking(aggregation="current_stage_only"),
            "promotion": {
                "scope": "per_arm",
                "count": 2,
                "selection_unit": "config",
            },
            "next_stage_id": "C",
        },
        {
            "stage_id": "C",
            "role": "held_out_operating_point_confirmation",
            "candidate_source": {
                "kind": "promoted_config_ids",
                "from_stage_id": "B",
                "expected_per_arm": 2,
            },
            "expected_config_count": 6,
            "expected_child_count": 6,
            "seeds": [1103],
            "requested_samples_per_seed": 96,
            "nfe": 128,
            "ranking": _ranking(aggregation="current_stage_only"),
            "promotion": {
                "scope": "per_arm",
                "count": 1,
                "selection_unit": "config",
            },
            "next_stage_id": "eligible",
        },
        {
            "stage_id": "eligible",
            "role": "registered_candidate_selection",
            "candidate_source": {
                "kind": "promoted_config_ids",
                "from_stage_id": "C",
                "expected_per_arm": 1,
            },
            "expected_config_count": 3,
            "expected_child_count": 6,
            "seeds": [1000, 1001],
            "requested_samples_per_seed": 256,
            "nfe": 128,
            "ranking": _ranking(aggregation="unweighted_mean_over_exact_seeds"),
            "promotion": {
                "scope": "global",
                "count": 1,
                "selection_unit": "config",
            },
            "next_stage_id": "final",
        },
        {
            "stage_id": "final",
            "role": "locked_final_superiority_evaluation",
            "candidate_source": {
                "kind": "promoted_config_ids",
                "from_stage_id": "eligible",
                "expected_global": 1,
            },
            "expected_config_count": 1,
            "expected_child_count": 3,
            "seeds": [0, 1, 2],
            "requested_samples_per_seed": 1000,
            "nfe": 128,
            "ranking": None,
            "promotion": None,
            "next_stage_id": None,
        },
    ]


def _live_payload(relative_path: str) -> bytes:
    try:
        _claim, payload = artifact_io.snapshot_file(
            REPOSITORY_ROOT, relative_path, capture_bytes=True
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise RegistryValidationError(f"cannot read stable {relative_path}") from error
    if payload is None:  # pragma: no cover - artifact_io contract
        raise AssertionError("snapshot bytes were not retained")
    return payload


def _reference(
    value: object,
    *,
    label: str,
    expected_path: str,
    expected_schema: int,
    payload_loader: Callable[[str], bytes],
) -> None:
    record = _exact_mapping(
        value, {"relative_path", "sha256", "schema_version"}, label=label
    )
    if (
        _relative_path(record["relative_path"], label=f"{label} path") != expected_path
        or record["schema_version"] != expected_schema
    ):
        raise RegistryValidationError(f"{label} path or schema differs")
    digest = _sha256(record["sha256"], label=f"{label} sha256")
    payload = payload_loader(expected_path)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise RegistryValidationError(f"{label} bytes differ")
    parsed = strict_json_loads(payload, label=label)
    if parsed.get("schema_version") != expected_schema:
        raise RegistryValidationError(f"{label} embedded schema differs")


def _config_record(
    value: object,
    *,
    expected_arm: str,
    expected_temperature: float,
    expected_top_p: float,
    payload_loader: Callable[[str], bytes],
    identity_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    record = _exact_mapping(
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
        label="config entry",
    )
    expected_id = config_id(expected_arm, expected_temperature, expected_top_p)
    identity = (expected_temperature, expected_top_p) == (1.0, 1.0)
    if (
        record["config_id"] != expected_id
        or record["arm_id"] != expected_arm
        or record["scale_up_member_slug"] != expected_arm.lower()
        or type(record["softmax_temp"]) not in (int, float)
        or float(record["softmax_temp"]) != expected_temperature
        or type(record["raw_loo_top_p"]) not in (int, float)
        or float(record["raw_loo_top_p"]) != expected_top_p
        or record["is_reused_identity"] is not identity
    ):
        raise RegistryValidationError(f"config entry differs for {expected_id}")
    reference = _exact_mapping(
        record["config"],
        {"relative_path", "sha256", "size_bytes", "storage"},
        label=f"{expected_id} config reference",
    )
    expected_path = (
        IDENTITY_CONFIGS[expected_arm]["relative_path"]
        if identity
        else f"{CONFIG_DIRECTORY}/{expected_id}.yaml"
    )
    expected_storage = "historical_identity_reuse" if identity else "generated"
    if (
        _relative_path(reference["relative_path"], label=f"{expected_id} path")
        != expected_path
        or reference["storage"] != expected_storage
        or type(reference["size_bytes"]) is not int
        or reference["size_bytes"] <= 0
    ):
        raise RegistryValidationError(f"{expected_id} config reference differs")
    payload = payload_loader(expected_path)
    digest = _sha256(reference["sha256"], label=f"{expected_id} config sha256")
    if (
        len(payload) != reference["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != digest
    ):
        raise RegistryValidationError(f"{expected_id} config byte identity differs")
    if identity:
        expected_payload = identity_payloads[expected_arm]
        identity_reference = IDENTITY_CONFIGS[expected_arm]
        if (
            payload != expected_payload
            or digest != identity_reference["sha256"]
            or len(payload) != identity_reference["size_bytes"]
        ):
            raise RegistryValidationError(f"{expected_id} historical identity changed")
    else:
        expected_payload = generated_config_payload(
            identity_payloads[expected_arm],
            softmax_temp=expected_temperature,
            raw_loo_top_p=expected_top_p,
        )
        if payload != expected_payload:
            raise RegistryValidationError(f"{expected_id} is not deterministic")

    import yaml

    normalized = benchmark.validate_sampling_config(yaml.safe_load(payload))
    normalized_digest = _sha256(
        record["normalized_sampling_sha256"],
        label=f"{expected_id} normalized sampling sha256",
    )
    if (
        canonical_json_sha256(normalized) != normalized_digest
        or normalized["softmax_temp"] != expected_temperature
        or normalized["raw_loo_top_p"] != expected_top_p
    ):
        raise RegistryValidationError(f"{expected_id} normalized sampling differs")
    return dict(record)


def validate_registry_document(
    value: object,
    *,
    payload_loader: Callable[[str], bytes] = _live_payload,
) -> Mapping[str, Any]:
    registry = _exact_mapping(
        value,
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
        registry["schema_version"] != SCHEMA_VERSION
        or registry["registry_id"] != REGISTRY_ID
        or registry["status"] != REGISTRY_STATUS
        or registry["claim_scope"] != CLAIM_SCOPE
    ):
        raise RegistryValidationError("registry identity differs")
    publication = _exact_mapping(
        registry["publication"],
        {
            "framework_revision",
            "config_revision",
            "config_directory",
            "registry_relative_path",
            "separate_config_and_registry_publications_required",
        },
        label="publication",
    )
    _revision(publication["framework_revision"], label="framework revision")
    _revision(publication["config_revision"], label="config revision")
    if (
        publication["framework_revision"] == publication["config_revision"]
        or publication["config_directory"] != CONFIG_DIRECTORY
        or publication["registry_relative_path"] != REGISTRY_RELATIVE_PATH
        or publication["separate_config_and_registry_publications_required"] is not True
    ):
        raise RegistryValidationError("publication firewall differs")
    authority = _exact_mapping(
        registry["authority"],
        {
            "superiority_protocol",
            "scale_up_registry",
            "benchmark_schema_version",
            "inference_weights",
            "nfe",
            "metric_branch",
        },
        label="authority",
    )
    if (
        authority["benchmark_schema_version"] != 8
        or authority["inference_weights"] != "ema"
        or authority["nfe"] != 128
        or authority["metric_branch"] != "released_comparable"
    ):
        raise RegistryValidationError("benchmark authority differs")
    _reference(
        authority["superiority_protocol"],
        label="superiority protocol",
        expected_path=SUPERIORITY_PROTOCOL_RELATIVE_PATH,
        expected_schema=4,
        payload_loader=payload_loader,
    )
    _reference(
        authority["scale_up_registry"],
        label="scale-up registry",
        expected_path=SCALE_UP_REGISTRY_RELATIVE_PATH,
        expected_schema=1,
        payload_loader=payload_loader,
    )
    expected_grid = {
        "arm_ids": list(ARM_IDS),
        "softmax_temperatures": list(SOFTMAX_TEMPERATURES),
        "raw_loo_top_p_values": list(RAW_LOO_TOP_P_VALUES),
        "cartesian_count": 36,
        "generated_config_count": 33,
        "reused_identity_count": 3,
    }
    if registry["grid"] != expected_grid:
        raise RegistryValidationError("grid differs from the frozen 36 points")
    identity_payloads = {
        arm: payload_loader(reference["relative_path"])
        for arm, reference in IDENTITY_CONFIGS.items()
    }
    configs = registry["configs"]
    if not isinstance(configs, list) or len(configs) != 36:
        raise RegistryValidationError("registry must contain exactly 36 configs")
    index = 0
    validated_configs = []
    for arm in ARM_IDS:
        for temperature in SOFTMAX_TEMPERATURES:
            for top_p in RAW_LOO_TOP_P_VALUES:
                validated_configs.append(
                    _config_record(
                        configs[index],
                        expected_arm=arm,
                        expected_temperature=temperature,
                        expected_top_p=top_p,
                        payload_loader=payload_loader,
                        identity_payloads=identity_payloads,
                    )
                )
                index += 1
    if [record["config_id"] for record in validated_configs] != list(all_config_ids()):
        raise RegistryValidationError("config ordering differs from the frozen grid")
    if registry["stages"] != expected_stages():
        raise RegistryValidationError("stage contract differs")
    return registry


def _git(
    arguments: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=check,
        capture_output=True,
    )


def git_blob_bytes(revision: str, relative_path: str) -> bytes:
    _revision(revision, label="Git blob revision")
    _relative_path(relative_path, label="Git blob path")
    object_name = f"{revision}:{relative_path}"
    try:
        kind = _git(["cat-file", "-t", object_name]).stdout.strip()
        if kind != b"blob":
            raise RegistryValidationError(f"Git object is not a blob: {object_name}")
        return _git(["cat-file", "blob", object_name]).stdout
    except subprocess.CalledProcessError as error:
        raise RegistryValidationError(f"missing Git blob: {object_name}") from error


def git_blob_absent(revision: str, relative_path: str) -> bool:
    result = _git(["cat-file", "-e", f"{revision}:{relative_path}"], check=False)
    if result.returncode == 0:
        return False
    if result.returncode == 128:
        return True
    raise RegistryValidationError("Git blob absence check was indeterminate")


def _single_parent(revision: str) -> str:
    fields = _git(["rev-list", "--parents", "-n", "1", revision]).stdout.split()
    if len(fields) != 2 or fields[0].decode("ascii") != revision:
        raise RegistryValidationError(f"{revision} is not a single-parent commit")
    return fields[1].decode("ascii")


def _changed_paths(parent: str, child: str) -> tuple[str, ...]:
    payload = _git(["diff", "--name-only", "-z", parent, child, "--"]).stdout
    if payload and not payload.endswith(b"\0"):
        raise RegistryValidationError("Git changed-path result is truncated")
    return (
        tuple(os.fsdecode(item) for item in payload[:-1].split(b"\0"))
        if payload
        else ()
    )


def validate_publication_history(registry: Mapping[str, Any]) -> None:
    publication = registry["publication"]
    framework_revision = publication["framework_revision"]
    config_revision = publication["config_revision"]
    if _single_parent(config_revision) != framework_revision:
        raise RegistryValidationError("C must have F as its exact sole parent")
    expected_paths = generated_config_relative_paths()
    if tuple(sorted(_changed_paths(framework_revision, config_revision))) != tuple(
        sorted(expected_paths)
    ):
        raise RegistryValidationError("F to C must change exactly the 33 config paths")
    if not git_blob_absent(config_revision, REGISTRY_RELATIVE_PATH):
        raise RegistryValidationError("registry must be absent from config revision C")
    for path in expected_paths:
        if not git_blob_absent(framework_revision, path):
            raise RegistryValidationError(
                f"generated config already existed at F: {path}"
            )
        if git_blob_bytes(config_revision, path) != _live_payload(path):
            raise RegistryValidationError(
                f"live generated config differs from C: {path}"
            )
    for reference in IDENTITY_CONFIGS.values():
        path = reference["relative_path"]
        framework_payload = git_blob_bytes(framework_revision, path)
        if framework_payload != git_blob_bytes(
            config_revision, path
        ) or framework_payload != _live_payload(path):
            raise RegistryValidationError(
                f"historical identity changed across F/C: {path}"
            )
    for path in (SUPERIORITY_PROTOCOL_RELATIVE_PATH, SCALE_UP_REGISTRY_RELATIVE_PATH):
        framework_payload = git_blob_bytes(framework_revision, path)
        if framework_payload != git_blob_bytes(
            config_revision, path
        ) or framework_payload != _live_payload(path):
            raise RegistryValidationError(f"authority input changed across F/C: {path}")


def load_and_verify_registry(
    path: Path,
    *,
    expected_raw_sha256: str,
    expected_canonical_sha256: str,
) -> Mapping[str, Any]:
    expected_raw_sha256 = _sha256(expected_raw_sha256, label="expected raw hash")
    expected_canonical_sha256 = _sha256(
        expected_canonical_sha256, label="expected canonical hash"
    )
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    if Path(os.path.abspath(candidate)) != REPOSITORY_ROOT / REGISTRY_RELATIVE_PATH:
        raise RegistryValidationError("registry path is not the fixed path")
    payload = _live_payload(REGISTRY_RELATIVE_PATH)
    if hashlib.sha256(payload).hexdigest() != expected_raw_sha256:
        raise RegistryValidationError("registry raw hash differs from caller pin")
    registry = strict_json_loads(payload, label="registry")
    if canonical_json_sha256(registry) != expected_canonical_sha256:
        raise RegistryValidationError("registry canonical hash differs from caller pin")
    validate_registry_document(registry)
    validate_publication_history(registry)
    return registry


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-canonical-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    registry = load_and_verify_registry(
        args.registry,
        expected_raw_sha256=args.expected_registry_sha256,
        expected_canonical_sha256=args.expected_registry_canonical_sha256,
    )
    print(
        json.dumps(
            {
                "status": "validated",
                "registry_id": registry["registry_id"],
                "framework_revision": registry["publication"]["framework_revision"],
                "config_revision": registry["publication"]["config_revision"],
                "config_count": len(registry["configs"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
