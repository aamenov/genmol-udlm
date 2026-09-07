"""Publish only the frozen candidate registry after the exact config revision C."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import benchmark  # noqa: E402
from scripts.udlm import (  # noqa: E402
    verify_denovo_candidate_config_registry as verifier,
)


class RegistryPreparationError(RuntimeError):
    """The config-revision C cannot authorize a registry-only publication."""


def _git(arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
    )


def require_clean_pushed_config_revision(
    expected_framework_revision: str, expected_config_revision: str
) -> tuple[str, str]:
    verifier._revision(expected_framework_revision, label="expected framework revision")
    verifier._revision(expected_config_revision, label="expected config revision")
    if _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"]).stdout:
        raise RegistryPreparationError("config worktree must be completely clean")
    head = _git(["rev-parse", "HEAD"]).stdout.decode("ascii").strip()
    upstream = _git(["rev-parse", "@{upstream}"]).stdout.decode("ascii").strip()
    if head != expected_config_revision or upstream != expected_config_revision:
        raise RegistryPreparationError("HEAD and upstream must equal caller-pinned C")
    if verifier._single_parent(expected_config_revision) != expected_framework_revision:
        raise RegistryPreparationError("C must have F as its exact sole parent")
    expected_paths = set(verifier.generated_config_relative_paths())
    if (
        set(
            verifier._changed_paths(
                expected_framework_revision, expected_config_revision
            )
        )
        != expected_paths
    ):
        raise RegistryPreparationError("F to C must change exactly 33 candidate YAMLs")
    if not verifier.git_blob_absent(
        expected_config_revision, verifier.REGISTRY_RELATIVE_PATH
    ):
        raise RegistryPreparationError("candidate registry must be absent from C")
    return expected_framework_revision, expected_config_revision


def _require_revision_unchanged(expected_config_revision: str) -> None:
    head = _git(["rev-parse", "HEAD"]).stdout.decode("ascii").strip()
    upstream = _git(["rev-parse", "@{upstream}"]).stdout.decode("ascii").strip()
    if head != expected_config_revision or upstream != expected_config_revision:
        raise RegistryPreparationError("config revision changed during publication")


def _json_reference(relative_path: str, *, expected_schema: int) -> dict[str, Any]:
    payload = verifier._live_payload(relative_path)
    parsed = verifier.strict_json_loads(payload, label=relative_path)
    if parsed.get("schema_version") != expected_schema:
        raise RegistryPreparationError(f"{relative_path} schema differs")
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema_version": expected_schema,
    }


def _config_entries() -> list[dict[str, Any]]:
    import yaml

    entries: list[dict[str, Any]] = []
    for arm in verifier.ARM_IDS:
        for temperature in verifier.SOFTMAX_TEMPERATURES:
            for top_p in verifier.RAW_LOO_TOP_P_VALUES:
                identity = (temperature, top_p) == (1.0, 1.0)
                config_id = verifier.config_id(arm, temperature, top_p)
                relative_path = (
                    verifier.IDENTITY_CONFIGS[arm]["relative_path"]
                    if identity
                    else f"{verifier.CONFIG_DIRECTORY}/{config_id}.yaml"
                )
                payload = verifier._live_payload(relative_path)
                source = yaml.safe_load(payload)
                normalized = benchmark.validate_sampling_config(source)
                if (
                    normalized["diffusion_type"] != "udlm"
                    or normalized["num_steps"] != 128
                    or normalized["softmax_temp"] != temperature
                    or normalized["raw_loo_top_p"] != top_p
                ):
                    raise RegistryPreparationError(
                        f"{config_id} does not normalize to its registered point"
                    )
                entries.append(
                    {
                        "config_id": config_id,
                        "arm_id": arm,
                        "scale_up_member_slug": arm.lower(),
                        "softmax_temp": temperature,
                        "raw_loo_top_p": top_p,
                        "is_reused_identity": identity,
                        "config": {
                            "relative_path": relative_path,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "size_bytes": len(payload),
                            "storage": (
                                "historical_identity_reuse" if identity else "generated"
                            ),
                        },
                        "normalized_sampling_sha256": (
                            verifier.canonical_json_sha256(normalized)
                        ),
                    }
                )
    if len(entries) != 36:
        raise AssertionError("registry config grid must contain exactly 36 entries")
    return entries


def build_registry_document(
    *, framework_revision: str, config_revision: str
) -> dict[str, Any]:
    verifier._revision(framework_revision, label="framework revision")
    verifier._revision(config_revision, label="config revision")
    registry = {
        "schema_version": verifier.SCHEMA_VERSION,
        "registry_id": verifier.REGISTRY_ID,
        "status": verifier.REGISTRY_STATUS,
        "claim_scope": verifier.CLAIM_SCOPE,
        "publication": {
            "framework_revision": framework_revision,
            "config_revision": config_revision,
            "config_directory": verifier.CONFIG_DIRECTORY,
            "registry_relative_path": verifier.REGISTRY_RELATIVE_PATH,
            "separate_config_and_registry_publications_required": True,
        },
        "authority": {
            "superiority_protocol": _json_reference(
                verifier.SUPERIORITY_PROTOCOL_RELATIVE_PATH, expected_schema=4
            ),
            "scale_up_registry": _json_reference(
                verifier.SCALE_UP_REGISTRY_RELATIVE_PATH, expected_schema=1
            ),
            "benchmark_schema_version": 8,
            "inference_weights": "ema",
            "nfe": 128,
            "metric_branch": "released_comparable",
        },
        "grid": {
            "arm_ids": list(verifier.ARM_IDS),
            "softmax_temperatures": list(verifier.SOFTMAX_TEMPERATURES),
            "raw_loo_top_p_values": list(verifier.RAW_LOO_TOP_P_VALUES),
            "cartesian_count": 36,
            "generated_config_count": 33,
            "reused_identity_count": 3,
        },
        "configs": _config_entries(),
        "stages": verifier.expected_stages(),
    }
    verifier.validate_registry_document(registry)
    return registry


def _status_paths() -> tuple[str, ...]:
    payload = _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"]).stdout
    if payload and not payload.endswith(b"\0"):
        raise RegistryPreparationError("Git status output is truncated")
    records = payload[:-1].split(b"\0") if payload else []
    paths: list[str] = []
    for record in records:
        if not record.startswith(b"?? "):
            raise RegistryPreparationError(
                "registry publication changed a tracked path"
            )
        paths.append(record[3:].decode("utf-8"))
    return tuple(paths)


def publish_candidate_registry(
    *,
    expected_framework_revision: str,
    expected_config_revision: str,
    source_validator=require_clean_pushed_config_revision,
) -> Mapping[str, object]:
    framework_revision, config_revision = source_validator(
        expected_framework_revision, expected_config_revision
    )
    if (
        framework_revision != expected_framework_revision
        or config_revision != expected_config_revision
    ):
        raise RegistryPreparationError("source validator returned different revisions")
    registry = build_registry_document(
        framework_revision=framework_revision,
        config_revision=config_revision,
    )
    verifier.validate_publication_history(registry)
    payload = verifier.json_bytes(registry)
    try:
        claim = artifact_io.publish_bytes_exclusive(
            REPOSITORY_ROOT, verifier.REGISTRY_RELATIVE_PATH, payload
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise RegistryPreparationError(
            "candidate registry path already exists or could not be published"
        ) from error
    _require_revision_unchanged(config_revision)
    if _status_paths() != (verifier.REGISTRY_RELATIVE_PATH,):
        raise RegistryPreparationError(
            "post-publication worktree is not exactly the registry-only G candidate"
        )
    return {
        "status": "candidate_registry_materialized",
        "registry_id": verifier.REGISTRY_ID,
        "framework_revision": framework_revision,
        "config_revision": config_revision,
        "relative_path": claim.relative_path,
        "sha256": claim.sha256,
        "canonical_sha256": verifier.canonical_json_sha256(registry),
        "size_bytes": claim.size_bytes,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-framework-revision", required=True)
    parser.add_argument("--expected-config-revision", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = publish_candidate_registry(
        expected_framework_revision=args.expected_framework_revision,
        expected_config_revision=args.expected_config_revision,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
