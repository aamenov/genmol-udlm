"""Materialize exactly 33 prospective de-novo candidate YAMLs at revision F."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import artifact_io  # noqa: E402
from scripts.udlm import (  # noqa: E402
    verify_denovo_candidate_config_registry as verifier,
)


class ConfigPreparationError(RuntimeError):
    """The config-only F -> C publication cannot be proved safe."""


def _git(arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
    )


def require_clean_pushed_framework(expected_revision: str) -> str:
    verifier._revision(expected_revision, label="expected framework revision")
    if _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"]).stdout:
        raise ConfigPreparationError("framework worktree must be completely clean")
    head = _git(["rev-parse", "HEAD"]).stdout.decode("ascii").strip()
    upstream = _git(["rev-parse", "@{upstream}"]).stdout.decode("ascii").strip()
    if head != expected_revision or upstream != expected_revision:
        raise ConfigPreparationError("HEAD and upstream must equal the caller-pinned F")
    return head


def _require_revision_unchanged(expected_revision: str) -> None:
    head = _git(["rev-parse", "HEAD"]).stdout.decode("ascii").strip()
    upstream = _git(["rev-parse", "@{upstream}"]).stdout.decode("ascii").strip()
    if head != expected_revision or upstream != expected_revision:
        raise ConfigPreparationError("framework revision changed during publication")


def _identity_payloads() -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for arm, reference in verifier.IDENTITY_CONFIGS.items():
        try:
            claim, payload = artifact_io.snapshot_file(
                REPOSITORY_ROOT, reference["relative_path"], capture_bytes=True
            )
        except (OSError, artifact_io.ArtifactIOError) as error:
            raise ConfigPreparationError(
                f"cannot read historical {arm} identity config"
            ) from error
        if payload is None:  # pragma: no cover - artifact_io contract
            raise AssertionError("identity bytes were not retained")
        if (
            claim.sha256 != reference["sha256"]
            or claim.size_bytes != reference["size_bytes"]
        ):
            raise ConfigPreparationError(
                f"historical {arm} identity config bytes changed"
            )
        payloads[arm] = payload
    return payloads


def candidate_config_payloads() -> dict[str, bytes]:
    identities = _identity_payloads()
    documents: dict[str, bytes] = {}
    for arm in verifier.ARM_IDS:
        for temperature in verifier.SOFTMAX_TEMPERATURES:
            for top_p in verifier.RAW_LOO_TOP_P_VALUES:
                if (temperature, top_p) == (1.0, 1.0):
                    continue
                config_id = verifier.config_id(arm, temperature, top_p)
                relative_path = f"{verifier.CONFIG_DIRECTORY}/{config_id}.yaml"
                documents[relative_path] = verifier.generated_config_payload(
                    identities[arm],
                    softmax_temp=temperature,
                    raw_loo_top_p=top_p,
                )
    if (
        len(documents) != 33
        or len(set(documents)) != 33
        or set(documents) != set(verifier.generated_config_relative_paths())
    ):
        raise AssertionError("candidate grid did not yield exactly 33 unique YAMLs")
    return documents


def _status_paths() -> tuple[str, ...]:
    payload = _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"]).stdout
    if payload and not payload.endswith(b"\0"):
        raise ConfigPreparationError("Git status output is truncated")
    records = payload[:-1].split(b"\0") if payload else []
    paths: list[str] = []
    for record in records:
        if not record.startswith(b"?? "):
            raise ConfigPreparationError("config publication changed a tracked path")
        paths.append(record[3:].decode("utf-8"))
    return tuple(paths)


def publish_candidate_configs(
    *,
    expected_framework_revision: str,
    source_validator=require_clean_pushed_framework,
) -> Mapping[str, object]:
    framework_revision = source_validator(expected_framework_revision)
    if framework_revision != expected_framework_revision:
        raise ConfigPreparationError("source validator returned a different revision")
    documents = candidate_config_payloads()
    try:
        directory_owner = artifact_io.create_directory_exclusive(
            REPOSITORY_ROOT, verifier.CONFIG_DIRECTORY
        )
    except (OSError, artifact_io.ArtifactIOError) as error:
        raise ConfigPreparationError(
            "candidate config directory already exists"
        ) from error

    ordered_paths = sorted(documents)
    completion_path = ordered_paths[-1]
    try:
        artifact_io.publish_bundle_exclusive(
            REPOSITORY_ROOT,
            [
                artifact_io.PublishItem(relative_path=path, payload=documents[path])
                for path in ordered_paths[:-1]
            ],
            completion=artifact_io.PublishItem(
                relative_path=completion_path,
                payload=documents[completion_path],
            ),
        )
    except artifact_io.CommitIndeterminateError as error:
        raise ConfigPreparationError(
            "config publication became visible but durability is indeterminate"
        ) from error
    except BaseException as error:
        try:
            artifact_io.remove_empty_directory_exact(REPOSITORY_ROOT, directory_owner)
        except BaseException as rollback_error:
            raise artifact_io.RollbackError(error, [rollback_error]) from error
        raise

    _require_revision_unchanged(expected_framework_revision)
    if set(_status_paths()) != set(documents):
        raise ConfigPreparationError(
            "post-publication worktree is not exactly the 33 generated YAMLs"
        )
    references = [
        {
            "relative_path": path,
            "sha256": hashlib.sha256(documents[path]).hexdigest(),
            "size_bytes": len(documents[path]),
        }
        for path in ordered_paths
    ]
    return {
        "status": "candidate_configs_materialized",
        "framework_revision": framework_revision,
        "config_directory": verifier.CONFIG_DIRECTORY,
        "generated_config_count": len(references),
        "configs": references,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-framework-revision", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = publish_candidate_configs(
        expected_framework_revision=args.expected_framework_revision
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
