"""Validate a frozen, selection-bound R/S/E UDLM scale-up registry.

The scale-up registry is intentionally created only after both optimization
screens have completed.  Validation recomputes the scheduler and conditioner
decisions from their immutable evidence, verifies the sequential publication
history, and proves that each of the three 1,000-update resolved configs is an
exact treatment-only derivative of the selected screen configuration.

This module is CPU-only.  It never probes GPUs, launches training, or writes an
artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.udlm import verify_optimization_screen as screen  # noqa: E402


REGISTRY_SCHEMA_VERSION = 1
MANIFEST_BINDING_SCHEMA_VERSION = 1
EXPECTED_REGISTRY_ID = "genmol_udlm_selection_bound_scale_up_v1"
EXPECTED_STATUS = "frozen_before_any_scale_up_training"
EXPECTED_CLAIM_SCOPE = (
    "matched_1000_update_training_only_not_superiority_without_registered_evaluation"
)
EXPECTED_MAX_STEPS = 1_000
EXPECTED_TRAINING_SEED = 17
EXPECTED_VARIANTS = ("udlm", "schedule_uniform", "udlm_categorical")
EXPECTED_SLUGS = ("r", "s", "e")
EXPECTED_PRIORS = {
    "udlm": "release_uniform",
    "schedule_uniform": "schedule_uniform",
    "udlm_categorical": "empirical_frequency",
}
EXPECTED_ROLES = {
    "udlm": "faithful_release_control",
    "schedule_uniform": "schedule_repair_uniform_control",
    "udlm_categorical": "empirical_prior_treatment",
}
EXPECTED_SCREEN_ARM_IDS = {
    "scheduler": ("E-L0", "E-L1"),
    "conditioning": ("E-A0", "E-A1"),
}
MAX_GPU_COUNT = 4
MAX_SAFE_UTILIZATION_PERCENT = 10
MIN_SAFE_FREE_MEMORY_MIB = 30_000
REGISTRY_RELATIVE_PATH_TEMPLATE = (
    "experiments/udlm/protocols/selection_bound_scale_up_registry_gpu{gpu_count}.json"
)
CONFIG_DIRECTORY_TEMPLATE = (
    "experiments/udlm/protocols/selection_bound_scale_up_configs_gpu{gpu_count}"
)
CONFIG_FILENAMES = {
    "udlm": "r_udlm.json",
    "schedule_uniform": "s_schedule_uniform.json",
    "udlm_categorical": "e_udlm_categorical.json",
}
FRAMEWORK_REQUIRED_PATHS = frozenset(
    {
        "scripts/udlm/verify_scale_up_registry.py",
        "scripts/udlm/prepare_scale_up_registry.py",
        "scripts/udlm/launch_scale_up_panel.py",
        "scripts/udlm/validate_scale_up_panel.py",
    }
)
GENERATION_READINESS_FRAMEWORK_PATHS = frozenset(
    {
        "scripts/exps/denovo/launch_benchmark.py",
        "scripts/exps/denovo/report.py",
        "scripts/exps/denovo/hparams_udlm_categorical_floor0002.yaml",
        "scripts/udlm/superiority_gate.py",
        "tests/test_denovo_benchmark.py",
        "tests/test_denovo_launcher.py",
        "tests/test_denovo_report.py",
        "tests/test_udlm_superiority_gate.py",
    }
)
R4_COMPATIBILITY_REPAIR_PATHS = frozenset(
    {
        "scripts/udlm/launch_health_panel.py",
        "scripts/udlm/validate_health_panel.py",
        "tests/test_checkpoint_io.py",
        "tests/test_udlm_health_panel_launcher.py",
        "tests/test_udlm_optimization_screen_launcher.py",
    }
)
FRAMEWORK_OPTIONAL_EXACT_PATHS = frozenset(
    {
        "README.md",
        "PROJECT_CONTEXT.md",
        "genmol_from_scratch.ipynb",
        "docs/udlm_experiment_plan.md",
        "scripts/train.py",
        "scripts/udlm/launch_train_pilot.py",
        "scripts/udlm/update_notebook.py",
        "scripts/udlm/write_pilot_evidence.py",
        "scripts/udlm/write_pilot_exit_status.py",
        "tests/test_udlm_pilot_exit_status.py",
        "tests/test_udlm_pilot_launcher.py",
        "tests/test_udlm_train_entrypoint.py",
        "tests/test_udlm_update_notebook.py",
        "tests/test_udlm_write_pilot_evidence.py",
    }
) | GENERATION_READINESS_FRAMEWORK_PATHS | R4_COMPATIBILITY_REPAIR_PATHS
SOURCE_PATHS = (
    "configs/base.yaml",
    "configs/udlm.yaml",
    "configs/udlm_categorical.yaml",
    "experiments/udlm/prior_geometry/floor_selection_train_rows_10001_30000.json",
    "scripts/exps/denovo/hparams_udlm_categorical_floor0002.yaml",
    "scripts/exps/denovo/launch_benchmark.py",
    "scripts/exps/denovo/report.py",
    "scripts/train.py",
    "scripts/udlm/launch_health_panel.py",
    "scripts/udlm/launch_optimization_screen.py",
    "scripts/udlm/launch_train_pilot.py",
    "scripts/udlm/launch_scale_up_panel.py",
    "scripts/udlm/prepare_scale_up_registry.py",
    "scripts/udlm/superiority_gate.py",
    "scripts/udlm/validate_health_panel.py",
    "scripts/udlm/validate_scale_up_panel.py",
    "scripts/udlm/verify_optimization_screen.py",
    "scripts/udlm/verify_scale_up_registry.py",
    "scripts/udlm/write_pilot_evidence.py",
    "scripts/udlm/write_pilot_exit_status.py",
    "src/genmol/backbone.py",
    "src/genmol/diffusion.py",
    "src/genmol/model.py",
    "src/genmol/utils/ema.py",
    "src/genmol/utils/utils_data.py",
)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_REVISION = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


BlobLoader = screen.BlobLoader
GitBlobLoader = screen.GitBlobLoader
GitAncestorChecker = screen.GitAncestorChecker
GitSoleParentChecker = screen.GitSoleParentChecker
GitTreePathsLoader = screen.GitTreePathsLoader
GitPushedChecker = screen.GitPushedChecker
GitDiffChecker = screen.GitDiffChecker
GitChangedPathsLoader = Callable[[str, str], frozenset[str]]


class ScaleUpValidationError(ValueError):
    """Raised when scale-up authority or registry bytes are not exact."""


@dataclass(frozen=True)
class ValidatedScaleUpRegistry:
    """Validated scale-up registry plus its recomputed screen authority."""

    data: Mapping[str, Any]
    relative_path: PurePosixPath
    raw_sha256: str
    raw_size_bytes: int
    canonical_sha256: str
    screen_registry: screen.ValidatedRegistry
    scheduler_selection: Mapping[str, Any]
    conditioning_selection: Mapping[str, Any]

    @property
    def reference(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path.as_posix(),
            "sha256": self.raw_sha256,
            "size_bytes": self.raw_size_bytes,
            "canonical_sha256": self.canonical_sha256,
            "schema_version": REGISTRY_SCHEMA_VERSION,
        }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScaleUpValidationError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ScaleUpValidationError(
            f"{label} fields differ; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ScaleUpValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ScaleUpValidationError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ScaleUpValidationError(f"{label} must be at most {maximum}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScaleUpValidationError(f"{label} must be a nonempty string")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ScaleUpValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _revision(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_REVISION.fullmatch(value) is None:
        raise ScaleUpValidationError(f"{label} must be a full lowercase Git revision")
    return value


def _relative_path(value: object, label: str, *, suffix: str | None = None) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or (suffix is not None and path.suffix != suffix)
    ):
        raise ScaleUpValidationError(f"{label} is not a normalized relative path")
    return text


def canonical_json_sha256(value: object) -> str:
    return screen.canonical_json_sha256(value)


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def validate_gpu_count(gpu_count: int) -> int:
    """Accept the user-selected world size, never a physical GPU identifier."""

    return _integer(gpu_count, "gpu_count", minimum=1, maximum=MAX_GPU_COUNT)


def exact_accumulation_steps(
    global_batch_size: int, micro_batch_size: int, gpu_count: int
) -> int:
    """Return exact accumulation and reject rounded global-batch arithmetic."""

    global_batch_size = _integer(global_batch_size, "global batch", minimum=1)
    micro_batch_size = _integer(micro_batch_size, "micro batch", minimum=1)
    gpu_count = validate_gpu_count(gpu_count)
    quotient, remainder = divmod(global_batch_size, micro_batch_size * gpu_count)
    if quotient < 1 or remainder:
        raise ScaleUpValidationError(
            "global batch must be an exact positive multiple of "
            "micro batch * gpu_count"
        )
    return quotient


def default_global_batch_size(gpu_count: int, *, micro_batch_size: int = 2) -> int:
    """Keep the established batch 16 when exact, otherwise use the nearest larger one."""

    gpu_count = validate_gpu_count(gpu_count)
    micro_batch_size = _integer(micro_batch_size, "micro batch", minimum=1)
    quantum = gpu_count * micro_batch_size
    return ((16 + quantum - 1) // quantum) * quantum


def _plain_json(value: object) -> object:
    """Round-trip Decimal leaves from the strict screen parser to JSON leaves."""

    return json.loads(json.dumps(value, default=float, allow_nan=False))


def _require_deterministic_json_bytes(
    payload: bytes, value: object, *, label: str
) -> None:
    """Require new scale artifacts to use this framework's exact JSON emitter."""

    if payload != json_bytes(_plain_json(value)):
        raise ScaleUpValidationError(
            f"{label} does not use the deterministic scale-up JSON serialization"
        )


def _set_existing_path(config: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    current: dict[str, Any] = config
    for component in path[:-1]:
        child = current.get(component)
        if not isinstance(child, dict):
            raise ScaleUpValidationError(
                f"selected screen config lacks {'.'.join(path)}"
            )
        current = child
    if path[-1] not in current:
        raise ScaleUpValidationError(f"selected screen config lacks {'.'.join(path)}")
    current[path[-1]] = value


def derive_scale_up_config(
    selected_screen_config: Mapping[str, Any],
    *,
    training_variant: str,
    gpu_count: int,
    global_batch_size: int,
    micro_batch_size: int,
    num_workers: int,
    output_directory: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Derive one exact 1,000-update member from the selected screen config."""

    if training_variant not in EXPECTED_VARIANTS:
        raise ScaleUpValidationError("training variant is not a registered R/S/E arm")
    gpu_count = validate_gpu_count(gpu_count)
    accumulation = exact_accumulation_steps(
        global_batch_size, micro_batch_size, gpu_count
    )
    num_workers = _integer(num_workers, "num_workers", minimum=0)
    output_directory = _relative_path(output_directory, "member output directory")
    if PurePosixPath(output_directory).parts[:2] != ("output", "udlm"):
        raise ScaleUpValidationError("member output directory must be below output/udlm")
    plain = _plain_json(selected_screen_config)
    if not isinstance(plain, dict):
        raise ScaleUpValidationError("selected screen config must be an object")
    config = copy.deepcopy(plain)
    replacements: tuple[tuple[tuple[str, ...], object], ...] = (
        (("trainer", "devices"), gpu_count),
        (("trainer", "max_steps"), EXPECTED_MAX_STEPS),
        (("trainer", "accumulate_grad_batches"), accumulation),
        (("loader", "global_batch_size"), global_batch_size),
        (("loader", "batch_size"), micro_batch_size),
        (("loader", "num_workers"), num_workers),
        (("callback", "every_n_train_steps"), EXPECTED_MAX_STEPS),
        (
            ("callback", "dirpath"),
            str(repository_root.resolve() / output_directory / "checkpoints"),
        ),
        (("training", "reseed_after_model_initialization"), True),
        (
            ("training", "udlm", "prior_variant"),
            EXPECTED_PRIORS[training_variant],
        ),
    )
    for path, value in replacements:
        _set_existing_path(config, path, value)
    if not screen._exact_json_equal(config.get("seed"), EXPECTED_TRAINING_SEED):
        raise ScaleUpValidationError("selected screen config changed training seed")
    return config


def matched_config_sha256(resolved_config: Mapping[str, Any]) -> str:
    """Hash a member after masking only treatment and output-directory leaves."""

    normalized = _plain_json(resolved_config)
    if not isinstance(normalized, dict):
        raise ScaleUpValidationError("resolved member config must be an object")
    try:
        prior = normalized["training"]["udlm"]["prior_variant"]
        callback = normalized["callback"]["dirpath"]
    except (KeyError, TypeError) as error:
        raise ScaleUpValidationError("member config lacks matched-panel leaves") from error
    if prior not in set(EXPECTED_PRIORS.values()):
        raise ScaleUpValidationError("member config has an unregistered prior treatment")
    if not isinstance(callback, str) or not callback:
        raise ScaleUpValidationError("member callback directory is invalid")
    normalized["training"]["udlm"]["prior_variant"] = "<REGISTERED_TREATMENT>"
    normalized["callback"]["dirpath"] = "<VARIANT_RUN_DIR>/checkpoints"
    return canonical_json_sha256(normalized)


def git_changed_paths_loader(ancestor: str, descendant: str) -> frozenset[str]:
    payload = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            ancestor,
            descendant,
            "--",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if not payload:
        return frozenset()
    if not payload.endswith(b"\0"):
        raise ScaleUpValidationError("Git changed-path listing is truncated")
    paths = tuple(os.fsdecode(item) for item in payload[:-1].split(b"\0"))
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise ScaleUpValidationError("Git changed-path listing is malformed")
    return frozenset(paths)


def _framework_path_allowed(path: str) -> bool:
    if path in FRAMEWORK_REQUIRED_PATHS or path in FRAMEWORK_OPTIONAL_EXACT_PATHS:
        return True
    return re.fullmatch(r"tests/test_udlm_scale_up_[A-Za-z0-9_.-]+\.py", path) is not None


def _load_reference(
    value: object,
    *,
    label: str,
    revision: str | None,
    loader: BlobLoader,
    git_blob_loader: GitBlobLoader,
    json_schema: int | None = None,
    require_canonical: bool = False,
) -> tuple[bytes | screen.BlobSnapshot, Mapping[str, Any] | None]:
    reference = _mapping(value, label)
    expected_keys = {"root", "relative_path", "sha256", "size_bytes"}
    if require_canonical:
        expected_keys.add("canonical_sha256")
    if json_schema is not None:
        expected_keys.add("schema_version")
    _exact_keys(reference, expected_keys, label)
    root = reference.get("root")
    if root not in {"repository", "project"}:
        raise ScaleUpValidationError(f"{label} root is invalid")
    relative = PurePosixPath(
        _relative_path(reference.get("relative_path"), f"{label} path")
    )
    digest = _sha256(reference.get("sha256"), f"{label} digest")
    size = _integer(reference.get("size_bytes"), f"{label} size", minimum=1)
    if json_schema is not None:
        schema_version = _integer(
            reference.get("schema_version"), f"{label} schema", minimum=1
        )
        if schema_version != json_schema:
            raise ScaleUpValidationError(f"{label} schema is unsupported")
    try:
        loaded = loader(str(root), relative)
    except Exception as error:
        raise ScaleUpValidationError(f"cannot load {label}") from error
    if isinstance(loaded, bytes):
        observed_size = len(loaded)
        observed_digest = hashlib.sha256(loaded).hexdigest()
    else:
        observed_size = loaded.size_bytes
        observed_digest = loaded.sha256
    if observed_size != size or observed_digest != digest:
        raise ScaleUpValidationError(f"{label} differs from its bound bytes")
    if revision is not None:
        if root != "repository":
            raise ScaleUpValidationError(f"committed {label} must use repository root")
        try:
            committed = git_blob_loader(revision, relative)
        except Exception as error:
            raise ScaleUpValidationError(f"{label} is not a Git blob at {revision}") from error
        if len(committed) != size or hashlib.sha256(committed).hexdigest() != digest:
            raise ScaleUpValidationError(f"committed {label} differs from its pin")
        if isinstance(loaded, bytes) and committed != loaded:
            raise ScaleUpValidationError(f"live {label} differs from committed bytes")
    parsed: Mapping[str, Any] | None = None
    if require_canonical or json_schema is not None:
        if not isinstance(loaded, bytes):
            raise ScaleUpValidationError(f"JSON {label} bytes were not retained")
        parsed = _mapping(screen.strict_json_loads(loaded, label=label), label)
        if require_canonical and canonical_json_sha256(parsed) != _sha256(
            reference.get("canonical_sha256"), f"{label} canonical digest"
        ):
            raise ScaleUpValidationError(f"{label} canonical digest differs")
        if json_schema is not None:
            payload_schema = _integer(
                parsed.get("schema_version"), f"{label} payload schema", minimum=1
            )
            if payload_schema != json_schema:
                raise ScaleUpValidationError(f"{label} payload schema differs")
    return loaded, parsed


def _git_blob_absent(
    revision: str,
    path: str,
    *,
    git_tree_paths_loader: GitTreePathsLoader,
    label: str,
) -> None:
    try:
        observed = git_tree_paths_loader(revision, PurePosixPath(path))
    except Exception as error:
        raise ScaleUpValidationError(
            f"cannot prove historical absence of {label}"
        ) from error
    if observed:
        raise ScaleUpValidationError(f"{label} unexpectedly existed at {revision}")


def _validate_revision_edge(
    *,
    parent: str,
    child: str,
    expected_paths: frozenset[str],
    label: str,
    git_ancestor_checker: GitAncestorChecker,
    git_sole_parent_checker: GitSoleParentChecker,
    git_pushed_checker: GitPushedChecker,
    git_diff_checker: GitDiffChecker,
    changed_paths_loader: GitChangedPathsLoader,
) -> None:
    if child == parent or not git_ancestor_checker(parent, child):
        raise ScaleUpValidationError(f"{label} revisions are not ordered")
    if not git_sole_parent_checker(child, parent):
        raise ScaleUpValidationError(f"{label} is not an exact single-parent transition")
    if not git_pushed_checker(parent) or not git_pushed_checker(child):
        raise ScaleUpValidationError(f"{label} revisions are not pushed")
    if changed_paths_loader(parent, child) != expected_paths:
        raise ScaleUpValidationError(f"{label} changed paths are not exact")
    if not git_diff_checker(parent, child, expected_paths):
        raise ScaleUpValidationError(f"{label} contains an unregistered change")


def _screen_registry_from_authority(
    authority: Mapping[str, Any],
    *,
    selection_revision: str,
    loader: BlobLoader,
    git_blob_loader: GitBlobLoader,
    git_ancestor_checker: GitAncestorChecker,
    git_sole_parent_checker: GitSoleParentChecker,
    git_tree_paths_loader: GitTreePathsLoader,
    git_pushed_checker: GitPushedChecker,
    git_diff_checker: GitDiffChecker,
) -> screen.ValidatedRegistry:
    reference = authority.get("screen_registry")
    loaded, parsed = _load_reference(
        reference,
        label="optimization-screen registry",
        revision=selection_revision,
        loader=loader,
        git_blob_loader=git_blob_loader,
        json_schema=screen.REGISTRY_SCHEMA_VERSION,
        require_canonical=True,
    )
    if not isinstance(loaded, bytes) or parsed is None:
        raise ScaleUpValidationError("optimization-screen registry bytes unavailable")
    ref = _mapping(reference, "optimization-screen registry reference")
    historical_source = _revision(
        _mapping(parsed.get("source"), "optimization-screen source").get("revision"),
        "optimization-screen source revision",
    )

    def historical_registry_loader(
        root: str, relative_path: PurePosixPath
    ) -> bytes | screen.BlobSnapshot:
        """Replay frozen registry inputs after later reviewed source changes.

        The optimization registry binds its implementation/config blobs at its
        own pre-screen source revision.  R4 necessarily changes some of those
        live paths, so validation must use their committed historical bytes.
        Noncommitted run artifacts and the project checkpoint still come from
        the stable local loader.
        """

        if root == "repository":
            try:
                return git_blob_loader(historical_source, relative_path)
            except Exception:
                pass
        return loader(root, relative_path)

    try:
        return screen.load_validated_registry(
            loaded,
            relative_path=str(ref["relative_path"]),
            expected_raw_sha256=str(ref["sha256"]),
            expected_canonical_sha256=str(ref["canonical_sha256"]),
            loader=historical_registry_loader,
            git_blob_loader=git_blob_loader,
            git_ancestor_checker=git_ancestor_checker,
            git_sole_parent_checker=git_sole_parent_checker,
            git_tree_paths_loader=git_tree_paths_loader,
            git_pushed_checker=git_pushed_checker,
            git_diff_checker=git_diff_checker,
        )
    except Exception as error:
        raise ScaleUpValidationError("optimization-screen registry is invalid") from error


def _validate_screen_authority(
    value: object,
    *,
    publication: Mapping[str, Any],
    loader: BlobLoader,
    git_blob_loader: GitBlobLoader,
    git_ancestor_checker: GitAncestorChecker,
    git_sole_parent_checker: GitSoleParentChecker,
    git_tree_paths_loader: GitTreePathsLoader,
    git_pushed_checker: GitPushedChecker,
    git_diff_checker: GitDiffChecker,
    changed_paths_loader: GitChangedPathsLoader,
) -> tuple[screen.ValidatedRegistry, Mapping[str, Any], Mapping[str, Any]]:
    authority = _mapping(value, "screen authority")
    _exact_keys(
        authority,
        {
            "screen_registry",
            "scheduler_evidence",
            "scheduler_selection",
            "conditioning_evidence",
            "conditioning_selection",
        },
        "screen authority",
    )
    selection_revision = _revision(
        publication.get("selection_revision"), "selection revision"
    )
    registry = _screen_registry_from_authority(
        authority,
        selection_revision=selection_revision,
        loader=loader,
        git_blob_loader=git_blob_loader,
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_tree_paths_loader=git_tree_paths_loader,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
    )
    records: dict[str, tuple[Mapping[str, Any], bytes, Mapping[str, Any]]] = {}
    for stage in ("scheduler", "conditioning"):
        evidence_ref = _mapping(
            authority.get(f"{stage}_evidence"), f"{stage} evidence reference"
        )
        selection_ref = _mapping(
            authority.get(f"{stage}_selection"), f"{stage} selection reference"
        )
        evidence_bytes, evidence = _load_reference(
            evidence_ref,
            label=f"{stage} evidence",
            revision=selection_revision,
            loader=loader,
            git_blob_loader=git_blob_loader,
            json_schema=screen.EVIDENCE_SCHEMA_VERSION,
            require_canonical=True,
        )
        _selection_bytes, declared = _load_reference(
            selection_ref,
            label=f"{stage} selection",
            revision=selection_revision,
            loader=loader,
            git_blob_loader=git_blob_loader,
            json_schema=screen.SELECTION_SCHEMA_VERSION,
            require_canonical=True,
        )
        if not isinstance(evidence_bytes, bytes) or evidence is None or declared is None:
            raise ScaleUpValidationError(f"{stage} decision bytes unavailable")
        recomputed = screen.evaluate_evidence_bytes(
            evidence_bytes,
            evidence_relative_path=str(evidence_ref["relative_path"]),
            stage_id=stage,
            registry=registry,
            loader=loader,
        )
        if recomputed.get("status") != "completed" or not screen._exact_json_equal(
            dict(declared), recomputed
        ):
            raise ScaleUpValidationError(f"{stage} selection is not reproducible")
        if recomputed.get("selected_arm_id") not in EXPECTED_SCREEN_ARM_IDS[stage]:
            raise ScaleUpValidationError(f"{stage} selection has no valid winner")
        records[stage] = (evidence_ref, evidence_bytes, declared)

    scheduler_selection = records["scheduler"][2]
    conditioning_selection = records["conditioning"][2]
    dependency = _mapping(
        conditioning_selection.get("dependency"), "conditioning dependency"
    )
    if dependency.get("selected_scheduler_arm_id") != scheduler_selection.get(
        "selected_arm_id"
    ):
        raise ScaleUpValidationError("conditioning selection used a different scheduler")

    registry_publication = _revision(
        scheduler_selection.get("run_source_revision"),
        "screen registry publication revision",
    )
    registry_source = _revision(
        registry.data["source"]["revision"], "screen registry source revision"
    )
    scheduler_publication = _revision(
        dependency.get("authorization_revision"), "scheduler publication revision"
    )
    conditioning_source = _revision(
        conditioning_selection.get("run_source_revision"),
        "conditioning run source revision",
    )
    if conditioning_source != scheduler_publication:
        raise ScaleUpValidationError("conditioning run did not use scheduler publication")
    registry_path = registry.relative_path.as_posix()
    scheduler_paths = frozenset(
        str(authority[name]["relative_path"])
        for name in ("scheduler_evidence", "scheduler_selection")
    )
    conditioning_paths = frozenset(
        str(authority[name]["relative_path"])
        for name in ("conditioning_evidence", "conditioning_selection")
    )
    _validate_revision_edge(
        parent=registry_source,
        child=registry_publication,
        expected_paths=frozenset({registry_path}),
        label="screen-registry publication",
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
        changed_paths_loader=changed_paths_loader,
    )
    _validate_revision_edge(
        parent=registry_publication,
        child=scheduler_publication,
        expected_paths=scheduler_paths,
        label="scheduler-decision publication",
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
        changed_paths_loader=changed_paths_loader,
    )
    _validate_revision_edge(
        parent=scheduler_publication,
        child=selection_revision,
        expected_paths=conditioning_paths,
        label="conditioning-decision publication",
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
        changed_paths_loader=changed_paths_loader,
    )
    for path in scheduler_paths:
        _git_blob_absent(
            registry_publication,
            path,
            git_tree_paths_loader=git_tree_paths_loader,
            label="scheduler decision artifact",
        )
    for path in conditioning_paths:
        _git_blob_absent(
            scheduler_publication,
            path,
            git_tree_paths_loader=git_tree_paths_loader,
            label="conditioning decision artifact",
        )
    return registry, scheduler_selection, conditioning_selection


def _validate_publication(
    value: object,
    *,
    git_ancestor_checker: GitAncestorChecker,
    git_sole_parent_checker: GitSoleParentChecker,
    git_tree_paths_loader: GitTreePathsLoader,
    git_pushed_checker: GitPushedChecker,
    git_diff_checker: GitDiffChecker,
    changed_paths_loader: GitChangedPathsLoader,
) -> Mapping[str, Any]:
    publication = _mapping(value, "publication")
    _exact_keys(
        publication,
        {
            "selection_revision",
            "framework_revision",
            "config_revision",
            "framework_paths",
            "config_paths",
        },
        "publication",
    )
    selection = _revision(publication.get("selection_revision"), "selection revision")
    framework = _revision(publication.get("framework_revision"), "framework revision")
    config = _revision(publication.get("config_revision"), "config revision")
    framework_paths_value = publication.get("framework_paths")
    config_paths_value = publication.get("config_paths")
    if not isinstance(framework_paths_value, list) or not isinstance(
        config_paths_value, list
    ):
        raise ScaleUpValidationError("publication paths must be arrays")
    framework_paths = frozenset(
        _relative_path(path, "framework changed path") for path in framework_paths_value
    )
    config_paths = frozenset(
        _relative_path(path, "config changed path", suffix=".json")
        for path in config_paths_value
    )
    if len(framework_paths) != len(framework_paths_value) or len(config_paths) != len(
        config_paths_value
    ):
        raise ScaleUpValidationError("publication paths contain duplicates")
    if not FRAMEWORK_REQUIRED_PATHS <= framework_paths or any(
        not _framework_path_allowed(path) for path in framework_paths
    ):
        raise ScaleUpValidationError("framework publication path set is invalid")
    if len(config_paths) != len(EXPECTED_VARIANTS):
        raise ScaleUpValidationError("config publication must contain three files")
    _validate_revision_edge(
        parent=selection,
        child=framework,
        expected_paths=framework_paths,
        label="selection-to-framework publication",
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
        changed_paths_loader=changed_paths_loader,
    )
    _validate_revision_edge(
        parent=framework,
        child=config,
        expected_paths=config_paths,
        label="framework-to-config publication",
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
        changed_paths_loader=changed_paths_loader,
    )
    for path in FRAMEWORK_REQUIRED_PATHS:
        _git_blob_absent(
            selection,
            path,
            git_tree_paths_loader=git_tree_paths_loader,
            label="scale-up framework path",
        )
    for path in config_paths:
        _git_blob_absent(
            framework,
            path,
            git_tree_paths_loader=git_tree_paths_loader,
            label="scale-up config",
        )
    for revision, boundary in (
        (selection, "selection revision"),
        (framework, "framework revision"),
    ):
        for gpu_count in range(1, MAX_GPU_COUNT + 1):
            _git_blob_absent(
                revision,
                CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count),
                git_tree_paths_loader=git_tree_paths_loader,
                label=f"scale-up config family at {boundary}",
            )
            _git_blob_absent(
                revision,
                REGISTRY_RELATIVE_PATH_TEMPLATE.format(gpu_count=gpu_count),
                git_tree_paths_loader=git_tree_paths_loader,
                label=f"scale-up registry family at {boundary}",
            )
    return publication


def _selected_screen_config(
    selected: Mapping[str, Any],
    *,
    screen_registry: screen.ValidatedRegistry,
    scheduler_selection: Mapping[str, Any],
    conditioning_selection: Mapping[str, Any],
) -> Mapping[str, Any]:
    _exact_keys(
        selected,
        {
            "scheduler_arm_id",
            "conditioning_arm_id",
            "scheduler",
            "conditioner",
            "screen_config",
        },
        "selected design",
    )
    scheduler_id = scheduler_selection["selected_arm_id"]
    conditioning_id = conditioning_selection["selected_arm_id"]
    if not screen._exact_json_equal(
        selected.get("scheduler_arm_id"), scheduler_id
    ) or not screen._exact_json_equal(
        selected.get("conditioning_arm_id"), conditioning_id
    ):
        raise ScaleUpValidationError("registry selected design differs from decisions")
    scheduler_stage = screen._stage(screen_registry, "scheduler")
    scheduler_arm = screen._arm(scheduler_stage, str(scheduler_id))
    conditioning_stage = screen._stage(screen_registry, "conditioning")
    conditioning_arm = screen._arm(conditioning_stage, str(conditioning_id))
    if not screen._exact_json_equal(
        selected.get("scheduler"), scheduler_arm.get("scheduler")
    ):
        raise ScaleUpValidationError("selected scheduler specification differs")
    if not screen._exact_json_equal(
        selected.get("conditioner"), conditioning_arm.get("conditioner")
    ):
        raise ScaleUpValidationError("selected conditioner specification differs")
    config_entry = screen._registered_config(
        conditioning_arm, scheduler_arm_id=str(scheduler_id)
    )
    if not screen._exact_json_equal(
        selected.get("screen_config"), config_entry.get("config")
    ):
        raise ScaleUpValidationError("selected screen config reference differs")
    parsed = config_entry.get("parsed_config")
    if not isinstance(parsed, Mapping):
        raise ScaleUpValidationError("selected screen config was not parsed")
    return parsed


def _validate_common_training(
    value: object, *, screen_registry: screen.ValidatedRegistry
) -> Mapping[str, Any]:
    common = _mapping(value, "common training")
    _exact_keys(
        common,
        {
            "optimizer_updates",
            "training_seed",
            "gpu_count",
            "global_batch_size",
            "micro_batch_size_per_process",
            "accumulate_grad_batches",
            "effective_global_batch_size",
            "num_workers",
            "exclude_special_tokens",
            "reseed_after_model_initialization",
            "initialization",
            "gpu_safety_policy",
            "common_resolved_config_sha256",
            "artifact_schema_versions",
        },
        "common training",
    )
    if _integer(common.get("optimizer_updates"), "optimizer updates", minimum=1) != (
        EXPECTED_MAX_STEPS
    ):
        raise ScaleUpValidationError("scale-up must use exactly 1,000 updates")
    if _integer(common.get("training_seed"), "training seed", minimum=0) != (
        EXPECTED_TRAINING_SEED
    ):
        raise ScaleUpValidationError("scale-up seed differs from the screen seed")
    gpu_count = validate_gpu_count(common.get("gpu_count"))
    global_batch = _integer(common.get("global_batch_size"), "global batch", minimum=1)
    micro_batch = _integer(
        common.get("micro_batch_size_per_process"), "micro batch", minimum=1
    )
    expected_global_batch = default_global_batch_size(
        gpu_count, micro_batch_size=2
    )
    if micro_batch != 2 or global_batch != expected_global_batch:
        raise ScaleUpValidationError(
            "scale-up batch plan must use m=2 and the registered W-specific global batch"
        )
    expected_accumulation = exact_accumulation_steps(
        global_batch, micro_batch, gpu_count
    )
    accumulation = _integer(
        common.get("accumulate_grad_batches"), "accumulation steps", minimum=1
    )
    effective_global_batch = _integer(
        common.get("effective_global_batch_size"),
        "effective global batch",
        minimum=1,
    )
    if accumulation != expected_accumulation or effective_global_batch != global_batch:
        raise ScaleUpValidationError("scale-up global-batch arithmetic is not exact")
    if _integer(common.get("num_workers"), "num_workers", minimum=0) != 1:
        raise ScaleUpValidationError("scale-up num_workers must equal one")
    if common.get("exclude_special_tokens") is not False:
        raise ScaleUpValidationError("scale-up must retain the full-vocabulary E process")
    if common.get("reseed_after_model_initialization") is not True:
        raise ScaleUpValidationError(
            "matched scale-up must retain the selected conditioning reseed protocol"
        )
    if not screen._exact_json_equal(
        common.get("initialization"),
        screen_registry.data["common_training"]["initialization"],
    ):
        raise ScaleUpValidationError("scale-up initialization differs from the screen")
    safety = _mapping(common.get("gpu_safety_policy"), "GPU safety policy")
    expected_safety = {
        "max_utilization_percent": MAX_SAFE_UTILIZATION_PERCENT,
        "utilization_comparison": "strictly_less_than",
        "min_free_memory_mib": MIN_SAFE_FREE_MEMORY_MIB,
        "active_compute_processes_allowed": True,
        "compute_mode_prohibited_allowed": False,
    }
    if not screen._exact_json_equal(dict(safety), expected_safety):
        raise ScaleUpValidationError("GPU safety policy is not the exact shared-host policy")
    _sha256(
        common.get("common_resolved_config_sha256"), "common resolved-config digest"
    )
    expected_schemas = {
        "launch_manifest": 2,
        "runtime_config": 2,
        "training_summary": 5,
        "exit_receipt": 5,
        "predecessor_receipt_binding": 1,
        "matched_panel": 2,
    }
    if not screen._exact_json_equal(
        common.get("artifact_schema_versions"), expected_schemas
    ):
        raise ScaleUpValidationError("scale-up artifact schema contract differs")
    return common


def _validate_members(
    value: object,
    *,
    selected_config: Mapping[str, Any],
    common: Mapping[str, Any],
    publication: Mapping[str, Any],
    loader: BlobLoader,
    git_blob_loader: GitBlobLoader,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_VARIANTS):
        raise ScaleUpValidationError("scale-up members must contain exact R/S/E entries")
    config_revision = str(publication["config_revision"])
    expected_config_paths = set(publication["config_paths"])
    observed_paths: set[str] = set()
    observed_outputs: set[str] = set()
    observed_runs: set[str] = set()
    common_hashes: set[str] = set()
    normalized: list[Mapping[str, Any]] = []
    for position, (raw_member, variant, slug) in enumerate(
        zip(value, EXPECTED_VARIANTS, EXPECTED_SLUGS, strict=True)
    ):
        member = _mapping(raw_member, f"member {position}")
        _exact_keys(
            member,
            {
                "position",
                "slug",
                "training_variant",
                "prior_variant",
                "comparison_role",
                "run_name",
                "output_directory",
                "config",
            },
            f"member {position}",
        )
        expected_values = {
            "position": position,
            "slug": slug,
            "training_variant": variant,
            "prior_variant": EXPECTED_PRIORS[variant],
            "comparison_role": EXPECTED_ROLES[variant],
        }
        for field, expected in expected_values.items():
            if not screen._exact_json_equal(member.get(field), expected):
                raise ScaleUpValidationError(f"member {position} {field} differs")
        run_name = _string(member.get("run_name"), f"member {position} run name")
        expected_run_name = (
            f"scaleup-w{common['gpu_count']}-{slug}-"
            f"{str(publication['framework_revision'])[:12]}"
        )
        if (
            SAFE_ID.fullmatch(run_name) is None
            or run_name in observed_runs
            or run_name != expected_run_name
        ):
            raise ScaleUpValidationError("scale-up run names are invalid or duplicated")
        output = _relative_path(
            member.get("output_directory"), f"member {position} output directory"
        )
        expected_output = f"output/udlm/{expected_run_name}"
        if (
            PurePosixPath(output).parts[:2] != ("output", "udlm")
            or output in observed_outputs
            or output != expected_output
        ):
            raise ScaleUpValidationError("scale-up output directories are invalid")
        config_ref = _mapping(member.get("config"), f"member {position} config")
        payload, _parsed_strict = _load_reference(
            config_ref,
            label=f"member {position} config",
            revision=config_revision,
            loader=loader,
            git_blob_loader=git_blob_loader,
            require_canonical=True,
        )
        if not isinstance(payload, bytes):
            raise ScaleUpValidationError("resolved config bytes were not retained")
        try:
            parsed = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ScaleUpValidationError("resolved config is not ordinary JSON") from error
        if not isinstance(parsed, dict):
            raise ScaleUpValidationError("resolved config must be an object")
        _require_deterministic_json_bytes(
            payload, parsed, label=f"member {position} config"
        )
        expected = derive_scale_up_config(
            selected_config,
            training_variant=variant,
            gpu_count=int(common["gpu_count"]),
            global_batch_size=int(common["global_batch_size"]),
            micro_batch_size=int(common["micro_batch_size_per_process"]),
            num_workers=int(common["num_workers"]),
            output_directory=output,
        )
        if not screen._exact_json_equal(
            parsed, expected
        ) or canonical_json_sha256(parsed) != canonical_json_sha256(expected):
            raise ScaleUpValidationError(
                f"member {position} is not the exact selected-design derivative"
            )
        common_hashes.add(matched_config_sha256(parsed))
        path = str(config_ref["relative_path"])
        expected_config_path = (
            PurePosixPath(
                CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=common["gpu_count"])
            )
            / CONFIG_FILENAMES[variant]
        ).as_posix()
        if path != expected_config_path:
            raise ScaleUpValidationError("member config path is not canonical")
        if path in observed_paths:
            raise ScaleUpValidationError("scale-up config references are duplicated")
        observed_paths.add(path)
        observed_outputs.add(output)
        observed_runs.add(run_name)
        normalized.append(member)
    if observed_paths != expected_config_paths:
        raise ScaleUpValidationError("member config paths differ from publication paths")
    if common_hashes != {common["common_resolved_config_sha256"]}:
        raise ScaleUpValidationError("R/S/E configs are not exactly matched")
    return tuple(normalized)


def validate_registry(
    value: object,
    *,
    loader: BlobLoader = screen.local_blob_loader,
    git_blob_loader: GitBlobLoader = screen.git_blob_loader,
    git_ancestor_checker: GitAncestorChecker = screen.git_ancestor_checker,
    git_sole_parent_checker: GitSoleParentChecker = screen.git_sole_parent_checker,
    git_tree_paths_loader: GitTreePathsLoader = screen.git_tree_paths_loader,
    git_pushed_checker: GitPushedChecker = screen.git_pushed_checker,
    git_diff_checker: GitDiffChecker = screen.git_diff_checker,
    changed_paths_loader: GitChangedPathsLoader = git_changed_paths_loader,
) -> tuple[
    Mapping[str, Any],
    screen.ValidatedRegistry,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    registry = _mapping(value, "scale-up registry")
    _exact_keys(
        registry,
        {
            "schema_version",
            "registry_id",
            "status",
            "claim_scope",
            "firewall",
            "publication",
            "source",
            "screen_authority",
            "selected_design",
            "common_training",
            "members",
        },
        "scale-up registry",
    )
    exact = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_id": EXPECTED_REGISTRY_ID,
        "status": EXPECTED_STATUS,
        "claim_scope": EXPECTED_CLAIM_SCOPE,
    }
    for field, expected in exact.items():
        if not screen._exact_json_equal(registry.get(field), expected):
            raise ScaleUpValidationError(f"scale-up registry {field} differs")
    expected_firewall = {
        "generation_metrics_allowed": False,
        "superiority_evidence_eligible": False,
        "candidate_lock_eligible": False,
        "failed_or_missing_receipt_policy": "incomplete_no_panel",
        "unregistered_attempts_allowed": False,
    }
    if not screen._exact_json_equal(registry.get("firewall"), expected_firewall):
        raise ScaleUpValidationError("scale-up firewall differs")
    publication = _validate_publication(
        registry.get("publication"),
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_tree_paths_loader=git_tree_paths_loader,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
        changed_paths_loader=changed_paths_loader,
    )
    source = _mapping(registry.get("source"), "source")
    _exact_keys(source, {"revision", "clean", "pushed", "blobs"}, "source")
    if (
        source.get("revision") != publication["config_revision"]
        or source.get("clean") is not True
        or source.get("pushed") is not True
    ):
        raise ScaleUpValidationError("source is not the clean pushed config revision")
    blobs = source.get("blobs")
    if not isinstance(blobs, list) or len(blobs) != len(SOURCE_PATHS):
        raise ScaleUpValidationError("source closure has the wrong size")
    observed_source_paths: list[str] = []
    for index, blob in enumerate(blobs):
        reference = _mapping(blob, f"source blob {index}")
        payload, _ = _load_reference(
            reference,
            label=f"source blob {index}",
            revision=str(source["revision"]),
            loader=loader,
            git_blob_loader=git_blob_loader,
        )
        del payload
        observed_source_paths.append(str(reference["relative_path"]))
    if tuple(observed_source_paths) != SOURCE_PATHS:
        raise ScaleUpValidationError("source closure paths or ordering differ")
    screen_registry, scheduler_selection, conditioning_selection = (
        _validate_screen_authority(
            registry.get("screen_authority"),
            publication=publication,
            loader=loader,
            git_blob_loader=git_blob_loader,
            git_ancestor_checker=git_ancestor_checker,
            git_sole_parent_checker=git_sole_parent_checker,
            git_tree_paths_loader=git_tree_paths_loader,
            git_pushed_checker=git_pushed_checker,
            git_diff_checker=git_diff_checker,
            changed_paths_loader=changed_paths_loader,
        )
    )
    selected = _mapping(registry.get("selected_design"), "selected design")
    selected_config = _selected_screen_config(
        selected,
        screen_registry=screen_registry,
        scheduler_selection=scheduler_selection,
        conditioning_selection=conditioning_selection,
    )
    common = _validate_common_training(
        registry.get("common_training"), screen_registry=screen_registry
    )
    _validate_members(
        registry.get("members"),
        selected_config=selected_config,
        common=common,
        publication=publication,
        loader=loader,
        git_blob_loader=git_blob_loader,
    )
    return registry, screen_registry, scheduler_selection, conditioning_selection


def load_validated_registry(
    payload: bytes,
    *,
    relative_path: str,
    expected_raw_sha256: str,
    expected_canonical_sha256: str,
    loader: BlobLoader = screen.local_blob_loader,
    git_blob_loader: GitBlobLoader = screen.git_blob_loader,
    git_ancestor_checker: GitAncestorChecker = screen.git_ancestor_checker,
    git_sole_parent_checker: GitSoleParentChecker = screen.git_sole_parent_checker,
    git_tree_paths_loader: GitTreePathsLoader = screen.git_tree_paths_loader,
    git_pushed_checker: GitPushedChecker = screen.git_pushed_checker,
    git_diff_checker: GitDiffChecker = screen.git_diff_checker,
    changed_paths_loader: GitChangedPathsLoader = git_changed_paths_loader,
) -> ValidatedScaleUpRegistry:
    raw = hashlib.sha256(payload).hexdigest()
    if raw != _sha256(expected_raw_sha256, "expected registry raw digest"):
        raise ScaleUpValidationError("scale-up registry raw digest differs")
    parsed = _mapping(
        screen.strict_json_loads(payload, label="scale-up registry"),
        "scale-up registry",
    )
    _require_deterministic_json_bytes(payload, parsed, label="scale-up registry")
    canonical = canonical_json_sha256(parsed)
    if canonical != _sha256(
        expected_canonical_sha256, "expected registry canonical digest"
    ):
        raise ScaleUpValidationError("scale-up registry canonical digest differs")
    normalized, screen_registry, scheduler, conditioning = validate_registry(
        parsed,
        loader=loader,
        git_blob_loader=git_blob_loader,
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_tree_paths_loader=git_tree_paths_loader,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
        changed_paths_loader=changed_paths_loader,
    )
    relative = PurePosixPath(
        _relative_path(relative_path, "scale-up registry path", suffix=".json")
    )
    expected_relative = REGISTRY_RELATIVE_PATH_TEMPLATE.format(
        gpu_count=normalized["common_training"]["gpu_count"]
    )
    if relative.as_posix() != expected_relative:
        raise ScaleUpValidationError(
            "scale-up registry path differs from its frozen world size"
        )
    return ValidatedScaleUpRegistry(
        data=normalized,
        relative_path=relative,
        raw_sha256=raw,
        raw_size_bytes=len(payload),
        canonical_sha256=canonical,
        screen_registry=screen_registry,
        scheduler_selection=scheduler,
        conditioning_selection=conditioning,
    )


def expected_manifest_binding(
    registry: ValidatedScaleUpRegistry, *, position: int
) -> dict[str, Any]:
    """Build the exact schema-1 ``selection_bound_scale_up`` manifest value."""

    position = _integer(position, "scale-up member position", minimum=0, maximum=2)
    member = _mapping(registry.data["members"][position], "scale-up member")
    authority = _mapping(registry.data["screen_authority"], "screen authority")
    selected = _mapping(registry.data["selected_design"], "selected design")
    return {
        "schema_version": MANIFEST_BINDING_SCHEMA_VERSION,
        "registry": registry.reference,
        "screen_authority": {
            name: dict(_mapping(authority[name], f"screen authority {name}"))
            for name in (
                "scheduler_evidence",
                "scheduler_selection",
                "conditioning_evidence",
                "conditioning_selection",
            )
        },
        "selected_design": {
            "scheduler_arm_id": selected["scheduler_arm_id"],
            "conditioning_arm_id": selected["conditioning_arm_id"],
        },
        "member": {
            "arm_id": EXPECTED_SLUGS[position].upper(),
            "arm_order": [slug.upper() for slug in EXPECTED_SLUGS],
            "position": position,
            "training_variant": EXPECTED_VARIANTS[position],
            "training_variant_order": list(EXPECTED_VARIANTS),
            "registered_config": dict(
                _mapping(member["config"], "registered member config")
            ),
            "registered_config_source_revision": registry.data["publication"][
                "config_revision"
            ],
        },
    }


def validate_manifest_binding(
    value: object,
    *,
    registry: ValidatedScaleUpRegistry,
    position: int,
) -> dict[str, Any]:
    """Require exact manifest authority, selected design, and member advance."""

    binding = _mapping(value, "selection-bound scale-up manifest binding")
    expected = expected_manifest_binding(registry, position=position)
    if not screen._exact_json_equal(dict(binding), expected):
        raise ScaleUpValidationError(
            "selection-bound scale-up manifest binding differs from the registry"
        )
    return expected


def _stable_bytes(path: Path) -> bytes:
    loaded = screen._read_stable_file(path, retain=True)
    if not isinstance(loaded, bytes):
        raise ScaleUpValidationError("registry bytes were not retained")
    return loaded


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-canonical-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = args.registry.resolve(strict=True)
    try:
        relative = path.relative_to(REPOSITORY_ROOT.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise ScaleUpValidationError("registry must be inside the repository") from error
    validated = load_validated_registry(
        _stable_bytes(path),
        relative_path=relative,
        expected_raw_sha256=args.expected_registry_sha256,
        expected_canonical_sha256=args.expected_registry_canonical_sha256,
    )
    summary = {
        "status": "validated",
        "registry": validated.reference,
        "source_revision": validated.data["source"]["revision"],
        "gpu_count": validated.data["common_training"]["gpu_count"],
        "optimizer_updates": EXPECTED_MAX_STEPS,
        "selected_scheduler_arm_id": validated.scheduler_selection["selected_arm_id"],
        "selected_conditioning_arm_id": validated.conditioning_selection[
            "selected_arm_id"
        ],
        "member_order": list(EXPECTED_VARIANTS),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
