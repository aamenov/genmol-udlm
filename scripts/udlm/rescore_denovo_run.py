"""Independently re-decode and re-score one schema-8 de-novo run.

The benchmark summary is not treated as a source of metric truth.  This module
retains the exact summary/CSV bytes, verifies their caller-pinned digests, takes
only ``raw_model_text`` from the CSV, and runs the current decoder and metric
implementation again.  Every one of the 21 CSV fields, both metric branches,
and all failure counts must agree with the retained evidence.

The byte-oriented :func:`rescore_denovo_run` callable performs no file writes,
model loading, sample generation, or GPU operation.  Production callers should
use :func:`invoke_rescore_worker`, which starts a fresh CPU-only interpreter for
each seed.  This is required because PyTDC diversity passes through set
iteration whose order is fixed only when ``PYTHONHASHSEED`` is set before
interpreter startup.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = REPOSITORY_ROOT / "src"
for _import_root in (REPOSITORY_SRC, REPOSITORY_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from scripts.exps.denovo import benchmark  # noqa: E402
from scripts.udlm import rescore_mdlm_baseline as baseline_rescore  # noqa: E402


SUMMARY_SCHEMA_VERSION = 8
HISTORICAL_MDLM_SUMMARY_SCHEMA_VERSION = 7
MAXIMUM_SUMMARY_SIZE_BYTES = 2 * 1024 * 1024
TOKEN_AUDIT_SCHEMA_VERSION = 1
MODEL_VOCAB_SIZE = 1_880
TOKENIZER_EFFECTIVE_SIZE = 1_882
MAXIMUM_TOKEN_AUDIT_ROWS = 1_000
MAXIMUM_TOKEN_AUDIT_COLUMNS = 256
CONTROL_TOKEN_IDS = {"unk": 0, "bos": 1, "eos": 2, "pad": 3, "mask": 4}
HISTORICAL_MDLM_CHECKPOINT_SHA256 = (
    "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
)
HISTORICAL_MDLM_CHECKPOINT_SIZE_BYTES = 1_396_998_679
HISTORICAL_MDLM_GLOBAL_STEP = 50_000
NUMERIC_ABSOLUTE_TOLERANCE = baseline_rescore.NUMERIC_ABSOLUTE_TOLERANCE
RescoreValidationError = baseline_rescore.RescoreValidationError
RetainedArtifact = baseline_rescore.RetainedArtifact
WORKER_RESULT_PREFIX = "GENMOL_DENOVO_RESCORE_WORKER_RESULT="

EXPECTED_IDENTITY_ARGUMENTS = (
    "expected_checkpoint_sha256",
    "expected_config_sha256",
    "expected_source_revision",
    "expected_runner_sha256",
    "expected_sampler_source_sha256",
    "expected_ema_source_sha256",
    "expected_implementation_inputs_sha256",
    "expected_metric_inputs_sha256",
)

SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "seed",
        "num_samples",
        "run",
        "checkpoint",
        "config",
        "metrics",
        "failure_counts",
        "runtime_seconds",
        "environment",
        "git",
        "implementation_inputs",
        "metric_inputs",
        "tokenizer",
        "artifacts",
        "sampled_token_control_audit",
    }
)
HISTORICAL_MDLM_SUMMARY_FIELDS = SUMMARY_FIELDS - {"sampled_token_control_audit"}
RUN_FIELDS = frozenset(
    {
        "seed",
        "requested_sample_count",
        "evaluation_tier",
        "final_protocol_eligible",
        "started_at_utc",
        "completed_at_utc",
        "one_seed_per_invocation",
        "single_generation_batch",
        "generation_protocol",
        "command",
        "seed_configuration",
        "execution_authority",
    }
)
HISTORICAL_MDLM_RUN_FIELDS = RUN_FIELDS - {"execution_authority"}
CHECKPOINT_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "mtime_utc",
        "byte_identity_verified_before_and_after_load",
        "global_step",
        "epoch",
        "diffusion_type",
        "udlm_inference_eps",
        "udlm_exclude_special_tokens",
        "udlm_prior_variant",
        "udlm_prior_metadata",
        "udlm_prior_metadata_sha256",
    }
)
CONFIG_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "git_tracking",
        "sampling_sha256",
        "effective_sha256",
        "source",
        "effective",
        "sampling",
    }
)
GIT_FIELDS = frozenset(
    {
        "repo_root",
        "commit",
        "branch",
        "remote_origin",
        "dirty",
        "status_porcelain",
        "runner_sha256",
        "upstream",
        "expected_source_revision",
        "clean_pushed_source_verified_before_and_after_run",
    }
)
GENERATION_PROTOCOL_FIELDS = frozenset(
    {
        "diffusion_type",
        "nfe",
        "nfe_definition",
        "num_steps",
        "num_steps_source",
        "inference_eps",
        "exclude_special_tokens",
        "prior_variant",
        "prior_metadata_sha256",
        "temperature",
        "randomness",
        "raw_loo_top_p",
        "randomness_used_by_sampler",
        "model_use_bracket_safe",
        "single_generation_batch",
        "released_safe_fix",
        "released_largest_component",
        "strict_safe_fix",
        "inference_weights",
    }
)
HISTORICAL_MDLM_GENERATION_PROTOCOL_FIELDS = GENERATION_PROTOCOL_FIELDS - {
    "raw_loo_top_p"
}
GIBBS_CORRECTOR_PROTOCOL_FIELDS = frozenset(
    {
        "gibbs_corrector",
        "predictor_transitions_per_molecule",
        "corrector_steps_per_molecule",
    }
)
HISTORICAL_MDLM_IMPLEMENTATION_INPUT_NAMES = frozenset(
    benchmark.IMPLEMENTATION_INPUT_PATHS
) - {"artifact_io_source"}
IMPLEMENTATION_INPUT_NAMES = HISTORICAL_MDLM_IMPLEMENTATION_INPUT_NAMES | {
    "artifact_io_source"
}
RUNTIME_FIELDS = frozenset(
    {
        "model_load_and_device_move",
        "model_sampling_and_tokenizer",
        "sampled_token_control_audit",
        "released_postprocessing",
        "generation",
        "decode_and_metrics",
        "total_before_summary_write",
    }
)
HISTORICAL_MDLM_RUNTIME_FIELDS = RUNTIME_FIELDS - {"sampled_token_control_audit"}
ARTIFACT_BUNDLE = {
    "publication_api": "scripts.artifact_io.publish_bundle_exclusive",
    "ordinary_members": ["raw_samples.csv"],
    "completion_member": "summary.json",
    "exclusive_no_clobber": True,
    "completion_linked_last": True,
    "precompletion_failure_rollback": "exact_owned_members_only",
}
EXECUTION_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "launch_authority",
        "launch_authority_canonical_sha256",
        "output_directory_descriptor_retained_until_after_bundle_publication",
        "validated_before_model_import",
        "revalidated_immediately_before_publication",
    }
)
LAUNCH_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "generation_lease",
        "artifact_io_source",
        "output_directory",
        "command",
        "command_sha256",
    }
)
GENERATION_LEASE_FIELDS = frozenset(
    {"path", "relative_path", "sha256", "device", "inode", "owner_token"}
)
AUTHORITY_SOURCE_FIELDS = frozenset({"path", "sha256", "device", "inode"})
AUTHORITY_OUTPUT_FIELDS = frozenset({"path", "relative_path", "device", "inode"})

TOKEN_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "rows",
        "columns",
        "model_vocab_size",
        "tokenizer_effective_size",
        "control_token_ids",
        "sampler_input_ids",
        "final_sampled_ids",
        "editable_mask",
        "control_token_counts",
    }
)
UINT16_ARRAY_FIELDS = frozenset(
    {
        "encoding",
        "dtype",
        "byte_order",
        "array_order",
        "compression",
        "element_count",
        "decoded_byte_count",
        "decoded_sha256",
        "data_base64",
    }
)
EDITABLE_MASK_FIELDS = frozenset(
    {
        "encoding",
        "packing",
        "bit_order",
        "array_order",
        "compression",
        "logical_bit_count",
        "decoded_byte_count",
        "unused_tail_bit_count",
        "decoded_sha256",
        "data_base64",
    }
)
CONTROL_COUNT_FIELDS = frozenset(
    {
        "sampler_input_all_positions",
        "final_sampled_all_positions",
        "final_sampled_editable_positions",
    }
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RescoreValidationError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise RescoreValidationError(
            f"{label} fields differ: {sorted(value)} != {sorted(expected)}"
        )


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise RescoreValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise RescoreValidationError(f"{label} must be at least {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RescoreValidationError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _git_revision(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RescoreValidationError(f"{label} must be 40 lowercase hexadecimal digits")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RescoreValidationError(f"{label} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RescoreValidationError(
            f"{label} must be a timezone-aware ISO timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RescoreValidationError(f"{label} must include a timezone")
    return parsed


def _assert_expected(actual: Any, expected: Any, label: str) -> None:
    if expected is not None and actual != expected:
        raise RescoreValidationError(f"{label} differs: {actual!r} != {expected!r}")


def _canonical_base64(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise RescoreValidationError(f"{label} must be an RFC 4648 base64 string")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RescoreValidationError(
            f"{label} must be valid RFC 4648 base64"
        ) from error
    canonical = base64.b64encode(decoded).decode("ascii")
    if value != canonical:
        raise RescoreValidationError(f"{label} is not canonical RFC 4648 base64")
    return decoded


def _validate_encoded_uint16_array(
    value: Any,
    *,
    label: str,
    expected_elements: int,
) -> list[int]:
    record = _mapping(value, label)
    _exact_keys(record, UINT16_ARRAY_FIELDS, label)
    expected_identity = {
        "encoding": "rfc4648_base64",
        "dtype": "uint16",
        "byte_order": "little",
        "array_order": "C",
        "compression": "none",
    }
    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            raise RescoreValidationError(f"{label}.{field} differs")
    element_count = _integer(
        record.get("element_count"), f"{label}.element_count", minimum=0
    )
    decoded_byte_count = _integer(
        record.get("decoded_byte_count"),
        f"{label}.decoded_byte_count",
        minimum=0,
    )
    if element_count != expected_elements:
        raise RescoreValidationError(f"{label}.element_count differs from shape")
    if decoded_byte_count != 2 * expected_elements:
        raise RescoreValidationError(f"{label}.decoded_byte_count differs from shape")
    decoded = _canonical_base64(record.get("data_base64"), f"{label}.data_base64")
    if len(decoded) != decoded_byte_count:
        raise RescoreValidationError(f"{label} decoded byte length differs")
    digest = _sha256(record.get("decoded_sha256"), f"{label}.decoded_sha256")
    if hashlib.sha256(decoded).hexdigest() != digest:
        raise RescoreValidationError(f"{label}.decoded_sha256 differs")
    return [
        int.from_bytes(decoded[offset : offset + 2], "little", signed=False)
        for offset in range(0, len(decoded), 2)
    ]


def _validate_editable_mask(
    value: Any,
    *,
    expected_bits: int,
) -> list[bool]:
    label = "sampled_token_control_audit.editable_mask"
    record = _mapping(value, label)
    _exact_keys(record, EDITABLE_MASK_FIELDS, label)
    expected_identity = {
        "encoding": "rfc4648_base64",
        "packing": "one_bit_per_position",
        "bit_order": "msb0",
        "array_order": "C",
        "compression": "none",
    }
    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            raise RescoreValidationError(f"{label}.{field} differs")
    logical_bits = _integer(
        record.get("logical_bit_count"), f"{label}.logical_bit_count", minimum=0
    )
    byte_count = _integer(
        record.get("decoded_byte_count"), f"{label}.decoded_byte_count", minimum=0
    )
    unused_tail_bits = _integer(
        record.get("unused_tail_bit_count"),
        f"{label}.unused_tail_bit_count",
        minimum=0,
    )
    expected_bytes = (expected_bits + 7) // 8
    expected_unused = expected_bytes * 8 - expected_bits
    if logical_bits != expected_bits or byte_count != expected_bytes:
        raise RescoreValidationError(f"{label} length differs from shape")
    if unused_tail_bits != expected_unused or unused_tail_bits > 7:
        raise RescoreValidationError(f"{label}.unused_tail_bit_count differs")
    decoded = _canonical_base64(record.get("data_base64"), f"{label}.data_base64")
    if len(decoded) != byte_count:
        raise RescoreValidationError(f"{label} decoded byte length differs")
    digest = _sha256(record.get("decoded_sha256"), f"{label}.decoded_sha256")
    if hashlib.sha256(decoded).hexdigest() != digest:
        raise RescoreValidationError(f"{label}.decoded_sha256 differs")
    if unused_tail_bits and decoded and decoded[-1] & ((1 << unused_tail_bits) - 1):
        raise RescoreValidationError(f"{label} unused tail bits must be zero")
    return [
        bool(decoded[index // 8] & (1 << (7 - index % 8)))
        for index in range(expected_bits)
    ]


def _control_counts(values: Sequence[int]) -> dict[str, int]:
    return {
        name: sum(value == token_id for value in values)
        for name, token_id in CONTROL_TOKEN_IDS.items()
    }


def _validate_control_counts(
    value: Any,
    *,
    sampler_input_ids: Sequence[int],
    final_sampled_ids: Sequence[int],
    editable: Sequence[bool],
) -> dict[str, dict[str, int]]:
    label = "sampled_token_control_audit.control_token_counts"
    record = _mapping(value, label)
    _exact_keys(record, CONTROL_COUNT_FIELDS, label)
    expected = {
        "sampler_input_all_positions": _control_counts(sampler_input_ids),
        "final_sampled_all_positions": _control_counts(final_sampled_ids),
        "final_sampled_editable_positions": _control_counts(
            [
                token_id
                for token_id, is_editable in zip(final_sampled_ids, editable)
                if is_editable
            ]
        ),
    }
    for scope, expected_counts in expected.items():
        counts = _mapping(record.get(scope), f"{label}.{scope}")
        _exact_keys(counts, frozenset(CONTROL_TOKEN_IDS), f"{label}.{scope}")
        for name, expected_count in expected_counts.items():
            observed = _integer(counts.get(name), f"{label}.{scope}.{name}", minimum=0)
            if observed != expected_count:
                raise RescoreValidationError(
                    f"{label}.{scope}.{name} differs from decoded IDs"
                )
    return expected


def validate_sampled_token_control_audit(
    value: Any,
    *,
    expected_rows: int,
    exclude_special_tokens: bool,
    raw_model_texts: Sequence[str],
    tokenizer_batch_decode: Callable[..., Sequence[str]],
) -> dict[str, Any]:
    """Validate and independently decode the complete schema-8 token audit."""

    label = "sampled_token_control_audit"
    audit = _mapping(value, label)
    _exact_keys(audit, TOKEN_AUDIT_FIELDS, label)
    if audit.get("schema_version") != TOKEN_AUDIT_SCHEMA_VERSION:
        raise RescoreValidationError(f"{label}.schema_version differs")
    rows = _integer(audit.get("rows"), f"{label}.rows", minimum=1)
    columns = _integer(audit.get("columns"), f"{label}.columns", minimum=1)
    if rows != expected_rows or rows > MAXIMUM_TOKEN_AUDIT_ROWS:
        raise RescoreValidationError(f"{label}.rows differs or exceeds 1000")
    if columns > MAXIMUM_TOKEN_AUDIT_COLUMNS:
        raise RescoreValidationError(f"{label}.columns exceeds 256")
    if audit.get("model_vocab_size") != MODEL_VOCAB_SIZE:
        raise RescoreValidationError(f"{label}.model_vocab_size differs")
    if audit.get("tokenizer_effective_size") != TOKENIZER_EFFECTIVE_SIZE:
        raise RescoreValidationError(f"{label}.tokenizer_effective_size differs")
    control_ids = _mapping(audit.get("control_token_ids"), f"{label}.control_token_ids")
    if dict(control_ids) != CONTROL_TOKEN_IDS:
        raise RescoreValidationError(f"{label}.control_token_ids differs")

    element_count = rows * columns
    sampler_input_ids = _validate_encoded_uint16_array(
        audit.get("sampler_input_ids"),
        label=f"{label}.sampler_input_ids",
        expected_elements=element_count,
    )
    final_sampled_ids = _validate_encoded_uint16_array(
        audit.get("final_sampled_ids"),
        label=f"{label}.final_sampled_ids",
        expected_elements=element_count,
    )
    editable = _validate_editable_mask(
        audit.get("editable_mask"), expected_bits=element_count
    )
    for name, values in (
        ("sampler_input_ids", sampler_input_ids),
        ("final_sampled_ids", final_sampled_ids),
    ):
        if any(value < 0 or value >= MODEL_VOCAB_SIZE for value in values):
            raise RescoreValidationError(f"{label}.{name} contains an out-of-range ID")

    final_rows: list[list[int]] = []
    for row_index in range(rows):
        start = row_index * columns
        stop = start + columns
        input_row = sampler_input_ids[start:stop]
        final_row = final_sampled_ids[start:stop]
        editable_row = editable[start:stop]
        if input_row[0] != CONTROL_TOKEN_IDS["bos"]:
            raise RescoreValidationError(f"{label} row {row_index} lacks BOS")
        try:
            eos_index = input_row.index(CONTROL_TOKEN_IDS["eos"], 1)
        except ValueError as error:
            raise RescoreValidationError(
                f"{label} row {row_index} lacks EOS"
            ) from error
        if eos_index <= 1:
            raise RescoreValidationError(
                f"{label} row {row_index} has no editable MASK body"
            )
        expected_input = (
            [CONTROL_TOKEN_IDS["bos"]]
            + [CONTROL_TOKEN_IDS["mask"]] * (eos_index - 1)
            + [CONTROL_TOKEN_IDS["eos"]]
            + [CONTROL_TOKEN_IDS["pad"]] * (columns - eos_index - 1)
        )
        if input_row != expected_input:
            raise RescoreValidationError(
                f"{label} row {row_index} is not BOS MASK+ EOS PAD*"
            )
        expected_editable = [value == CONTROL_TOKEN_IDS["mask"] for value in input_row]
        if editable_row != expected_editable:
            raise RescoreValidationError(
                f"{label} row {row_index} editable mask differs from input MASK"
            )
        for column_index, (before, after, is_editable) in enumerate(
            zip(input_row, final_row, editable_row)
        ):
            if not is_editable and before != after:
                raise RescoreValidationError(
                    f"{label} immutable ID changed at row {row_index}, "
                    f"column {column_index}"
                )
            if (
                is_editable
                and exclude_special_tokens
                and after in CONTROL_TOKEN_IDS.values()
            ):
                raise RescoreValidationError(
                    f"{label} editable special ID violates exclusion policy"
                )
        final_rows.append(final_row)

    counts = _validate_control_counts(
        audit.get("control_token_counts"),
        sampler_input_ids=sampler_input_ids,
        final_sampled_ids=final_sampled_ids,
        editable=editable,
    )
    if len(raw_model_texts) != rows or any(
        not isinstance(value, str) for value in raw_model_texts
    ):
        raise RescoreValidationError("raw_model_text rows differ from token audit")
    decoded = tokenizer_batch_decode(final_rows, skip_special_tokens=True)
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        raise RescoreValidationError("tokenizer batch_decode returned an invalid value")
    normalized_decoded = [str(value) for value in decoded]
    if normalized_decoded != list(raw_model_texts):
        raise RescoreValidationError(
            "tokenizer batch_decode(final_sampled_ids, skip_special_tokens=True) "
            "differs from raw_samples.csv raw_model_text order"
        )
    return {
        "schema_version": TOKEN_AUDIT_SCHEMA_VERSION,
        "rows": rows,
        "columns": columns,
        "sampler_input_ids_sha256": audit["sampler_input_ids"]["decoded_sha256"],
        "final_sampled_ids_sha256": audit["final_sampled_ids"]["decoded_sha256"],
        "editable_mask_sha256": audit["editable_mask"]["decoded_sha256"],
        "control_token_counts": counts,
        "exact_batch_decode_match": True,
    }


def _nonnegative_identity(value: Any, label: str) -> int:
    return _integer(value, label, minimum=0)


def _validate_execution_authority(
    value: Any,
    *,
    run_command: Sequence[str],
    checkpoint_path: str,
    checkpoint_sha256: str,
    source_revision: str,
    config_path: str,
    config_sha256: str,
    expected_sample_count: int,
    expected_seed: int,
    environment: Mapping[str, Any],
    implementation_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    label = "run.execution_authority"
    execution = _mapping(value, label)
    _exact_keys(execution, EXECUTION_AUTHORITY_FIELDS, label)
    if execution.get("schema_version") != 1:
        raise RescoreValidationError(f"{label}.schema_version differs")
    for flag in (
        "output_directory_descriptor_retained_until_after_bundle_publication",
        "validated_before_model_import",
        "revalidated_immediately_before_publication",
    ):
        if execution.get(flag) is not True:
            raise RescoreValidationError(f"{label}.{flag} must be true")

    authority = _mapping(execution.get("launch_authority"), f"{label}.launch_authority")
    _exact_keys(authority, LAUNCH_AUTHORITY_FIELDS, f"{label}.launch_authority")
    if authority.get("schema_version") != 1:
        raise RescoreValidationError(f"{label}.launch_authority.schema_version differs")
    authority_digest = _sha256(
        execution.get("launch_authority_canonical_sha256"),
        f"{label}.launch_authority_canonical_sha256",
    )
    if baseline_rescore.canonical_json_sha256(authority) != authority_digest:
        raise RescoreValidationError(f"{label} launch-authority digest differs")

    lease = _mapping(
        authority.get("generation_lease"), f"{label}.launch_authority.generation_lease"
    )
    _exact_keys(lease, GENERATION_LEASE_FIELDS, "generation lease authority")
    if lease.get("relative_path") != "output/.single_generation_job.lock":
        raise RescoreValidationError("generation lease relative path differs")
    lease_path = lease.get("path")
    if not isinstance(lease_path, str) or not Path(lease_path).is_absolute():
        raise RescoreValidationError("generation lease path must be absolute")
    if (
        Path(lease_path).as_posix()
        != (
            Path(lease_path).parents[1] / "output/.single_generation_job.lock"
        ).as_posix()
    ):
        raise RescoreValidationError("generation lease absolute/relative paths differ")
    lease_sha256 = _sha256(lease.get("sha256"), "generation lease sha256")
    owner_token = _sha256(lease.get("owner_token"), "generation lease owner token")
    _nonnegative_identity(lease.get("device"), "generation lease device")
    _nonnegative_identity(lease.get("inode"), "generation lease inode")

    artifact_source = _mapping(
        authority.get("artifact_io_source"),
        f"{label}.launch_authority.artifact_io_source",
    )
    _exact_keys(artifact_source, AUTHORITY_SOURCE_FIELDS, "artifact_io authority")
    artifact_source_path = artifact_source.get("path")
    if (
        not isinstance(artifact_source_path, str)
        or not Path(artifact_source_path).is_absolute()
        or Path(artifact_source_path).as_posix().endswith("/scripts/artifact_io.py")
        is not True
    ):
        raise RescoreValidationError("artifact_io authority path differs")
    artifact_source_sha256 = _sha256(
        artifact_source.get("sha256"), "artifact_io authority sha256"
    )
    _nonnegative_identity(artifact_source.get("device"), "artifact_io source device")
    _nonnegative_identity(artifact_source.get("inode"), "artifact_io source inode")
    implementation_artifact_source = _mapping(
        implementation_inputs.get("artifact_io_source"),
        "implementation_inputs.artifact_io_source",
    )
    if (
        implementation_artifact_source.get("path") != artifact_source_path
        or implementation_artifact_source.get("sha256") != artifact_source_sha256
    ):
        raise RescoreValidationError(
            "artifact_io launch authority differs from implementation input"
        )

    output = _mapping(
        authority.get("output_directory"),
        f"{label}.launch_authority.output_directory",
    )
    _exact_keys(output, AUTHORITY_OUTPUT_FIELDS, "output-directory authority")
    output_path = output.get("path")
    output_relative = output.get("relative_path")
    if (
        not isinstance(output_path, str)
        or not Path(output_path).is_absolute()
        or not isinstance(output_relative, str)
        or not output_relative
        or Path(output_relative).is_absolute()
        or ".." in Path(output_relative).parts
        or not output_relative.startswith("output/")
        or not output_path.endswith("/" + output_relative)
    ):
        raise RescoreValidationError("output-directory authority paths differ")
    output_device = _nonnegative_identity(
        output.get("device"), "output-directory device"
    )
    output_inode = _nonnegative_identity(output.get("inode"), "output-directory inode")

    authority_command = authority.get("command")
    if authority_command != list(run_command):
        raise RescoreValidationError(
            "launch-authority command differs from run.command"
        )
    expected_command = [
        run_command[0],
        run_command[1],
        "--checkpoint",
        checkpoint_path,
        "--expected-checkpoint-sha256",
        checkpoint_sha256,
        "--expected-source-revision",
        source_revision,
        "--config",
        config_path,
        "--expected-config-sha256",
        config_sha256,
        "--num-samples",
        str(expected_sample_count),
        "--seed",
        str(expected_seed),
        "--device",
        "cuda:0",
        "--output-dir",
        output_path,
        "--expected-output-directory-device",
        str(output_device),
        "--expected-output-directory-inode",
        str(output_inode),
    ]
    if list(run_command) != expected_command:
        raise RescoreValidationError("schema-8 run.command ordering or value differs")
    command_digest = _sha256(
        authority.get("command_sha256"), "launch-authority command sha256"
    )
    encoded_command = json.dumps(
        list(run_command), separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if hashlib.sha256(encoded_command).hexdigest() != command_digest:
        raise RescoreValidationError("launch-authority command digest differs")

    launch_environment = _mapping(
        environment.get("launch_environment"), "environment.launch_environment"
    )
    authority_json = json.dumps(
        dict(authority), separators=(",", ":"), sort_keys=True, ensure_ascii=True
    )
    expected_environment = {
        "GENMOL_BENCHMARK_GENERATION_LEASE_PATH": lease_path,
        "GENMOL_BENCHMARK_EXPECTED_GENERATION_LEASE_SHA256": lease_sha256,
        "GENMOL_BENCHMARK_GENERATION_LEASE_OWNER_TOKEN": owner_token,
        "GENMOL_BENCHMARK_LAUNCH_AUTHORITY_JSON": authority_json,
    }
    for key, expected in expected_environment.items():
        if launch_environment.get(key) != expected:
            raise RescoreValidationError(
                f"environment.launch_environment.{key} differs"
            )
    try:
        parsed_environment_authority = json.loads(
            launch_environment["GENMOL_BENCHMARK_LAUNCH_AUTHORITY_JSON"]
        )
    except (TypeError, json.JSONDecodeError) as error:
        raise RescoreValidationError(
            "launch authority environment JSON is invalid"
        ) from error
    if parsed_environment_authority != dict(authority):
        raise RescoreValidationError("parsed environment launch authority differs")
    return {
        "generation_lease_sha256": lease_sha256,
        "artifact_io_source_sha256": artifact_source_sha256,
        "output_directory": dict(output),
        "command_sha256": command_digest,
        "launch_authority_canonical_sha256": authority_digest,
    }


def _validate_summary_identity(
    summary: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_sample_count: int,
    raw_sha256: str,
    expected_checkpoint_sha256: str | None,
    expected_config_sha256: str | None,
    expected_source_revision: str | None,
    expected_runner_sha256: str | None,
    expected_sampler_source_sha256: str | None,
    expected_ema_source_sha256: str | None,
    expected_implementation_inputs_sha256: str | None,
    expected_metric_inputs_sha256: str | None,
) -> dict[str, Any]:
    """Validate identity-bearing schema-8 fields without trusting metrics."""

    for value, label in (
        (expected_checkpoint_sha256, "expected checkpoint SHA-256"),
        (expected_config_sha256, "expected config SHA-256"),
        (expected_runner_sha256, "expected runner SHA-256"),
        (expected_sampler_source_sha256, "expected sampler-source SHA-256"),
        (expected_ema_source_sha256, "expected EMA-source SHA-256"),
        (
            expected_implementation_inputs_sha256,
            "expected implementation-input map SHA-256",
        ),
        (expected_metric_inputs_sha256, "expected metric-input map SHA-256"),
    ):
        if value is not None:
            _sha256(value, label)
    if expected_source_revision is not None:
        _git_revision(expected_source_revision, "expected source revision")

    schema_version = summary.get("schema_version")
    historical_mdlm = schema_version == HISTORICAL_MDLM_SUMMARY_SCHEMA_VERSION
    _exact_keys(
        summary,
        HISTORICAL_MDLM_SUMMARY_FIELDS if historical_mdlm else SUMMARY_FIELDS,
        "benchmark summary",
    )
    if schema_version not in {
        SUMMARY_SCHEMA_VERSION,
        HISTORICAL_MDLM_SUMMARY_SCHEMA_VERSION,
    }:
        raise RescoreValidationError(
            "benchmark summary schema must equal 8, except for the exact pinned "
            "historical MDLM schema-7 baseline"
        )
    if summary.get("status") != "completed":
        raise RescoreValidationError("benchmark summary is not completed")
    if summary.get("seed") != expected_seed:
        raise RescoreValidationError("benchmark summary seed differs")
    if summary.get("num_samples") != expected_sample_count:
        raise RescoreValidationError("benchmark summary sample count differs")

    run = _mapping(summary.get("run"), "benchmark summary.run")
    _exact_keys(
        run,
        HISTORICAL_MDLM_RUN_FIELDS if historical_mdlm else RUN_FIELDS,
        "benchmark summary.run",
    )
    if (
        run.get("seed") != expected_seed
        or run.get("requested_sample_count") != expected_sample_count
    ):
        raise RescoreValidationError("benchmark run seed/sample aliases differ")
    expected_tier = "final" if expected_sample_count == 1_000 else "pilot"
    if (
        run.get("evaluation_tier") != expected_tier
        or run.get("final_protocol_eligible") is not (expected_sample_count == 1_000)
        or run.get("one_seed_per_invocation") is not True
        or run.get("single_generation_batch") is not True
    ):
        raise RescoreValidationError("benchmark run tier/batch contract differs")
    started_at = _timestamp(run.get("started_at_utc"), "run.started_at_utc")
    completed_at = _timestamp(run.get("completed_at_utc"), "run.completed_at_utc")
    if completed_at < started_at:
        raise RescoreValidationError("benchmark completion predates its start")
    command = run.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise RescoreValidationError("run.command must be a nonempty string list")
    seed_configuration = _mapping(
        run.get("seed_configuration"), "run.seed_configuration"
    )
    expected_seed_configuration = {
        "seed": expected_seed,
        "seed_applied_immediately_before_generation": True,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda_all": True,
        "python_hash_seed": str(expected_seed),
    }
    if dict(seed_configuration) != expected_seed_configuration:
        raise RescoreValidationError("run.seed_configuration differs")

    checkpoint = _mapping(summary.get("checkpoint"), "benchmark checkpoint")
    expected_checkpoint_fields = CHECKPOINT_FIELDS
    if "udlm_denoiser_metadata" in checkpoint:
        expected_checkpoint_fields |= {"udlm_denoiser_metadata"}
    _exact_keys(checkpoint, expected_checkpoint_fields, "benchmark checkpoint")
    checkpoint_sha256 = _sha256(checkpoint.get("sha256"), "checkpoint.sha256")
    _assert_expected(checkpoint_sha256, expected_checkpoint_sha256, "checkpoint.sha256")
    checkpoint_size = _integer(
        checkpoint.get("size_bytes"), "checkpoint.size_bytes", minimum=1
    )
    checkpoint_step = _integer(
        checkpoint.get("global_step"), "checkpoint.global_step", minimum=0
    )
    epoch = checkpoint.get("epoch")
    if epoch is not None:
        _integer(epoch, "checkpoint.epoch", minimum=0)
    _timestamp(checkpoint.get("mtime_utc"), "checkpoint.mtime_utc")
    if checkpoint.get("byte_identity_verified_before_and_after_load") is not True:
        raise RescoreValidationError(
            "checkpoint bytes were not verified around loading"
        )
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise RescoreValidationError("checkpoint.path must be recorded")

    config = _mapping(summary.get("config"), "benchmark config")
    _exact_keys(config, CONFIG_FIELDS, "benchmark config")
    config_path = config.get("path")
    if not isinstance(config_path, str) or not config_path:
        raise RescoreValidationError("config.path must be recorded")
    config_sha256 = _sha256(config.get("sha256"), "config.sha256")
    sampling_sha256 = _sha256(config.get("sampling_sha256"), "config.sampling_sha256")
    effective_sha256 = _sha256(
        config.get("effective_sha256"), "config.effective_sha256"
    )
    _assert_expected(config_sha256, expected_config_sha256, "config.sha256")
    source_config = _mapping(config.get("source"), "config.source")
    sampling = _mapping(config.get("sampling"), "config.sampling")
    effective = _mapping(config.get("effective"), "config.effective")
    if baseline_rescore.canonical_json_sha256(sampling) != sampling_sha256:
        raise RescoreValidationError("config.sampling_sha256 is invalid")
    if baseline_rescore.canonical_json_sha256(effective) != effective_sha256:
        raise RescoreValidationError("config.effective_sha256 is invalid")
    if not historical_mdlm:
        try:
            normalized_sampling = benchmark.validate_sampling_config(sampling)
        except ValueError as error:
            raise RescoreValidationError(
                f"sampling config is invalid: {error}"
            ) from error
        if dict(sampling) != normalized_sampling:
            raise RescoreValidationError("sampling config is not canonical")
    x0_denoiser = sampling.get("parameterization", "raw_loo") == "x0_denoiser"
    if source_config.get("parameterization", "raw_loo") != sampling.get(
        "parameterization", "raw_loo"
    ):
        raise RescoreValidationError(
            "source/sampling denoiser parameterization differs"
        )
    try:
        benchmark.validate_denoiser_sampling_identity(checkpoint, sampling)
    except ValueError as error:
        raise RescoreValidationError(str(error)) from error
    gibbs_corrector = sampling.get("gibbs_corrector", False)
    source_gibbs_corrector = source_config.get("gibbs_corrector", False)
    if (
        type(source_gibbs_corrector) is not bool
        or source_gibbs_corrector is not gibbs_corrector
    ):
        raise RescoreValidationError("source/sampling gibbs_corrector setting differs")
    expected_effective = dict(source_config)
    if not historical_mdlm:
        expected_effective["raw_loo_top_p"] = sampling["raw_loo_top_p"]
    expected_effective.update(
        {
            "model_path": checkpoint_path,
            "num_samples": expected_sample_count,
            "device": "cuda:0",
        }
    )
    if dict(effective) != expected_effective:
        raise RescoreValidationError(
            "effective config does not bind checkpoint/count/device"
        )

    protocol = _mapping(run.get("generation_protocol"), "run.generation_protocol")
    _exact_keys(
        protocol,
        (
            HISTORICAL_MDLM_GENERATION_PROTOCOL_FIELDS
            if historical_mdlm
            else GENERATION_PROTOCOL_FIELDS
            | (GIBBS_CORRECTOR_PROTOCOL_FIELDS if gibbs_corrector else frozenset())
        ),
        "run.generation_protocol",
    )
    if protocol.get("diffusion_type") != sampling["diffusion_type"]:
        raise RescoreValidationError("generation/checkpoint config diffusion differs")
    if checkpoint.get("diffusion_type") != sampling["diffusion_type"]:
        raise RescoreValidationError("checkpoint/config diffusion differs")
    diffusion_type = sampling["diffusion_type"]
    if diffusion_type == "mdlm":
        if any(
            checkpoint.get(field) is not None
            for field in (
                "udlm_inference_eps",
                "udlm_exclude_special_tokens",
                "udlm_prior_variant",
                "udlm_prior_metadata",
                "udlm_prior_metadata_sha256",
            )
        ):
            raise RescoreValidationError("MDLM checkpoint has non-null UDLM metadata")
    else:
        checkpoint_config_pairs = {
            "udlm_inference_eps": "inference_eps",
            "udlm_exclude_special_tokens": "exclude_special_tokens",
            "udlm_prior_variant": "prior_variant",
            "udlm_prior_metadata_sha256": "prior_metadata_sha256",
        }
        for checkpoint_key, sampling_key in checkpoint_config_pairs.items():
            if checkpoint.get(checkpoint_key) != sampling[sampling_key]:
                raise RescoreValidationError(
                    f"checkpoint.{checkpoint_key} differs from sampling config"
                )
        prior_variant = sampling["prior_variant"]
        prior_metadata = checkpoint.get("udlm_prior_metadata")
        prior_digest = checkpoint.get("udlm_prior_metadata_sha256")
        if prior_variant == "release_uniform":
            if prior_metadata is not None or prior_digest is not None:
                raise RescoreValidationError(
                    "release_uniform checkpoint has categorical prior metadata"
                )
        else:
            try:
                validated_prior = benchmark.validate_udlm_prior_metadata_record(
                    prior_metadata,
                    expected_variant=prior_variant,
                    expected_exclude_special_tokens=sampling["exclude_special_tokens"],
                )
            except (RuntimeError, ValueError) as error:
                raise RescoreValidationError(
                    f"checkpoint UDLM prior metadata is invalid: {error}"
                ) from error
            if baseline_rescore.canonical_json_sha256(validated_prior) != prior_digest:
                raise RescoreValidationError(
                    "checkpoint prior metadata digest is invalid"
                )
    nfe = _integer(protocol.get("nfe"), "generation_protocol.nfe", minimum=1)
    if sampling["diffusion_type"] == "udlm" and nfe != sampling["num_steps"]:
        raise RescoreValidationError("UDLM NFE differs from sampling num_steps")
    expected_num_steps_source = (
        benchmark.GIBBS_CORRECTOR_NUM_STEPS_SOURCE
        if gibbs_corrector
        else (
            "explicit UDLM reverse-transition count"
            if diffusion_type == "udlm"
            else "MDLM.get_num_steps_confidence on the single padded generation batch"
        )
    )
    if protocol.get("num_steps_source") != expected_num_steps_source:
        raise RescoreValidationError("generation_protocol.num_steps_source differs")
    protocol_config_pairs = {
        "num_steps": "num_steps",
        "inference_eps": "inference_eps",
        "exclude_special_tokens": "exclude_special_tokens",
        "prior_variant": "prior_variant",
        "prior_metadata_sha256": "prior_metadata_sha256",
        "temperature": "softmax_temp",
        "randomness": "randomness",
    }
    for protocol_key, sampling_key in protocol_config_pairs.items():
        if protocol.get(protocol_key) != sampling[sampling_key]:
            raise RescoreValidationError(
                f"generation_protocol.{protocol_key} differs from config"
            )
    if not historical_mdlm:
        if protocol.get("raw_loo_top_p") != sampling.get("raw_loo_top_p"):
            raise RescoreValidationError(
                "generation_protocol.raw_loo_top_p differs from config"
            )
    expected_common_protocol = {
        "nfe_definition": (
            benchmark.GIBBS_CORRECTOR_NFE_DEFINITION
            if gibbs_corrector
            else "one full backbone forward evaluation per reverse step"
        ),
        "model_use_bracket_safe": False,
        "single_generation_batch": True,
        "released_safe_fix": True,
        "released_largest_component": "maximum SMILES string length",
        "strict_safe_fix": False,
        "randomness_used_by_sampler": sampling["diffusion_type"] == "mdlm",
    }
    for key, expected in expected_common_protocol.items():
        if protocol.get(key) != expected:
            raise RescoreValidationError(f"generation_protocol.{key} differs")
    if gibbs_corrector:
        if protocol.get("gibbs_corrector") is not True:
            raise RescoreValidationError(
                "generation_protocol.gibbs_corrector must be true"
            )
        for key in (
            "predictor_transitions_per_molecule",
            "corrector_steps_per_molecule",
        ):
            count = _integer(protocol.get(key), f"generation_protocol.{key}", minimum=1)
            if count != nfe // 2:
                raise RescoreValidationError(
                    f"generation_protocol.{key} differs from NFE budget"
                )
    try:
        inference_weights = benchmark.validate_inference_weights(
            protocol.get("inference_weights"), require_ema=True
        )
    except ValueError as error:
        raise RescoreValidationError(
            f"inference weights are invalid: {error}"
        ) from error

    if historical_mdlm:
        expected_sampling = {
            "diffusion_type": "mdlm",
            "softmax_temp": 0.5,
            "randomness": 0.5,
            "min_add_len": 40,
            "num_steps": None,
            "inference_eps": None,
            "exclude_special_tokens": None,
            "prior_variant": None,
            "prior_metadata_sha256": None,
        }
        if (
            expected_seed not in (0, 1, 2)
            or expected_sample_count != 1_000
            or checkpoint_sha256 != HISTORICAL_MDLM_CHECKPOINT_SHA256
            or checkpoint_size != HISTORICAL_MDLM_CHECKPOINT_SIZE_BYTES
            or checkpoint_step != HISTORICAL_MDLM_GLOBAL_STEP
            or dict(sampling) != expected_sampling
            or diffusion_type != "mdlm"
            or len(command) != 20
        ):
            raise RescoreValidationError(
                "schema-7 input is not the exact pinned historical MDLM baseline"
            )
    elif len(command) != 24:
        raise RescoreValidationError(
            "schema-8 benchmark run.command must contain exactly 24 elements"
        )

    git = _mapping(summary.get("git"), "benchmark git provenance")
    _exact_keys(git, GIT_FIELDS, "benchmark git provenance")
    source_revision = _git_revision(git.get("commit"), "git.commit")
    if not (
        git.get("upstream") == source_revision
        and git.get("expected_source_revision") == source_revision
        and git.get("dirty") is False
        and git.get("clean_pushed_source_verified_before_and_after_run") is True
    ):
        raise RescoreValidationError("benchmark clean pushed source binding differs")
    _assert_expected(source_revision, expected_source_revision, "git.commit")
    runner_sha256 = _sha256(git.get("runner_sha256"), "git.runner_sha256")
    _assert_expected(runner_sha256, expected_runner_sha256, "git.runner_sha256")

    tracking = _mapping(config.get("git_tracking"), "config.git_tracking")
    expected_tracking_fields = frozenset(
        {
            "path",
            "relative_path",
            "source_revision",
            "sha256",
            "tracked_at_source_revision",
        }
    )
    _exact_keys(tracking, expected_tracking_fields, "config.git_tracking")
    for path_field in ("path", "relative_path"):
        if not isinstance(tracking.get(path_field), str) or not tracking[path_field]:
            raise RescoreValidationError(
                f"config.git_tracking.{path_field} must be recorded"
            )
    if (
        tracking.get("source_revision") != source_revision
        or tracking.get("sha256") != config_sha256
        or tracking.get("tracked_at_source_revision") is not True
    ):
        raise RescoreValidationError("config tracked-source binding differs")

    implementation_inputs = _mapping(
        summary.get("implementation_inputs"), "implementation_inputs"
    )
    expected_implementation_names = (
        HISTORICAL_MDLM_IMPLEMENTATION_INPUT_NAMES
        if historical_mdlm
        else IMPLEMENTATION_INPUT_NAMES
    )
    if gibbs_corrector:
        expected_implementation_names |= {"corrector_source"}
    if x0_denoiser:
        expected_implementation_names |= {"denoiser_source"}
    if set(implementation_inputs) != expected_implementation_names:
        raise RescoreValidationError("implementation input map is incomplete")
    source_hashes: dict[str, str] = {}
    for name, raw_item in implementation_inputs.items():
        item = _mapping(raw_item, f"implementation_inputs.{name}")
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise RescoreValidationError(
                f"implementation_inputs.{name}.path must be recorded"
            )
        source_hashes[name] = _sha256(
            item.get("sha256"), f"implementation_inputs.{name}.sha256"
        )
        _integer(
            item.get("size_bytes"),
            f"implementation_inputs.{name}.size_bytes",
            minimum=1,
        )
    if gibbs_corrector:
        expected_corrector_path = (
            Path(str(git.get("repo_root"))) / "src/genmol/corrector.py"
        )
        if implementation_inputs["corrector_source"]["path"] != str(
            expected_corrector_path
        ):
            raise RescoreValidationError(
                "implementation_inputs.corrector_source.path differs"
            )
    if x0_denoiser:
        expected_denoiser_path = (
            Path(str(git.get("repo_root"))) / "src/genmol/denoiser.py"
        )
        if implementation_inputs["denoiser_source"]["path"] != str(
            expected_denoiser_path
        ):
            raise RescoreValidationError(
                "implementation_inputs.denoiser_source.path differs"
            )
    _assert_expected(
        source_hashes["sampler_source"],
        expected_sampler_source_sha256,
        "implementation_inputs.sampler_source.sha256",
    )
    _assert_expected(
        source_hashes["ema_source"],
        expected_ema_source_sha256,
        "implementation_inputs.ema_source.sha256",
    )
    implementation_inputs_sha256 = baseline_rescore.canonical_json_sha256(
        implementation_inputs
    )
    _assert_expected(
        implementation_inputs_sha256,
        expected_implementation_inputs_sha256,
        "implementation_inputs canonical SHA-256",
    )
    metric_inputs = _mapping(summary.get("metric_inputs"), "metric_inputs")
    metric_inputs_sha256 = baseline_rescore.canonical_json_sha256(metric_inputs)
    _assert_expected(
        metric_inputs_sha256,
        expected_metric_inputs_sha256,
        "metric_inputs canonical SHA-256",
    )

    runtime = _mapping(summary.get("runtime_seconds"), "runtime_seconds")
    _exact_keys(
        runtime,
        HISTORICAL_MDLM_RUNTIME_FIELDS if historical_mdlm else RUNTIME_FIELDS,
        "runtime_seconds",
    )
    for name, value in runtime.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RescoreValidationError(f"runtime_seconds.{name} must be finite")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise RescoreValidationError(
                f"runtime_seconds.{name} must be finite and nonnegative"
            )

    artifacts = _mapping(summary.get("artifacts"), "benchmark artifacts")
    _exact_keys(
        artifacts,
        (
            frozenset({"raw_samples_csv", "summary_json"})
            if historical_mdlm
            else frozenset({"raw_samples_csv", "summary_json", "bundle"})
        ),
        "benchmark artifacts",
    )
    raw_artifact = _mapping(
        artifacts.get("raw_samples_csv"), "artifacts.raw_samples_csv"
    )
    _exact_keys(
        raw_artifact,
        frozenset({"path", "sha256", "row_count", "fields"}),
        "artifacts.raw_samples_csv",
    )
    if (
        raw_artifact.get("sha256") != raw_sha256
        or raw_artifact.get("row_count") != expected_sample_count
        or raw_artifact.get("fields") != list(benchmark.RAW_SAMPLE_FIELDS)
    ):
        raise RescoreValidationError("summary raw artifact binding differs")
    summary_artifact = _mapping(artifacts.get("summary_json"), "artifacts.summary_json")
    _exact_keys(summary_artifact, frozenset({"path"}), "artifacts.summary_json")
    if (
        not isinstance(summary_artifact.get("path"), str)
        or not summary_artifact["path"]
    ):
        raise RescoreValidationError("artifacts.summary_json.path must be recorded")
    execution_authority = None
    if not historical_mdlm:
        bundle = _mapping(artifacts.get("bundle"), "artifacts.bundle")
        if dict(bundle) != ARTIFACT_BUNDLE:
            raise RescoreValidationError("artifacts.bundle differs")
        environment = _mapping(summary.get("environment"), "environment")
        execution_authority = _validate_execution_authority(
            run.get("execution_authority"),
            run_command=command,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            source_revision=source_revision,
            config_path=config_path,
            config_sha256=config_sha256,
            expected_sample_count=expected_sample_count,
            expected_seed=expected_seed,
            environment=environment,
            implementation_inputs=implementation_inputs,
        )

    return {
        "seed": expected_seed,
        "sample_count": expected_sample_count,
        "started_at_utc": run["started_at_utc"],
        "completed_at_utc": run["completed_at_utc"],
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
            "global_step": checkpoint_step,
        },
        "config": {
            "path": config_path,
            "sha256": config_sha256,
            "sampling": dict(sampling),
            "sampling_sha256": sampling_sha256,
            "effective_sha256": effective_sha256,
            "git_tracking": dict(tracking),
        },
        "generation": {
            "nfe": nfe,
            **(
                {"udlm_denoiser_metadata": dict(checkpoint["udlm_denoiser_metadata"])}
                if x0_denoiser
                else {}
            ),
            "metric_branches": ["released_comparable", "strict"],
            "inference_weights": inference_weights,
            "raw_loo_top_p": protocol.get("raw_loo_top_p"),
            **(
                {key: protocol[key] for key in GIBBS_CORRECTOR_PROTOCOL_FIELDS}
                if gibbs_corrector
                else {}
            ),
        },
        "source": {
            "revision": source_revision,
            "runner_sha256": runner_sha256,
            "sampler_source_sha256": source_hashes["sampler_source"],
            "ema_source_sha256": source_hashes["ema_source"],
            **(
                {"denoiser_source_sha256": source_hashes["denoiser_source"]}
                if x0_denoiser
                else {}
            ),
            **(
                {"corrector_source_sha256": source_hashes["corrector_source"]}
                if gibbs_corrector
                else {}
            ),
            "implementation_inputs_sha256": implementation_inputs_sha256,
            "metric_inputs_sha256": metric_inputs_sha256,
        },
        "artifacts": {
            "summary_json": {"recorded_path": summary_artifact["path"]},
            "raw_samples_csv": {"recorded_path": raw_artifact["path"]},
        },
        "summary_schema_version": schema_version,
        "historical_mdlm_compatibility": historical_mdlm,
        "execution_authority": execution_authority,
    }


def rescore_denovo_run(
    *,
    summary_payload: bytes,
    raw_samples_payload: bytes,
    expected_summary_sha256: str,
    expected_raw_samples_sha256: str,
    expected_seed: int,
    expected_sample_count: int,
    decode_function: Callable[..., list[dict[str, Any]]],
    evaluate_function: Callable[..., tuple[dict[str, Any], dict[str, int]]],
    oracle_qed: Callable[[Sequence[str]], Any],
    oracle_sa: Callable[[Sequence[str]], Any],
    diversity_evaluator: Callable[[Sequence[str]], Any],
    tokenizer_batch_decode: Callable[..., Sequence[str]] | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_source_revision: str | None = None,
    expected_runner_sha256: str | None = None,
    expected_sampler_source_sha256: str | None = None,
    expected_ema_source_sha256: str | None = None,
    expected_implementation_inputs_sha256: str | None = None,
    expected_metric_inputs_sha256: str | None = None,
) -> dict[str, Any]:
    """Purely validate and independently re-score retained benchmark bytes."""

    _integer(expected_seed, "expected_seed", minimum=0)
    _integer(expected_sample_count, "expected_sample_count", minimum=1)
    expected_summary_sha256 = _sha256(
        expected_summary_sha256, "expected summary SHA-256"
    )
    expected_raw_samples_sha256 = _sha256(
        expected_raw_samples_sha256, "expected raw CSV SHA-256"
    )
    if len(summary_payload) > MAXIMUM_SUMMARY_SIZE_BYTES:
        raise RescoreValidationError("benchmark summary exceeds the 2 MiB limit")
    summary_sha256 = hashlib.sha256(summary_payload).hexdigest()
    raw_sha256 = hashlib.sha256(raw_samples_payload).hexdigest()
    if summary_sha256 != expected_summary_sha256:
        raise RescoreValidationError(
            "summary payload SHA-256 differs from its expected digest"
        )
    if raw_sha256 != expected_raw_samples_sha256:
        raise RescoreValidationError(
            "raw CSV payload SHA-256 differs from its expected digest"
        )

    rows = baseline_rescore.parse_raw_rows(
        raw_samples_payload,
        expected_count=expected_sample_count,
        label=f"seed {expected_seed} raw_samples.csv",
    )
    summary = _mapping(
        baseline_rescore.strict_json_loads(
            summary_payload, label=f"seed {expected_seed} summary.json"
        ),
        f"seed {expected_seed} summary.json",
    )
    identity = _validate_summary_identity(
        summary,
        expected_seed=expected_seed,
        expected_sample_count=expected_sample_count,
        raw_sha256=raw_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_source_revision=expected_source_revision,
        expected_runner_sha256=expected_runner_sha256,
        expected_sampler_source_sha256=expected_sampler_source_sha256,
        expected_ema_source_sha256=expected_ema_source_sha256,
        expected_implementation_inputs_sha256=expected_implementation_inputs_sha256,
        expected_metric_inputs_sha256=expected_metric_inputs_sha256,
    )
    if identity["summary_schema_version"] == SUMMARY_SCHEMA_VERSION:
        if tokenizer_batch_decode is None:
            raise RescoreValidationError(
                "schema-8 validation requires the exact tokenizer batch_decode"
            )
        sampling = identity["config"]["sampling"]
        exclude_special_tokens = sampling.get("exclude_special_tokens")
        if type(exclude_special_tokens) is not bool:
            raise RescoreValidationError(
                "schema-8 candidate exclude_special_tokens must be boolean"
            )
        identity["sampled_token_control_audit"] = validate_sampled_token_control_audit(
            summary.get("sampled_token_control_audit"),
            expected_rows=expected_sample_count,
            exclude_special_tokens=exclude_special_tokens,
            raw_model_texts=[row["raw_model_text"] for row in rows],
            tokenizer_batch_decode=tokenizer_batch_decode,
        )

    decoded = decode_function(
        [row["raw_model_text"] for row in rows],
        use_bracket_safe=False,
    )
    rescored_metrics, rescored_failures = evaluate_function(
        decoded,
        requested_count=expected_sample_count,
        oracle_qed=oracle_qed,
        oracle_sa=oracle_sa,
        diversity_evaluator=diversity_evaluator,
    )
    row_comparison = baseline_rescore.compare_raw_records(rows, decoded)
    baseline_rescore._assert_exact_or_close(  # noqa: SLF001
        rescored_metrics,
        summary.get("metrics"),
        label=f"seed {expected_seed}.metrics",
    )
    baseline_rescore._assert_exact_or_close(  # noqa: SLF001
        rescored_failures,
        summary.get("failure_counts"),
        label=f"seed {expected_seed}.failure_counts",
    )

    released = _mapping(
        rescored_metrics.get("released_comparable"),
        "rescored metrics.released_comparable",
    )
    strict = _mapping(rescored_metrics.get("strict"), "rescored metrics.strict")
    for branch_name, branch in (("released_comparable", released), ("strict", strict)):
        diversity = branch.get("diversity")
        zero_unique_sentinel = (
            type(branch.get("unique_count")) is int
            and branch["unique_count"] == 0
            and branch.get("diversity_undefined_reason") == "no_unique_valid_molecules"
        )
        if diversity is None:
            if not zero_unique_sentinel:
                raise RescoreValidationError(
                    f"{branch_name} diversity may be null only for the "
                    "no_unique_valid_molecules zero-unique sentinel"
                )
            continue
        if zero_unique_sentinel:
            raise RescoreValidationError(
                f"{branch_name} zero-unique diversity sentinel must be null"
            )
        if isinstance(diversity, bool) or not isinstance(diversity, (int, float)):
            raise RescoreValidationError(f"{branch_name} diversity must be finite")
        if not math.isfinite(float(diversity)) or not 0.0 <= float(diversity) <= 1.0:
            raise RescoreValidationError(f"{branch_name} diversity must lie in [0, 1]")

    return {
        "status": "exact_match",
        "seed": expected_seed,
        "summary_sha256": summary_sha256,
        "raw_samples_sha256": raw_sha256,
        "row_comparison": row_comparison,
        "metrics": rescored_metrics,
        "failure_counts": rescored_failures,
        "identity": identity,
        "independent_recomputation": {
            "raw_input_field": "raw_model_text",
            "decoder": "benchmark.decode_records",
            "metric_evaluator": "benchmark.evaluate_records",
            "qed_recomputed": True,
            "sa_recomputed": True,
            "released_diversity_recomputed": True,
            "strict_branch_recomputed": True,
            "all_21_raw_fields_compared": True,
            "numeric_absolute_tolerance": NUMERIC_ABSOLUTE_TOLERANCE,
        },
    }


def _load_pinned_tokenizer_batch_decode() -> Callable[..., Sequence[str]]:
    """Load the already-cached, checksum-pinned SAFE tokenizer without downloads."""

    from huggingface_hub import try_to_load_from_cache
    from safe.tokenizer import SAFETokenizer

    tokenizer_path = try_to_load_from_cache(
        benchmark.TOKENIZER_REQUESTED_IDENTIFIER,
        "tokenizer.json",
        revision=benchmark.SAFE_GPT_TOKENIZER_REVISION,
    )
    if not isinstance(tokenizer_path, str):
        raise RescoreValidationError("pinned SAFE tokenizer is absent from local cache")
    payload = Path(tokenizer_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != benchmark.SAFE_GPT_TOKENIZER_SHA256:
        raise RescoreValidationError("cached SAFE tokenizer checksum differs")
    tokenizer = SAFETokenizer.from_pretrained(tokenizer_path).get_pretrained()
    tokenizer.add_tokens(["<", ">"])
    return tokenizer.batch_decode


def read_run_artifacts(
    *,
    summary_path: Path,
    raw_samples_path: Path,
    allowed_root: Path,
    expected_summary_sha256: str,
    expected_raw_samples_sha256: str,
) -> tuple[RetainedArtifact, RetainedArtifact]:
    """Stable-read a summary/CSV pair without following symlinks."""

    summary = baseline_rescore.read_stable_regular_file(
        summary_path,
        allowed_root=allowed_root,
        label="schema-7 de-novo summary",
        expected_sha256=_sha256(expected_summary_sha256, "expected summary SHA-256"),
    )
    raw = baseline_rescore.read_stable_regular_file(
        raw_samples_path,
        allowed_root=allowed_root,
        label="schema-7 de-novo raw CSV",
        expected_sha256=_sha256(
            expected_raw_samples_sha256, "expected raw CSV SHA-256"
        ),
    )
    return summary, raw


def _validate_recorded_artifact_paths(
    *,
    summary_artifact: RetainedArtifact,
    raw_artifact: RetainedArtifact,
) -> dict[str, dict[str, str | bool]]:
    """Bind summary-internal labels to the exact live paths that were read."""

    summary = _mapping(
        baseline_rescore.strict_json_loads(
            summary_artifact.payload, label="schema-7 de-novo summary"
        ),
        "schema-7 de-novo summary",
    )
    artifacts = _mapping(summary.get("artifacts"), "benchmark artifacts")
    recorded = {
        "summary_json": _mapping(
            artifacts.get("summary_json"), "artifacts.summary_json"
        ).get("path"),
        "raw_samples_csv": _mapping(
            artifacts.get("raw_samples_csv"), "artifacts.raw_samples_csv"
        ).get("path"),
    }
    retained = {
        "summary_json": summary_artifact,
        "raw_samples_csv": raw_artifact,
    }
    result: dict[str, dict[str, str | bool]] = {}
    for name, artifact in retained.items():
        recorded_value = recorded[name]
        if not isinstance(recorded_value, str) or not recorded_value:
            raise RescoreValidationError(
                f"artifacts.{name}.path must be a nonempty absolute path"
            )
        recorded_path = Path(recorded_value)
        if not recorded_path.is_absolute():
            raise RescoreValidationError(
                f"artifacts.{name}.path must be an absolute path"
            )
        try:
            supplied_resolved_path = artifact.path.resolve(strict=True)
        except OSError as error:
            raise RescoreValidationError(
                f"supplied {name} path became unavailable"
            ) from error
        if recorded_path != supplied_resolved_path:
            raise RescoreValidationError(
                f"artifacts.{name}.path differs from supplied resolved path: "
                f"{recorded_path} != {supplied_resolved_path}"
            )
        result[name] = {
            "recorded_path": str(recorded_path),
            "supplied_resolved_path": str(supplied_resolved_path),
            "exact_path_match": True,
        }
    return result


def rescore_denovo_run_files(
    *,
    summary_path: Path,
    raw_samples_path: Path,
    allowed_root: Path,
    expected_summary_sha256: str,
    expected_raw_samples_sha256: str,
    expected_seed: int,
    expected_sample_count: int,
    **expected_identity: Any,
) -> dict[str, Any]:
    """Re-score in a seed-at-start CPU worker with pinned metric inputs.

    Direct callers must already be running in the seed-specific environment.
    Multi-seed callers must use :func:`invoke_rescore_worker` once per seed.
    """

    _integer(expected_seed, "expected_seed", minimum=0)
    _integer(expected_sample_count, "expected_sample_count", minimum=1)
    worker_environment = baseline_rescore.validate_worker_environment(expected_seed)
    summary_artifact, raw_artifact = read_run_artifacts(
        summary_path=summary_path,
        raw_samples_path=raw_samples_path,
        allowed_root=allowed_root,
        expected_summary_sha256=expected_summary_sha256,
        expected_raw_samples_sha256=expected_raw_samples_sha256,
    )
    path_bindings = _validate_recorded_artifact_paths(
        summary_artifact=summary_artifact,
        raw_artifact=raw_artifact,
    )
    sa_snapshot = benchmark.load_pinned_sa_metric_input()
    runtime_metric_inputs_sha256 = baseline_rescore.canonical_json_sha256(
        sa_snapshot.provenance
    )
    caller_metric_digest = expected_identity.pop("expected_metric_inputs_sha256", None)
    if (
        caller_metric_digest is not None
        and caller_metric_digest != runtime_metric_inputs_sha256
    ):
        raise RescoreValidationError(
            "caller metric-input digest differs from the pinned runtime inputs"
        )
    benchmark.assert_local_genmol_import()
    tokenizer_batch_decode = _load_pinned_tokenizer_batch_decode()
    with baseline_rescore.network_disabled():
        from tdc import Evaluator, Oracle

        with benchmark.pinned_tdc_sa_oracle(sa_snapshot, Oracle) as sa_oracle:
            result = rescore_denovo_run(
                summary_payload=summary_artifact.payload,
                raw_samples_payload=raw_artifact.payload,
                expected_summary_sha256=expected_summary_sha256,
                expected_raw_samples_sha256=expected_raw_samples_sha256,
                expected_seed=expected_seed,
                expected_sample_count=expected_sample_count,
                decode_function=benchmark.decode_records,
                evaluate_function=benchmark.evaluate_records,
                oracle_qed=Oracle("qed"),
                oracle_sa=sa_oracle,
                diversity_evaluator=Evaluator("diversity"),
                tokenizer_batch_decode=tokenizer_batch_decode,
                expected_metric_inputs_sha256=runtime_metric_inputs_sha256,
                **expected_identity,
            )
    benchmark.assert_runtime_tdc_metric_provenance(sa_snapshot.provenance)

    final_summary, final_raw = read_run_artifacts(
        summary_path=summary_path,
        raw_samples_path=raw_samples_path,
        allowed_root=allowed_root,
        expected_summary_sha256=expected_summary_sha256,
        expected_raw_samples_sha256=expected_raw_samples_sha256,
    )
    if (
        final_summary.sha256 != summary_artifact.sha256
        or final_raw.sha256 != raw_artifact.sha256
    ):
        raise RescoreValidationError("de-novo run artifacts changed during rescoring")
    result["stable_inputs"] = {
        "summary_json": summary_artifact.provenance(),
        "raw_samples_csv": raw_artifact.provenance(),
        "recorded_path_bindings": path_bindings,
        "revalidated_unchanged_after_rescore": True,
    }
    result["worker_environment"] = worker_environment
    return result


def _worker_command(
    *,
    summary_path: Path,
    raw_samples_path: Path,
    allowed_root: Path,
    expected_summary_sha256: str,
    expected_raw_samples_sha256: str,
    expected_seed: int,
    expected_sample_count: int,
    expected_identity: Mapping[str, str | None],
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--summary-path",
        str(summary_path),
        "--raw-samples-path",
        str(raw_samples_path),
        "--allowed-root",
        str(allowed_root),
        "--expected-summary-sha256",
        expected_summary_sha256,
        "--expected-raw-samples-sha256",
        expected_raw_samples_sha256,
        "--expected-seed",
        str(expected_seed),
        "--expected-sample-count",
        str(expected_sample_count),
    ]
    for name in EXPECTED_IDENTITY_ARGUMENTS:
        value = expected_identity.get(name)
        if value is not None:
            command.extend((f"--{name.replace('_', '-')}", value))
    return command


def invoke_rescore_worker(
    *,
    summary_path: Path,
    raw_samples_path: Path,
    allowed_root: Path,
    expected_summary_sha256: str,
    expected_raw_samples_sha256: str,
    expected_seed: int,
    expected_sample_count: int,
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_source_revision: str | None = None,
    expected_runner_sha256: str | None = None,
    expected_sampler_source_sha256: str | None = None,
    expected_ema_source_sha256: str | None = None,
    expected_implementation_inputs_sha256: str | None = None,
    expected_metric_inputs_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-score one seed in a fresh deterministic, offline, CPU subprocess."""

    _integer(expected_seed, "expected_seed", minimum=0)
    _integer(expected_sample_count, "expected_sample_count", minimum=1)
    expected_summary_sha256 = _sha256(
        expected_summary_sha256, "expected summary SHA-256"
    )
    expected_raw_samples_sha256 = _sha256(
        expected_raw_samples_sha256, "expected raw CSV SHA-256"
    )
    allowed_root = allowed_root.resolve(strict=True)
    expected_identity = {
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "expected_config_sha256": expected_config_sha256,
        "expected_source_revision": expected_source_revision,
        "expected_runner_sha256": expected_runner_sha256,
        "expected_sampler_source_sha256": expected_sampler_source_sha256,
        "expected_ema_source_sha256": expected_ema_source_sha256,
        "expected_implementation_inputs_sha256": (
            expected_implementation_inputs_sha256
        ),
        "expected_metric_inputs_sha256": expected_metric_inputs_sha256,
    }
    command = _worker_command(
        summary_path=summary_path,
        raw_samples_path=raw_samples_path,
        allowed_root=allowed_root,
        expected_summary_sha256=expected_summary_sha256,
        expected_raw_samples_sha256=expected_raw_samples_sha256,
        expected_seed=expected_seed,
        expected_sample_count=expected_sample_count,
        expected_identity=expected_identity,
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=baseline_rescore.worker_environment(expected_seed),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        raise RescoreValidationError(
            f"seed {expected_seed} rescore worker failed with status "
            f"{completed.returncode}\nstdout:\n{stdout[-4000:]}\n"
            f"stderr:\n{stderr[-8000:]}"
        )
    marker_lines = [
        line for line in stdout.splitlines() if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if len(marker_lines) != 1:
        raise RescoreValidationError(
            f"seed {expected_seed} worker emitted {len(marker_lines)} result records"
        )
    payload = marker_lines[0][len(WORKER_RESULT_PREFIX) :].encode("utf-8")
    result = _mapping(
        baseline_rescore.strict_json_loads(
            payload, label=f"seed {expected_seed} de-novo rescore worker result"
        ),
        f"seed {expected_seed} de-novo rescore worker result",
    )
    if result.get("seed") != expected_seed or result.get("status") != "exact_match":
        raise RescoreValidationError(
            f"seed {expected_seed} worker returned an invalid completion record"
        )
    if (
        result.get("summary_sha256") != expected_summary_sha256
        or result.get("raw_samples_sha256") != expected_raw_samples_sha256
    ):
        raise RescoreValidationError(
            f"seed {expected_seed} worker returned different artifact identities"
        )
    environment = _mapping(
        result.get("worker_environment"),
        f"seed {expected_seed} worker_environment",
    )
    if (
        environment.get("python_hash_seed") != str(expected_seed)
        or environment.get("device") != "cpu"
        or environment.get("cuda_visible_devices") != ""
        or environment.get("nvidia_visible_devices") != ""
    ):
        raise RescoreValidationError(
            f"seed {expected_seed} worker returned invalid CPU/seed evidence"
        )
    return dict(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--raw-samples-path", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-raw-samples-sha256", required=True)
    parser.add_argument("--expected-seed", required=True, type=int)
    parser.add_argument("--expected-sample-count", required=True, type=int)
    for name in EXPECTED_IDENTITY_ARGUMENTS:
        parser.add_argument(f"--{name.replace('_', '-')}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_identity = {
        name: getattr(args, name)
        for name in EXPECTED_IDENTITY_ARGUMENTS
        if getattr(args, name) is not None
    }
    result = rescore_denovo_run_files(
        summary_path=args.summary_path,
        raw_samples_path=args.raw_samples_path,
        allowed_root=args.allowed_root,
        expected_summary_sha256=args.expected_summary_sha256,
        expected_raw_samples_sha256=args.expected_raw_samples_sha256,
        expected_seed=args.expected_seed,
        expected_sample_count=args.expected_sample_count,
        **expected_identity,
    )
    print(
        WORKER_RESULT_PREFIX
        + json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
