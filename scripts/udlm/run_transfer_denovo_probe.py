"""Separate, unpromotable de novo probe for audited zero-update MDLM transfers.

Default operation is a hash/config/proof preview without checkpoint loading,
model construction, GPU discovery, or metric evaluation. Existing trained
benchmark schemas and their positive-update EMA requirement are unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import benchmark as bench  # noqa: E402

ARTIFACT_KIND = "zero_update_transfer_denovo_probe"
TRANSFER_ARM = "frozen_mdlm_ema_transfer"
REFERENCE_ARM = "mdlm_ema_reference"
SPEC_FIELDS = {
    "schema_version",
    "artifact_kind",
    "entry_id",
    "arm_kind",
    "seed",
    "num_samples",
    "checkpoint",
    "sampling_config",
    "export_manifest",
    "design",
    "input_sha256",
}
SAMPLING_FIELDS = {
    "model_path",
    "num_samples",
    "device",
    "diffusion_type",
    "parameterization",
    "softmax_temp",
    "randomness",
    "min_add_len",
    "num_steps",
    "inference_eps",
    "exclude_special_tokens",
    "prior_variant",
    "prior_metadata_sha256",
    "raw_loo_top_p",
    "gibbs_corrector",
    "temperature_space",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def canonical_digest(value):
    return bench._canonical_json_sha256(value)


def sha256(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        "invalid SHA-256",
    )
    return value


def strict_json(payload):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON field {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON constant {value}")

    return json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)


def project_path(value, input_root=ROOT, *, exists=True):
    path = Path(value)
    path = (path if path.is_absolute() else Path(input_root) / path).resolve(
        strict=exists
    )
    require(
        path.is_relative_to(PROJECT) and path != PROJECT,
        "artifact must remain inside project",
    )
    return path


def file_record(path, *, capture=False):
    path = project_path(path)
    claim, payload = artifact_io.snapshot_file(
        PROJECT, path.relative_to(PROJECT), capture_bytes=capture
    )
    return {
        "path": str(path),
        "sha256": claim.sha256,
        "size_bytes": claim.size_bytes,
    }, payload


def read_reference(reference, input_root, *, capture=False):
    require(
        isinstance(reference, dict)
        and {"path", "sha256"} <= set(reference) <= {"path", "sha256", "size_bytes"},
        "invalid artifact reference",
    )
    sha256(reference["sha256"])
    record, payload = file_record(
        project_path(reference["path"], input_root), capture=capture
    )
    require(
        record["sha256"] == reference["sha256"],
        "artifact hash differs: " + record["path"],
    )
    if "size_bytes" in reference:
        require(
            type(reference["size_bytes"]) is int
            and record["size_bytes"] == reference["size_bytes"],
            "artifact size differs",
        )
    return record, payload


def direct_input_records(sampling):
    """Hash source/metric assets only; do not deserialize length or SA pickles."""
    paths = dict(bench.IMPLEMENTATION_INPUT_PATHS)
    if sampling.get("prior_variant") == "mask_rich_empirical":
        paths["empirical_frequency"] = ROOT / bench.EMPIRICAL_FREQUENCY_RELATIVE_PATH
    if sampling.get("parameterization", "raw_loo") == "x0_denoiser":
        paths["denoiser_source"] = bench.DENOISER_SOURCE_PATH
    paths.update(
        {
            "probe_source": Path(__file__),
            "benchmark_source": Path(bench.__file__),
            "exporter_source": ROOT / "scripts/udlm/export_zero_update_transfer.py",
            "gpu_probe_source": ROOT / "scripts/exps/denovo/launch_benchmark.py",
            "sa_fragment_scores": ROOT / bench.SA_FRAGMENT_SCORES_RELATIVE_PATH,
        }
    )
    metric = bench._tdc_metric_implementation_provenance()
    paths.update(
        {
            "tdc_" + name: Path(value["path"])
            for name, value in metric["implementation_files"].items()
        }
    )
    records = {name: file_record(path)[0] for name, path in paths.items()}
    require(
        records["sa_fragment_scores"]["sha256"] == bench.SA_FRAGMENT_SCORES_SHA256,
        "SA scores differ from existing benchmark pin",
    )
    if "empirical_frequency" in records:
        require(
            records["empirical_frequency"]["sha256"]
            == bench.EMPIRICAL_FREQUENCY_SHA256,
            "empirical frequency differs from existing prior pin",
        )
    return records


def validate_spec(spec):
    require(
        isinstance(spec, dict) and set(spec) == SPEC_FIELDS, "probe spec fields differ"
    )
    require(
        type(spec["schema_version"]) is int
        and spec["schema_version"] == 1
        and spec["artifact_kind"] == ARTIFACT_KIND,
        "wrong probe artifact identity",
    )
    require(spec["arm_kind"] in (TRANSFER_ARM, REFERENCE_ARM), "unknown probe arm kind")
    require(
        isinstance(spec["entry_id"], str)
        and re.fullmatch(r"[a-z0-9][a-z0-9_]{0,79}", spec["entry_id"]),
        "invalid entry ID",
    )
    require(type(spec["seed"]) is int and 0 <= spec["seed"] < 2**32, "invalid seed")
    require(
        type(spec["num_samples"]) is int
        and 1 <= spec["num_samples"] <= bench.MAX_AUDIT_ROWS,
        "invalid requested sample count",
    )
    require(
        (spec["export_manifest"] is None) == (spec["arm_kind"] == REFERENCE_ARM),
        "transfer proof presence differs from arm kind",
    )
    require(isinstance(spec["input_sha256"], dict), "input hashes must be a mapping")
    for value in spec["input_sha256"].values():
        sha256(value)


def read_export_proof(reference, checkpoint, sampling, input_root):
    """Verify the published export closure, without reopening the source MDLM."""
    from scripts.udlm import export_zero_update_transfer as exporter

    manifest_record, payload = read_reference(reference, input_root, capture=True)
    manifest = strict_json(payload)
    require(
        manifest.get("status") == "completed"
        and manifest.get("roundtrip") == "strict_CPU_GenMol.load_from_checkpoint",
        "export did not complete its strict roundtrip",
    )
    require(
        manifest.get("checkpoint") == checkpoint,
        "export checkpoint differs from probe checkpoint",
    )
    transfer = manifest["transfer_metadata"]
    require(
        transfer.get("artifact_kind") == "inference_only_zero_update_transfer",
        "wrong transfer artifact kind",
    )
    for key in ("udlm_optimizer_updates", "udlm_example_exposures"):
        require(
            type(transfer.get(key)) is int and transfer[key] == 0,
            "transfer training counters must be exactly zero",
        )
    require(
        transfer.get("trained_as_ct_or_ce") is False
        and transfer.get("training_objective_applied") is None,
        "transfer falsely claims CT/CE training",
    )
    require(
        transfer.get("selected_parameterization")
        == sampling.get("parameterization", "raw_loo"),
        "transfer interpretation differs from sampling",
    )
    require(
        transfer.get("prior_metadata_sha256") == sampling["prior_metadata_sha256"],
        "transfer prior differs from sampling",
    )
    source = transfer["source_mdlm"]
    require(
        type(source.get("global_step")) is int
        and source["global_step"] == 50000
        and source.get("diffusion") == "mdlm"
        and source.get("weights") == "ema"
        and source.get("config_sha256_scope")
        == "complete_unresolved_source_configuration_no_eager_interpolation",
        "transfer source is not MDLM50000 EMA",
    )
    require(
        type(source["ema"]["num_updates"]) is int and source["ema"]["num_updates"] > 0,
        "source EMA has no positive history",
    )
    bench.validate_inference_weights(
        {"source": "ema", "ema_applied": True, "ema": source["ema"]},
        require_ema=True,
    )
    sha256(source["checkpoint"]["sha256"])
    require(
        exporter.digest(source["backbone_state"]) == source["backbone_state_sha256"],
        "source tensor proof digest differs",
    )
    outputs = manifest["outputs"]
    require(
        set(outputs)
        == {
            "request.json",
            "transfer.ckpt",
            "resolved_config.json",
            "exporter_source.py",
        },
        "export output closure differs",
    )
    closure = {"export_manifest": manifest_record}
    retained = {}
    directory = Path(manifest_record["path"]).parent
    for name, record in outputs.items():
        require(
            Path(record["path"]).resolve() == directory / name,
            "export output relocated without declaration",
        )
        observed, data = read_reference(
            record, input_root, capture=name != "transfer.ckpt"
        )
        closure["export_" + name] = observed
        retained[name] = data
    require(outputs["transfer.ckpt"] == checkpoint, "export output checkpoint mismatch")
    request = strict_json(retained["request.json"])
    config = exporter.validate_config(strict_json(retained["resolved_config.json"]))
    require(
        hashlib.sha256(retained["request.json"]).hexdigest()
        == manifest["request_sha256"],
        "export request hash differs",
    )
    require(
        request.get("operation") == "zero_update_mdlm_ema_transfer"
        and request.get("config") == config,
        "export request/config mismatch",
    )
    require(
        request.get("expected_source_sha256") == source["checkpoint"]["sha256"]
        and request.get("seed") == transfer["initialization_seed"],
        "export initialization source/seed differs",
    )
    require(
        exporter.digest(config) == transfer["config_sha256"],
        "resolved export config digest differs",
    )
    require(
        config["training"]["udlm"]["parameterization"]
        == transfer["selected_parameterization"],
        "export config interpretation differs",
    )
    source_code = transfer["exporter_source"]
    require(
        type(source_code.get("head")) is str
        and re.fullmatch(r"[0-9a-f]{40}", source_code["head"])
        and set(source_code["files"]) == set(exporter.SOURCE_FILES),
        "export source closure differs",
    )
    require(request.get("source") == source_code, "export source identity differs")
    require(
        outputs["exporter_source.py"]["sha256"]
        == source_code["files"]["scripts/udlm/export_zero_update_transfer.py"][
            "sha256"
        ],
        "exporter archived source differs",
    )
    # Bind the original exporter implementation to the actual helper being reused.
    require(
        file_record(Path(exporter.__file__))[0]["sha256"]
        == outputs["exporter_source.py"]["sha256"],
        "runtime exporter helper differs from certified exporter",
    )
    for name, record in source_code["files"].items():
        current = ROOT / name
        require(
            file_record(current)[0]["sha256"] == record["sha256"],
            "export/generation implementation differs: " + name,
        )
    closure["export_input_config"], _ = read_reference(
        request["config_input"], input_root
    )
    require(
        exporter.validate_config(
            bench.load_yaml_config(Path(closure["export_input_config"]["path"]))
        )
        == config,
        "original export config differs from resolved configuration",
    )
    validation = manifest["validation"]
    require(
        type(validation["ema"]["num_updates"]) is int
        and validation["ema"]["num_updates"] == 0,
        "export EMA count is not zero",
    )
    require(
        validation["source_base_state_sha256"] == source["backbone_state_sha256"],
        "export base-source proof differs",
    )
    return {"manifest": manifest, "config": config, "closure": closure}


def preflight(*, spec_path, expected_spec_sha256, input_root, expected_source_revision):
    source = bench.require_clean_pushed_source(expected_source_revision)
    spec_record, payload = read_reference(
        {"path": str(spec_path), "sha256": expected_spec_sha256},
        input_root,
        capture=True,
    )
    spec = strict_json(payload)
    validate_spec(spec)
    records = {"spec": spec_record}
    for key in ("checkpoint", "sampling_config", "design"):
        records[key], _ = read_reference(spec[key], input_root)
    config = bench.load_yaml_config(Path(records["sampling_config"]["path"]))
    require(set(config) <= SAMPLING_FIELDS, "unsupported probe sampling fields")
    if "model_path" in config:
        require(
            project_path(config["model_path"], input_root)
            == Path(records["checkpoint"]["path"]),
            "sampling YAML model_path differs from spec checkpoint",
        )
    if "num_samples" in config:
        require(
            type(config["num_samples"]) is int
            and config["num_samples"] == spec["num_samples"],
            "sampling YAML sample count differs",
        )
    # Historical YAML normalization accepts unrelated extra fields; explicitly
    # reject this unimplemented guidance surface in this separate probe as well.
    require(
        "context_guidance" not in config
        and config.get("method") != "posterior_context",
        "context guidance is not part of this probe",
    )
    sampling = bench.validate_sampling_config(config)
    require(not sampling.get("gibbs_corrector", False), "probe is predictor-only")
    if spec["arm_kind"] == TRANSFER_ARM:
        require(
            sampling["diffusion_type"] == "udlm"
            and sampling["prior_variant"] == "mask_rich_empirical"
            and sampling["exclude_special_tokens"] is False
            and sampling["raw_loo_top_p"] == 1.0,
            "transfer requires full-alphabet MASK prior and top-p1",
        )
        proof = read_export_proof(
            spec["export_manifest"], records["checkpoint"], sampling, input_root
        )
        records.update(proof["closure"])
    else:
        require(
            sampling["diffusion_type"] == "mdlm", "reference must use original MDLM"
        )
        proof = None
    direct = direct_input_records(sampling)
    require(
        {key: record["sha256"] for key, record in direct.items()}
        == spec["input_sha256"],
        "pinned implementation/metric input hashes differ",
    )
    records.update({"direct_" + key: value for key, value in direct.items()})
    revalidate(records)
    return {
        "spec": spec,
        "source": source,
        "source_config": config,
        "sampling": sampling,
        "proof": proof,
        "inputs": records,
        "direct_inputs": direct,
        "input_root": str(Path(input_root).resolve()),
    }


def revalidate(records):
    for record in records.values():
        current, _ = file_record(record["path"])
        require(current == record, "input changed during probe: " + record["path"])


def validate_loaded_checkpoint(checkpoint, sampler, plan):
    """Separate exact-zero transfer policy; never modify the trained policy."""
    from scripts.udlm import export_zero_update_transfer as exporter
    from omegaconf import OmegaConf

    spec, sampling = plan["spec"], plan["sampling"]
    receipt = bench.validate_inference_weights(
        sampler.inference_weights, require_ema=spec["arm_kind"] == REFERENCE_ARM
    )
    if spec["arm_kind"] == REFERENCE_ARM:
        require(
            type(checkpoint.get("global_step")) is int
            and checkpoint["global_step"] == 50000,
            "reference must be MDLM50000",
        )
        require(
            exporter.TRANSFER_KEY not in checkpoint
            and sampler.diffusion_type == "mdlm",
            "reference has transfer/UDLM identity",
        )
        return {
            "kind": REFERENCE_ARM,
            "inference_weights": receipt,
            "global_step": 50000,
        }
    proof = plan["proof"]
    manifest, config = proof["manifest"], proof["config"]
    transfer = manifest["transfer_metadata"]
    require(
        exporter.encode(checkpoint.get(exporter.TRANSFER_KEY))
        == exporter.encode(transfer),
        "embedded transfer identity differs",
    )
    for key in ("global_step", "epoch"):
        require(
            type(checkpoint.get(key)) is int and checkpoint[key] == 0,
            "transfer checkpoint counters must be exactly zero",
        )
    require(
        exporter.encode(checkpoint.get("loops"))
        == exporter.encode(exporter.COMPATIBILITY_LOOPS)
        and not any(
            key in checkpoint
            for key in ("optimizer_states", "lr_schedulers", "callbacks", "sampler")
        ),
        "transfer contains training history",
    )
    require(
        exporter.encode(
            OmegaConf.to_container(
                checkpoint["hyper_parameters"]["config"], resolve=True
            )
        )
        == exporter.encode(config),
        "embedded export configuration differs",
    )
    require(
        exporter.tensor_manifest(checkpoint["state_dict"])
        == manifest["validation"]["state"],
        "serialized transfer tensors differ before loader normalization",
    )
    require(
        receipt["source"] == "ema"
        and type(receipt["ema"]["num_updates"]) is int
        and receipt["ema"]["num_updates"] == 0,
        "transfer requires actual applied zero-update EMA",
    )
    require(
        sampler.diffusion_type == "udlm"
        and sampler.model.udlm_parameterization
        == sampling.get("parameterization", "raw_loo"),
        "loaded transfer interpretation differs",
    )
    observed = exporter.validate_initialized(sampler.model, transfer["source_mdlm"])
    require(observed == manifest["validation"], "loaded export validation differs")
    require(
        canonical_digest(sampler.model.udlm_prior_metadata.to_dict())
        == sampling["prior_metadata_sha256"],
        "loaded transfer prior differs",
    )
    return {
        "kind": TRANSFER_ARM,
        "inference_weights": receipt,
        "global_step": 0,
        "transfer_metadata": transfer,
        "loaded_validation_sha256": canonical_digest(observed),
        "source_checkpoint_revalidated_this_run": False,
        "source_proof_scope": "pinned completed exporter proof; loaded target tensors independently match its source-EMA tensor map",
    }


def load_sampler(plan, snapshot):
    import torch
    from genmol.model import GenMol
    from genmol.sampler import (
        Sampler,
        _ema_metadata_and_parameters,
        _inference_weights_receipt,
    )
    from genmol.utils.checkpoint_io import verified_checkpoint_file

    record = plan["inputs"]["checkpoint"]
    with verified_checkpoint_file(record["path"], expected_sha256=record["sha256"]) as (
        stream,
        _,
    ):
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        stream.seek(0)
        model = GenMol.load_from_checkpoint(
            stream, map_location="cpu", strict=True, weights_only=False
        ).eval()
    # The ordinary constructor permits Lightning's default storage placement.
    # Bind only existing de-novo methods to an explicitly CPU-loaded model so
    # proof validation cannot allocate CUDA before the fresh capacity check.
    require(model.ema is not None, "probe requires actual EMA state")
    parameters, trainable, shadows, metadata = _ema_metadata_and_parameters(model)
    model.ema.copy_to(iter(parameters))
    require(
        all(torch.equal(p.detach(), s.detach()) for p, s in zip(trainable, shadows)),
        "EMA copy differed",
    )
    sampler = Sampler.__new__(Sampler)
    sampler.model = model
    sampler._inference_weights = _inference_weights_receipt("ema", True, metadata)
    sampler.mdlm = model.mdlm
    sampler.diffusion_type = model.diffusion_type
    sampler.pad_index = model.tokenizer.pad_token_id
    sampler.length_distribution = snapshot.length_distribution
    identity = validate_loaded_checkpoint(checkpoint, sampler, plan)
    del checkpoint
    return sampler, identity


def prepare_device(device):
    if device == "cpu":
        return {"device": "cpu", "gpu": None}
    require(device == "cuda:0", "probe CUDA must use isolated logical cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(
        re.fullmatch(r"GPU-[0-9a-fA-F-]+", visible) is not None,
        "probe requires exactly one explicit GPU UUID mapping",
    )
    from scripts.exps.denovo import launch_benchmark

    state = launch_benchmark._probe_gpu(visible)
    require(state.uuid == visible, "fresh physical GPU UUID differs")
    require(
        not state.rejection_reasons(
            max_utilization_percent=10, min_free_memory_mib=30000
        ),
        "selected GPU no longer meets launch policy",
    )
    bench.validate_device(device)
    return {
        "device": device,
        "cuda_visible_devices": visible,
        "gpu": state.as_dict(),
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
        "scheduling_scope": "one externally leased worker; campaign controller owns global concurrency and lease",
    }


@contextmanager
def observed_backbone_work(sampler):
    batches = []

    def observe(_module, args, kwargs):
        values = args[0] if args else kwargs.get("input_ids")
        require(values is not None and values.ndim == 2, "unrecognized backbone input")
        batches.append(int(values.shape[0]))

    hook = sampler.model.backbone.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        yield batches
    finally:
        hook.remove()


def score_records(records, count, snapshot):
    from tdc import Oracle, Evaluator

    with bench.pinned_tdc_sa_oracle(snapshot, Oracle) as sa:
        result = bench.evaluate_records(
            records,
            requested_count=count,
            oracle_qed=Oracle("qed"),
            oracle_sa=sa,
            diversity_evaluator=Evaluator("diversity"),
        )
    bench.assert_runtime_tdc_metric_provenance(snapshot.provenance)
    return result


def write_new(path, payload):
    with Path(path).open("xb") as stream:
        stream.write(payload)


def csv_payload(records):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=bench.RAW_SAMPLE_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(records)
    return text.getvalue().encode()


def execute(plan, *, device, output_dir):
    output_dir = project_path(output_dir, exists=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "spec": plan["spec"],
        "source": plan["source"],
        "inputs": plan["inputs"],
        "device": device,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "engineering_only": True,
        "final_promotion_eligible": False,
    }
    write_new(output_dir / "request.json", encoded(request))
    terminal = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "status": "failed",
        "request_sha256": hashlib.sha256(encoded(request)).hexdigest(),
        "verification_status": "pending_independent_rescore",
        "final_promotion_eligible": False,
    }
    started = time.perf_counter()
    try:
        require(
            Path(sys.prefix).resolve() == PROJECT / ".venv", "use the project .venv"
        )
        require(
            os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "probe requires offline assets",
        )
        implementation = bench.load_implementation_input_snapshot(
            **bench.sampling_implementation_options(plan["sampling"])
        )
        sa_snapshot = bench.load_pinned_sa_metric_input()
        for key, value in implementation.provenance.items():
            require(
                value["sha256"] == plan["direct_inputs"][key]["sha256"],
                "resident implementation input differs",
            )
        bench.assert_local_genmol_import()
        load_start = time.perf_counter()
        sampler, identity = load_sampler(plan, implementation)
        bench.assert_runtime_module_provenance(implementation.provenance)
        # The expensive CPU checkpoint/proof work precedes this fresh GPU check.
        device_receipt = prepare_device(device)
        sampler.model.to(device)
        sampler.mdlm.to_device(sampler.model.device)
        load_seconds = time.perf_counter() - load_start
        seed = bench.seed_sampling(plan["spec"]["seed"], device)
        bench.synchronize_device(sampler.model.device)
        generation_start = time.perf_counter()
        with observed_backbone_work(sampler) as batches:
            texts, protocol, input_ids, sampled_ids = bench.generate_raw_model_text(
                sampler, plan["spec"]["num_samples"], **plan["sampling"]
            )
        protocol["model_use_bracket_safe"] = bool(
            sampler.model.config.training.get("use_bracket_safe")
        )
        bench.synchronize_device(sampler.model.device)
        sampling_seconds = time.perf_counter() - generation_start
        count = plan["spec"]["num_samples"]
        require(
            type(protocol["nfe"]) is int
            and protocol["nfe"] > 0
            and batches == [count] * protocol["nfe"],
            "observed backbone NFE/batch count differs",
        )
        token_audit = bench.build_sampled_token_control_audit(
            sampler, input_ids, sampled_ids, texts
        )
        raw_generation = {
            "raw_model_texts": texts,
            "generation_protocol": protocol,
            "sampled_token_control_audit": token_audit,
        }
        write_new(output_dir / "raw_generation.json", encoded(raw_generation))
        decode_start = time.perf_counter()
        decode_timing = {}
        records = bench.decode_records(
            texts,
            use_bracket_safe=protocol["model_use_bracket_safe"],
            timing=decode_timing,
        )
        metrics, failures = score_records(records, count, sa_snapshot)
        metric_seconds = time.perf_counter() - decode_start
        require(
            len(records) == count
            and [r["sample_index"] for r in records] == list(range(count))
            and [r["raw_model_text"] for r in records] == texts,
            "raw CSV slot/text identity differs",
        )
        summary = {
            "schema_version": 1,
            "artifact_kind": ARTIFACT_KIND,
            "arm_kind": plan["spec"]["arm_kind"],
            "entry_id": plan["spec"]["entry_id"],
            "verification_status": "pending_independent_rescore",
            "final_promotion_eligible": False,
            "run": {
                "seed": plan["spec"]["seed"],
                "requested_sample_count": count,
                "single_generation_batch": True,
                "seed_configuration": seed,
                "generation_protocol": {
                    **protocol,
                    "inference_weights": identity["inference_weights"],
                },
            },
            "checkpoint_identity": identity,
            "config": {"source": plan["source_config"], "sampling": plan["sampling"]},
            "source": plan["source"],
            "inputs": plan["inputs"],
            "device": device_receipt,
            "software": {
                "python": sys.version,
                "executable": sys.executable,
                "packages": {
                    name: bench._package_version(name)
                    for name in (
                        "torch",
                        "lightning",
                        "transformers",
                        "omegaconf",
                        "numpy",
                        "safe-mol",
                        "rdkit",
                        "PyTDC",
                        "bionemo-moco",
                    )
                },
            },
            "tokenizer": bench.tokenizer_provenance(sampler.model.tokenizer),
            "implementation_inputs": dict(implementation.provenance),
            "metric_inputs": dict(sa_snapshot.provenance),
            "sampled_token_control_audit": token_audit,
            "metrics": metrics,
            "failure_counts": failures,
            "observed_work": {
                "backbone_calls": len(batches),
                "backbone_batch_sizes": batches,
                "candidate_equivalent_evaluations": sum(batches),
                "nfe_per_candidate": sum(batches) // count,
                "padded_sequence_length": token_audit["columns"],
                "padding_caveat": "batch-weighted NFE counts every padded row; token lengths and attention cost can differ",
            },
            "runtime_seconds": {
                "model_load_and_device_move": load_seconds,
                "model_sampling_and_tokenizer": sampling_seconds,
                "released_postprocessing": decode_timing["released_postprocessing"],
                "generation": sampling_seconds
                + decode_timing["released_postprocessing"],
                "decode_and_metrics": metric_seconds,
            },
        }
        payload = csv_payload(records)
        summary["raw_samples_csv"] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "row_count": count,
            "fields": list(bench.RAW_SAMPLE_FIELDS),
        }
        revalidate(plan["inputs"])
        require(
            bench.require_clean_pushed_source(plan["source"]["head"]) == plan["source"],
            "generation source changed",
        )
        bench.assert_runtime_module_provenance(implementation.provenance)
        write_new(output_dir / "raw_samples.csv", payload)
        write_new(output_dir / "summary.json", encoded(summary))
        terminal.update(
            status="completed",
            source=plan["source"],
            entry_id=plan["spec"]["entry_id"],
            seed=plan["spec"]["seed"],
            requested_sample_count=count,
        )
    except BaseException as error:
        terminal["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            revalidate(plan["inputs"])
            require(
                bench.require_clean_pushed_source(plan["source"]["head"])
                == plan["source"],
                "generation source changed before terminal",
            )
            terminal["final_input_validation"] = {"status": "verified"}
        except BaseException as error:
            terminal["status"] = "failed"
            terminal["final_input_validation"] = {
                "status": "failed",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            terminal.setdefault("error", terminal["final_input_validation"]["error"])
        terminal["runtime_seconds"] = time.perf_counter() - started
        terminal["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        terminal["outputs"] = {
            path.name: file_record(path)[0]
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        }
        write_new(output_dir / "terminal_manifest.json", encoded(terminal))
    require(terminal["status"] == "completed", "probe failed final input validation")
    return terminal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = preflight(
        spec_path=args.spec,
        expected_spec_sha256=args.expected_spec_sha256,
        input_root=args.input_root,
        expected_source_revision=args.expected_source_revision,
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "artifact_kind": ARTIFACT_KIND,
                    "spec": plan["spec"],
                    "inputs": plan["inputs"],
                    "source": plan["source"],
                    "checkpoint_deserialized": False,
                    "gpu_queried": False,
                    "metrics_evaluated": False,
                },
                indent=2,
            )
        )
        return 0
    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    result = execute(plan, device=args.device, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "verification_status": result["verification_status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
