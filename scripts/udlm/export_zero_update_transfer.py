"""Export an explicitly untrained categorical interpretation of MDLM EMA weights.

No Trainer, optimizer, data loader, model forward, or molecular evaluation is used.
The positive-update EMA requirement in existing benchmark validators is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
TRANSFER_KEY = "zero_update_transfer_metadata"
SOURCE_STEP = 50000
COMPATIBILITY_LOOPS = {
    "fit_loop": {
        "epoch_progress": {"current": {"completed": 0}},
        "epoch_loop.batch_progress": {"current": {"completed": 0}},
    }
}
SOURCE_FILES = (
    "scripts/udlm/export_zero_update_transfer.py",
    "src/genmol/model.py",
    "src/genmol/backbone.py",
    "src/genmol/diffusion.py",
    "src/genmol/denoiser.py",
    "src/genmol/utils/checkpoint_io.py",
    "src/genmol/utils/ema.py",
    "src/genmol/utils/utils_data.py",
    "src/genmol/utils/utils_save.py",
    "src/genmol/utils/utils_moco.py",
)


def encode(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def artifact(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return {
        "path": str(path),
        "sha256": h.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def project_path(value, *, existing=True):
    path = Path(value).resolve(strict=existing)
    if not path.is_relative_to(PROJECT) or path == PROJECT:
        raise ValueError("all input/output artifacts must be inside the project")
    return path


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    head = git("rev-parse", "HEAD")
    if git("status", "--porcelain") or git("rev-parse", "@{upstream}") != head:
        raise ValueError("export requires a clean, pushed source checkout")
    return {
        "head": head,
        "files": {name: artifact(ROOT / name) for name in SOURCE_FILES},
    }


def validate_config(config):
    """Accept a resolved inference architecture, never a training launch config."""
    if not isinstance(config, dict) or set(config) != {"model", "training"}:
        raise ValueError("config must contain only model and training")
    if "${" in json.dumps(config, allow_nan=False):
        raise ValueError("config must already be fully resolved")
    training = config["training"]
    keys = {
        "diffusion",
        "ema",
        "antithetic_sampling",
        "sampling_eps",
        "global_mean_loss",
        "use_bracket_safe",
        "udlm",
    }
    if not isinstance(training, dict) or set(training) != keys:
        raise ValueError("training contains missing or unsupported inference fields")
    udlm = training["udlm"]
    required = {
        "prior_variant",
        "mask_mixture_weight",
        "parameterization",
        "empirical_uniform_mix",
        "noise_eps",
        "time_embedding_size",
        "conditioning_variant",
        "zero_init_conditioning",
        "mask_all_special_tokens",
        "exclude_special_tokens",
    }
    if not isinstance(udlm, dict) or set(udlm) != required:
        raise ValueError("udlm requires all explicit export settings")
    if (
        training["diffusion"] != "udlm"
        or udlm["prior_variant"] != "mask_rich_empirical"
        or udlm["parameterization"] not in {"raw_loo", "x0_denoiser"}
        or udlm["conditioning_variant"] != "film_adaln"
        or udlm["zero_init_conditioning"] is not False
        or udlm["exclude_special_tokens"] is not False
        or udlm["mask_all_special_tokens"] is not True
    ):
        raise ValueError(
            "export requires explicit MASK-rich raw/denoiser FiLM interpretation"
        )
    for name, value in (
        ("mask_mixture_weight", udlm["mask_mixture_weight"]),
        ("empirical_uniform_mix", udlm["empirical_uniform_mix"]),
        ("noise_eps", udlm["noise_eps"]),
        ("sampling_eps", training["sampling_eps"]),
        ("ema", training["ema"]),
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError(
                f"{name} must be a finite scalar strictly between zero and one"
            )
    for name in ("antithetic_sampling", "global_mean_loss", "use_bracket_safe"):
        if type(training[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if type(udlm["time_embedding_size"]) is not int or udlm["time_embedding_size"] <= 0:
        raise ValueError("time_embedding_size must be a positive integer")
    if not isinstance(config["model"], dict) or not config["model"]:
        raise ValueError("model must contain an explicit BERT architecture")
    return json.loads(encode(config))


def tensor_record(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise ValueError("all serialized tensors must be CPU tensors")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("nonfinite tensor in transfer state")
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def tensor_manifest(state):
    return {name: tensor_record(value) for name, value in state.items()}


def _ema_check(model, *, expected_updates):
    import torch

    if (
        model.ema is None
        or type(model.ema.num_updates) is not int
        or model.ema.num_updates != expected_updates
    ):
        raise ValueError("EMA update count differs from its truthful export identity")
    parameters = [
        (n, p) for n, p in model.backbone.named_parameters() if p.requires_grad
    ]
    shadows = model.ema.shadow_params
    if not parameters or len(parameters) != len(shadows):
        raise ValueError("EMA shadow count differs from named backbone parameters")
    for (name, parameter), shadow in zip(parameters, shadows, strict=True):
        if tensor_record(parameter) != tensor_record(shadow) or not torch.equal(
            parameter, shadow
        ):
            raise ValueError(f"EMA shadow does not equal initialized parameter {name}")
    return {
        "decay": float(model.ema.decay),
        "num_updates": expected_updates,
        "shadow_parameter_count": len(shadows),
    }


def inspect_source(path, expected_sha256):
    """Reconstruct MDLM independently, apply its EMA, then name ALL state entries."""
    import torch
    from omegaconf import OmegaConf
    from genmol.model import GenMol
    from genmol.utils.checkpoint_io import verified_checkpoint_file

    with verified_checkpoint_file(path, expected_sha256=expected_sha256) as (
        stream,
        identity,
    ):
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    if (
        type(checkpoint.get("global_step")) is not int
        or checkpoint["global_step"] != SOURCE_STEP
    ):
        raise ValueError("source must record the actual MDLM50000 global step")
    config = OmegaConf.to_container(
        OmegaConf.create(checkpoint["hyper_parameters"]["config"]), resolve=False
    )
    if config["training"].get("diffusion", "mdlm") != "mdlm":
        raise ValueError("source checkpoint is not MDLM")
    if any(key.startswith("udlm_") or key == TRANSFER_KEY for key in checkpoint):
        raise ValueError("source checkpoint already carries a UDLM/transfer identity")
    source = GenMol(OmegaConf.create(config)).cpu().eval()
    source.on_load_checkpoint(checkpoint)
    source.load_state_dict(checkpoint["state_dict"], strict=True)
    if tensor_manifest(source.state_dict()) != tensor_manifest(
        checkpoint["state_dict"]
    ):
        raise ValueError("source raw state has casting or inconsistent tied aliases")
    if (
        source.ema is None
        or type(source.ema.num_updates) is not int
        or source.ema.num_updates <= 0
    ):
        raise ValueError("source requires an EMA with a positive actual update count")
    decay = source.ema.decay
    if type(decay) not in (float, int) or not math.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("source EMA decay is invalid")
    params = [p for p in source.backbone.parameters() if p.requires_grad]
    if len(params) != len(source.ema.shadow_params) or not params:
        raise ValueError("source EMA count differs from strict MDLM parameter topology")
    # Shape/dtype checks precede copy_to so broadcasting/casting cannot hide defects.
    for parameter, shadow in zip(params, source.ema.shadow_params, strict=True):
        p, s = tensor_record(parameter), tensor_record(shadow)
        if (p["shape"], p["dtype"]) != (s["shape"], s["dtype"]):
            raise ValueError("source EMA shape/dtype differs from MDLM parameters")
    source.ema.copy_to(source.backbone.parameters())
    ema = _ema_check(source, expected_updates=source.ema.num_updates)
    # state_dict includes persistent buffers AND every tied state alias.
    state = tensor_manifest(source.backbone.state_dict())
    return {
        "checkpoint": {
            "path": identity.resolved_path,
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        },
        "global_step": SOURCE_STEP,
        "diffusion": "mdlm",
        "weights": "ema",
        "config_sha256": digest(config),
        "config_sha256_scope": "complete_unresolved_source_configuration_no_eager_interpolation",
        "backbone_config": source.backbone.config.to_dict(),
        "ema": ema,
        "backbone_state": state,
        "backbone_state_sha256": digest(state),
        "backbone_buffers": tensor_manifest(
            dict(source.backbone.named_buffers(remove_duplicate=False))
        ),
        "parameter_names": [name for name, _ in source.backbone.named_parameters()],
        "buffer_names": [name for name, _ in source.backbone.named_buffers()],
    }


def validate_initialized(model, reference):
    import torch
    from genmol.backbone import is_conditioning_parameter_name

    if encode(model.backbone.config.to_dict()) != encode(reference["backbone_config"]):
        raise ValueError("target BERT architecture differs from source MDLM")
    state = model.backbone.state_dict()
    base = {n: t for n, t in state.items() if not is_conditioning_parameter_name(n)}
    if tensor_manifest(base) != reference["backbone_state"]:
        raise ValueError(
            "base backbone parameters, buffers, or tied aliases differ from source EMA"
        )
    buffers = {
        n: t
        for n, t in model.backbone.named_buffers(remove_duplicate=False)
        if not is_conditioning_parameter_name(n)
    }
    if tensor_manifest(buffers) != reference["backbone_buffers"]:
        raise ValueError("base backbone buffer values differ from source MDLM")
    film = {
        n: p for n, p in model.backbone.named_parameters() if ".film_modulation." in n
    }
    if len(film) != 2 * model.config.model.num_hidden_layers or not film:
        raise ValueError("FiLM output projection topology differs")
    for name, value in film.items():
        tensor_record(value)
        if torch.count_nonzero(value).item():
            raise ValueError(f"FiLM has nonzero effect: {name}")
    model._validate_runtime_udlm_prior_identity()
    model._validate_runtime_udlm_conditioning_identity()
    model._validate_udlm_parameterization_state_dict(model.state_dict())
    return {
        "source_base_state_sha256": digest(tensor_manifest(base)),
        "film_zero_projection_tensors": len(film),
        "ema": _ema_check(model, expected_updates=0),
        "state": tensor_manifest(model.state_dict()),
    }


def make_checkpoint(model, transfer):
    import lightning
    from genmol.model import (
        UDLM_PRIOR_CHECKPOINT_KEY,
        UDLM_CONDITIONING_CHECKPOINT_KEY,
        UDLM_DENOISER_CHECKPOINT_KEY,
        UDLM_DENOISER_METADATA,
    )

    checkpoint = {
        "pytorch-lightning_version": lightning.__version__,
        "state_dict": model.state_dict(),
        "hyper_parameters": {"config": model.config},
        "global_step": 0,
        "epoch": 0,
        "loops": json.loads(encode(COMPATIBILITY_LOOPS)),
        "ema": model.ema.state_dict(),
        TRANSFER_KEY: transfer,
        UDLM_PRIOR_CHECKPOINT_KEY: model.udlm_prior_metadata.to_dict(),
        UDLM_CONDITIONING_CHECKPOINT_KEY: model.udlm_conditioning_metadata.to_dict(),
    }
    if model.udlm_parameterization == "x0_denoiser":
        checkpoint[UDLM_DENOISER_CHECKPOINT_KEY] = dict(UDLM_DENOISER_METADATA)
    return checkpoint


def validate_checkpoint(
    path, expected_sha256, config, transfer, reference, initialized
):
    """Validate exporter provenance, then exercise the unchanged strict CPU loader."""
    import torch
    from omegaconf import OmegaConf
    from genmol.model import GenMol
    from genmol.utils.checkpoint_io import verified_checkpoint_file

    with verified_checkpoint_file(path, expected_sha256=expected_sha256) as (stream, _):
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        # Strict PyTorch loading still permits dtype casts and last-write-wins
        # tied aliases. Inspect actual serialized bytes before either can hide a defect.
        if tensor_manifest(checkpoint["state_dict"]) != initialized["state"]:
            raise ValueError(
                "serialized state differs before strict loader casts or aliases"
            )
        if (
            encode(checkpoint.get(TRANSFER_KEY)) != encode(transfer)
            or type(checkpoint.get("global_step")) is not int
            or checkpoint["global_step"] != 0
            or type(checkpoint.get("epoch")) is not int
            or checkpoint["epoch"] != 0
            or encode(checkpoint.get("loops")) != encode(COMPATIBILITY_LOOPS)
            or any(
                k in checkpoint
                for k in ("optimizer_states", "lr_schedulers", "callbacks", "sampler")
            )
            or encode(
                OmegaConf.to_container(
                    checkpoint["hyper_parameters"]["config"], resolve=True
                )
            )
            != encode(config)
        ):
            raise ValueError("inference-only transfer checkpoint identity differs")
        del checkpoint
        stream.seek(0)
        restored = GenMol.load_from_checkpoint(
            stream, map_location="cpu", strict=True, weights_only=False
        ).eval()
    observed = validate_initialized(restored, reference)
    if observed != initialized:
        raise ValueError("strict serialization roundtrip altered initialized state")
    return observed


def export_checkpoint(
    *, config_path, source_checkpoint, expected_source_sha256, seed, output_dir
):
    if (
        Path(sys.prefix).resolve() != PROJECT / ".venv"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise ValueError(
            "export requires project .venv, no exposed CUDA, and offline assets"
        )
    import torch
    from omegaconf import OmegaConf
    from genmol.model import GenMol
    from genmol.utils.checkpoint_io import validate_sha256, verified_checkpoint_file
    import inspect

    if Path(inspect.getfile(GenMol)).resolve() != ROOT / "src/genmol/model.py":
        raise ValueError(
            "GenMol import does not come from the recorded source checkout"
        )

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("initialization seed must be an integer in [0, 2**32)")
    validate_sha256(expected_source_sha256)
    config_path = project_path(config_path)
    source_checkpoint = project_path(source_checkpoint)
    output_dir = project_path(output_dir, existing=False)
    config_input = artifact(config_path)
    config = validate_config(
        OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    )
    source = source_identity()
    # Exclusive reservation precedes checkpoint deserialization/model construction.
    output_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "schema_version": 1,
        "operation": "zero_update_mdlm_ema_transfer",
        "source": source,
        "config_input": config_input,
        "config": config,
        "source_checkpoint": str(source_checkpoint),
        "expected_source_sha256": expected_source_sha256,
        "seed": seed,
        "device": "cpu",
        "training_updates": 0,
        "example_exposures": 0,
    }
    (output_dir / "request.json").write_bytes(encode(request))
    started = time.monotonic()
    terminal = {
        "schema_version": 1,
        "status": "failed",
        "request_sha256": hashlib.sha256(encode(request)).hexdigest(),
    }
    try:
        reference = inspect_source(source_checkpoint, expected_source_sha256)
        torch.manual_seed(seed)
        model = GenMol(OmegaConf.create(config)).cpu().eval()
        initialization = model.initialize_from_mdlm_checkpoint(
            source_checkpoint, use_ema=True, expected_sha256=expected_source_sha256
        )
        initialized = validate_initialized(model, reference)
        transfer = {
            "schema_version": 1,
            "artifact_kind": "inference_only_zero_update_transfer",
            "udlm_optimizer_updates": 0,
            "udlm_example_exposures": 0,
            "trained_as_ct_or_ce": False,
            "training_objective_applied": None,
            "selected_parameterization": config["training"]["udlm"]["parameterization"],
            "interpretation": "frozen_MDLM_EMA_logits_reinterpreted_without_calibration_or_UDLM_training",
            "source_mdlm": reference,
            "initialization_seed": seed,
            "config_sha256": digest(config),
            "exporter_source": source,
            "prior_metadata_sha256": digest(model.udlm_prior_metadata.to_dict()),
            "conditioning_metadata_sha256": digest(
                model.udlm_conditioning_metadata.to_dict()
            ),
            "loop_counter_scope": "zero_loader_compatibility_only_no_Trainer_or_training_history",
            "benchmark_status": "not_accepted_by_existing_positive_update_EMA_gate",
        }
        checkpoint_path = output_dir / "transfer.ckpt"
        with checkpoint_path.open("xb") as stream:
            torch.save(make_checkpoint(model, transfer), stream)
        checkpoint_record = artifact(checkpoint_path)
        del model
        validate_checkpoint(
            checkpoint_path,
            checkpoint_record["sha256"],
            config,
            transfer,
            reference,
            initialized,
        )
        with verified_checkpoint_file(
            source_checkpoint, expected_sha256=expected_source_sha256
        ):
            pass
        if source_identity() != source or artifact(config_path) != config_input:
            raise ValueError("source/config bytes changed during export")
        (output_dir / "resolved_config.json").write_bytes(encode(config))
        (output_dir / "exporter_source.py").write_bytes(Path(__file__).read_bytes())
        terminal.update(
            status="completed",
            checkpoint=checkpoint_record,
            transfer_metadata=transfer,
            initialization=initialization,
            validation=initialized,
            roundtrip="strict_CPU_GenMol.load_from_checkpoint",
            outputs={p.name: artifact(p) for p in output_dir.iterdir()},
            software={
                name: importlib.metadata.version(name)
                for name in ("torch", "lightning", "transformers", "omegaconf")
            },
        )
    except BaseException as error:
        terminal["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        terminal["runtime_seconds"] = time.monotonic() - started
        with (output_dir / "manifest.json").open("xb") as stream:
            stream.write(encode(terminal))
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != PROJECT / ".venv":
        parser.error("use the project .venv")
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    result = export_checkpoint(
        config_path=args.config,
        source_checkpoint=args.source_checkpoint,
        expected_source_sha256=args.expected_source_sha256,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "checkpoint": result["checkpoint"]}))


if __name__ == "__main__":
    main()
