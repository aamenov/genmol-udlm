"""Adversarial tiny CPU exports; no production weights, forwards, or oracles."""

import copy
import json
from pathlib import Path
import subprocess

import pytest
import torch

import genmol.model as model_module
from scripts.exps.denovo import benchmark
from scripts.udlm import export_zero_update_transfer as exporter
import test_udlm_zero_update_export as export_tests

export_case = export_tests.export_case
run_export = export_tests.run_export


@pytest.mark.parametrize("mutation", ["buffer_dtype", "tied_alias"])
def test_source_state_cannot_be_normalized_silently_before_ema_reference(
    export_case, mutation, monkeypatch
):
    payload = export_case["source_payload"]
    state = payload["state_dict"]
    if mutation == "buffer_dtype":
        state["backbone.synthetic_buffer"] = state["backbone.synthetic_buffer"].double()
    else:
        key = "backbone.bert.embeddings.word_embeddings.weight"
        state[key] = state[key].clone() + 1
    torch.save(payload, export_case["source_path"])
    export_case["kwargs"]["expected_source_sha256"] = exporter.artifact(
        export_case["source_path"]
    )["sha256"]
    monkeypatch.setattr(
        model_module.GenMol,
        "initialize_from_mdlm_checkpoint",
        lambda *_a, **_k: pytest.fail("invalid source reached transfer initialization"),
    )
    with pytest.raises((ValueError, RuntimeError)):
        run_export(export_case)
    output = export_case["kwargs"]["output_dir"]
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["status"] == "failed" and "checkpoint" not in manifest
    assert not (output / "transfer.ckpt").exists()


def _validate_saved(path, case, result, *, expected_sha256=None, transfer=None):
    return exporter.validate_checkpoint(
        path,
        expected_sha256 or exporter.artifact(path)["sha256"],
        case["config"],
        result["transfer_metadata"] if transfer is None else transfer,
        result["transfer_metadata"]["source_mdlm"],
        result["validation"],
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "transfer_updates",
        "transfer_trained",
        "transfer_missing",
        "step",
        "boolean_step",
        "epoch",
        "loops",
        "optimizer",
        "scheduler",
        "callbacks",
        "sampler",
        "ema_updates",
        "ema_decay",
        "ema_shadow",
        "base_parameter",
        "persistent_buffer",
        "buffer_dtype",
        "tied_alias",
        "film_projection",
        "ce_marker",
        "ce_metadata",
        "prior_marker",
        "prior_metadata",
        "stationary_prior",
        "active_mapping",
    ],
)
def test_rehashed_checkpoint_tampering_fails_semantic_validation(export_case, mutation):
    result = run_export(export_case, "x0_denoiser")
    original = Path(result["checkpoint"]["path"])
    payload = torch.load(original, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    if mutation == "transfer_updates":
        payload[exporter.TRANSFER_KEY]["udlm_optimizer_updates"] = 1
    elif mutation == "transfer_trained":
        payload[exporter.TRANSFER_KEY]["trained_as_ct_or_ce"] = True
    elif mutation == "transfer_missing":
        del payload[exporter.TRANSFER_KEY]
    elif mutation == "step":
        payload["global_step"] = 1
    elif mutation == "boolean_step":
        payload["global_step"] = False
    elif mutation == "epoch":
        payload["epoch"] = 1
    elif mutation == "loops":
        payload["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] = 1
    elif mutation in {"optimizer", "scheduler", "callbacks", "sampler"}:
        key = {"optimizer": "optimizer_states", "scheduler": "lr_schedulers"}.get(
            mutation, mutation
        )
        payload[key] = []
    elif mutation == "ema_updates":
        payload["ema"]["num_updates"] = 1
    elif mutation == "ema_decay":
        payload["ema"]["decay"] = 0.8
    elif mutation == "ema_shadow":
        payload["ema"]["shadow_params"][0] = payload["ema"]["shadow_params"][0] + 1
    elif mutation == "base_parameter":
        key = "backbone.bert.encoder.layer.0.attention.self.query.weight"
        state[key] = state[key] + 1
    elif mutation == "persistent_buffer":
        state["backbone.synthetic_buffer"] = torch.tensor([17, 24])
    elif mutation == "buffer_dtype":
        state["backbone.synthetic_buffer"] = state["backbone.synthetic_buffer"].double()
    elif mutation == "tied_alias":
        # Break one serialized alias only. Strict load alone silently lets the
        # later decoder alias overwrite this inconsistent embeddings entry.
        key = "backbone.bert.embeddings.word_embeddings.weight"
        state[key] = state[key].clone() + 1
    elif mutation == "film_projection":
        key = next(key for key in state if ".film_modulation." in key)
        state[key] = state[key] + 1
    elif mutation == "ce_marker":
        state[model_module.UDLM_DENOISER_STATE_KEY] = torch.tensor(0, dtype=torch.int64)
    elif mutation == "ce_metadata":
        del payload[model_module.UDLM_DENOISER_CHECKPOINT_KEY]
    elif mutation == "prior_marker":
        state[model_module.UDLM_MASK_RICH_STATE_KEY] = torch.tensor(
            0, dtype=torch.int64
        )
    elif mutation == "prior_metadata":
        payload[model_module.UDLM_PRIOR_CHECKPOINT_KEY]["mask_mixture_weight"] = 0.9
    elif mutation == "stationary_prior":
        state["mdlm.stationary_probs"] = state["mdlm.stationary_probs"].roll(1)
    elif mutation == "active_mapping":
        state["mdlm.token_to_diffusion_index"] = state[
            "mdlm.token_to_diffusion_index"
        ].roll(1)
    else:
        raise AssertionError(mutation)
    changed = original.parent / "tampered.ckpt"
    torch.save(payload, changed)
    assert exporter.artifact(changed)["sha256"] != result["checkpoint"]["sha256"]
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        _validate_saved(changed, export_case, result)
    assert exporter.artifact(original) == result["checkpoint"]


def test_changed_checkpoint_digest_rejected_before_deserialization(
    export_case, monkeypatch
):
    result = run_export(export_case)
    path = Path(result["checkpoint"]["path"])
    path.write_bytes(path.read_bytes() + b"tampered trailing bytes")
    monkeypatch.setattr(
        torch, "load", lambda *_a, **_k: pytest.fail("unpinned deserialization")
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        _validate_saved(
            path, export_case, result, expected_sha256=result["checkpoint"]["sha256"]
        )


def test_independent_manifest_claim_cannot_relabel_zero_update_checkpoint(export_case):
    result = run_export(export_case, "x0_denoiser")
    relabeled = copy.deepcopy(result["transfer_metadata"])
    relabeled["trained_as_ct_or_ce"] = True
    relabeled["training_objective_applied"] = "clean_token_cross_entropy"
    with pytest.raises(ValueError, match="checkpoint identity"):
        _validate_saved(
            result["checkpoint"]["path"], export_case, result, transfer=relabeled
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "nonpersistent_buffer",
        "film",
        "ema",
        "prior",
        "mapping",
        "mask_marker",
        "ce_marker",
    ],
)
def test_live_identity_validation_rejects_runtime_mutations(export_case, mutation):
    result = run_export(export_case, "x0_denoiser")
    model = model_module.GenMol.load_from_checkpoint(
        result["checkpoint"]["path"],
        map_location="cpu",
        strict=True,
        weights_only=False,
    ).eval()
    reference = result["transfer_metadata"]["source_mdlm"]
    assert exporter.validate_initialized(model, reference) == result["validation"]
    with torch.no_grad():
        if mutation == "nonpersistent_buffer":
            model.backbone.bert.embeddings.position_ids.add_(1)
        elif mutation == "film":
            next(
                p
                for n, p in model.backbone.named_parameters()
                if ".film_modulation." in n
            ).add_(1)
        elif mutation == "ema":
            model.ema.num_updates = 1
        elif mutation == "prior":
            model.mdlm.stationary_probs.copy_(model.mdlm.stationary_probs.roll(1))
        elif mutation == "mapping":
            model.mdlm.token_to_diffusion_index.copy_(
                model.mdlm.token_to_diffusion_index.roll(1)
            )
        elif mutation == "mask_marker":
            getattr(model, model_module.UDLM_MASK_RICH_STATE_KEY).zero_()
        elif mutation == "ce_marker":
            getattr(model, model_module.UDLM_DENOISER_STATE_KEY).zero_()
    with pytest.raises((ValueError, RuntimeError)):
        exporter.validate_initialized(model, reference)


@pytest.mark.parametrize("parameterization", ["raw_loo", "x0_denoiser"])
def test_existing_trained_ema_gate_rejects_truthful_zero_update_receipt(
    export_case, parameterization
):
    result = run_export(export_case, parameterization)
    weights = {"source": "ema", "ema_applied": True, "ema": result["validation"]["ema"]}
    assert weights["ema"]["num_updates"] == 0
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="positive"):
        benchmark.validate_inference_weights(weights, require_ema=True)


def test_export_does_not_change_existing_model_sampling_or_acceptance_source():
    # The exporter has no authority to weaken historical sampling/checkpoint gates.
    baseline = "48473d4febbd06d9bc07986ca96ceb93926c91ec"
    for relative in (
        "src/genmol/model.py",
        "src/genmol/sampler.py",
        "src/genmol/denoiser.py",
        "scripts/exps/denovo/benchmark.py",
        "scripts/exps/denovo/launch_benchmark.py",
        "scripts/udlm/rescore_denovo_run.py",
    ):
        previous = subprocess.check_output(
            ["git", "show", f"{baseline}:{relative}"], cwd=exporter.ROOT
        )
        assert (exporter.ROOT / relative).read_bytes() == previous, relative
