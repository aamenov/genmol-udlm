"""Real tiny CPU serialization; all models/tokenizers/artifacts are synthetic."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import genmol.model as model_module
from scripts.udlm import export_zero_update_transfer as exporter
from test_udlm_model import (
    _Tokenizer,
    _config,
    _frequency_payload,
    _install_frequency_artifact,
)


@pytest.fixture
def export_case(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(model_module, "get_tokenizer", lambda: _Tokenizer())
    _install_frequency_artifact(monkeypatch, tmp_path, _frequency_payload())
    # Both source and target constructors get a persistent buffer in addition to
    # real BERT's nonpersistent position/token-type buffers. No fake backbone.
    original_init = model_module.BertForMaskedLM.__init__

    def init_with_buffer(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.register_buffer(
            "synthetic_buffer", torch.tensor([8, 11], dtype=torch.int64)
        )

    monkeypatch.setattr(model_module.BertForMaskedLM, "__init__", init_with_buffer)
    source_config = _config()
    source_config.training.ema = 0.9
    source = model_module.GenMol(source_config)
    source.ema.num_updates = (
        123  # Deliberately differs from global_step: preserve actual field.
    )
    for i, shadow in enumerate(source.ema.shadow_params):
        shadow.add_(0.002 * (i + 1))  # Prove EMA is copied instead of raw model state.
    source.backbone.synthetic_buffer.copy_(torch.tensor([17, 23]))
    payload = {
        "global_step": 50000,
        "state_dict": source.state_dict(),
        "hyper_parameters": {"config": source_config},
        "ema": source.ema.state_dict(),
        "loops": copy.deepcopy(exporter.COMPATIBILITY_LOOPS),
    }
    source_path = tmp_path / "synthetic_mdlm.ckpt"
    torch.save(payload, source_path)
    target = _config(
        diffusion="udlm",
        prior_variant="mask_rich_empirical",
        conditioning_variant="film_adaln",
        zero_init_conditioning=False,
    )
    del target.optim
    target.training.ema = 0.9
    target.training.udlm.mask_mixture_weight = 0.9999
    target.training.udlm.parameterization = "raw_loo"
    target.training.udlm.mask_all_special_tokens = True
    config = OmegaConf.to_container(target, resolve=True)
    config_path = tmp_path / "config.json"
    config_path.write_bytes(exporter.encode(config))
    monkeypatch.setattr(
        exporter,
        "source_identity",
        lambda: {
            "head": "synthetic_test_only",
            "files": {
                name: exporter.artifact(exporter.ROOT / name)
                for name in exporter.SOURCE_FILES
            },
        },
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Trainer/optimizer/forward/CUDA/oracle work is forbidden")

    monkeypatch.setattr(model_module.GenMol, "configure_optimizers", forbidden)
    monkeypatch.setattr(model_module.GenMol, "on_save_checkpoint", forbidden)
    monkeypatch.setattr(model_module.GenMol, "forward", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    return {
        "config": config,
        "config_path": config_path,
        "source_path": source_path,
        "source_payload": payload,
        "source_model": source,
        "kwargs": {
            "config_path": config_path,
            "source_checkpoint": source_path,
            "expected_source_sha256": exporter.artifact(source_path)["sha256"],
            "seed": 2800,
            "output_dir": tmp_path / "export",
        },
    }


def run_export(case, parameterization="raw_loo"):
    case["config"]["training"]["udlm"]["parameterization"] = parameterization
    case["config_path"].write_bytes(exporter.encode(case["config"]))
    return exporter.export_checkpoint(**case["kwargs"])


@pytest.mark.parametrize("parameterization", ["raw_loo", "x0_denoiser"])
def test_real_strict_lightning_roundtrip_copies_ema_and_all_named_buffers(
    export_case, parameterization
):
    result = run_export(export_case, parameterization)
    assert result["status"] == "completed"
    transfer = result["transfer_metadata"]
    assert transfer["source_mdlm"]["ema"]["num_updates"] == 123
    assert transfer["source_mdlm"]["global_step"] == 50000
    assert transfer["udlm_optimizer_updates"] == transfer["udlm_example_exposures"] == 0
    assert transfer["trained_as_ct_or_ce"] is False
    assert transfer["training_objective_applied"] is None
    assert transfer["selected_parameterization"] == parameterization
    checkpoint = torch.load(
        result["checkpoint"]["path"], weights_only=False, map_location="cpu"
    )
    assert checkpoint["ema"]["num_updates"] == checkpoint["global_step"] == 0
    assert (
        not {"optimizer_states", "lr_schedulers", "callbacks", "sampler"}
        & checkpoint.keys()
    )
    expected = export_case["source_model"]
    expected.ema.copy_to(expected.backbone.parameters())
    state = checkpoint["state_dict"]
    for name, value in expected.backbone.state_dict().items():
        assert torch.equal(state["backbone." + name], value), name
    for alias in (
        "backbone.bert.embeddings.word_embeddings.weight",
        "backbone.cls.predictions.decoder.weight",
    ):
        assert torch.equal(
            state[alias], expected.backbone.bert.embeddings.word_embeddings.weight
        )
    assert "bert.embeddings.position_ids" in transfer["source_mdlm"]["backbone_buffers"]
    assert "synthetic_buffer" in transfer["source_mdlm"]["backbone_buffers"]
    assert (model_module.UDLM_DENOISER_STATE_KEY in state) == (
        parameterization == "x0_denoiser"
    )
    assert result["validation"]["film_zero_projection_tensors"] == 4
    for record in result["outputs"].values():
        assert exporter.artifact(record["path"]) == record
    request = Path(export_case["kwargs"]["output_dir"]) / "request.json"
    assert hashlib.sha256(request.read_bytes()).hexdigest() == result["request_sha256"]


def test_preexisting_output_fails_before_any_source_deserialization(
    export_case, monkeypatch
):
    output = export_case["kwargs"]["output_dir"]
    output.mkdir()
    (output / "keep").write_text("preserve")
    monkeypatch.setattr(
        exporter, "inspect_source", lambda *_: pytest.fail("source read")
    )
    with pytest.raises(FileExistsError):
        run_export(export_case)
    assert (output / "keep").read_text() == "preserve"


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_step",
        "source_udlm",
        "missing_ema",
        "nonfinite_ema",
        "wrong_ema_dtype",
        "missing_base_state",
        "unknown_state",
    ],
)
def test_source_tamper_is_rejected_before_transfer_initialization(
    export_case, mutation, monkeypatch
):
    payload = export_case["source_payload"]
    if mutation == "wrong_step":
        payload["global_step"] = 49999
    elif mutation == "source_udlm":
        payload["hyper_parameters"]["config"].training.diffusion = "udlm"
    elif mutation == "missing_ema":
        del payload["ema"]
    elif mutation == "nonfinite_ema":
        payload["ema"]["shadow_params"][0][0, 0] = float("nan")
    elif mutation == "wrong_ema_dtype":
        payload["ema"]["shadow_params"][0] = payload["ema"]["shadow_params"][0].double()
    elif mutation == "missing_base_state":
        del payload["state_dict"]["backbone.synthetic_buffer"]
    else:
        payload["state_dict"]["unknown_extra"] = torch.tensor(0)
    torch.save(payload, export_case["source_path"])
    export_case["kwargs"]["expected_source_sha256"] = exporter.artifact(
        export_case["source_path"]
    )["sha256"]
    monkeypatch.setattr(
        model_module.GenMol,
        "initialize_from_mdlm_checkpoint",
        lambda *_args, **_kw: pytest.fail("initialization called"),
    )
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        run_export(export_case)
    terminal = json.loads(
        (export_case["kwargs"]["output_dir"] / "manifest.json").read_bytes()
    )
    assert terminal["status"] == "failed" and "checkpoint" not in terminal


def test_wrong_source_digest_never_deserializes(export_case, monkeypatch):
    export_case["kwargs"]["expected_source_sha256"] = "0" * 64
    monkeypatch.setattr(
        torch, "load", lambda *_args, **_kw: pytest.fail("deserialized unpinned input")
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        run_export(export_case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mask_mixture_weight", 0),
        ("mask_mixture_weight", 1),
        ("mask_mixture_weight", True),
        ("parameterization", "unknown"),
        ("exclude_special_tokens", True),
        ("zero_init_conditioning", True),
        ("empirical_uniform_mix", 0),
        ("mask_all_special_tokens", False),
    ],
)
def test_invalid_contract_fails_before_output_or_model(
    export_case, field, value, monkeypatch
):
    config = export_case["config"]
    config["training"]["udlm"][field] = value
    export_case["config_path"].write_bytes(exporter.encode(config))
    monkeypatch.setattr(
        exporter, "inspect_source", lambda *_: pytest.fail("source read")
    )
    with pytest.raises(ValueError):
        exporter.export_checkpoint(**export_case["kwargs"])
    assert not export_case["kwargs"]["output_dir"].exists()


def test_post_roundtrip_config_race_leaves_failure_receipt(export_case, monkeypatch):
    validate = exporter.validate_checkpoint

    def change_config(*args, **kwargs):
        result = validate(*args, **kwargs)
        export_case["config_path"].write_text("{}")
        return result

    monkeypatch.setattr(exporter, "validate_checkpoint", change_config)
    with pytest.raises(ValueError, match="source/config bytes changed"):
        run_export(export_case)
    terminal = json.loads(
        (export_case["kwargs"]["output_dir"] / "manifest.json").read_bytes()
    )
    assert terminal["status"] == "failed" and "checkpoint" not in terminal
    assert (export_case["kwargs"]["output_dir"] / "transfer.ckpt").is_file()


def test_same_shape_activation_change_is_not_a_frozen_backbone(export_case):
    export_case["config"]["model"]["hidden_act"] = "relu"
    export_case["config_path"].write_bytes(exporter.encode(export_case["config"]))
    with pytest.raises(ValueError, match="BERT architecture differs"):
        exporter.export_checkpoint(**export_case["kwargs"])
    manifest = json.loads(
        (export_case["kwargs"]["output_dir"] / "manifest.json").read_bytes()
    )
    assert manifest["status"] == "failed"
    assert not (export_case["kwargs"]["output_dir"] / "transfer.ckpt").exists()


def test_unused_source_interpolations_remain_unresolved_and_fingerprinted(export_case):
    config = export_case["source_payload"]["hyper_parameters"]["config"]
    config.callback = {"dirpath": "${cwd:}"}
    config.unused_schedule = {"future_horizon": "${unused_schedule_resolver:steps}"}
    assert not OmegaConf.has_resolver("cwd")
    assert not OmegaConf.has_resolver("unused_schedule_resolver")
    torch.save(export_case["source_payload"], export_case["source_path"])
    export_case["kwargs"]["expected_source_sha256"] = exporter.artifact(
        export_case["source_path"]
    )["sha256"]
    result = run_export(export_case, "x0_denoiser")
    reference = result["transfer_metadata"]["source_mdlm"]
    assert reference["config_sha256"] == exporter.digest(
        OmegaConf.to_container(config, resolve=False)
    )
    assert (
        reference["config_sha256_scope"]
        == "complete_unresolved_source_configuration_no_eager_interpolation"
    )
    assert not OmegaConf.has_resolver("cwd")
    assert not OmegaConf.has_resolver("unused_schedule_resolver")
    assert result["status"] == "completed"


def test_unresolved_required_source_model_field_still_fails(export_case, monkeypatch):
    from omegaconf.errors import UnsupportedInterpolationType

    config = export_case["source_payload"]["hyper_parameters"]["config"]
    config.model.hidden_size = "${unknown_required_model:width}"
    torch.save(export_case["source_payload"], export_case["source_path"])
    export_case["kwargs"]["expected_source_sha256"] = exporter.artifact(
        export_case["source_path"]
    )["sha256"]
    monkeypatch.setattr(
        model_module.GenMol,
        "initialize_from_mdlm_checkpoint",
        lambda *_a, **_k: pytest.fail("required interpolation bypassed"),
    )
    with pytest.raises(UnsupportedInterpolationType):
        run_export(export_case)
    manifest = json.loads(
        (export_case["kwargs"]["output_dir"] / "manifest.json").read_bytes()
    )
    assert manifest["status"] == "failed"
    assert not (export_case["kwargs"]["output_dir"] / "transfer.ckpt").exists()
