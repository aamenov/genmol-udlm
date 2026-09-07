"""Synthetic probe boundaries: no real checkpoints, chemistry, or GPU work."""

import builtins
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import genmol.model as model_module
from scripts.udlm import run_transfer_denovo_probe as probe
from scripts.udlm import export_zero_update_transfer as exporter
import test_udlm_zero_update_export as export_tests

export_case = export_tests.export_case


def write_json(path, value):
    path.write_bytes(probe.encoded(value))
    return probe.file_record(path)[0]


def sampling(parameterization="raw_loo", prior="0" * 64):
    return probe.bench.validate_sampling_config(
        {
            "diffusion_type": "udlm",
            "softmax_temp": 0.5,
            "randomness": 0,
            "min_add_len": 1,
            "num_steps": 4,
            "inference_eps": 1e-5,
            "exclude_special_tokens": False,
            "prior_variant": "mask_rich_empirical",
            "prior_metadata_sha256": prior,
            "raw_loo_top_p": 1,
            "parameterization": parameterization,
        }
    )


@pytest.fixture
def reference_spec(tmp_path, monkeypatch):
    checkpoint = tmp_path / "not_a_checkpoint.ckpt"
    checkpoint.write_bytes(b"metadata-only test; must never deserialize")
    config = write_json(
        tmp_path / "sampling.yaml",
        {
            "diffusion_type": "mdlm",
            "softmax_temp": 1,
            "randomness": 2,
            "min_add_len": 1,
        },
    )
    design = write_json(tmp_path / "design.json", {"purpose": "synthetic"})
    direct = {"synthetic_input": design}
    monkeypatch.setattr(probe, "direct_input_records", lambda _: direct)
    source = {"head": "1" * 40, "upstream": "1" * 40}
    monkeypatch.setattr(probe.bench, "require_clean_pushed_source", lambda _: source)
    spec = {
        "schema_version": 1,
        "artifact_kind": probe.ARTIFACT_KIND,
        "entry_id": "synthetic_reference",
        "arm_kind": probe.REFERENCE_ARM,
        "seed": 2800,
        "num_samples": 2,
        "checkpoint": probe.file_record(checkpoint)[0],
        "sampling_config": config,
        "export_manifest": None,
        "design": design,
        "input_sha256": {k: v["sha256"] for k, v in direct.items()},
    }
    path = tmp_path / "spec.json"
    record = write_json(path, spec)
    return spec, path, record, source


def preview(case):
    _spec, path, record, source = case
    return probe.preflight(
        spec_path=path,
        expected_spec_sha256=record["sha256"],
        input_root=path.parent,
        expected_source_revision=source["head"],
    )


def test_default_cli_is_metadata_only_even_when_torch_import_is_forbidden(
    reference_spec, monkeypatch, capsys
):
    _spec, path, record, source = reference_spec
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "genmol", "rdkit", "tdc"}:
            pytest.fail("heavy import during dry-run: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for name in ("execute", "load_sampler", "prepare_device", "score_records"):
        monkeypatch.setattr(
            probe, name, lambda *_a, **_k: pytest.fail("non-preview work")
        )
    assert (
        probe.main(
            [
                "--input-root",
                str(path.parent),
                "--spec",
                str(path),
                "--expected-spec-sha256",
                record["sha256"],
                "--expected-source-revision",
                source["head"],
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run"
    assert (
        result["checkpoint_deserialized"]
        is result["gpu_queried"]
        is result["metrics_evaluated"]
        is False
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("num_samples", 0),
        ("num_samples", 1001),
        ("arm_kind", "trained_ce"),
        ("export_manifest", {}),
    ],
)
def test_invalid_spec_rejected_without_runtime_work(reference_spec, field, value):
    spec, path, record, source = reference_spec
    spec[field] = value
    changed = write_json(path, spec)
    with pytest.raises(ValueError):
        preview((spec, path, changed, source))


def test_hash_and_guidance_rejections_happen_in_preflight(reference_spec):
    spec, path, record, source = reference_spec
    with pytest.raises(ValueError, match="hash differs"):
        preview((spec, path, {**record, "sha256": "0" * 64}, source))
    config_path = Path(spec["sampling_config"]["path"])
    config = json.loads(config_path.read_bytes())
    config["context_guidance"] = {"method": "posterior_context"}
    spec["sampling_config"] = write_json(config_path, config)
    record = write_json(path, spec)
    with pytest.raises(ValueError, match="context guidance|unsupported probe sampling"):
        preview((spec, path, record, source))


@pytest.fixture
def transferred(export_case, monkeypatch):
    original_source = exporter.source_identity
    monkeypatch.setattr(
        exporter, "source_identity", lambda: {**original_source(), "head": "2" * 40}
    )
    result = export_tests.run_export(export_case, "x0_denoiser")
    normalized = sampling(
        "x0_denoiser", result["transfer_metadata"]["prior_metadata_sha256"]
    )
    manifest = Path(result["checkpoint"]["path"]).parent / "manifest.json"
    proof = probe.read_export_proof(
        probe.file_record(manifest)[0],
        result["checkpoint"],
        normalized,
        manifest.parent,
    )
    checkpoint = torch.load(
        result["checkpoint"]["path"], map_location="cpu", weights_only=False
    )
    model = model_module.GenMol.load_from_checkpoint(
        result["checkpoint"]["path"],
        map_location="cpu",
        strict=True,
        weights_only=False,
    ).eval()
    sampler = SimpleNamespace(
        model=model,
        diffusion_type="udlm",
        inference_weights={
            "source": "ema",
            "ema_applied": True,
            "ema": result["validation"]["ema"],
        },
    )
    plan = {
        "spec": {"arm_kind": probe.TRANSFER_ARM},
        "sampling": normalized,
        "proof": proof,
    }
    return result, checkpoint, sampler, plan, manifest


def test_exact_zero_transfer_passes_separate_probe_gate_but_not_trained_gate(
    transferred,
):
    _result, checkpoint, sampler, plan, _path = transferred
    value = probe.validate_loaded_checkpoint(checkpoint, sampler, plan)
    assert value["global_step"] == 0 and value["kind"] == probe.TRANSFER_ARM
    assert value["source_checkpoint_revalidated_this_run"] is False
    with pytest.raises(probe.bench.BenchmarkConfigurationError, match="positive"):
        probe.bench.validate_inference_weights(
            sampler.inference_weights, require_ema=True
        )


def test_actual_tiny_transfer_loader_places_all_state_on_cpu_before_device_probe(
    transferred, monkeypatch
):
    result, _checkpoint, _sampler, plan, _path = transferred
    plan["inputs"] = {"checkpoint": result["checkpoint"]}
    original_load = torch.load
    placements = []

    def cpu_load(*args, **kwargs):
        placements.append(kwargs.get("map_location"))
        assert kwargs.get("map_location") == "cpu"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", cpu_load)
    monkeypatch.setattr(
        probe, "prepare_device", lambda *_: pytest.fail("early device probe")
    )
    sampler, identity = probe.load_sampler(
        plan, SimpleNamespace(length_distribution={1: 1.0})
    )
    assert placements and all(value == "cpu" for value in placements)
    assert all(t.device.type == "cpu" for t in sampler.model.state_dict().values())
    assert identity["global_step"] == 0
    assert sampler.inference_weights["ema_applied"] is True


@pytest.mark.parametrize(
    "mutation",
    ["step", "training", "state_alias", "ema_positive", "film", "prior", "ce_marker"],
)
def test_loaded_transfer_semantic_mismatches_rejected(transferred, mutation):
    _result, checkpoint, sampler, plan, _path = transferred
    if mutation == "step":
        checkpoint["global_step"] = 1
    elif mutation == "training":
        checkpoint["optimizer_states"] = []
    elif mutation == "state_alias":
        key = "backbone.bert.embeddings.word_embeddings.weight"
        checkpoint["state_dict"][key] = checkpoint["state_dict"][key].clone() + 1
    elif mutation == "ema_positive":
        sampler.inference_weights["ema"]["num_updates"] = 1
    elif mutation == "film":
        with torch.no_grad():
            next(
                p
                for n, p in sampler.model.backbone.named_parameters()
                if ".film_modulation." in n
            ).add_(1)
    elif mutation == "prior":
        plan["sampling"]["prior_metadata_sha256"] = "0" * 64
    elif mutation == "ce_marker":
        getattr(sampler.model, model_module.UDLM_DENOISER_STATE_KEY).zero_()
    with pytest.raises((ValueError, RuntimeError)):
        probe.validate_loaded_checkpoint(checkpoint, sampler, plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "failed",
        "trained",
        "updates",
        "parameterization",
        "prior",
        "source_history",
        "tensor_proof",
        "outputs",
        "ema_positive",
    ],
)
def test_rehashed_export_proof_cannot_relabel_transfer(transferred, mutation):
    result, _checkpoint, _sampler, plan, path = transferred
    manifest = copy.deepcopy(result)
    transfer = manifest["transfer_metadata"]
    if mutation == "failed":
        manifest["status"] = "failed"
    elif mutation == "trained":
        transfer["trained_as_ct_or_ce"] = True
    elif mutation == "updates":
        transfer["udlm_optimizer_updates"] = 1
    elif mutation == "parameterization":
        transfer["selected_parameterization"] = "raw_loo"
    elif mutation == "prior":
        transfer["prior_metadata_sha256"] = "0" * 64
    elif mutation == "source_history":
        transfer["source_mdlm"]["ema"]["num_updates"] = 0
    elif mutation == "tensor_proof":
        transfer["source_mdlm"]["backbone_state_sha256"] = "0" * 64
    elif mutation == "outputs":
        del manifest["outputs"]["request.json"]
    elif mutation == "ema_positive":
        manifest["validation"]["ema"]["num_updates"] = 1
    reference = write_json(path, manifest)
    with pytest.raises((ValueError, RuntimeError)):
        probe.read_export_proof(
            reference, result["checkpoint"], plan["sampling"], path.parent
        )


@pytest.mark.parametrize(
    "updates,step,transfer,accepted",
    [
        (123, 50000, False, True),
        (0, 50000, False, False),
        (123, 0, False, False),
        (123, 50000, True, False),
    ],
)
def test_mdlm_reference_keeps_positive_ema_and_original_step_gate(
    updates, step, transfer, accepted
):
    checkpoint = {"global_step": step}
    if transfer:
        checkpoint[exporter.TRANSFER_KEY] = {}
    sampler = SimpleNamespace(
        diffusion_type="mdlm",
        inference_weights={
            "source": "ema",
            "ema_applied": True,
            "ema": {"decay": 0.9, "num_updates": updates, "shadow_parameter_count": 1},
        },
    )
    plan = {"spec": {"arm_kind": probe.REFERENCE_ARM}, "sampling": {}}
    if accepted:
        assert (
            probe.validate_loaded_checkpoint(checkpoint, sampler, plan)["global_step"]
            == 50000
        )
    else:
        with pytest.raises(ValueError):
            probe.validate_loaded_checkpoint(checkpoint, sampler, plan)


class Tokenizer:
    unk_token_id, bos_token_id, eos_token_id, pad_token_id, mask_token_id = range(5)

    def __len__(self):
        return 1882

    def batch_decode(self, values, *, skip_special_tokens):
        assert skip_special_tokens is True
        return [" ".join(str(x) for x in row if x >= 5) for row in values.tolist()]


class SyntheticModel(torch.nn.Module):
    bos_index, eos_index, mask_index = 1, 2, 4
    device = torch.device("cpu")
    udlm_parameterization = "raw_loo"

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Identity()
        self.tokenizer = Tokenizer()
        self.config = SimpleNamespace(
            model=SimpleNamespace(vocab_size=1880),
            training={
                "use_bracket_safe": False,
                "udlm": {
                    "inference_eps": 1e-5,
                    "exclude_special_tokens": False,
                    "prior_variant": "mask_rich_empirical",
                },
            },
        )


class SyntheticSampler:
    pad_index, diffusion_type = 3, "udlm"

    def __init__(self):
        self.model = SyntheticModel()
        self.mdlm = SimpleNamespace(to_device=lambda _: None)
        self.calls = 4
        self.corrupt_framing = False

    def _insert_mask(self, ids, count, *, min_add_len):
        assert ids.tolist() == [[1, 2]] and count == 2 and min_add_len == 1
        return torch.tensor([[1, 4, 4, 2, 3], [1, 4, 2, 3, 3]])

    def generate(self, ids, **kwargs):
        assert kwargs == {
            "softmax_temp": 0.5,
            "randomness": 0.0,
            "num_steps": 4,
            "return_token_ids": True,
            "raw_loo_top_p": 1.0,
        }
        for _ in range(self.calls):
            self.model.backbone(ids)
        result = torch.tensor([[1, 7, 4, 2, 3], [1, 8, 2, 3, 3]])
        if self.corrupt_framing:
            result[0, 0] = 9
        return result


@pytest.fixture
def execution_case(reference_spec, monkeypatch):
    plan = preview(reference_spec)
    plan["sampling"] = sampling()
    plan["source_config"] = dict(plan["sampling"])
    sampler = SyntheticSampler()
    implementation = SimpleNamespace(
        provenance=plan["direct_inputs"], length_distribution=None
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(
        probe.bench, "load_implementation_input_snapshot", lambda **_: implementation
    )
    monkeypatch.setattr(
        probe.bench,
        "load_pinned_sa_metric_input",
        lambda: SimpleNamespace(provenance={}),
    )
    for name in (
        "assert_local_genmol_import",
        "assert_runtime_module_provenance",
        "synchronize_device",
    ):
        monkeypatch.setattr(probe.bench, name, lambda *_a, **_k: None)
    monkeypatch.setattr(
        probe.bench, "_validate_loaded_udlm_prior_identity", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        probe.bench,
        "seed_sampling",
        lambda seed, device: {"seed": seed, "device": device},
    )
    monkeypatch.setattr(
        probe.bench, "tokenizer_provenance", lambda _: {"synthetic": True}
    )
    monkeypatch.setattr(
        probe, "prepare_device", lambda device: {"device": device, "gpu": None}
    )
    identity = {
        "kind": probe.TRANSFER_ARM,
        "inference_weights": {
            "source": "ema",
            "ema_applied": True,
            "ema": {"decay": 0.9, "num_updates": 0, "shadow_parameter_count": 1},
        },
    }
    monkeypatch.setattr(probe, "load_sampler", lambda *_: (sampler, identity))
    decode = probe.bench.decode_records
    monkeypatch.setattr(
        probe.bench,
        "decode_records",
        lambda texts, **kwargs: decode(
            texts,
            strict_decoder=lambda text: "C" if text == "7" else None,
            released_decoder=lambda text: "C" if text == "7" else None,
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        probe,
        "score_records",
        lambda records, count, snapshot: (
            {"synthetic_only": {"requested": count}},
            {"synthetic_failed_rows": 1},
        ),
    )
    return plan, sampler


def test_execute_preserves_raw_rows_controls_and_observed_batch_nfe(
    execution_case, tmp_path
):
    plan, sampler = execution_case
    output = tmp_path / "run"
    terminal = probe.execute(plan, device="cpu", output_dir=output)
    assert (
        terminal["status"] == "completed"
        and terminal["final_promotion_eligible"] is False
    )
    assert terminal["verification_status"] == "pending_independent_rescore"
    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["run"]["generation_protocol"]["model_use_bracket_safe"] is False
    raw = json.loads((output / "raw_generation.json").read_bytes())
    assert raw["generation_protocol"]["model_use_bracket_safe"] is False
    work = summary["observed_work"]
    assert work["backbone_batch_sizes"] == [2] * 4
    assert (
        work["candidate_equivalent_evaluations"] == 8 and work["nfe_per_candidate"] == 4
    )
    audit = summary["sampled_token_control_audit"]
    assert (audit["rows"], audit["columns"]) == (2, 5)
    assert (
        audit["control_token_counts"]["final_sampled_editable_positions"]["mask"] == 1
    )
    with (output / "raw_samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["sample_index"] for r in rows] == ["0", "1"]
    assert [r["raw_model_text"] for r in rows] == ["7", "8"]
    assert rows[1]["strict_decode_error"] == "decode_returned_none"
    assert not sampler.model.backbone._forward_pre_hooks
    for record in terminal["outputs"].values():
        assert probe.file_record(record["path"])[0] == record


@pytest.mark.parametrize(
    "failure", ["nfe", "framing", "scoring", "row_order", "source"]
)
def test_execute_retains_failed_terminal_without_accepted_summary(
    execution_case, tmp_path, monkeypatch, failure
):
    plan, sampler = execution_case
    if failure == "nfe":
        sampler.calls = 3
    elif failure == "framing":
        sampler.corrupt_framing = True
    elif failure == "scoring":
        monkeypatch.setattr(
            probe,
            "score_records",
            lambda *_: (_ for _ in ()).throw(ValueError("synthetic metric failure")),
        )
    elif failure == "row_order":
        decode = probe.bench.decode_records
        monkeypatch.setattr(
            probe.bench,
            "decode_records",
            lambda *a, **k: list(reversed(decode(*a, **k))),
        )
    elif failure == "source":
        monkeypatch.setattr(
            probe.bench, "require_clean_pushed_source", lambda _: {"head": "changed"}
        )
    output = tmp_path / "failed_run"
    with pytest.raises((ValueError, RuntimeError)):
        probe.execute(plan, device="cpu", output_dir=output)
    terminal = json.loads((output / "terminal_manifest.json").read_bytes())
    assert (
        terminal["status"] == "failed" and terminal["final_promotion_eligible"] is False
    )
    assert "error" in terminal and not (output / "summary.json").exists()
    assert not sampler.model.backbone._forward_pre_hooks
    if failure in {"scoring", "row_order", "source"}:
        assert (output / "raw_generation.json").is_file()


def test_execute_never_reuses_output_namespace(execution_case, tmp_path, monkeypatch):
    plan, _sampler = execution_case
    output = tmp_path / "existing"
    output.mkdir()
    (output / "preserve").write_text("existing evidence")
    monkeypatch.setattr(probe, "load_sampler", lambda *_: pytest.fail("model load"))
    with pytest.raises(FileExistsError):
        probe.execute(plan, device="cpu", output_dir=output)
    assert list(output.iterdir()) == [output / "preserve"]


def test_late_input_drift_retains_outputs_but_fails_terminal(
    execution_case, tmp_path, monkeypatch
):
    plan, _sampler = execution_case
    original = probe.write_new

    def write_then_drift(path, payload):
        original(path, payload)
        if Path(path).name == "summary.json":
            Path(plan["inputs"]["design"]["path"]).write_text("changed after summary")

    monkeypatch.setattr(probe, "write_new", write_then_drift)
    output = tmp_path / "late_drift"
    with pytest.raises(ValueError, match="final input validation"):
        probe.execute(plan, device="cpu", output_dir=output)
    terminal = json.loads((output / "terminal_manifest.json").read_bytes())
    assert terminal["status"] == "failed"
    assert terminal["final_input_validation"]["status"] == "failed"
    assert (output / "summary.json").is_file()
    assert (output / "raw_samples.csv").is_file()
    assert terminal["final_promotion_eligible"] is False


def test_reference_loader_does_not_resolve_unused_callback(export_case):
    from omegaconf import OmegaConf
    import lightning

    payload = export_case["source_payload"]
    payload["hyper_parameters"]["config"].callback = {"dirpath": "${cwd:}"}
    payload["pytorch-lightning_version"] = lightning.__version__
    assert not OmegaConf.has_resolver("cwd")
    torch.save(payload, export_case["source_path"])
    plan = {
        "spec": {"arm_kind": probe.REFERENCE_ARM},
        "sampling": {},
        "inputs": {"checkpoint": probe.file_record(export_case["source_path"])[0]},
    }
    sampler, identity = probe.load_sampler(
        plan, SimpleNamespace(length_distribution=[40])
    )
    assert identity["global_step"] == 50000
    assert sampler.inference_weights["ema"]["num_updates"] == 123
    assert OmegaConf.to_container(sampler.model.config, resolve=False)["callback"] == {
        "dirpath": "${cwd:}"
    }
    assert not OmegaConf.has_resolver("cwd")
