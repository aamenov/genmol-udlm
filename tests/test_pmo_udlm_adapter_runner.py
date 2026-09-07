"""CPU integration of opt-in sampling with the real PMO budget/population loop.

Checkpoint preparation and the scoring function are synthetic. No TDC evaluator,
checkpoint, GPU inventory, or diffusion model is constructed by these tests.
"""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from scripts.exps.pmo import run_ablation as runner
from scripts.exps.pmo import udlm_sampling as sampling


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    model = tmp_path / "synthetic.ckpt"
    model.write_bytes(b"synthetic checkpoint; never deserialized")
    vocab = tmp_path / "vocab.csv"
    vocab.write_text("frag,score,size\n[1*]CC,0.9,2\n[1*]CN,0.8,2\n[1*]CO,0.7,2\n")
    config = tmp_path / "sampling.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "diffusion_type": "udlm",
                "parameterization": "x0_denoiser",
                "temperature_space": "x0_denoiser",
                "softmax_temp": 0.5,
                "randomness": 0,
                "min_add_len": 18,
                "num_steps": 2,
                "inference_eps": 1e-5,
                "exclude_special_tokens": False,
                "prior_variant": "schedule_uniform",
                "prior_metadata_sha256": "a" * 64,
                "raw_loo_top_p": 1.0,
            }
        )
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        runner,
        "_git_metadata",
        lambda: {"commit": "synthetic", "tracked_diff_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        runner, "_runtime_metadata", lambda device: {"logical_device": device}
    )
    monkeypatch.setattr(runner, "_attach_fragments", lambda *_: "CCO")
    monkeypatch.setattr(runner, "_molecule_size_bounds", lambda *_: (1, 100))
    return SimpleNamespace(root=tmp_path, model=model, vocab=vocab, config=config)


def arguments(inputs, **overrides):
    values = dict(
        oracle="qed",
        variant="released",
        model_path=inputs.model,
        vocab_path=inputs.vocab,
        sampling_config=inputs.config,
        device="cpu",
        seed=19,
        max_oracle_calls=2,
        reporting_frequency=1,
        checkpoint_every=1,
        max_iterations=5,
        population_size=3,
        warmup=0,
        legacy_warmup_off_by_one=False,
        gamma=0.0,
        softmax_temp=1.2,
        randomness=2.0,
        guidance_scale=2.0,
        min_mol_size=1,
        max_mol_size=100,
        legacy_seed_count=1,
        prior_mean=None,
        prior_mean_source=None,
        delta_attribution="novel_vs_parent",
        experiment_id="synthetic_adapter",
        scientific_status="synthetic CPU test; no molecular benchmark claim",
        output_root=inputs.root / "output",
        resume=False,
        durable_events=False,
    )
    values.update(overrides)
    return Namespace(**values)


class SyntheticSampler:
    def __init__(self, *, outputs=("CCN", "NCC", "CCC"), failure=None):
        self.diffusion_type = "udlm"
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Identity()
        self.outputs = iter(outputs)
        self.calls = []
        self.failure = failure

    def generate(self, smiles, **kwargs):
        self.calls.append((smiles, kwargs))
        if self.failure:
            raise self.failure
        for _ in range(kwargs.get("num_steps", 2)):
            self.model.backbone(torch.tensor([1.0]))
        return next(self.outputs)

    def mask_modification(self, smiles, **kwargs):
        return self.generate(smiles, **kwargs)


def install_adapter(monkeypatch, *, sampler=None):
    observed = SimpleNamespace(events=[], scored=[], adapters=[])
    sampler = sampler or SyntheticSampler()

    def prepare(contract, **kwargs):
        observed.events.append("prepare")
        observed.contract = contract
        observed.prepare_kwargs = kwargs
        receipt = {
            "contract": contract,
            "implementation_inputs": {},
            "sampler_kwargs": sampling.modification_kwargs(
                contract, gamma=kwargs["gamma"], guidance_scale=kwargs["guidance_scale"]
            ),
            "configured_nfe_per_generation": contract["configuration"]["num_steps"],
            "test_only": "synthetic preparation; no checkpoint acceptance",
        }
        adapter = sampling.SamplingAdapter(sampler, receipt)
        observed.adapters.append(adapter)
        return adapter

    def evaluator(smiles):
        assert type(smiles) is list and len(smiles) == 1
        observed.scored.append(smiles[0])
        return [len(smiles[0]) / 100]

    def oracle_factory(**kwargs):
        assert kwargs == {"name": "qed"}
        observed.events.append("oracle_factory")
        return evaluator

    monkeypatch.setattr(sampling, "prepare", prepare)
    monkeypatch.setattr(runner, "TDCOracle", oracle_factory)
    observed.sampler = sampler
    return observed


def artifacts(run_dir):
    event_path = run_dir / "events.jsonl"
    return (
        json.loads((run_dir / "manifest.json").read_text()),
        json.loads((run_dir / "summary.json").read_text()),
        [
            json.loads(line)
            for line in (
                event_path.read_text().splitlines() if event_path.exists() else []
            )
        ],
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"gamma": 0.1}, "gamma=0"),
        ({"gamma": None, "oracle": "albuterol_similarity"}, "gamma=0"),
        ({"variant": "running_mean"}, "released policy"),
        ({"resume": True}, "no resume"),
    ],
)
def test_unsupported_opt_in_fails_before_prepare_and_oracle(
    inputs, monkeypatch, overrides, message
):
    prepare, factory = Mock(), Mock()
    monkeypatch.setattr(sampling, "prepare", prepare)
    monkeypatch.setattr(runner, "TDCOracle", factory)
    with pytest.raises(ValueError, match=message):
        runner.run(arguments(inputs, **overrides))
    prepare.assert_not_called()
    factory.assert_not_called()
    assert not (inputs.root / "output").exists()


@pytest.mark.parametrize(
    "failure", ["unknown_field", "checkpoint_mismatch", "prepare_failure"]
)
def test_opt_in_identity_errors_precede_oracle_construction(
    inputs, monkeypatch, failure
):
    if failure != "prepare_failure":
        value = yaml.safe_load(inputs.config.read_text())
        value["unexpected" if failure == "unknown_field" else "checkpoint_sha256"] = (
            True if failure == "unknown_field" else "f" * 64
        )
        inputs.config.write_text(yaml.safe_dump(value))
    prepare = Mock(side_effect=ValueError("synthetic failed checkpoint acceptance"))
    factory = Mock()
    monkeypatch.setattr(sampling, "prepare", prepare)
    monkeypatch.setattr(runner, "TDCOracle", factory)
    with pytest.raises(ValueError):
        runner.run(arguments(inputs))
    factory.assert_not_called()
    assert prepare.call_count == int(failure == "prepare_failure")


def test_adapter_forwards_yaml_controls_and_preserves_unique_oracle_budget(
    inputs, monkeypatch
):
    observed = install_adapter(monkeypatch)
    run_dir = runner.run(arguments(inputs))
    manifest, summary, events = artifacts(run_dir)
    assert observed.events == ["prepare", "oracle_factory"]
    assert observed.prepare_kwargs == {
        "model_path": str(inputs.model),
        "device": "cpu",
        "gamma": 0.0,
        "guidance_scale": 2.0,
        "sampler_class": runner.Sampler,
    }
    assert observed.scored == ["CCN", "CCC"]
    assert [event["child_oracle"]["charged"] for event in events] == [True, False, True]
    assert [event["oracle_calls"] for event in events] == [1, 1, 2]
    assert summary["status"] == "completed"
    assert summary["scores"]["all_charged_molecules"]["oracle_calls"] == 2
    assert summary["sampling"]["observed"] == {
        "modification_attempts": 3,
        "generation_calls": 3,
        "backbone_evaluations": 6,
        "pre_generation_fallbacks": 0,
    }
    assert manifest["extra"]["sampling"] == summary["sampling"]["identity"]
    assert manifest["config"]["softmax_temp"] == 0.5
    assert manifest["config"]["randomness"] == 0.0
    assert manifest["config"]["pmo_sampling"] == observed.contract
    expected = {
        "gamma": 0.0,
        "w": 2.0,
        "softmax_temp": 0.5,
        "randomness": 0.0,
        "num_steps": 2,
        "raw_loo_top_p": 1.0,
        "temperature_space": "x0_denoiser",
    }
    assert observed.sampler.calls == [("CCO", expected)] * 3
    assert all(
        event["sampling"]
        == {
            "generation_calls": 1,
            "backbone_evaluations": 2,
            "pre_generation_fallbacks": 0,
        }
        for event in events
    )


def test_attaching_only_warmup_does_not_claim_generation_work(inputs, monkeypatch):
    observed = install_adapter(monkeypatch)
    _, summary, events = artifacts(
        runner.run(arguments(inputs, warmup=100, max_oracle_calls=1))
    )
    assert observed.sampler.calls == []
    assert observed.scored == ["CCO"]
    assert summary["sampling"]["observed"] == {
        "modification_attempts": 0,
        "generation_calls": 0,
        "backbone_evaluations": 0,
        "pre_generation_fallbacks": 0,
    }
    assert events[0]["sampling"] == {
        "generation_calls": 0,
        "backbone_evaluations": 0,
        "pre_generation_fallbacks": 0,
    }


def test_generation_failure_is_terminal_without_a_charge(inputs, monkeypatch):
    observed = install_adapter(
        monkeypatch,
        sampler=SyntheticSampler(failure=RuntimeError("synthetic generation failure")),
    )
    args = arguments(inputs)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        runner.run(args)
    run_dir = runner._run_directory(runner._resolved_config(args), args.output_root)
    _, summary, events = artifacts(run_dir)
    assert observed.scored == []
    assert events == []
    assert summary["status"] == "failed"
    assert summary["checkpoint_consistent"] is False
    assert summary["scores"]["all_charged_molecules"]["oracle_calls"] == 0
    assert summary["sampling"]["observed"]["generation_calls"] == 1


def test_default_artifacts_and_released_population_match_opt_in_loop(
    inputs, monkeypatch
):
    observed = install_adapter(monkeypatch)
    opt_dir = runner.run(arguments(inputs, experiment_id="explicit"))
    _, opt_summary, opt_events = artifacts(opt_dir)
    order = []
    legacy = SyntheticSampler()
    legacy.diffusion_type = "mdlm"
    legacy.model.device = "cpu"
    legacy.mdlm = SimpleNamespace(to_device=lambda device: None)

    def factory(**_kwargs):
        order.append("oracle_factory")
        return lambda smiles: len(smiles) / 100

    def constructor(_path):
        order.append("sampler")
        return legacy

    monkeypatch.setattr(runner, "TDCOracle", factory)
    monkeypatch.setattr(runner, "Sampler", constructor)
    monkeypatch.setattr(
        sampling,
        "prepare",
        Mock(side_effect=AssertionError("default must not prepare adapter")),
    )
    legacy_dir = runner.run(
        arguments(inputs, sampling_config=None, experiment_id="default")
    )
    manifest, summary, events = artifacts(legacy_dir)
    assert order == ["oracle_factory", "sampler"]
    assert "pmo_sampling" not in manifest["config"]
    assert "sampling" not in manifest["extra"]
    assert "sampling" not in summary
    assert all("sampling" not in event for event in events)
    assert manifest["config"]["softmax_temp"] == 1.2
    assert manifest["config"]["randomness"] == 2.0
    assert summary["population"] == opt_summary["population"]
    assert summary["scores"] == opt_summary["scores"]

    def stripped(event):
        return {
            key: value
            for key, value in event.items()
            if key not in {"sampling", "elapsed_seconds"}
        }

    assert [stripped(event) for event in events] == [
        stripped(event) for event in opt_events
    ]
    assert observed.adapters[0].statistics["modification_attempts"] == 3


def test_implicit_udlm_preserves_legacy_factory_order_but_never_scores(
    inputs, monkeypatch
):
    order, scored = [], []

    def factory(**_kwargs):
        order.append("oracle_factory")
        return lambda smiles: scored.append(smiles)

    def constructor(_path):
        order.append("sampler")
        return SyntheticSampler()

    monkeypatch.setattr(runner, "TDCOracle", factory)
    monkeypatch.setattr(runner, "Sampler", constructor)
    with pytest.raises(ValueError, match="explicit --sampling-config"):
        runner.run(arguments(inputs, sampling_config=None))
    assert order == ["oracle_factory", "sampler"]
    assert scored == []
