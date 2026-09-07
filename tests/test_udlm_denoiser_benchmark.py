"""Bind CE checkpoint semantics and helper bytes through independent rescoring."""

from __future__ import annotations

import copy
import hashlib

import pytest
import torch

import test_denovo_benchmark as benchmark_cases
import test_denovo_launcher as launcher_cases
import test_udlm_rescore_denovo_run as rescore_cases
from scripts.exps.denovo import benchmark
from scripts.exps.denovo import launch_benchmark as launcher
from scripts.udlm import rescore_denovo_run as rescore


def _sampling(**overrides):
    return benchmark.validate_sampling_config(
        {
            "diffusion_type": "udlm",
            "softmax_temp": 0.5,
            "randomness": 0,
            "min_add_len": 40,
            "num_steps": 128,
            "inference_eps": 1e-5,
            "exclude_special_tokens": False,
            "prior_variant": "schedule_uniform",
            "prior_metadata_sha256": "a" * 64,
            **overrides,
        }
    )


def _checkpoint():
    metadata, state, config = benchmark_cases._categorical_checkpoint_parts(
        "schedule_uniform"
    )
    config["training"]["udlm"]["parameterization"] = "x0_denoiser"
    state["_udlm_denoiser_ce_version"] = torch.tensor(1, dtype=torch.int64)
    return {
        "global_step": 20,
        "epoch": 0,
        "hyper_parameters": {"config": config},
        "state_dict": state,
        "udlm_prior_metadata": metadata,
        "udlm_denoiser_metadata": dict(benchmark.UDLM_DENOISER_METADATA),
    }


def test_ce_checkpoint_inspection_and_config_binding(tmp_path):
    path = tmp_path / "ce.ckpt"
    torch.save(_checkpoint(), path)
    record = benchmark.checkpoint_metadata(path)
    assert record["udlm_denoiser_metadata"] == benchmark.UDLM_DENOISER_METADATA
    benchmark.validate_denoiser_sampling_identity(
        record, _sampling(parameterization="x0_denoiser")
    )
    with pytest.raises(ValueError, match="raw-LOO"):
        benchmark.validate_denoiser_sampling_identity(record, _sampling())


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_marker",
        "boolean_marker",
        "missing_metadata",
        "raw_config",
        "wrong_objective",
        "boolean_schema",
    ],
)
def test_ce_checkpoint_cannot_be_reinterpreted(tmp_path, tamper):
    checkpoint = _checkpoint()
    if tamper == "missing_marker":
        del checkpoint["state_dict"]["_udlm_denoiser_ce_version"]
    elif tamper == "boolean_marker":
        checkpoint["state_dict"]["_udlm_denoiser_ce_version"] = torch.tensor(True)
    elif tamper == "missing_metadata":
        del checkpoint["udlm_denoiser_metadata"]
    elif tamper == "raw_config":
        checkpoint["hyper_parameters"]["config"]["training"]["udlm"][
            "parameterization"
        ] = "raw_loo"
    elif tamper == "wrong_objective":
        checkpoint["udlm_denoiser_metadata"]["objective"] = "ct"
    else:
        checkpoint["udlm_denoiser_metadata"]["schema_version"] = True
    path = tmp_path / "bad.ckpt"
    torch.save(checkpoint, path)
    with pytest.raises((RuntimeError, ValueError)):
        benchmark.checkpoint_metadata(path)


def test_ce_source_is_conditional_and_runtime_bound():
    old = benchmark.implementation_input_provenance()
    assert _sampling() == _sampling(parameterization="raw_loo")
    new = benchmark.implementation_input_provenance(x0_denoiser=True)
    assert set(new) == set(old) | {"denoiser_source"}
    assert {k: v for k, v in new.items() if k != "denoiser_source"} == old
    benchmark.assert_runtime_module_provenance(new)
    new["denoiser_source"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="denoiser source changed"):
        benchmark.assert_runtime_module_provenance(new)


def _ce_fixture():
    fixture = rescore_cases._fixture()
    summary = fixture["summary"]
    prior, _, _ = benchmark_cases._categorical_checkpoint_parts("schedule_uniform")
    prior_sha = benchmark._canonical_json_sha256(prior)
    for name in ("source", "sampling", "effective"):
        summary["config"][name].update(
            parameterization="x0_denoiser",
            prior_variant="schedule_uniform",
            prior_metadata_sha256=prior_sha,
        )
    for name in ("sampling", "effective"):
        summary["config"][name + "_sha256"] = benchmark._canonical_json_sha256(
            summary["config"][name]
        )
    summary["checkpoint"].update(
        udlm_prior_variant="schedule_uniform",
        udlm_prior_metadata=prior,
        udlm_prior_metadata_sha256=prior_sha,
        udlm_denoiser_metadata=dict(benchmark.UDLM_DENOISER_METADATA),
    )
    summary["run"]["generation_protocol"].update(
        prior_variant="schedule_uniform", prior_metadata_sha256=prior_sha
    )
    summary["implementation_inputs"]["denoiser_source"] = {
        "path": "/project/src/genmol/denoiser.py",
        "sha256": "9" * 64,
        "size_bytes": 5000,
    }
    fixture["implementation_inputs_sha256"] = benchmark._canonical_json_sha256(
        summary["implementation_inputs"]
    )
    return _refresh(fixture)


def _refresh(fixture):
    fixture["summary_payload"] = rescore_cases._json_bytes(fixture["summary"])
    fixture["summary_sha256"] = hashlib.sha256(fixture["summary_payload"]).hexdigest()
    return fixture


def test_ce_independent_rescore_retains_conversion_source():
    identity = rescore_cases._rescore(_ce_fixture())["identity"]
    assert identity["source"]["denoiser_source_sha256"] == "9" * 64
    assert (
        "denoiser_source_sha256"
        not in rescore_cases._rescore(rescore_cases._fixture())["identity"]["source"]
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_source",
        "wrong_path",
        "wrong_hash",
        "missing_metadata",
        "source_flag",
        "wrong_objective",
    ],
)
def test_ce_rescore_rejects_inconsistent_semantics_or_helper(tamper):
    fixture = _ce_fixture()
    summary = fixture["summary"]
    if tamper == "missing_source":
        del summary["implementation_inputs"]["denoiser_source"]
    elif tamper == "wrong_path":
        summary["implementation_inputs"]["denoiser_source"][
            "path"
        ] = "/project/other.py"
    elif tamper == "wrong_hash":
        summary["implementation_inputs"]["denoiser_source"]["sha256"] = "0" * 64
    elif tamper == "missing_metadata":
        del summary["checkpoint"]["udlm_denoiser_metadata"]
    elif tamper == "source_flag":
        del summary["config"]["source"]["parameterization"]
    else:
        summary["checkpoint"]["udlm_denoiser_metadata"]["objective"] = "ct"
    with pytest.raises(rescore.RescoreValidationError):
        rescore_cases._rescore(_refresh(fixture))


def test_launcher_binds_ce_checkpoint_and_helper(tmp_path):
    fixture = launcher_cases._expected(tmp_path)
    source = _sampling(parameterization="x0_denoiser")
    fixture.config_path.write_text(rescore_cases._json_bytes(source).decode())
    info = {
        "sha256": fixture.checkpoint_sha256,
        "global_step": 20,
        "size_bytes": fixture.checkpoint_size_bytes,
        "diffusion_type": "udlm",
        "udlm_inference_eps": 1e-5,
        "udlm_exclude_special_tokens": False,
        "udlm_prior_variant": "schedule_uniform",
        "udlm_prior_metadata": {},
        "udlm_prior_metadata_sha256": "a" * 64,
        "udlm_denoiser_metadata": dict(benchmark.UDLM_DENOISER_METADATA),
    }
    identity = launcher._build_expected_run_identity(
        fixture.checkpoint_path,
        fixture.config_path,
        32,
        checkpoint_info=info,
        metric_inputs=fixture.metric_inputs,
    )
    assert (
        identity.checkpoint_udlm_denoiser_metadata == benchmark.UDLM_DENOISER_METADATA
    )
    assert "denoiser_source" in identity.implementation_inputs
    wrong = copy.deepcopy(info)
    del wrong["udlm_denoiser_metadata"]
    with pytest.raises(ValueError, match="denoiser identity"):
        launcher._build_expected_run_identity(
            fixture.checkpoint_path,
            fixture.config_path,
            32,
            checkpoint_info=wrong,
            metric_inputs=fixture.metric_inputs,
        )
