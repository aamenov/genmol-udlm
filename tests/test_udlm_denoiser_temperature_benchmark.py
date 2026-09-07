"""Bind the optional CE-temperature inference law without changing old receipts."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import test_denovo_launcher as launcher_cases
import test_udlm_denoiser_benchmark as ce_cases
import test_udlm_rescore_denovo_run as rescore_cases
from scripts.exps.denovo import benchmark
from scripts.exps.denovo import launch_benchmark as launcher
from scripts.udlm import rescore_denovo_run as rescore


def config(**overrides):
    return {
        "diffusion_type": "udlm",
        "parameterization": "x0_denoiser",
        "prior_variant": "schedule_uniform",
        "prior_metadata_sha256": "a" * 64,
        "softmax_temp": 0.5,
        "randomness": 0.0,
        "min_add_len": 40,
        "num_steps": 128,
        "inference_eps": 1e-5,
        "exclude_special_tokens": False,
        **overrides,
    }


@pytest.mark.parametrize("kind", ["mdlm", "raw_loo", "x0_denoiser"])
def test_default_temperature_preserves_canonical_identity(kind):
    source = (
        {"softmax_temp": 0.5, "randomness": 0.0, "min_add_len": 40}
        if kind == "mdlm"
        else config(parameterization=kind)
    )
    absent = benchmark.validate_sampling_config(source)
    explicit = benchmark.validate_sampling_config(
        dict(source, temperature_space="raw_loo")
    )
    assert absent == explicit and "temperature_space" not in absent
    assert benchmark._canonical_json_sha256(absent) == benchmark._canonical_json_sha256(
        explicit
    )


@pytest.mark.parametrize(
    "value", [None, True, False, 0, 1, "clean", "X0_DENOISER", [], {}]
)
def test_temperature_space_is_an_explicit_string_enum(value):
    with pytest.raises(ValueError, match="temperature_space"):
        benchmark.validate_sampling_config(config(temperature_space=value))


@pytest.mark.parametrize(
    "changes",
    [
        {"parameterization": "raw_loo"},
        {"raw_loo_top_p": 0.9},
        {"gibbs_corrector": True},
        {
            "diffusion_type": "mdlm",
            "parameterization": "raw_loo",
            "prior_variant": None,
            "prior_metadata_sha256": None,
            "num_steps": None,
            "inference_eps": None,
            "exclude_special_tokens": None,
        },
    ],
)
def test_denoiser_temperature_rejects_unsupported_laws(changes):
    with pytest.raises(ValueError, match="temperature_space"):
        benchmark.validate_sampling_config(
            config(temperature_space="x0_denoiser", **changes)
        )


def test_raw_handoff_keeps_default_kwargs_and_records_opt_in_order():
    prior = ce_cases._checkpoint()["udlm_prior_metadata"]
    calls = []
    sampler = SimpleNamespace(
        diffusion_type="udlm",
        model=SimpleNamespace(
            udlm_parameterization="x0_denoiser",
            udlm_prior_metadata=prior,
            bos_index=1,
            eos_index=2,
            device=torch.device("cpu"),
            config=SimpleNamespace(
                training={
                    "udlm": {
                        "inference_eps": 1e-5,
                        "prior_variant": "schedule_uniform",
                        "exclude_special_tokens": False,
                    }
                }
            ),
            tokenizer=SimpleNamespace(batch_decode=lambda ids, **kw: ["C"] * len(ids)),
        ),
        _insert_mask=lambda values, count, **kw: values.repeat(count, 1),
        generate=lambda values, **kw: calls.append(kw) or values,
    )
    outputs = []
    for mode in (None, "raw_loo", "x0_denoiser"):
        source = config(prior_metadata_sha256=benchmark._canonical_json_sha256(prior))
        if mode is not None:
            source["temperature_space"] = mode
        result = benchmark.generate_raw_model_text(
            sampler, 3, **benchmark.validate_sampling_config(source)
        )
        outputs.append(result)
        assert result[0] == ["C"] * 3
        assert torch.equal(result[2], result[3])
        assert result[1]["nfe"] == 128
    assert calls[0] == calls[1] and "temperature_space" not in calls[0]
    assert outputs[0][1] == outputs[1][1]
    assert calls[2] == dict(calls[0], temperature_space="x0_denoiser")
    assert outputs[2][1] == dict(
        outputs[0][1], **benchmark.DENOISER_TEMPERATURE_PROTOCOL
    )
    sampler.model.udlm_parameterization = "raw_loo"
    with pytest.raises(ValueError, match="parameterization"):
        benchmark.generate_raw_model_text(
            sampler, 3, **benchmark.validate_sampling_config(source)
        )
    assert len(calls) == 3


def temperature_fixture():
    fixture = ce_cases._ce_fixture()
    for name in ("source", "sampling", "effective"):
        fixture["summary"]["config"][name]["temperature_space"] = "x0_denoiser"
    fixture["summary"]["run"]["generation_protocol"].update(
        benchmark.DENOISER_TEMPERATURE_PROTOCOL
    )
    return refresh(fixture)


def refresh(fixture):
    for name in ("sampling", "effective"):
        fixture["summary"]["config"][name + "_sha256"] = (
            benchmark._canonical_json_sha256(fixture["summary"]["config"][name])
        )
    return ce_cases._refresh(fixture)


def test_independent_rescore_retains_space_order_and_bound_denoiser_source():
    old = rescore_cases._rescore(ce_cases._ce_fixture())["identity"]
    new = rescore_cases._rescore(temperature_fixture())["identity"]
    assert not set(benchmark.DENOISER_TEMPERATURE_PROTOCOL).intersection(
        old["generation"]
    )
    assert new["generation"] == dict(
        old["generation"], **benchmark.DENOISER_TEMPERATURE_PROTOCOL
    )
    assert new["source"]["denoiser_source_sha256"] == "9" * 64
    assert new["config"]["sampling"]["temperature_space"] == "x0_denoiser"


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperature_space", "raw_loo"),
        ("temperature_application", "after_loo_conversion"),
        ("reverse_bridge_temperature", 0.5),
        ("reverse_bridge_temperature", True),
    ],
)
def test_rescore_rejects_forged_temperature_order(key, value):
    fixture = temperature_fixture()
    fixture["summary"]["run"]["generation_protocol"][key] = value
    with pytest.raises(rescore.RescoreValidationError, match=key):
        rescore_cases._rescore(refresh(fixture))


@pytest.mark.parametrize(
    "mutation", ["missing_field", "source_mode", "sampling_mode", "missing_helper"]
)
def test_rescore_rejects_missing_or_relabelled_mode(mutation):
    fixture = temperature_fixture()
    summary = fixture["summary"]
    if mutation == "missing_field":
        del summary["run"]["generation_protocol"]["temperature_space"]
    elif mutation == "source_mode":
        summary["config"]["source"]["temperature_space"] = "raw_loo"
        summary["config"]["effective"]["temperature_space"] = "raw_loo"
    elif mutation == "sampling_mode":
        del summary["config"]["sampling"]["temperature_space"]
    else:
        del summary["implementation_inputs"]["denoiser_source"]
    with pytest.raises(rescore.RescoreValidationError):
        rescore_cases._rescore(refresh(fixture))


def test_historical_rescore_rejects_unconfigured_temperature_receipt_fields():
    fixture = ce_cases._ce_fixture()
    fixture["summary"]["run"]["generation_protocol"].update(
        benchmark.DENOISER_TEMPERATURE_PROTOCOL
    )
    with pytest.raises(rescore.RescoreValidationError, match="generation_protocol"):
        rescore_cases._rescore(refresh(fixture))


def test_launcher_hashes_mode_and_binds_existing_ce_sources(tmp_path):
    fixture = launcher_cases._expected(tmp_path)
    torch.save(ce_cases._checkpoint(), fixture.checkpoint_path)
    checkpoint = benchmark.checkpoint_metadata(fixture.checkpoint_path)
    identities = []
    for mode in ("raw_loo", "x0_denoiser"):
        source = config(
            temperature_space=mode,
            prior_metadata_sha256=checkpoint["udlm_prior_metadata_sha256"],
        )
        fixture.config_path.write_text(json.dumps(source))
        identities.append(
            launcher._build_expected_run_identity(
                fixture.checkpoint_path,
                fixture.config_path,
                32,
                checkpoint_info=checkpoint,
                metric_inputs=fixture.metric_inputs,
            )
        )
    old, new = identities
    assert old.sampling_config_sha256 != new.sampling_config_sha256
    assert old.source_config_sha256 != new.source_config_sha256
    assert new.sampling_config == dict(
        old.sampling_config, temperature_space="x0_denoiser"
    )
    assert new.implementation_inputs == old.implementation_inputs
    helper = new.implementation_inputs["denoiser_source"]
    assert (
        helper["sha256"]
        == hashlib.sha256(benchmark.DENOISER_SOURCE_PATH.read_bytes()).hexdigest()
    )
    assert new.effective_config["temperature_space"] == "x0_denoiser"


@pytest.mark.parametrize("configured", [False, True])
def test_launcher_completion_rejects_mode_receipt_drift(tmp_path, configured):
    fixture = launcher_cases._expected(tmp_path)
    torch.save(ce_cases._checkpoint(), fixture.checkpoint_path)
    checkpoint = benchmark.checkpoint_metadata(fixture.checkpoint_path)
    fixture.config_path.write_text(
        json.dumps(
            config(
                temperature_space="x0_denoiser" if configured else "raw_loo",
                prior_metadata_sha256=checkpoint["udlm_prior_metadata_sha256"],
            )
        )
    )
    expected = launcher._build_expected_run_identity(
        fixture.checkpoint_path,
        fixture.config_path,
        3,
        checkpoint_info=checkpoint,
        metric_inputs=fixture.metric_inputs,
    )
    expected = replace(
        expected,
        source_revision=fixture.source_revision,
        config_git_tracking=dict(
            fixture.config_git_tracking, sha256=expected.source_config_sha256
        ),
    )
    root = tmp_path / "runs"
    _, summary_path = launcher_cases._write_matching_artifacts(root, 1000, expected)
    summary = launcher_cases._read_summary(summary_path)
    summary["checkpoint"]["udlm_denoiser_metadata"] = checkpoint[
        "udlm_denoiser_metadata"
    ]
    if configured:
        summary["run"]["generation_protocol"].update(
            benchmark.DENOISER_TEMPERATURE_PROTOCOL
        )
    launcher_cases._write_summary(summary_path, summary)
    assert launcher._completed(root, 1000, expected)
    if configured:
        del summary["run"]["generation_protocol"]["temperature_application"]
    else:
        summary["run"]["generation_protocol"].update(
            benchmark.DENOISER_TEMPERATURE_PROTOCOL
        )
    launcher_cases._write_summary(summary_path, summary)
    with pytest.raises(launcher.CompletionArtifactError, match="temperature"):
        launcher._completed(root, 1000, expected)
