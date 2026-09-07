"""Audit the optional Gibbs sampler's config, NFE, and retained evidence contract."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import test_denovo_launcher as launcher_cases
import test_udlm_rescore_denovo_run as rescore_cases
from scripts.exps.denovo import benchmark
from scripts.exps.denovo import launch_benchmark as launcher
from scripts.udlm import rescore_denovo_run as rescore


def _config(**overrides):
    return {
        "diffusion_type": "udlm",
        "softmax_temp": 0.7,
        "randomness": 0.0,
        "min_add_len": 40,
        "num_steps": 128,
        "inference_eps": 1e-5,
        "exclude_special_tokens": False,
        **overrides,
    }


def test_inactive_corrector_preserves_canonical_sampling_identity():
    historical = benchmark.validate_sampling_config(_config())
    explicit_false = benchmark.validate_sampling_config(_config(gibbs_corrector=False))
    assert historical == explicit_false
    assert "gibbs_corrector" not in historical
    assert benchmark._canonical_json_sha256(historical) == (
        benchmark._canonical_json_sha256(explicit_false)
    )
    enabled = benchmark.validate_sampling_config(_config(gibbs_corrector=True))
    assert enabled.pop("gibbs_corrector") is True
    assert enabled == historical


@pytest.mark.parametrize("flag", [None, 0, 1, "false", "true", [], {}])
def test_corrector_config_requires_a_boolean(flag):
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="boolean"):
        benchmark.validate_sampling_config(_config(gibbs_corrector=flag))


@pytest.mark.parametrize("steps", [0, 1, 3, 127, True, 2.5])
def test_corrector_config_rejects_invalid_total_nfe(steps):
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="num_steps"):
        benchmark.validate_sampling_config(_config(gibbs_corrector=True, num_steps=steps))


def test_mdlm_cannot_enable_corrector():
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="requires UDLM"):
        benchmark.validate_sampling_config(
            _config(diffusion_type="mdlm", gibbs_corrector=True)
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("nfe", [2, 128])
def test_raw_benchmark_handoff_and_nfe_metadata(enabled, nfe):
    import torch

    calls = []

    def generate(values, **kwargs):
        calls.append(kwargs)
        return values

    sampler = SimpleNamespace(
        diffusion_type="udlm",
        model=SimpleNamespace(
            bos_index=1,
            eos_index=2,
            device=torch.device("cpu"),
            config=SimpleNamespace(training={"udlm": {"inference_eps": 1e-5}}),
            tokenizer=SimpleNamespace(
                batch_decode=lambda values, **kwargs: ["C"] * len(values)
            ),
        ),
        _insert_mask=lambda values, count, **kwargs: values.repeat(count, 1),
        generate=generate,
    )
    sampling = benchmark.validate_sampling_config(
        _config(gibbs_corrector=enabled, num_steps=nfe)
    )
    texts, protocol, input_ids, final_ids = benchmark.generate_raw_model_text(
        sampler, 3, **sampling
    )
    assert texts == ["C"] * 3
    assert torch.equal(input_ids, final_ids)
    assert calls[0]["num_steps"] == protocol["nfe"] == nfe
    if enabled:
        assert calls[0]["gibbs_corrector"] is True
        assert protocol["gibbs_corrector"] is True
        assert protocol["predictor_transitions_per_molecule"] == nfe // 2
        assert protocol["corrector_steps_per_molecule"] == nfe // 2
        assert protocol["num_steps_source"] == (
            "explicit UDLM total predictor-plus-corrector NFE budget"
        )
        assert "fresh Gibbs corrector" in protocol["nfe_definition"]
    else:
        assert "gibbs_corrector" not in calls[0]
        assert not rescore.GIBBS_CORRECTOR_PROTOCOL_FIELDS.intersection(protocol)
        assert protocol["num_steps_source"] == "explicit UDLM reverse-transition count"


def test_corrector_implementation_hash_is_conditional_and_binds_runtime():
    historical = benchmark.implementation_input_provenance()
    corrected = benchmark.implementation_input_provenance(gibbs_corrector=True)
    assert set(corrected) == set(historical) | {"corrector_source"}
    item = corrected.pop("corrector_source")
    assert corrected == historical
    path = benchmark.REPO_ROOT / "src/genmol/corrector.py"
    assert item == {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    corrected["corrector_source"] = item
    benchmark.assert_runtime_module_provenance(corrected)
    item["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="corrector source changed"):
        benchmark.assert_runtime_module_provenance(corrected)


@pytest.mark.parametrize("enabled", [False, True])
def test_launcher_matches_effective_config_and_conditional_sources(
    tmp_path, monkeypatch, enabled
):
    fixture = launcher_cases._expected(tmp_path)
    source = _config(gibbs_corrector=enabled)
    fixture.config_path.write_text(json.dumps(source))
    source = benchmark.load_yaml_config(fixture.config_path)
    captures = []
    original = benchmark.implementation_input_provenance

    def capture(**kwargs):
        captures.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(benchmark, "implementation_input_provenance", capture)
    identity = launcher._build_expected_run_identity(
        fixture.checkpoint_path,
        fixture.config_path,
        32,
        checkpoint_info={
            "sha256": fixture.checkpoint_sha256,
            "global_step": 1000,
            "size_bytes": fixture.checkpoint_size_bytes,
            "diffusion_type": "udlm",
            "udlm_inference_eps": 1e-5,
            "udlm_exclude_special_tokens": False,
        },
        metric_inputs=fixture.metric_inputs,
    )
    assert captures == ([{"gibbs_corrector": True}] if enabled else [{}])
    assert identity.effective_config == {
        **source,
        "model_path": str(fixture.checkpoint_path),
        "num_samples": 32,
        "device": "cuda:0",
        "raw_loo_top_p": 1.0,
    }
    assert ("gibbs_corrector" in identity.sampling_config) is enabled
    assert ("corrector_source" in identity.implementation_inputs) is enabled


def _refresh_summary(fixture):
    fixture["summary_payload"] = rescore_cases._json_bytes(fixture["summary"])
    fixture["summary_sha256"] = hashlib.sha256(fixture["summary_payload"]).hexdigest()
    return fixture


def _corrector_fixture():
    fixture = rescore_cases._fixture()
    summary = fixture["summary"]
    for branch in ("source", "sampling", "effective"):
        summary["config"][branch]["gibbs_corrector"] = True
    for branch in ("sampling", "effective"):
        summary["config"][branch + "_sha256"] = benchmark._canonical_json_sha256(
            summary["config"][branch]
        )
    summary["run"]["generation_protocol"].update(
        {
            "gibbs_corrector": True,
            "predictor_transitions_per_molecule": 64,
            "corrector_steps_per_molecule": 64,
            "num_steps_source": "explicit UDLM total predictor-plus-corrector NFE budget",
            "nfe_definition": (
                "one full backbone forward evaluation per predictor transition "
                "and per fresh Gibbs corrector"
            ),
        }
    )
    summary["implementation_inputs"]["corrector_source"] = {
        "path": "/project/src/genmol/corrector.py",
        "sha256": "9" * 64,
        "size_bytes": 9000,
    }
    fixture["implementation_inputs_sha256"] = benchmark._canonical_json_sha256(
        summary["implementation_inputs"]
    )
    return _refresh_summary(fixture)


def test_independent_rescore_accepts_old_and_corrector_schema8():
    historical = rescore_cases._rescore(rescore_cases._fixture())["identity"]
    corrected = rescore_cases._rescore(_corrector_fixture())["identity"]
    assert historical["generation"]["nfe"] == corrected["generation"]["nfe"] == 128
    assert "gibbs_corrector" not in historical["generation"]
    assert "corrector_source_sha256" not in historical["source"]
    assert corrected["generation"]["gibbs_corrector"] is True
    assert corrected["generation"]["predictor_transitions_per_molecule"] == 64
    assert corrected["generation"]["corrector_steps_per_molecule"] == 64
    assert corrected["source"]["corrector_source_sha256"] == "9" * 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("gibbs_corrector", False),
        ("gibbs_corrector", 1),
        ("predictor_transitions_per_molecule", 128),
        ("corrector_steps_per_molecule", 0),
        ("corrector_steps_per_molecule", True),
        ("nfe", 256),
        ("nfe_definition", "one full backbone forward evaluation per reverse step"),
        ("num_steps_source", "explicit UDLM reverse-transition count"),
    ],
)
def test_independent_rescore_rejects_forged_corrector_protocol(field, value):
    fixture = _corrector_fixture()
    fixture["summary"]["run"]["generation_protocol"][field] = value
    with pytest.raises(rescore.RescoreValidationError):
        rescore_cases._rescore(_refresh_summary(fixture))


@pytest.mark.parametrize(
    "tamper",
    ["missing_source", "wrong_path", "wrong_hash", "source_flag", "missing_count"],
)
def test_independent_rescore_rejects_missing_or_mismatched_corrector_binding(tamper):
    fixture = _corrector_fixture()
    summary = fixture["summary"]
    if tamper == "missing_source":
        del summary["implementation_inputs"]["corrector_source"]
    elif tamper == "wrong_path":
        summary["implementation_inputs"]["corrector_source"]["path"] = "/project/other.py"
    elif tamper == "wrong_hash":
        summary["implementation_inputs"]["corrector_source"]["sha256"] = "0" * 64
    elif tamper == "source_flag":
        del summary["config"]["source"]["gibbs_corrector"]
    else:
        del summary["run"]["generation_protocol"]["corrector_steps_per_molecule"]
    with pytest.raises(rescore.RescoreValidationError):
        rescore_cases._rescore(_refresh_summary(fixture))


def test_independent_rescore_rejects_corrector_fields_for_inactive_config():
    fixture = rescore_cases._fixture()
    fixture["summary"]["run"]["generation_protocol"]["gibbs_corrector"] = False
    with pytest.raises(rescore.RescoreValidationError, match="generation_protocol"):
        rescore_cases._rescore(_refresh_summary(fixture))
