"""Independent prototype boundaries; synthetic CPU sampling, no checkpoints."""

import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import types

import pytest
import torch
import yaml

from genmol import context_guidance as guidance
from genmol import sampler as sampler_module
from scripts.exps.denovo import benchmark
from scripts.exps.pmo import udlm_sampling as pmo
from test_udlm_context_guidance import (
    configuration,
    inputs,
    make_sampler,
    rng_state,
    seed_all,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "48473d4febbd06d9bc07986ca96ceb93926c91ec"


def committed(relative):
    return subprocess.check_output(
        ["git", "show", f"{BASE_REVISION}:{relative}"], cwd=ROOT, text=True
    )


def method(source, name):
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Sampler"
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def assert_rng_equal(left, right):
    assert left[:2] == right[:2]
    assert torch.equal(left[2], right[2])


def sampling_config(**changes):
    value = dict(
        diffusion_type="udlm",
        softmax_temp=0.5,
        randomness=0,
        min_add_len=18,
        num_steps=3,
        inference_eps=1e-5,
        exclude_special_tokens=False,
        prior_variant="schedule_uniform",
        prior_metadata_sha256="a" * 64,
        raw_loo_top_p=1.0,
    )
    value.update(changes)
    return value


def test_existing_generate_source_is_identical_except_exact_reserved_guard():
    previous = committed("src/genmol/sampler.py")
    current = Path(sampler_module.__file__).read_text()
    old, new = method(previous, "generate"), method(current, "generate")
    guard = new.body[1]  # after the unchanged docstring, before any operations
    expected_guard = ast.parse("""
if 'context_guidance' in kwargs or (
    isinstance(kwargs.get('method'), str) and kwargs['method'] == 'posterior_context'
):
    raise ValueError('Experimental posterior context guidance requires the separate '
                     'generate_context_guided API; it is not an option of generate.')
""").body[0]
    assert ast.dump(guard, include_attributes=False) == ast.dump(
        expected_guard, include_attributes=False
    )
    without_guard = deepcopy(new)
    without_guard.body.pop(1)
    assert ast.dump(old, include_attributes=False) == ast.dump(
        without_guard, include_attributes=False
    )
    lines = current.splitlines(keepends=True)
    del lines[guard.lineno - 1 : guard.end_lineno]
    stripped = "".join(lines)
    assert ast.get_source_segment(previous, old) == ast.get_source_segment(
        stripped, method(stripped, "generate")
    )
    assert "config" not in inspect.signature(sampler_module.Sampler.generate).parameters


@pytest.mark.parametrize("parameterization", ["raw_loo", "x0_denoiser"])
def test_default_generate_ids_and_all_rng_match_committed_method(parameterization):
    old = method(committed("src/genmol/sampler.py"), "generate")
    namespace = dict(vars(sampler_module))
    exec(
        compile(
            ast.Module(body=[old], type_ignores=[]), "<committed-generate>", "exec"
        ),
        namespace,
    )
    previous, current = make_sampler(parameterization), make_sampler(parameterization)
    old_method = types.MethodType(namespace["generate"], previous)
    seed_all()
    expected = old_method(
        inputs(), num_steps=3, softmax_temp=0.5, randomness=0, return_token_ids=True
    )
    old_rng = rng_state()
    seed_all()
    actual = current.generate(
        inputs(), num_steps=3, softmax_temp=0.5, randomness=0, return_token_ids=True
    )
    assert torch.equal(actual, expected)
    assert_rng_equal(old_rng, rng_state())


def test_benchmark_defaults_and_unknown_key_behavior_remain_historical():
    expected = {
        "diffusion_type": "udlm",
        "softmax_temp": 0.5,
        "randomness": 0.0,
        "min_add_len": 18,
        "num_steps": 3,
        "inference_eps": 1e-5,
        "exclude_special_tokens": False,
        "prior_variant": "schedule_uniform",
        "prior_metadata_sha256": "a" * 64,
        "raw_loo_top_p": 1.0,
    }
    normalized = benchmark.validate_sampling_config(sampling_config())
    assert normalized == expected
    # Unrelated historical unknown fields retain their previous treatment.
    assert (
        benchmark.validate_sampling_config(sampling_config(gamma=0.7, scale=4))
        == normalized
    )
    for unrelated in ("other_method", None, ["posterior_context"]):
        assert (
            benchmark.validate_sampling_config(sampling_config(method=unrelated))
            == normalized
        )
    assert (
        benchmark.validate_sampling_config(
            sampling_config(
                parameterization="raw_loo",
                gibbs_corrector=False,
                temperature_space="raw_loo",
            )
        )
        == normalized
    )
    with pytest.raises(benchmark.BenchmarkConfigurationError):
        benchmark.validate_sampling_config(
            sampling_config(temperature_space="posterior_context")
        )


def test_benchmark_normalizer_has_only_the_exact_reserved_guard_added():
    def find(source):
        return next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "validate_sampling_config"
        )

    old = find(committed("scripts/exps/denovo/benchmark.py"))
    new = find(Path(benchmark.__file__).read_text())
    expected_guard = ast.parse("""
if "context_guidance" in config or (
    isinstance(config.get("method"), str) and config["method"] == "posterior_context"
):
    raise BenchmarkConfigurationError("Experimental posterior context guidance requires the separate "
                                      "sampler API and is not supported by benchmark YAMLs")
""").body[0]
    assert ast.dump(new.body.pop(0), include_attributes=False) == ast.dump(
        expected_guard, include_attributes=False
    )
    assert ast.dump(old, include_attributes=False) == ast.dump(
        new, include_attributes=False
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"context_guidance": None},
        {"context_guidance": configuration()},
        {"method": "posterior_context"},
    ],
)
def test_reserved_settings_reject_in_old_api_and_benchmark_before_rng_or_model(
    settings,
):
    sampler = make_sampler()
    seed_all()
    before = rng_state()
    with pytest.raises(ValueError, match="separate"):
        sampler.generate(inputs(), return_token_ids=True, **settings)
    assert not sampler.model.backbone.calls
    assert_rng_equal(before, rng_state())
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="separate"):
        benchmark.validate_sampling_config(sampling_config(**settings))
    assert_rng_equal(before, rng_state())


def test_unrelated_generate_unknown_keys_retain_ids_and_rng():
    old, new = make_sampler(), make_sampler()
    seed_all()
    expected = old.generate(inputs(), num_steps=3, return_token_ids=True)
    before = rng_state()
    seed_all()
    result = new.generate(
        inputs(),
        num_steps=3,
        return_token_ids=True,
        method=["posterior_context"],
        unrelated="ignored historically",
    )
    assert torch.equal(expected, result)
    assert_rng_equal(before, rng_state())


@pytest.mark.parametrize(
    "field", ["context_guidance", "method", "context_seed", "execution"]
)
def test_pmo_yaml_rejects_prototype_fields_without_checkpoint_load(tmp_path, field):
    path = tmp_path / "synthetic_sampling.yaml"
    values = sampling_config(checkpoint_sha256="b" * 64)
    values[field] = (
        configuration() if field == "context_guidance" else "posterior_context"
    )
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="only the declared"):
        pmo.read_contract(path, gamma=0, variant="released")


def test_pmo_and_legacy_generate_keep_nonzero_udlm_gamma_guard(tmp_path):
    path = tmp_path / "synthetic_sampling.yaml"
    path.write_text(yaml.safe_dump(sampling_config(checkpoint_sha256="b" * 64)))
    with pytest.raises(ValueError, match="requires gamma=0"):
        pmo.read_contract(path, gamma=0.5, variant="released")
    sampler = make_sampler()
    seed_all()
    before = rng_state()
    with pytest.raises(ValueError, match="not a valid UDLM"):
        sampler.generate(inputs(), gamma=0.5, w=2, return_token_ids=True)
    assert not sampler.model.backbone.calls
    assert_rng_equal(before, rng_state())


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"method": "clean_logits"},
        {"gamma": True},
        {"gamma": -0.1},
        {"gamma": 1.1},
        {"gamma": float("nan")},
        {"scale": 0.9},
        {"scale": True},
        {"scale": float("inf")},
        {"context_seed": -1},
        {"context_seed": True},
        {"context_seed": 2**63},
        {"predictor_steps": 0},
        {"predictor_steps": True},
        {"execution": "parallel"},
        {"extra": "not allowed"},
    ],
)
def test_bad_config_rejected_before_identity_rng_and_model(changes, monkeypatch):
    sampler = make_sampler()
    seed_all()
    before = rng_state()
    monkeypatch.setattr(
        guidance,
        "_runtime_identity",
        lambda *_: pytest.fail("runtime identity checked before config rejection"),
    )
    with pytest.raises(ValueError):
        sampler.generate_context_guided(inputs(), config=configuration(**changes))
    assert not sampler.model.backbone.calls
    assert_rng_equal(before, rng_state())


@pytest.mark.parametrize("config", [None, {}, {"schema_version": 1}])
def test_missing_config_fields_fail_before_rng(config):
    sampler = make_sampler()
    seed_all()
    before = rng_state()
    with pytest.raises(ValueError, match="exactly"):
        sampler.generate_context_guided(inputs(), config=config)
    assert not sampler.model.backbone.calls
    assert_rng_equal(before, rng_state())


@pytest.mark.parametrize("parameterization", ["raw_loo", "x0_denoiser"])
def test_receipt_is_defensive_and_does_not_advertise_checkpoint_acceptance(
    parameterization,
):
    sampler = make_sampler(parameterization)
    config = configuration()
    original_input = inputs()
    model_identity = deepcopy(sampler.model.udlm_prior_metadata.to_dict())
    weights = deepcopy(sampler.inference_weights)
    seed_all()
    result = sampler.generate_context_guided(original_input, config=config)
    receipt = result["receipt"]
    normalized = {
        **config,
        "gamma": float(config["gamma"]),
        "scale": float(config["scale"]),
    }
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert receipt["config"] == normalized
    assert receipt["config_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["identity"]["checkpoint_bytes_revalidated"] is False
    assert receipt["fixed_controls"] == {
        "temperature": 1.0,
        "raw_loo_top_p": 1.0,
        "gibbs_corrector": False,
    }
    assert "no benchmark acceptance" in receipt["scientific_status"]
    assert set(receipt["source_sha256"]) == set(guidance.SOURCE_PATHS)
    for relative, digest in receipt["source_sha256"].items():
        assert digest == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    config["gamma"] = 0
    assert receipt["config"]["gamma"] == 1
    receipt["identity"]["inference_weights"]["ema"]["decay"] = -1
    receipt["identity"]["prior_metadata"]["variant"] = "altered"
    receipt["config"]["scale"] = 99
    assert sampler.inference_weights == weights
    assert sampler.model.udlm_prior_metadata.to_dict() == model_identity
    assert config["scale"] == 2
    result["token_ids"][0, 0] = 99
    assert torch.equal(original_input, inputs())
