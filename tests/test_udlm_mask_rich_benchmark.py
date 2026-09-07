"""Bind a MASK mixture's law, identity and state through benchmark evidence."""

import copy

import pytest
import torch

import test_denovo_benchmark as prior_cases
import test_udlm_denoiser_benchmark as ce_cases
import test_udlm_rescore_denovo_run as rescore_cases
from scripts.exps.denovo import benchmark
from scripts.exps.denovo import launch_benchmark as launcher


def checkpoint(weight=0.8, *, ce=False):
    metadata, state, config = prior_cases._categorical_checkpoint_parts(
        "empirical_frequency", mixture_weight=0.0002
    )
    base = state["mdlm.stationary_probs"].clone()
    mixture = base.clone()
    if weight:
        mixture *= 1.0 - weight
        mixture[benchmark.CONTROL_TOKEN_IDS["mask"]] += weight
        mixture /= mixture.sum()
    metadata.update(
        variant="mask_rich_empirical",
        **benchmark.UDLM_PRIOR_VARIANT_IDENTITIES["mask_rich_empirical"],
        mask_mixture_weight=weight,
        mask_token_id=4,
        base_stationary_probs_sha256=benchmark._canonical_numeric_sequence_sha256(
            base.tolist()
        ),
        stationary_probs_sha256=benchmark._canonical_numeric_sequence_sha256(
            mixture.tolist()
        ),
    )
    state["mdlm.stationary_probs"] = mixture
    state[benchmark.UDLM_MASK_RICH_STATE_KEY] = torch.tensor(
        weight, dtype=torch.float64
    ).view(torch.int64)
    config["training"]["udlm"].update(
        prior_variant="mask_rich_empirical", mask_mixture_weight=weight
    )
    result = {
        "global_step": 20,
        "epoch": 0,
        "state_dict": state,
        "hyper_parameters": {"config": config},
        benchmark.UDLM_PRIOR_CHECKPOINT_KEY: metadata,
    }
    if ce:
        config["training"]["udlm"]["parameterization"] = "x0_denoiser"
        state["_udlm_denoiser_ce_version"] = torch.tensor(1, dtype=torch.int64)
        result["udlm_denoiser_metadata"] = dict(benchmark.UDLM_DENOISER_METADATA)
    return result


@pytest.mark.parametrize("weight", [0.0, 0.8, 0.95])
@pytest.mark.parametrize("ce", [False, True])
def test_checkpoint_inspection_and_metadata_only_rebuild(tmp_path, weight, ce):
    value = checkpoint(weight, ce=ce)
    path = tmp_path / "mask.ckpt"
    torch.save(value, path)
    inspected = benchmark.checkpoint_metadata(path)
    metadata = inspected["udlm_prior_metadata"]
    assert metadata["mask_mixture_weight"] == weight
    assert benchmark.validate_udlm_prior_metadata_record(metadata) == metadata
    sampling = ce_cases._sampling(
        prior_variant="mask_rich_empirical",
        prior_metadata_sha256=inspected["udlm_prior_metadata_sha256"],
        **({"parameterization": "x0_denoiser"} if ce else {}),
    )
    benchmark.validate_denoiser_sampling_identity(inspected, sampling)


def test_zero_weight_preserves_empirical_bytes_but_changes_identity():
    plain, state, _ = prior_cases._categorical_checkpoint_parts(
        "empirical_frequency", mixture_weight=0.0002
    )
    masked = checkpoint(0.0)
    assert torch.equal(
        state["mdlm.stationary_probs"], masked["state_dict"]["mdlm.stationary_probs"]
    )
    assert benchmark._canonical_json_sha256(plain) != benchmark._canonical_json_sha256(
        masked["udlm_prior_metadata"]
    )


def test_checkpoint_variant_retains_existing_case_normalization(tmp_path):
    value = checkpoint()
    value["hyper_parameters"]["config"]["training"]["udlm"]["prior_variant"] = (
        "MASK_RICH_EMPIRICAL"
    )
    path = tmp_path / "mask-uppercase.ckpt"
    torch.save(value, path)
    assert (
        benchmark.checkpoint_metadata(path)["udlm_prior_variant"]
        == "mask_rich_empirical"
    )


def test_launcher_binds_mask_prior_hash_without_extra_inference_fields(tmp_path):
    path = tmp_path / "mask.ckpt"
    torch.save(checkpoint(ce=True), path)
    record = benchmark.checkpoint_metadata(path)
    config_path = tmp_path / "sampling.json"
    sampling = ce_cases._sampling(
        prior_variant="mask_rich_empirical",
        prior_metadata_sha256=record["udlm_prior_metadata_sha256"],
        parameterization="x0_denoiser",
    )
    config_path.write_bytes(rescore_cases._json_bytes(sampling))
    identity = launcher._build_expected_run_identity(
        path,
        config_path,
        32,
        checkpoint_info=record,
        metric_inputs={},
        implementation_inputs={},
    )
    assert identity.checkpoint_udlm_prior_metadata == record["udlm_prior_metadata"]
    assert "mask_mixture_weight" not in identity.sampling_config
    sampling["prior_metadata_sha256"] = "0" * 64
    config_path.write_bytes(rescore_cases._json_bytes(sampling))
    with pytest.raises(ValueError, match="prior_metadata_sha256"):
        launcher._build_expected_run_identity(
            path,
            config_path,
            32,
            checkpoint_info=record,
            metric_inputs={},
            implementation_inputs={},
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_marker",
        "wrong_bits",
        "float_marker",
        "vector_marker",
        "missing_weight",
        "weight_config",
        "wrong_mask",
        "base_hash",
        "stationary_hash",
        "old_variant",
        "excluded_mask",
        "extra_field",
    ],
)
def test_mixture_checkpoint_cannot_be_reinterpreted(tmp_path, tamper):
    value = checkpoint()
    metadata, state, config = (
        value["udlm_prior_metadata"],
        value["state_dict"],
        value["hyper_parameters"]["config"],
    )
    if tamper == "missing_marker":
        del state[benchmark.UDLM_MASK_RICH_STATE_KEY]
    elif tamper == "wrong_bits":
        state[benchmark.UDLM_MASK_RICH_STATE_KEY] = torch.tensor(0, dtype=torch.int64)
    elif tamper == "float_marker":
        state[benchmark.UDLM_MASK_RICH_STATE_KEY] = torch.tensor(
            0.8, dtype=torch.float64
        )
    elif tamper == "vector_marker":
        state[benchmark.UDLM_MASK_RICH_STATE_KEY] = state[
            benchmark.UDLM_MASK_RICH_STATE_KEY
        ].reshape(1)
    elif tamper == "missing_weight":
        del config["training"]["udlm"]["mask_mixture_weight"]
    elif tamper == "weight_config":
        config["training"]["udlm"]["mask_mixture_weight"] = 0.9
    elif tamper == "wrong_mask":
        metadata["mask_token_id"] = 3
    elif tamper == "base_hash":
        metadata["base_stationary_probs_sha256"] = "0" * 64
    elif tamper == "stationary_hash":
        metadata["stationary_probs_sha256"] = "0" * 64
    elif tamper == "old_variant":
        config["training"]["udlm"]["prior_variant"] = "empirical_frequency"
    elif tamper == "excluded_mask":
        config["training"]["udlm"]["exclude_special_tokens"] = True
    else:
        metadata["undeclared"] = 1
    path = tmp_path / "bad.ckpt"
    torch.save(value, path)
    with pytest.raises(RuntimeError):
        benchmark.checkpoint_metadata(path)


@pytest.mark.parametrize("weight", [True, None, -0.1, 1.0, float("nan"), float("inf")])
def test_metadata_rejects_invalid_mask_weights(weight):
    metadata = checkpoint()["udlm_prior_metadata"]
    metadata["mask_mixture_weight"] = weight
    with pytest.raises(RuntimeError):
        benchmark.validate_udlm_prior_metadata_record(metadata)


@pytest.mark.parametrize("variant", ["schedule_uniform", "empirical_frequency"])
def test_old_categorical_priors_reject_new_state_marker(variant):
    metadata, state, _ = prior_cases._categorical_checkpoint_parts(variant)
    state[benchmark.UDLM_MASK_RICH_STATE_KEY] = torch.tensor(0, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="another prior"):
        benchmark.validate_udlm_prior_metadata_record(metadata, state_dict=state)


def test_ce_independent_rescore_accepts_new_prior_and_rejects_changed_law():
    fixture = ce_cases._ce_fixture()
    metadata = checkpoint(ce=True)["udlm_prior_metadata"]
    prior_sha = benchmark._canonical_json_sha256(metadata)
    summary = fixture["summary"]
    for section in ("source", "sampling", "effective"):
        summary["config"][section].update(
            prior_variant="mask_rich_empirical", prior_metadata_sha256=prior_sha
        )
    for section in ("sampling", "effective"):
        summary["config"][section + "_sha256"] = benchmark._canonical_json_sha256(
            summary["config"][section]
        )
    summary["checkpoint"].update(
        udlm_prior_variant="mask_rich_empirical",
        udlm_prior_metadata=metadata,
        udlm_prior_metadata_sha256=prior_sha,
    )
    summary["run"]["generation_protocol"].update(
        prior_variant="mask_rich_empirical", prior_metadata_sha256=prior_sha
    )
    assert rescore_cases._rescore(ce_cases._refresh(fixture))["status"] == "exact_match"
    changed = copy.deepcopy(fixture)
    changed["summary"]["checkpoint"]["udlm_prior_metadata"]["mask_mixture_weight"] = 0.7
    with pytest.raises(Exception, match="prior"):
        rescore_cases._rescore(ce_cases._refresh(changed))
