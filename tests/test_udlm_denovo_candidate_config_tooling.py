from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from scripts.udlm import prepare_denovo_candidate_config_registry as registry_prepare
from scripts.udlm import prepare_denovo_candidate_configs as config_prepare
from scripts.udlm import verify_denovo_candidate_config_registry as verifier


def _identity_bytes() -> dict[str, bytes]:
    return {
        arm: (verifier.REPOSITORY_ROOT / reference["relative_path"]).read_bytes()
        for arm, reference in verifier.IDENTITY_CONFIGS.items()
    }


def _populate_candidate_tree(root: Path, identities: dict[str, bytes]) -> None:
    for arm, reference in verifier.IDENTITY_CONFIGS.items():
        path = root / reference["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(identities[arm])
    for arm in verifier.ARM_IDS:
        for temperature in verifier.SOFTMAX_TEMPERATURES:
            for top_p in verifier.RAW_LOO_TOP_P_VALUES:
                if (temperature, top_p) == (1.0, 1.0):
                    continue
                config_id = verifier.config_id(arm, temperature, top_p)
                path = root / verifier.CONFIG_DIRECTORY / f"{config_id}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    verifier.generated_config_payload(
                        identities[arm],
                        softmax_temp=temperature,
                        raw_loo_top_p=top_p,
                    )
                )
    protocol = root / verifier.SUPERIORITY_PROTOCOL_RELATIVE_PATH
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_bytes(b'{"schema_version":4}\n')
    scale_up = root / verifier.SCALE_UP_REGISTRY_RELATIVE_PATH
    scale_up.parent.mkdir(parents=True, exist_ok=True)
    scale_up.write_bytes(b'{"schema_version":1}\n')


def _patch_roots(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(verifier, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(config_prepare, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(registry_prepare, "REPOSITORY_ROOT", root)


def test_config_ids_and_grid_are_exact_and_canonical() -> None:
    config_ids = verifier.all_config_ids()

    assert len(config_ids) == 36
    assert len(set(config_ids)) == 36
    assert all(
        re.fullmatch(r"[rse]_t(?:050|070|085|100)_p(?:095|098|100)", value)
        for value in config_ids
    )
    assert config_ids[:3] == (
        "r_t050_p100",
        "r_t050_p098",
        "r_t050_p095",
    )
    assert len(verifier.generated_config_relative_paths()) == 33
    assert all(
        path.endswith(".yaml") for path in verifier.generated_config_relative_paths()
    )


def test_candidate_yaml_derivation_is_deterministic_and_changes_only_two_controls() -> (
    None
):
    identities = _identity_bytes()
    first = config_prepare.candidate_config_payloads()
    second = config_prepare.candidate_config_payloads()

    assert first == second
    assert len(first) == 33
    for relative_path, payload in first.items():
        config_id = Path(relative_path).stem
        arm = config_id[0].upper()
        base = yaml.safe_load(identities[arm])
        generated = yaml.safe_load(payload)
        differing = {
            key
            for key in set(base) | set(generated)
            if base.get(key) != generated.get(key)
        }
        assert differing <= {"softmax_temp", "raw_loo_top_p"}
        assert "raw_loo_top_p" in generated
        assert generated["num_steps"] == 128
        assert generated["prior_variant"] == base["prior_variant"]

    assert {
        arm: hashlib.sha256(payload).hexdigest() for arm, payload in identities.items()
    } == {
        arm: reference["sha256"] for arm, reference in verifier.IDENTITY_CONFIGS.items()
    }


def test_registry_stage_contract_is_six_stage_machine_readable_and_frozen() -> None:
    stages = verifier.expected_stages()

    assert [stage["stage_id"] for stage in stages] == [
        "D",
        "A",
        "B",
        "C",
        "eligible",
        "final",
    ]
    assert [stage["expected_child_count"] for stage in stages] == [1, 12, 18, 6, 6, 3]
    assert [stage["requested_samples_per_seed"] for stage in stages] == [
        32,
        32,
        64,
        96,
        256,
        1000,
    ]
    assert stages[0]["candidate_source"] == {
        "kind": "fixed_config_ids",
        "config_ids": ["e_t100_p100"],
    }
    assert len(stages[1]["candidate_source"]["config_ids"]) == 12
    assert stages[2]["candidate_source"] == {
        "kind": "cross_product_of_promoted_temperatures",
        "from_stage_id": "A",
        "raw_loo_top_p_values": [1.0, 0.98, 0.95],
        "expected_per_arm": 6,
    }
    for stage in stages[1:5]:
        assert stage["ranking"]["order"] == list(verifier.RANKING_ORDER)
        assert stage["ranking"]["on_failed_or_undefined_child"] == ("retain_unrankable")
        assert stage["ranking"]["on_insufficient_rankable_quota"] == (
            "campaign_incomplete_without_promotion"
        )
    assert stages[0]["ranking"] is stages[0]["promotion"] is None
    assert stages[-1]["ranking"] is stages[-1]["promotion"] is None


def test_registry_builder_and_verifier_agree_on_exact_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = _identity_bytes()
    _populate_candidate_tree(tmp_path, identities)
    _patch_roots(monkeypatch, tmp_path)

    registry = registry_prepare.build_registry_document(
        framework_revision="a" * 40,
        config_revision="b" * 40,
    )

    assert set(registry) == {
        "schema_version",
        "registry_id",
        "status",
        "claim_scope",
        "publication",
        "authority",
        "grid",
        "configs",
        "stages",
    }
    assert registry["status"] == "frozen_before_candidate_generation"
    assert len(registry["configs"]) == 36
    assert [entry["config_id"] for entry in registry["configs"]] == list(
        verifier.all_config_ids()
    )
    assert sum(entry["is_reused_identity"] for entry in registry["configs"]) == 3
    assert verifier.validate_registry_document(registry) is registry
    assert verifier.json_bytes(registry) == verifier.json_bytes(copy.deepcopy(registry))

    tampered = copy.deepcopy(registry)
    tampered["configs"][0]["softmax_temp"] = 0.7
    with pytest.raises(verifier.RegistryValidationError, match="differs"):
        verifier.validate_registry_document(tampered)


def test_config_publication_is_exactly_33_no_clobber_and_preserves_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = _identity_bytes()
    for arm, reference in verifier.IDENTITY_CONFIGS.items():
        path = tmp_path / reference["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(identities[arm])
    (tmp_path / verifier.CONFIG_DIRECTORY).parent.mkdir(parents=True, exist_ok=True)
    _patch_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        config_prepare, "_require_revision_unchanged", lambda _value: None
    )
    monkeypatch.setattr(
        config_prepare,
        "_status_paths",
        lambda: verifier.generated_config_relative_paths(),
    )

    def source(revision):
        return revision

    receipt = config_prepare.publish_candidate_configs(
        expected_framework_revision="a" * 40,
        source_validator=source,
    )

    config_directory = tmp_path / verifier.CONFIG_DIRECTORY
    assert receipt["generated_config_count"] == 33
    assert len(list(config_directory.iterdir())) == 33
    assert not (tmp_path / verifier.REGISTRY_RELATIVE_PATH).exists()
    assert {
        arm: (tmp_path / reference["relative_path"]).read_bytes()
        for arm, reference in verifier.IDENTITY_CONFIGS.items()
    } == identities
    before = {path.name: path.read_bytes() for path in config_directory.iterdir()}
    with pytest.raises(config_prepare.ConfigPreparationError):
        config_prepare.publish_candidate_configs(
            expected_framework_revision="a" * 40,
            source_validator=source,
        )
    assert {
        path.name: path.read_bytes() for path in config_directory.iterdir()
    } == before


def test_registry_publication_is_registry_only_and_never_clobbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = _identity_bytes()
    _populate_candidate_tree(tmp_path, identities)
    _patch_roots(monkeypatch, tmp_path)
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        registry_prepare.verifier, "validate_publication_history", lambda _value: None
    )
    monkeypatch.setattr(
        registry_prepare, "_require_revision_unchanged", lambda _value: None
    )
    monkeypatch.setattr(
        registry_prepare,
        "_status_paths",
        lambda: (verifier.REGISTRY_RELATIVE_PATH,),
    )

    def source(framework, config):
        return framework, config

    receipt = registry_prepare.publish_candidate_registry(
        expected_framework_revision="a" * 40,
        expected_config_revision="b" * 40,
        source_validator=source,
    )

    registry_path = tmp_path / verifier.REGISTRY_RELATIVE_PATH
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert set(after) - set(before) == {verifier.REGISTRY_RELATIVE_PATH}
    assert all(after[path] == payload for path, payload in before.items())
    assert receipt["relative_path"] == verifier.REGISTRY_RELATIVE_PATH
    assert receipt["sha256"] == hashlib.sha256(registry_path.read_bytes()).hexdigest()
    registry = verifier.strict_json_loads(registry_path.read_bytes(), label="registry")
    verifier.validate_registry_document(registry)

    with pytest.raises(registry_prepare.RegistryPreparationError):
        registry_prepare.publish_candidate_registry(
            expected_framework_revision="a" * 40,
            expected_config_revision="b" * 40,
            source_validator=source,
        )


def test_tool_clis_expose_revision_pins_but_no_scientific_overrides() -> None:
    assert vars(
        config_prepare._parse_args(["--expected-framework-revision", "a" * 40])
    ) == {"expected_framework_revision": "a" * 40}
    assert vars(
        registry_prepare._parse_args(
            [
                "--expected-framework-revision",
                "a" * 40,
                "--expected-config-revision",
                "b" * 40,
            ]
        )
    ) == {
        "expected_framework_revision": "a" * 40,
        "expected_config_revision": "b" * 40,
    }
    with pytest.raises(SystemExit):
        config_prepare._parse_args(
            [
                "--expected-framework-revision",
                "a" * 40,
                "--softmax-temp",
                "0.6",
            ]
        )


def test_registry_strict_json_rejects_duplicates_and_nonfinite_values() -> None:
    with pytest.raises(verifier.RegistryValidationError, match="duplicate"):
        verifier.strict_json_loads(
            b'{"schema_version":1,"schema_version":1}', label="x"
        )
    with pytest.raises(verifier.RegistryValidationError, match="non-finite"):
        verifier.strict_json_loads(b'{"schema_version":NaN}', label="x")


def test_registry_never_contains_G_or_a_self_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = _identity_bytes()
    _populate_candidate_tree(tmp_path, identities)
    _patch_roots(monkeypatch, tmp_path)
    registry = registry_prepare.build_registry_document(
        framework_revision="a" * 40,
        config_revision="b" * 40,
    )

    serialized = json.dumps(registry, sort_keys=True)
    assert "registry_revision" not in serialized
    assert "registry_sha256" not in serialized
    assert set(registry["publication"]) == {
        "framework_revision",
        "config_revision",
        "config_directory",
        "registry_relative_path",
        "separate_config_and_registry_publications_required",
    }
