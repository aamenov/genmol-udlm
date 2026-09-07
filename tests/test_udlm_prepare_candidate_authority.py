from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.udlm import prepare_candidate_authority as authority


def _config_values(config_id: str) -> tuple[str, float, float]:
    arm = config_id[0].upper()
    temperature = int(config_id.split("_t", 1)[1].split("_p", 1)[0]) / 100
    top_p = int(config_id.rsplit("_p", 1)[1]) / 100
    return arm, temperature, top_p


def _entry(stage_id: str, config_id: str, seeds: tuple[int, ...], index: int) -> dict:
    arm, temperature, top_p = _config_values(config_id)
    attempt_id = f"stage-{stage_id.lower()}-{config_id}"
    return {
        "config_id": config_id,
        "attempt_id": attempt_id,
        "candidate_id": authority.campaign.CANDIDATE_IDS[arm],
        "arm_id": arm,
        "softmax_temp": temperature,
        "raw_loo_top_p": top_p,
        "status": "completed",
        "rankable": stage_id != "D",
        "selection_score": (
            None
            if stage_id == "D"
            else {
                "released_quality": 0.99 - index / 1_000,
                "released_diversity": 0.8 - index / 10_000,
            }
        ),
        "child_outcomes": [
            {
                "artifact_kind": "pilot_evaluation",
                "pilot_seed": seed,
                "relative_path": (
                    f"experiments/udlm/pilots/{attempt_id}/seed_{seed}.json"
                ),
                "sha256": f"{(index + child_index + 1) % 16:x}" * 64,
                "schema_version": 2,
            }
            for child_index, seed in enumerate(seeds)
        ],
    }


def _decision() -> dict:
    stages = []
    promotions: dict[str, list[str]] = {}
    predecessor = None
    for stage_number, stage_id in enumerate(authority.campaign.STAGE_IDS):
        contract = authority.campaign.STAGE_CONTRACT[stage_id]
        if stage_id == "D":
            config_ids = ["e_t100_p100"]
        elif stage_id == "A":
            config_ids = sorted(
                f"{arm.lower()}_t{temperature}_p100"
                for arm in authority.campaign.ARM_IDS
                for temperature in ("050", "070", "085", "100")
            )
        elif stage_id == "B":
            config_ids = sorted(
                f"{config_id.rsplit('_p', 1)[0]}_p{top_p}"
                for config_id in promotions["A"]
                for top_p in ("095", "098", "100")
            )
        else:
            prior = "B" if stage_id == "C" else "C"
            config_ids = sorted(promotions[prior])
        entries = [
            _entry(stage_id, config_id, contract["seeds"], index)
            for index, config_id in enumerate(config_ids)
        ]
        promoted, global_winner = authority._promotions(stage_id, entries)
        source = {
            "relative_path": authority._stage_path(stage_id),
            "sha256": f"{stage_number + 1:x}" * 64,
            "schema_version": 1,
        }
        stages.append(
            {
                "stage_id": stage_id,
                "role": authority.STAGE_ROLES[stage_id],
                "seed_values": list(contract["seeds"]),
                "requested_samples_per_child": contract["samples"],
                "scheduled_entry_count": contract["entries"],
                "scheduled_child_count": contract["children"],
                "entries": entries,
                "advancement": {
                    "predecessor_stage_decision": predecessor,
                    "ranking_order": (
                        [] if stage_id == "D" else list(authority.RANKING_ORDER)
                    ),
                    "promoted_config_ids": promoted,
                    "global_winner_config_id": global_winner,
                    "required_promotions_per_arm": authority.PROMOTION_QUOTA[stage_id],
                    "no_retry_or_substitution": True,
                    "on_failed_or_undefined_child": "retain_unrankable",
                    "on_insufficient_rankable_quota": (
                        "campaign_incomplete_without_promotion"
                    ),
                    "accounting": {
                        "scheduled_entry_count": contract["entries"],
                        "terminal_entry_count": contract["entries"],
                        "scheduled_child_count": contract["children"],
                        "terminal_child_count": contract["children"],
                        "requested_molecule_count": (
                            contract["children"] * contract["samples"]
                        ),
                    },
                    "source_stage_decision": source,
                },
            }
        )
        promotions[stage_id] = promoted
        predecessor = source
    winner = promotions["eligible"][0]
    winner_entry = next(
        entry for entry in stages[-1]["entries"] if entry["config_id"] == winner
    )
    return {
        "schema_version": 1,
        "protocol_id": authority.PROTOCOL_ID,
        "status": "closed_before_candidate_ledger",
        "final_seed_results_included": False,
        "registry": {
            "relative_path": authority.campaign.REGISTRY_RELATIVE_PATH,
            "sha256": "a" * 64,
            "schema_version": 1,
            "registry_id": "de-novo-candidate-config-registry-v1",
            "registry_revision": "b" * 40,
        },
        "campaign": {
            "grid_universe_entry_count": 36,
            "executed_entry_count": 40,
            "child_outcome_count": 43,
            "requested_molecule_count": 3680,
            "nfe": 128,
            "no_cross_stage_pooling": True,
            "no_retries_or_substitutions": True,
            "failed_or_undefined_children_retained_unrankable": True,
            "quota_failure_policy": "campaign_incomplete_and_candidate_lock_forbidden",
            "shared_seed_inference": (
                "blocking_or_common_random_number_control_only_not_paired_inference"
            ),
            "ranking": list(authority.RANKING_ORDER),
        },
        "stages": stages,
        "selection": {
            "candidate_id": winner_entry["candidate_id"],
            "selected_attempt_id": winner_entry["attempt_id"],
            "selected_config_id": winner,
            "rule": authority.SELECTION_RULE,
            "checkpoint_selection_rule": authority.CHECKPOINT_SELECTION_RULE,
            "selected_without_final_seed_results": True,
        },
    }


def _install_lock_builder_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, SimpleNamespace, SimpleNamespace, dict]:
    from scripts.udlm import superiority_gate

    decision = _decision()
    selected = decision["selection"]
    arm, temperature, top_p = _config_values(selected["selected_config_id"])
    sampling = authority.benchmark.validate_sampling_config(
        {
            "diffusion_type": "udlm",
            "softmax_temp": temperature,
            "randomness": 0.0,
            "min_add_len": 40,
            "num_steps": 128,
            "inference_eps": 1e-5,
            "exclude_special_tokens": False,
            "prior_variant": "release_uniform",
            "prior_metadata_sha256": None,
            "raw_loo_top_p": top_p,
        }
    )
    config_sha = "c" * 64
    sampling_sha = authority.canonical_json_sha256(sampling)
    config = authority.campaign.CandidateConfig(
        config_id=selected["selected_config_id"],
        arm_id=arm,
        scale_up_member_slug=arm.lower(),
        softmax_temp=temperature,
        raw_loo_top_p=top_p,
        is_reused_identity=False,
        relative_path=f"experiments/configs/{selected['selected_config_id']}.yaml",
        sha256=config_sha,
        size_bytes=10,
        storage="generated",
        normalized_sampling_sha256=sampling_sha,
    )
    registry = SimpleNamespace(
        raw_sha256=decision["registry"]["sha256"],
        data={"registry_id": decision["registry"]["registry_id"]},
        configs=(config,),
    )
    checkpoint = {"sha256": "4" * 64, "size_bytes": 1234, "global_step": 1000}
    exit_receipt = {
        "relative_path": f"output/udlm/scaleup-{arm.lower()}/pilot_exit_status.json",
        "sha256": "5" * 64,
        "schema_version": 5,
    }
    inference_weights = {
        "source": "ema",
        "ema_applied": True,
        "ema": {
            "shadow_parameter_count": 230,
            "decay": 0.9999,
            "num_updates": 1000,
        },
    }
    selected_pilot = {
        "identity": {
            "candidate_id": selected["candidate_id"],
            "requested_samples": 256,
            "nfe": 128,
            "metric_branch": "released_comparable",
            "checkpoint": checkpoint,
            "evaluation_config": {
                "relative_path": config.relative_path,
                "sha256": config.sha256,
            },
            "sampling": {"config": sampling, "sha256": sampling_sha},
            "inference_weights": inference_weights,
            "runner_sha256": "6" * 64,
            "sampler_source_sha256": "7" * 64,
            "implementation_inputs_sha256": "8" * 64,
            "metric_inputs_sha256": "9" * 64,
            "benchmark_revision": decision["registry"]["registry_revision"],
            "training_exit_receipt": {
                **exit_receipt,
                "recorded_at_utc": "2026-09-07T00:00:00+00:00",
            },
        },
        "completed_at_utc": "2026-09-07T00:10:00+00:00",
        "eligible_stage_completed_at_utc": "2026-09-07T00:20:00+00:00",
    }
    training = {
        "source_revision": "a" * 40,
        "training_summary": {
            "relative_path": f"output/udlm/scaleup-{arm.lower()}/training_summary.json",
            "sha256": "a" * 64,
            "schema_version": 5,
        },
        "exit_receipt": exit_receipt,
        "runtime_config": {
            "relative_path": f"output/udlm/scaleup-{arm.lower()}/runtime_config.json",
            "sha256": "b" * 64,
            "schema_version": 2,
        },
        "launch_manifest": {
            "relative_path": f"output/udlm/scaleup-{arm.lower()}/launch_manifest.json",
            "sha256": "d" * 64,
            "schema_version": 2,
        },
        "resolved_training_config_sha256": "e" * 64,
        "training_argv_sha256": "f" * 64,
        "checkpoint": {
            "relative_path": f"output/udlm/scaleup-{arm.lower()}/checkpoints/1000.ckpt",
            **checkpoint,
            "weights": "ema",
        },
        "startup": {
            "mode": "warm_start",
            "initialization_checkpoint_sha256": "1" * 64,
        },
        "training_seed": 17,
        "optimizer_updates": 1000,
        "world_size": 1,
        "data_exposure": {
            "global_examples_per_optimizer_step": 16,
            "optimizer_updates": 1000,
            "total_requested_examples": 16000,
            "stream_partition_policy": (
                "huggingface_split_dataset_by_node_disjoint_rank_streams"
            ),
        },
        "parameter_counts": {
            "base_model_trainable": 10,
            "time_conditioner_trainable": 2,
            "total_trainable": 12,
        },
    }
    terminal_e = {
        "successful_exit_receipt": {
            "relative_path": "output/udlm/scaleup-e/pilot_exit_status.json",
            "sha256": "2" * 64,
            "schema_version": 5,
        }
    }
    protocol_payload = b"{}\n"
    monkeypatch.setattr(
        authority,
        "_snapshot",
        lambda path, _label: (
            SimpleNamespace(
                sha256=superiority_gate.PROTOCOL_SHA256,
                size_bytes=len(protocol_payload),
            ),
            protocol_payload,
        ),
    )
    monkeypatch.setattr(superiority_gate, "validate_protocol", lambda _value: None)
    monkeypatch.setattr(
        authority, "_selected_pilot_authority", lambda _decision: selected_pilot
    )
    monkeypatch.setattr(
        authority,
        "_terminal_training_authority",
        lambda *_args, **_kwargs: (
            training,
            {"arm_id": arm},
            terminal_e,
            inference_weights,
        ),
    )

    def source_digest(_revision: str, path: str) -> str:
        if path == config.relative_path:
            return config.sha256
        if path == "scripts/exps/denovo/benchmark.py":
            return selected_pilot["identity"]["runner_sha256"]
        if path == "src/genmol/sampler.py":
            return selected_pilot["identity"]["sampler_source_sha256"]
        return hashlib.sha256(path.encode()).hexdigest()

    monkeypatch.setattr(authority, "_source_digest_at_revision", source_digest)
    ledger_claim = SimpleNamespace(sha256="3" * 64)
    return decision, registry, ledger_claim, selected_pilot


def test_exact_decision_projects_all_and_only_executed_attempts() -> None:
    decision = _decision()
    assert authority.validate_candidate_decision(decision) == decision
    ledger = authority.project_candidate_ledger(decision)
    assert ledger["status"] == "closed_before_final_evaluation"
    assert len(ledger["attempts"]) == 40
    assert sum(len(row["artifact_refs"]) for row in ledger["attempts"]) == 43
    assert "selected_config_id" not in ledger["selection"]
    assert ledger["selection"]["candidate_id"] == decision["selection"]["candidate_id"]
    assert all(
        row["ineligibility_reason"] == "engineering_or_nonregistered_operating_point"
        for row in ledger["attempts"][:-3]
    )


def test_lock_builder_derives_fixed_payload_without_manual_draft_or_final_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision, registry, ledger_claim, selected_pilot = _install_lock_builder_mocks(
        monkeypatch
    )
    captured = []

    def validate(value, **kwargs):
        captured.append((value, kwargs))
        return dict(value)

    monkeypatch.setattr(authority, "validate_lock_draft", validate)
    lock = authority.build_candidate_lock(
        decision=decision,
        ledger_claim=ledger_claim,
        registry=registry,
        locked_at_utc=datetime(2026, 9, 7, 1, tzinfo=timezone.utc),
    )
    selected = decision["selection"]
    assert lock["candidate_id"] == selected["candidate_id"]
    assert lock["selection"]["candidate_ledger"] == {
        "relative_path": authority.LEDGER_RELATIVE_PATH,
        "sha256": ledger_claim.sha256,
        "schema_version": 2,
    }
    assert lock["inference"]["final_run_directories_by_seed"] == [
        {
            "seed": seed,
            "relative_path": (
                f"output/udlm/final/{selected['candidate_id']}/"
                f"{selected['selected_config_id']}/seed_{seed}"
            ),
        }
        for seed in authority.FINAL_SEEDS
    ]
    assert captured[0][1]["selected_pilot_authority"] is selected_pilot
    assert not hasattr(authority, "LOCK_DRAFT_RELATIVE_PATH")


def test_lock_builder_rejects_checkpoint_swap_and_preeligible_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision, registry, ledger_claim, selected_pilot = _install_lock_builder_mocks(
        monkeypatch
    )
    monkeypatch.setattr(
        authority, "validate_lock_draft", lambda value, **_kwargs: value
    )
    selected_pilot["identity"]["checkpoint"] = {
        **selected_pilot["identity"]["checkpoint"],
        "sha256": "0" * 64,
    }
    with pytest.raises(authority.CandidateAuthorityError, match="checkpoint/training"):
        authority.build_candidate_lock(
            decision=decision,
            ledger_claim=ledger_claim,
            registry=registry,
            locked_at_utc=datetime(2026, 9, 7, 1, tzinfo=timezone.utc),
        )

    decision, registry, ledger_claim, _selected_pilot = _install_lock_builder_mocks(
        monkeypatch
    )
    monkeypatch.setattr(
        authority, "validate_lock_draft", lambda value, **_kwargs: value
    )
    with pytest.raises(authority.CandidateAuthorityError, match="must follow"):
        authority.build_candidate_lock(
            decision=decision,
            ledger_claim=ledger_claim,
            registry=registry,
            locked_at_utc=datetime(2026, 9, 7, 0, 20, tzinfo=timezone.utc),
        )


def test_selected_pilot_authority_rejects_cross_seed_identity_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = _decision()
    selected = decision["selection"]
    eligible = decision["stages"][-1]
    entry = next(
        row
        for row in eligible["entries"]
        if row["attempt_id"] == selected["selected_attempt_id"]
    )
    source_children = [
        {
            **child,
            "relative_path": (
                "output/udlm/de_novo_candidate_campaign_v1/evidence/"
                f"{entry['attempt_id']}/seed_{child['pilot_seed']}.json"
            ),
        }
        for child in entry["child_outcomes"]
    ]
    source = {
        "schema_version": 1,
        "stage_id": "eligible",
        "status": "completed",
        "entries": [
            {
                "attempt_id": entry["attempt_id"],
                "candidate_id": entry["candidate_id"],
                "config_id": entry["config_id"],
                "status": "completed",
                "rankable": True,
                "released_quality": entry["selection_score"]["released_quality"],
                "released_diversity": entry["selection_score"]["released_diversity"],
                "child_outcomes": source_children,
            }
        ],
        "completed_at_utc": "2026-09-07T00:20:00+00:00",
    }
    payload = authority.canonical_json_bytes(source)
    eligible["advancement"]["source_stage_decision"]["sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    monkeypatch.setattr(
        authority,
        "_snapshot",
        lambda _path, _label: (
            SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()),
            payload,
        ),
    )
    mismatch_second_seed = False

    def artifact_reference(_value, _label, **expected):
        index = list(authority.campaign.STAGE_CONTRACT["eligible"]["seeds"]).index(
            expected["expected_seed"]
        )
        child = entry["child_outcomes"][index]
        return child, {
            "attempt_id": entry["attempt_id"],
            "candidate_id": entry["candidate_id"],
            "pilot_seed": child["pilot_seed"],
            "requested_samples": 256,
            "nfe": 128,
            "metric_branch": "released_comparable",
            "checkpoint": {"sha256": "1" * 64, "size_bytes": 1, "global_step": 1},
            "evaluation_config": {"relative_path": "config.yaml", "sha256": "2" * 64},
            "sampling": {"config": {}, "sha256": "3" * 64},
            "inference_weights": {"source": "ema"},
            "runner_sha256": (
                "0" * 64 if mismatch_second_seed and index == 1 else "4" * 64
            ),
            "sampler_source_sha256": "5" * 64,
            "implementation_inputs_sha256": "6" * 64,
            "metric_inputs_sha256": "7" * 64,
            "benchmark_revision": "8" * 40,
            "training_exit_receipt": {
                "relative_path": "output/train/pilot_exit_status.json",
                "sha256": "9" * 64,
                "schema_version": 5,
            },
            "started_at_utc": f"2026-09-07T00:0{index}:00+00:00",
            "completed_at_utc": f"2026-09-07T00:1{index}:00+00:00",
            "quality": entry["selection_score"]["released_quality"],
            "diversity": entry["selection_score"]["released_diversity"],
        }

    monkeypatch.setattr(authority, "_artifact_reference", artifact_reference)
    result = authority._selected_pilot_authority(decision)
    assert result["identity"]["runner_sha256"] == "4" * 64

    mismatch_second_seed = True
    with pytest.raises(authority.CandidateAuthorityError, match="share one"):
        authority._selected_pilot_authority(decision)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["selection"].__setitem__("candidate_id", "r_t050_p100"),
        lambda value: value["stages"][1]["entries"][0].__setitem__(
            "attempt_id", "stage-a-r_t050_p100-seed1101"
        ),
        lambda value: value["campaign"].__setitem__("executed_entry_count", 39),
        lambda value: value["stages"][2]["advancement"][
            "promoted_config_ids"
        ].reverse(),
        lambda value: value["campaign"].__setitem__(
            "candidate_decision_sha256", "0" * 64
        ),
    ],
)
def test_decision_rejects_aliases_schedule_tampering_and_hash_cycles(mutator) -> None:
    decision = _decision()
    mutator(decision)
    with pytest.raises(authority.CandidateAuthorityError):
        authority.validate_candidate_decision(decision)


def test_ledger_uses_exact_failure_and_undefined_reason_enums() -> None:
    decision = _decision()
    failed = decision["stages"][1]["entries"][-1]
    failed["child_outcomes"][0]["artifact_kind"] = "pilot_failure"
    failed["status"] = "failed"
    failed["rankable"] = False
    failed["selection_score"] = None
    undefined = decision["stages"][-1]["entries"][-1]
    undefined["status"] = "undefined"
    undefined["rankable"] = False
    undefined["selection_score"] = None
    promoted, winner = authority._promotions(
        "eligible", decision["stages"][-1]["entries"]
    )
    decision["stages"][-1]["advancement"]["promoted_config_ids"] = promoted
    decision["stages"][-1]["advancement"]["global_winner_config_id"] = winner
    winner_entry = next(
        entry
        for entry in decision["stages"][-1]["entries"]
        if entry["config_id"] == winner
    )
    decision["selection"].update(
        {
            "candidate_id": winner_entry["candidate_id"],
            "selected_attempt_id": winner_entry["attempt_id"],
            "selected_config_id": winner,
        }
    )
    ledger = authority.project_candidate_ledger(decision)
    by_attempt = {row["attempt_id"]: row for row in ledger["attempts"]}
    assert by_attempt[failed["attempt_id"]]["ineligibility_reason"] == "pilot_failed"
    assert by_attempt[undefined["attempt_id"]]["ineligibility_reason"] == (
        "undefined_released_diversity_no_unique_molecules"
    )
    assert by_attempt[undefined["attempt_id"]]["status"] == "completed"

    noneligible_undefined = decision["stages"][1]["entries"][-2]
    noneligible_undefined["status"] = "undefined"
    noneligible_undefined["rankable"] = False
    noneligible_undefined["selection_score"] = None
    promoted, _winner = authority._promotions("A", decision["stages"][1]["entries"])
    decision["stages"][1]["advancement"]["promoted_config_ids"] = promoted
    ledger = authority.project_candidate_ledger(decision)
    by_attempt = {row["attempt_id"]: row for row in ledger["attempts"]}
    assert by_attempt[noneligible_undefined["attempt_id"]]["status"] == "completed"
    assert by_attempt[noneligible_undefined["attempt_id"]]["ineligibility_reason"] == (
        "engineering_or_nonregistered_operating_point"
    )


def _registry_in_rse_iteration_order(
    tmp_path: Path,
) -> authority.campaign.ValidatedRegistry:
    configs = tuple(
        authority.campaign.CandidateConfig(
            config_id=(f"{arm.lower()}_t{temperature_code}_p{top_p_code}"),
            arm_id=arm,
            scale_up_member_slug=arm.lower(),
            softmax_temp=temperature,
            raw_loo_top_p=top_p,
            is_reused_identity=(temperature, top_p) == (1.0, 1.0),
            relative_path=f"experiments/configs/{arm.lower()}-{temperature_code}-{top_p_code}.yaml",
            sha256="1" * 64,
            size_bytes=1,
            storage=(
                "historical_identity_reuse"
                if (temperature, top_p) == (1.0, 1.0)
                else "generated"
            ),
            normalized_sampling_sha256="2" * 64,
        )
        for arm in authority.campaign.ARM_IDS
        for temperature, temperature_code in (
            (0.5, "050"),
            (0.7, "070"),
            (0.85, "085"),
            (1.0, "100"),
        )
        for top_p, top_p_code in ((1.0, "100"), (0.98, "098"), (0.95, "095"))
    )
    assert [config.arm_id for config in configs[:13:12]] == ["R", "S"]
    return authority.campaign.ValidatedRegistry(
        path=tmp_path / authority.campaign.REGISTRY_RELATIVE_PATH,
        raw_sha256="3" * 64,
        canonical_sha256="4" * 64,
        size_bytes=1,
        data={
            "registry_id": "de-novo-candidate-config-registry-v1",
            "stages": authority.campaign.registry_verifier.expected_stages(),
        },
        configs=configs,
        checkpoint_by_arm={},
        training_receipt_by_arm={},
    )


def test_builder_accepts_ascii_stage_entries_from_rse_ordered_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_in_rse_iteration_order(tmp_path)
    stage_payloads: dict[str, bytes] = {}
    decisions: dict[str, dict] = {}
    predecessor = None
    for stage_index, stage_id in enumerate(authority.campaign.STAGE_IDS):
        attempts = authority.campaign.derive_attempts(registry, stage_id, decisions)
        assert [attempt.config_id for attempt in attempts] == sorted(
            (attempt.config_id for attempt in attempts), key=str.encode
        )
        outcomes = tuple(
            authority.campaign.AttemptOutcome(
                spec=attempt,
                status="completed",
                child_outcomes=tuple(
                    {
                        "artifact_kind": "pilot_evaluation",
                        "pilot_seed": seed,
                        "relative_path": (
                            f"experiments/udlm/pilots/{attempt.attempt_id}/seed_{seed}.json"
                        ),
                        "sha256": hashlib.sha256(
                            f"{attempt.attempt_id}:{seed}".encode()
                        ).hexdigest(),
                        "schema_version": 2,
                    }
                    for seed in attempt.seeds
                ),
                released_quality=None if stage_id == "D" else 0.8,
                released_diversity=None if stage_id == "D" else 0.7,
                diagnostic_structural_passed=True if stage_id == "D" else None,
            )
            for attempt in attempts
        )
        stage_decision = authority.campaign.derive_decision(
            registry=registry,
            stage_id=stage_id,
            attempts=attempts,
            outcomes=outcomes,
            prior_decisions=decisions,
            predecessor=predecessor,
            source_revision="5" * 40,
            decided_at_utc=f"2026-09-07T00:0{stage_index}:00+00:00",
        )
        decisions[stage_id] = stage_decision
        payload = json.dumps(stage_decision, sort_keys=True).encode()
        stage_payloads[authority._stage_path(stage_id)] = payload
        predecessor = {
            "relative_path": authority._stage_path(stage_id),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "schema_version": 1,
        }

    def snapshot(path: str, _label: str):
        payload = stage_payloads[path]
        return SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()), payload

    def artifact_reference(value, _label: str, **expected):
        stage_token = expected["expected_attempt_id"].split("-", 2)[1]
        stage_id = "eligible" if stage_token == "eligible" else stage_token.upper()
        stage_index = authority.campaign.STAGE_IDS.index(stage_id)
        if stage_index == 0:
            started_at = "2026-09-06T23:58:00+00:00"
            completed_at = "2026-09-06T23:59:00+00:00"
        else:
            started_at = f"2026-09-07T00:0{stage_index - 1}:10+00:00"
            completed_at = f"2026-09-07T00:0{stage_index - 1}:50+00:00"
        result = {
            "attempt_id": expected["expected_attempt_id"],
            "candidate_id": expected["expected_candidate_id"],
            "pilot_seed": expected["expected_seed"],
            "requested_samples": expected["expected_samples"],
            "nfe": 128,
            "metric_branch": "released_comparable",
            "quality": 0.8,
            "diversity": 0.7,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
        }
        return dict(value), result

    monkeypatch.setattr(authority, "_snapshot", snapshot)
    monkeypatch.setattr(authority, "_artifact_reference", artifact_reference)
    decision = authority.build_candidate_decision(registry, registry_revision="6" * 40)
    assert decision["stages"][1]["entries"][0]["config_id"] == "e_t050_p100"


def test_source_reference_is_live_but_decision_reference_is_tracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_id = "stage-a-e_t050_p100"
    candidate_id = "e-w1-1000u-dcb271453411"
    seed = 1101
    live_path = (
        "output/udlm/de_novo_candidate_campaign_v1/evidence/"
        f"{attempt_id}/seed_{seed}.json"
    )
    tracked_path = f"experiments/udlm/pilots/{attempt_id}/seed_{seed}.json"
    envelope = {
        "schema_version": 2,
        "artifact_kind": "pilot_evaluation",
        "status": "completed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": seed,
        "final_seed_results_included": False,
        "training_exit_receipt": {},
        "benchmark_artifacts": {},
    }
    payload = authority.canonical_json_bytes(envelope)
    digest = hashlib.sha256(payload).hexdigest()

    def snapshot(path: str, _label: str):
        assert path in {live_path, tracked_path}
        return SimpleNamespace(sha256=digest, size_bytes=len(payload)), payload

    monkeypatch.setattr(authority, "_snapshot", snapshot)
    monkeypatch.setattr(
        authority,
        "_validate_completed_envelope_live",
        lambda _envelope: {
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "pilot_seed": seed,
            "requested_samples": 32,
            "nfe": 128,
            "metric_branch": "released_comparable",
        },
    )
    tracked, _validated = authority._artifact_reference(
        {
            "artifact_kind": "pilot_evaluation",
            "pilot_seed": seed,
            "relative_path": live_path,
            "sha256": digest,
            "schema_version": 2,
        },
        "child",
        expected_attempt_id=attempt_id,
        expected_candidate_id=candidate_id,
        expected_seed=seed,
        expected_samples=32,
    )
    assert tracked == {
        "artifact_kind": "pilot_evaluation",
        "pilot_seed": seed,
        "relative_path": tracked_path,
        "sha256": digest,
        "schema_version": 2,
    }

    def mismatched_snapshot(path: str, _label: str):
        if path == tracked_path:
            return SimpleNamespace(sha256="0" * 64, size_bytes=len(payload)), b"changed"
        return SimpleNamespace(sha256=digest, size_bytes=len(payload)), payload

    monkeypatch.setattr(authority, "_snapshot", mismatched_snapshot)
    with pytest.raises(authority.CandidateAuthorityError, match="live/tracked"):
        authority._artifact_reference(
            {
                "artifact_kind": "pilot_evaluation",
                "pilot_seed": seed,
                "relative_path": live_path,
                "sha256": digest,
                "schema_version": 2,
            },
            "child",
            expected_attempt_id=attempt_id,
            expected_candidate_id=candidate_id,
            expected_seed=seed,
            expected_samples=32,
        )


def test_evidence_revision_is_exact_addition_only_child_of_g(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g_revision = "1" * 40
    e_revision = "2" * 40
    stage_path = authority._stage_path("D")
    live_path = (
        "output/udlm/de_novo_candidate_campaign_v1/evidence/"
        "stage-d-e_t100_p100/seed_1100.json"
    )
    tracked_path = "experiments/udlm/pilots/stage-d-e_t100_p100/seed_1100.json"
    support_path = "output/support.json"
    blobs = {
        stage_path: b"stage-decision",
        live_path: b"same-envelope",
        tracked_path: b"same-envelope",
        support_path: b"support",
        authority.campaign.REGISTRY_RELATIVE_PATH: b"registry",
        **{
            path: path.encode()
            for path in authority.materialize_candidate_evidence.LAUNCHER_RELATIVE_PATHS
        },
    }
    registry_reference = {
        "relative_path": authority.campaign.REGISTRY_RELATIVE_PATH,
        "sha256": hashlib.sha256(
            blobs[authority.campaign.REGISTRY_RELATIVE_PATH]
        ).hexdigest(),
        "canonical_sha256": "3" * 64,
        "size_bytes": len(blobs[authority.campaign.REGISTRY_RELATIVE_PATH]),
        "schema_version": 1,
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": authority.PROTOCOL_ID,
        "status": "complete_before_candidate_decision",
        "registry": registry_reference,
        "source_revision": {"head": g_revision, "upstream": g_revision},
        "counts": {},
        "stage_decisions": [
            {
                "stage_id": "D",
                "relative_path": stage_path,
                "sha256": hashlib.sha256(blobs[stage_path]).hexdigest(),
                "schema_version": 1,
            }
        ],
        "children": [
            {
                "live_envelope": {
                    "relative_path": live_path,
                    "sha256": hashlib.sha256(blobs[live_path]).hexdigest(),
                    "size_bytes": len(blobs[live_path]),
                    "schema_version": 2,
                },
                "tracked_envelope": {
                    "relative_path": tracked_path,
                    "sha256": hashlib.sha256(blobs[tracked_path]).hexdigest(),
                    "size_bytes": len(blobs[tracked_path]),
                    "schema_version": 2,
                },
                "supporting_artifacts": [
                    {
                        "artifact_role": "support",
                        "relative_path": support_path,
                        "sha256": hashlib.sha256(blobs[support_path]).hexdigest(),
                        "size_bytes": len(blobs[support_path]),
                        "schema_version": None,
                    }
                ],
            }
        ],
        "required_git_paths": [stage_path, support_path, tracked_path],
    }
    manifest_path = authority.materialize_candidate_evidence.MANIFEST_RELATIVE_PATH
    blobs[manifest_path] = authority.canonical_json_bytes(manifest)
    registry = SimpleNamespace(
        reference=registry_reference,
        raw_sha256=registry_reference["sha256"],
        size_bytes=registry_reference["size_bytes"],
        configs=(),
    )

    monkeypatch.setattr(
        authority.materialize_candidate_evidence,
        "validate_manifest",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        authority,
        "_git",
        lambda *arguments: (
            f"{e_revision} {g_revision}"
            if arguments[:3] == ("rev-list", "--parents", "-n")
            else ""
        ),
    )
    monkeypatch.setattr(authority, "_git_blob_absent", lambda *_args: True)
    monkeypatch.setattr(authority, "_git_blob_mode", lambda *_args: "100644")
    expected_changes = tuple(
        ("A", path) for path in (*manifest["required_git_paths"], manifest_path)
    )
    monkeypatch.setattr(authority, "_git_changed_entries", lambda *_: expected_changes)
    monkeypatch.setattr(authority, "_git_blob", lambda _revision, path: blobs[path])

    def snapshot(path: str, _label: str):
        payload = blobs[path]
        return (
            SimpleNamespace(
                sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)
            ),
            payload,
        )

    monkeypatch.setattr(authority, "_snapshot", snapshot)
    assert (
        authority._validate_evidence_publication(
            e_revision, registry_revision=g_revision, registry=registry
        )
        == manifest
    )

    monkeypatch.setattr(
        authority,
        "_git_changed_entries",
        lambda *_: (("M", tracked_path), *expected_changes[1:]),
    )
    with pytest.raises(authority.CandidateAuthorityError, match="add only"):
        authority._validate_evidence_publication(
            e_revision, registry_revision=g_revision, registry=registry
        )

    monkeypatch.setattr(authority, "_git_changed_entries", lambda *_: expected_changes)
    monkeypatch.setattr(authority, "_git_blob_mode", lambda *_args: "100755")
    with pytest.raises(authority.CandidateAuthorityError, match="non-executable"):
        authority._validate_evidence_publication(
            e_revision, registry_revision=g_revision, registry=registry
        )


def test_lock_publication_builds_from_committed_chain_without_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_revision = "1" * 40
    decision_revision = "2" * 40
    evidence_revision = "3" * 40
    decision = _decision()
    decision_payload = authority.canonical_json_bytes(decision)
    ledger_payload = authority.canonical_json_bytes(
        authority.project_candidate_ledger(decision)
    )
    ledger_claim = SimpleNamespace(
        sha256=hashlib.sha256(ledger_payload).hexdigest(),
        size_bytes=len(ledger_payload),
    )
    monkeypatch.setattr(authority, "_require_clean_pushed_source", lambda value: value)
    monkeypatch.setattr(authority, "_require_git_blob_absent", lambda *_args: None)

    def publication_parent(revision: str, path: str) -> str:
        if (revision, path) == (ledger_revision, authority.LEDGER_RELATIVE_PATH):
            return decision_revision
        if (revision, path) == (decision_revision, authority.DECISION_RELATIVE_PATH):
            return evidence_revision
        raise AssertionError((revision, path))

    monkeypatch.setattr(
        authority, "_require_exact_publication_commit", publication_parent
    )

    def snapshot(path: str, _label: str):
        if path == authority.DECISION_RELATIVE_PATH:
            return SimpleNamespace(), decision_payload
        if path == authority.LEDGER_RELATIVE_PATH:
            return ledger_claim, ledger_payload
        raise AssertionError(f"unexpected lock-publication read: {path}")

    monkeypatch.setattr(authority, "_snapshot", snapshot)
    monkeypatch.setattr(
        authority,
        "_git_blob",
        lambda revision, path: (
            decision_payload
            if (revision, path) == (decision_revision, authority.DECISION_RELATIVE_PATH)
            else ledger_payload
        ),
    )
    registry = SimpleNamespace()
    monkeypatch.setattr(
        authority, "_load_registry_for_decision", lambda _value: registry
    )
    evidence_checks = []
    monkeypatch.setattr(
        authority,
        "_validate_evidence_publication",
        lambda *args, **kwargs: evidence_checks.append((args, kwargs)),
    )
    build_calls = []
    lock = {"schema_version": 2}
    monkeypatch.setattr(
        authority,
        "build_candidate_lock",
        lambda **kwargs: build_calls.append(kwargs) or lock,
    )
    published = SimpleNamespace(relative_path=authority.LOCK_RELATIVE_PATH)
    monkeypatch.setattr(
        authority,
        "_publish",
        lambda path, value: (
            published if (path, value) == (authority.LOCK_RELATIVE_PATH, lock) else None
        ),
    )
    assert authority.publish_lock(expected_source_revision=ledger_revision) is published
    assert evidence_checks == [
        (
            (evidence_revision,),
            {
                "registry_revision": decision["registry"]["registry_revision"],
                "registry": registry,
            },
        )
    ]
    assert build_calls == [
        {"decision": decision, "ledger_claim": ledger_claim, "registry": registry}
    ]


def test_cli_phase_inputs_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        authority,
        "publish_ledger",
        lambda **kwargs: calls.append(kwargs)
        or type(
            "Claim",
            (),
            {
                "relative_path": authority.LEDGER_RELATIVE_PATH,
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
        )(),
    )
    assert (
        authority.main(["--phase", "ledger", "--expected-source-revision", "b" * 40])
        == 0
    )
    assert calls == [{"expected_source_revision": "b" * 40}]
    with pytest.raises(authority.CandidateAuthorityError, match="decision-only"):
        authority.main(
            [
                "--phase",
                "ledger",
                "--expected-source-revision",
                "b" * 40,
                "--registry-revision",
                "c" * 40,
            ]
        )
