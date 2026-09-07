from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from scripts import artifact_io
from scripts.exps.denovo import launch_benchmark
from scripts.exps.denovo import report as benchmark_report
from scripts.udlm import launch_candidate_campaign as campaign
from scripts.udlm import superiority_gate
from scripts.udlm import verify_denovo_candidate_config_registry as registry_verifier


def _registry(tmp_path: Path) -> campaign.ValidatedRegistry:
    configs = tuple(
        campaign.CandidateConfig(
            config_id=campaign._config_id(arm, temperature, top_p),
            arm_id=arm,
            scale_up_member_slug=arm.lower(),
            softmax_temp=temperature,
            raw_loo_top_p=top_p,
            is_reused_identity=(temperature, top_p) == (1.0, 1.0),
            relative_path=(
                f"experiments/configs/{campaign._config_id(arm, temperature, top_p)}.yaml"
            ),
            sha256="a" * 64,
            size_bytes=1,
            storage=(
                "historical_identity_reuse"
                if (temperature, top_p) == (1.0, 1.0)
                else "generated"
            ),
            normalized_sampling_sha256="b" * 64,
        )
        for arm in campaign.ARM_IDS
        for temperature in campaign.TEMPERATURES
        for top_p in campaign.RAW_TOP_P_VALUES
    )
    checkpoints = {
        arm: {
            "relative_path": f"output/{arm.lower()}/1000.ckpt",
            "sha256": "c" * 64,
            "size_bytes": 10,
            "global_step": 1000,
        }
        for arm in campaign.ARM_IDS
    }
    receipts = {
        arm: {
            "relative_path": f"output/{arm.lower()}/pilot_exit_status.json",
            "sha256": "d" * 64,
            "size_bytes": 10,
            "schema_version": 5,
        }
        for arm in campaign.ARM_IDS
    }
    return campaign.ValidatedRegistry(
        path=tmp_path / campaign.REGISTRY_RELATIVE_PATH,
        raw_sha256="e" * 64,
        canonical_sha256="f" * 64,
        size_bytes=123,
        data={
            "publication": {
                "framework_revision": "1" * 40,
                "config_revision": "2" * 40,
            },
            "stages": registry_verifier.expected_stages(),
        },
        configs=configs,
        checkpoint_by_arm=checkpoints,
        training_receipt_by_arm=receipts,
    )


def _child_ref(seed: int, *, failed: bool = False, suffix: str = "") -> dict[str, Any]:
    return {
        "artifact_kind": "pilot_failure" if failed else "pilot_evaluation",
        "pilot_seed": seed,
        "relative_path": f"output/evidence/seed_{seed}{suffix}.json",
        "sha256": hashlib.sha256(f"{seed}:{failed}:{suffix}".encode()).hexdigest(),
        "schema_version": 2,
    }


def _successful_outcomes(
    attempts: tuple[campaign.AttemptSpec, ...],
    *,
    equal_scores: bool = False,
) -> tuple[campaign.AttemptOutcome, ...]:
    outcomes = []
    for index, spec in enumerate(attempts):
        if spec.stage_id == "D":
            quality = diversity = None
            diagnostic = True
        else:
            quality = 0.5 if equal_scores else 1.0 - index / 100.0
            diversity = 0.5 if equal_scores else 0.9 - index / 100.0
            diagnostic = None
        outcomes.append(
            campaign.AttemptOutcome(
                spec=spec,
                status="completed",
                child_outcomes=tuple(
                    _child_ref(seed, suffix=f"-{spec.config_id}") for seed in spec.seeds
                ),
                released_quality=quality,
                released_diversity=diversity,
                diagnostic_structural_passed=diagnostic,
            )
        )
    return tuple(outcomes)


def _decision(
    registry: campaign.ValidatedRegistry,
    stage_id: str,
    decisions: dict[str, dict[str, Any]],
    *,
    outcomes: tuple[campaign.AttemptOutcome, ...] | None = None,
) -> tuple[tuple[campaign.AttemptSpec, ...], dict[str, Any]]:
    attempts = campaign.derive_attempts(registry, stage_id, decisions)
    result = campaign.derive_decision(
        registry=registry,
        stage_id=stage_id,
        attempts=attempts,
        outcomes=outcomes or _successful_outcomes(attempts),
        prior_decisions=decisions,
        predecessor=None,
        source_revision="9" * 40,
        decided_at_utc="2026-09-07T00:00:00+00:00",
    )
    decisions[stage_id] = result
    return attempts, result


def _complete_through(
    registry: campaign.ValidatedRegistry, final_stage: str
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[campaign.AttemptSpec, ...]]]:
    decisions: dict[str, dict[str, Any]] = {}
    attempts: dict[str, tuple[campaign.AttemptSpec, ...]] = {}
    for stage_id in campaign.STAGE_IDS:
        attempts[stage_id], _ = _decision(registry, stage_id, decisions)
        if stage_id == final_stage:
            break
    return decisions, attempts


def _gpu(
    index: int,
    *,
    utilization: int,
    uuid: str | None = None,
    used: int = 1_000,
    processes: tuple[dict[str, object], ...] = (),
) -> launch_benchmark.GPUState:
    return launch_benchmark.GPUState(
        index=index,
        uuid=uuid or f"GPU-{index}",
        name="test",
        memory_used_mib=used,
        memory_total_mib=49_140,
        utilization_percent=utilization,
        compute_mode="Default",
        compute_processes=processes,
    )


def _expected_ranked_identity(
    tmp_path: Path,
    *,
    config: campaign.CandidateConfig,
    sample_count: int,
) -> launch_benchmark.ExpectedRunIdentity:
    sampling = {
        "softmax_temp": config.softmax_temp,
        "raw_loo_top_p": config.raw_loo_top_p,
    }
    implementation = {
        "sampler_source": {"sha256": "7" * 64},
        "ema_source": {"sha256": "8" * 64},
    }
    metric_inputs = {"fixture": "metric-inputs"}
    return launch_benchmark.ExpectedRunIdentity(
        checkpoint_path=tmp_path / f"output/{config.arm_id.lower()}/1000.ckpt",
        checkpoint_sha256="c" * 64,
        checkpoint_global_step=1_000,
        checkpoint_size_bytes=10,
        checkpoint_diffusion_type="absorbing_state",
        checkpoint_udlm_inference_eps=None,
        checkpoint_udlm_exclude_special_tokens=None,
        checkpoint_udlm_prior_variant=None,
        checkpoint_udlm_prior_metadata=None,
        checkpoint_udlm_prior_metadata_sha256=None,
        config_path=tmp_path / config.relative_path,
        source_config={},
        source_config_sha256=config.sha256,
        config_git_tracking={"relative_path": config.relative_path},
        sampling_config=sampling,
        sampling_config_sha256=config.normalized_sampling_sha256,
        effective_config={},
        effective_config_sha256="9" * 64,
        benchmark_runner_sha256="6" * 64,
        implementation_inputs=implementation,
        metric_inputs=metric_inputs,
        num_samples=sample_count,
        source_revision="5" * 40,
    )


def _independent_gate_result(
    envelope: dict[str, Any],
    *,
    expected: launch_benchmark.ExpectedRunIdentity,
    config: campaign.CandidateConfig,
    quality: float | None,
    diversity: float | None,
) -> dict[str, Any]:
    summary = envelope["benchmark_artifacts"]["summary_json"]
    raw = envelope["benchmark_artifacts"]["raw_samples_csv"]
    training = envelope["training_exit_receipt"]
    return {
        "attempt_id": envelope["attempt_id"],
        "candidate_id": envelope["candidate_id"],
        "pilot_seed": envelope["pilot_seed"],
        "requested_samples": expected.num_samples,
        "nfe": 128,
        "metric_branch": "released_comparable",
        "checkpoint": {
            "sha256": expected.checkpoint_sha256,
            "size_bytes": expected.checkpoint_size_bytes,
            "global_step": expected.checkpoint_global_step,
        },
        "evaluation_config": {
            "relative_path": config.relative_path,
            "sha256": expected.source_config_sha256,
        },
        "sampling": {
            "config": expected.sampling_config,
            "sha256": expected.sampling_config_sha256,
        },
        "inference_weights": "ema",
        "runner_sha256": expected.benchmark_runner_sha256,
        "sampler_source_sha256": expected.implementation_inputs["sampler_source"][
            "sha256"
        ],
        "implementation_inputs_sha256": campaign._canonical_sha256(
            expected.implementation_inputs
        ),
        "metric_inputs_sha256": campaign._canonical_sha256(expected.metric_inputs),
        "benchmark_revision": expected.source_revision,
        "started_at_utc": "2026-09-07T00:00:00+00:00",
        "completed_at_utc": "2026-09-07T00:00:01+00:00",
        "training_exit_receipt": {
            **training,
            "recorded_at_utc": "2026-09-06T00:00:00+00:00",
        },
        "summary_json_sha256": summary["sha256"],
        "raw_samples_csv_sha256": raw["sha256"],
        "quality": quality,
        "diversity": diversity,
        "independent_rescore": {
            "all_21_fields_match": True,
            "both_metric_branches_match": True,
            "failure_counts_match": True,
            "raw_model_text_redecoded": True,
        },
    }


def _write_ranked_child_inputs(
    tmp_path: Path,
    *,
    registry: campaign.ValidatedRegistry,
    spec: campaign.AttemptSpec,
    producer_quality: float,
) -> tuple[launch_benchmark.ExpectedRunIdentity, Path]:
    config = next(item for item in registry.configs if item.config_id == spec.config_id)
    expected = _expected_ranked_identity(
        tmp_path,
        config=config,
        sample_count=spec.requested_samples_per_seed,
    )
    run_directory = campaign._attempt_root(spec) / f"seed_{spec.seeds[0]}"
    run_directory.mkdir(parents=True)
    (run_directory / "summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "released_comparable": {
                        "quality": producer_quality,
                        "diversity": producer_quality,
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw_path = run_directory / "raw_samples.csv"
    raw_path.write_text("producer-shaped fixture\n", encoding="utf-8")
    training_path = tmp_path / str(
        registry.training_receipt_by_arm[spec.arm_id]["relative_path"]
    )
    training_path.parent.mkdir(parents=True)
    training_payload = b'{"fixture":"training-receipt"}\n'
    training_path.write_bytes(training_payload)
    training = registry.training_receipt_by_arm[spec.arm_id]
    assert isinstance(training, dict)
    training["sha256"] = hashlib.sha256(training_payload).hexdigest()
    training["size_bytes"] = len(training_payload)
    return expected, raw_path


def _completion_reference(stage_id: str) -> dict[str, Any]:
    return {
        "relative_path": (
            f"{campaign.CAMPAIGN_RELATIVE_ROOT}/stages/"
            f"{stage_id.lower()}/stage_decision.json"
        ),
        "sha256": hashlib.sha256(stage_id.encode("ascii")).hexdigest(),
        "schema_version": campaign.DECISION_SCHEMA_VERSION,
    }


def _install_execution_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initial_decisions: dict[str, dict[str, Any]],
) -> dict[str, list[Any]]:
    """Replace every mutating/external boundary while retaining campaign logic."""

    initial_completions = {
        stage_id: _completion_reference(stage_id) for stage_id in initial_decisions
    }
    trace: dict[str, list[Any]] = {
        "fresh": [],
        "reserved": [],
        "executed": [],
        "published": [],
        "acquired": [],
        "revalidated": [],
        "released": [],
        "source_revalidated": [],
        "events": [],
    }

    class FakeIdentityFactory:
        def __init__(
            self, _registry: campaign.ValidatedRegistry, _source_revision: str
        ) -> None:
            pass

        def expected(self, _spec: campaign.AttemptSpec) -> object:
            return object()

    monkeypatch.setattr(
        campaign,
        "replay_decisions",
        lambda _registry: (dict(initial_decisions), dict(initial_completions)),
    )
    monkeypatch.setattr(campaign, "_IdentityFactory", FakeIdentityFactory)
    monkeypatch.setattr(
        campaign,
        "_fresh_stage_paths",
        lambda attempts: trace["fresh"].append(attempts[0].stage_id),
    )
    monkeypatch.setattr(
        campaign,
        "_reserve_attempt_roots",
        lambda attempts: trace["reserved"].append(attempts[0].stage_id),
    )

    def execute_stage_children(**kwargs: Any) -> tuple[campaign.AttemptSpec, ...]:
        attempts = tuple(kwargs["attempts"])
        trace["executed"].append(attempts[0].stage_id)
        trace["events"].append(f"children:{attempts[0].stage_id}")
        return attempts

    monkeypatch.setattr(campaign, "_execute_stage_children", execute_stage_children)
    monkeypatch.setattr(
        campaign,
        "_publish_child_evidence",
        lambda item, *, registry: item,
    )
    monkeypatch.setattr(
        campaign,
        "_attempt_outcomes",
        lambda attempts, _evidence: _successful_outcomes(tuple(attempts)),
    )

    def publish_decision(**kwargs: Any) -> dict[str, Any]:
        decision = kwargs["decision"]
        trace["published"].append(decision["stage_id"])
        trace["events"].append(f"decision:{decision['stage_id']}")
        return _completion_reference(decision["stage_id"])

    monkeypatch.setattr(campaign, "publish_decision", publish_decision)
    lease = object()
    monkeypatch.setattr(
        launch_benchmark,
        "_acquire_generation_lease",
        lambda **_kwargs: trace["acquired"].append(True) or lease,
    )
    monkeypatch.setattr(
        launch_benchmark,
        "_revalidate_generation_lease",
        lambda value: trace["revalidated"].append(value),
    )

    def release_lease(value: object) -> None:
        trace["released"].append(value)
        trace["events"].append("lease:released")

    monkeypatch.setattr(
        launch_benchmark, "_release_generation_lease_exact", release_lease
    )

    def require_source(_registry: campaign.ValidatedRegistry) -> dict[str, str]:
        trace["source_revalidated"].append(True)
        trace["events"].append("source:revalidated")
        return {"head": "9" * 40, "upstream": "9" * 40}

    monkeypatch.setattr(campaign, "_require_exact_registry_revision", require_source)
    return trace


def test_registry_stage_contract_is_exact_and_includes_nonlaunched_final() -> None:
    stages = registry_verifier.expected_stages()
    campaign._validate_stages(stages)
    assert [stage["stage_id"] for stage in stages] == [
        "D",
        "A",
        "B",
        "C",
        "eligible",
        "final",
    ]
    assert stages[0]["ranking"] is None
    assert stages[-1]["promotion"] is None
    mutated = json.loads(json.dumps(stages))
    mutated[0]["role"] = "opaque"
    with pytest.raises(campaign.CampaignValidationError, match="stage contract"):
        campaign._validate_stages(mutated)


def test_parser_requires_explicit_real_stage_limit_and_never_accepts_final() -> None:
    parsed = campaign._parse_args(
        [
            "--registry",
            campaign.REGISTRY_RELATIVE_PATH,
            "--expected-registry-sha256",
            "a" * 64,
            "--expected-registry-canonical-sha256",
            "b" * 64,
            "--dry-run",
        ]
    )
    assert parsed.dry_run is True
    assert parsed.through_stage is None
    real = campaign._parse_args(
        [
            "--registry",
            campaign.REGISTRY_RELATIVE_PATH,
            "--expected-registry-sha256",
            "a" * 64,
            "--expected-registry-canonical-sha256",
            "b" * 64,
            "--through-stage",
            "D",
        ]
    )
    assert real.through_stage == "D"
    with pytest.raises(SystemExit):
        campaign._parse_args(
            [
                "--registry",
                campaign.REGISTRY_RELATIVE_PATH,
                "--expected-registry-sha256",
                "a" * 64,
                "--expected-registry-canonical-sha256",
                "b" * 64,
            ]
        )
    with pytest.raises(SystemExit):
        campaign._parse_args(
            [
                "--registry",
                campaign.REGISTRY_RELATIVE_PATH,
                "--expected-registry-sha256",
                "a" * 64,
                "--expected-registry-canonical-sha256",
                "b" * 64,
                "--through-stage",
                "final",
            ]
        )
    with pytest.raises(SystemExit):
        campaign._parse_args(
            [
                "--registry",
                campaign.REGISTRY_RELATIVE_PATH,
                "--expected-registry-sha256",
                "a" * 64,
                "--expected-registry-canonical-sha256",
                "b" * 64,
                "--gpu-count",
                "2",
            ]
        )


def test_exact_config_and_attempt_id_grammars() -> None:
    assert campaign._config_id("R", 0.5, 1.0) == "r_t050_p100"
    assert campaign._config_id("S", 0.85, 0.98) == "s_t085_p098"
    assert campaign._attempt_id("eligible", "e_t100_p095", (1000, 1001)) == (
        "stage-eligible-e_t100_p095"
    )
    with pytest.raises(campaign.CampaignValidationError, match="seed tuple"):
        campaign._attempt_id("eligible", "e_t100_p095", (1000,))


def test_exact_stage_populations_and_prefinal_arithmetic(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    decisions, attempts = _complete_through(registry, "eligible")
    assert {stage: len(values) for stage, values in attempts.items()} == {
        "D": 1,
        "A": 12,
        "B": 18,
        "C": 6,
        "eligible": 3,
    }
    assert sum(len(values) for values in attempts.values()) == 40
    assert (
        sum(len(attempt.seeds) for values in attempts.values() for attempt in values)
        == 43
    )
    assert (
        sum(
            len(attempt.seeds) * attempt.requested_samples_per_seed
            for values in attempts.values()
            for attempt in values
        )
        == 3680
    )
    assert decisions["eligible"]["advancement"]["global_winner_config_id"]
    assert all(
        not attempt.attempt_id.endswith("-1000")
        and not attempt.attempt_id.endswith("-1001")
        for attempt in attempts["eligible"]
    )


def test_diagnostic_never_ranks_or_reads_chemistry(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    attempts = campaign.derive_attempts(registry, "D", {})
    bad = campaign.AttemptOutcome(
        spec=attempts[0],
        status="completed",
        child_outcomes=(_child_ref(1100),),
        released_quality=0.9,
        released_diversity=0.8,
        diagnostic_structural_passed=True,
    )
    with pytest.raises(campaign.CampaignValidationError, match="diagnostic"):
        campaign.derive_decision(
            registry=registry,
            stage_id="D",
            attempts=attempts,
            outcomes=(bad,),
            prior_decisions={},
            predecessor=None,
            source_revision="9" * 40,
            decided_at_utc="2026-09-07T00:00:00+00:00",
        )


def test_ranked_promotions_use_independent_gate_metrics_not_producer_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launch_benchmark, "REPOSITORY_ROOT", tmp_path)
    decisions: dict[str, dict[str, Any]] = {}
    _decision(registry, "D", decisions)
    attempts = campaign.derive_attempts(registry, "A", decisions)
    spec = attempts[0]
    expected, _raw_path = _write_ranked_child_inputs(
        tmp_path,
        registry=registry,
        spec=spec,
        producer_quality=1.0,
    )
    terminal = campaign.TerminalChild(
        spec=spec,
        seed=spec.seeds[0],
        expected=expected,
        failed=False,
        command=("synthetic-child",),
    )
    captured: list[dict[str, Any]] = []

    def independent(envelope: dict[str, Any]) -> dict[str, Any]:
        captured.append(envelope)
        config = next(
            item for item in registry.configs if item.config_id == spec.config_id
        )
        return _independent_gate_result(
            envelope,
            expected=expected,
            config=config,
            quality=0.01,
            diversity=0.02,
        )

    monkeypatch.setattr(
        superiority_gate, "_validate_completed_pilot_evidence_live", independent
    )
    monkeypatch.setattr(
        benchmark_report,
        "validate_run_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "campaign read producer metrics outside the independent gate"
        ),
    )
    evidence = campaign._publish_child_evidence(terminal, registry=registry)
    assert len(captured) == 1
    assert captured[0]["artifact_kind"] == "pilot_evaluation"
    assert evidence.released_quality == pytest.approx(0.01)
    assert evidence.released_diversity == pytest.approx(0.02)

    outcomes = list(_successful_outcomes(attempts))
    outcomes[0] = campaign._attempt_outcomes((spec,), (evidence,))[0]
    decision = campaign.derive_decision(
        registry=registry,
        stage_id="A",
        attempts=attempts,
        outcomes=tuple(outcomes),
        prior_decisions=decisions,
        predecessor=None,
        source_revision="9" * 40,
        decided_at_utc="2026-09-07T00:00:00+00:00",
    )
    assert spec.config_id not in decision["advancement"]["promoted_config_ids"]
    assert decision["entries"][0]["released_quality"] == pytest.approx(0.01)


def test_ranked_gate_input_replacement_fails_before_envelope_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launch_benchmark, "REPOSITORY_ROOT", tmp_path)
    decisions: dict[str, dict[str, Any]] = {}
    _decision(registry, "D", decisions)
    spec = campaign.derive_attempts(registry, "A", decisions)[0]
    expected, raw_path = _write_ranked_child_inputs(
        tmp_path,
        registry=registry,
        spec=spec,
        producer_quality=1.0,
    )
    terminal = campaign.TerminalChild(
        spec=spec,
        seed=spec.seeds[0],
        expected=expected,
        failed=False,
        command=("synthetic-child",),
    )

    def replace_input(envelope: dict[str, Any]) -> dict[str, Any]:
        config = next(
            item for item in registry.configs if item.config_id == spec.config_id
        )
        raw_path.unlink()
        raw_path.write_text("foreign replacement\n", encoding="utf-8")
        return _independent_gate_result(
            envelope,
            expected=expected,
            config=config,
            quality=0.9,
            diversity=0.8,
        )

    monkeypatch.setattr(
        superiority_gate, "_validate_completed_pilot_evidence_live", replace_input
    )
    with pytest.raises(campaign.CampaignValidationError, match="changed before"):
        campaign._publish_child_evidence(terminal, registry=registry)
    assert not (
        tmp_path / campaign._evidence_relative_path(spec, spec.seeds[0])
    ).exists()


def test_diagnostic_publication_does_not_invoke_ranked_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launch_benchmark, "REPOSITORY_ROOT", tmp_path)
    spec = campaign.derive_attempts(registry, "D", {})[0]
    expected, _raw_path = _write_ranked_child_inputs(
        tmp_path,
        registry=registry,
        spec=spec,
        producer_quality=1.0,
    )
    terminal = campaign.TerminalChild(
        spec=spec,
        seed=spec.seeds[0],
        expected=expected,
        failed=False,
        command=("synthetic-child",),
    )
    monkeypatch.setattr(
        campaign, "_diagnostic_structural_validation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        superiority_gate,
        "_validate_completed_pilot_evidence_live",
        lambda *_args, **_kwargs: pytest.fail("D invoked chemistry rescore"),
    )
    evidence = campaign._publish_child_evidence(terminal, registry=registry)
    assert evidence.released_quality is None
    assert evidence.released_diversity is None
    assert evidence.diagnostic_structural_passed is True


def test_ascii_ties_are_config_major_and_stage_local(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    decisions: dict[str, dict[str, Any]] = {}
    _decision(registry, "D", decisions)
    attempts = campaign.derive_attempts(registry, "A", decisions)
    decision = campaign.derive_decision(
        registry=registry,
        stage_id="A",
        attempts=attempts,
        outcomes=_successful_outcomes(attempts, equal_scores=True),
        prior_decisions=decisions,
        predecessor=None,
        source_revision="9" * 40,
        decided_at_utc="2026-09-07T00:00:00+00:00",
    )
    assert decision["advancement"]["promoted_config_ids"] == [
        "r_t050_p100",
        "r_t070_p100",
        "s_t050_p100",
        "s_t070_p100",
        "e_t050_p100",
        "e_t070_p100",
    ]
    assert all(attempt.attempt_id.endswith(attempt.config_id) for attempt in attempts)


def test_failures_are_retained_unrankable_without_retry_or_partial_promotion(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    decisions: dict[str, dict[str, Any]] = {}
    _decision(registry, "D", decisions)
    attempts = campaign.derive_attempts(registry, "A", decisions)
    outcomes = list(_successful_outcomes(attempts))
    r_indices = [index for index, spec in enumerate(attempts) if spec.arm_id == "R"]
    for index in r_indices[:3]:
        spec = attempts[index]
        outcomes[index] = campaign.AttemptOutcome(
            spec=spec,
            status="failed",
            child_outcomes=(_child_ref(1101, failed=True, suffix=spec.config_id),),
            released_quality=None,
            released_diversity=None,
        )
    decision = campaign.derive_decision(
        registry=registry,
        stage_id="A",
        attempts=attempts,
        outcomes=tuple(outcomes),
        prior_decisions=decisions,
        predecessor=None,
        source_revision="9" * 40,
        decided_at_utc="2026-09-07T00:00:00+00:00",
    )
    assert decision["status"] == "campaign_incomplete"
    assert decision["advancement"]["promoted_config_ids"] == []
    assert decision["advancement"]["no_retry_or_substitution"] is True
    assert len(decision["entries"]) == 12
    assert sum(entry["rankable"] for entry in decision["entries"]) == 9


def test_undefined_metric_is_retained_and_unrankable(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    decisions: dict[str, dict[str, Any]] = {}
    _decision(registry, "D", decisions)
    attempts = campaign.derive_attempts(registry, "A", decisions)
    outcomes = list(_successful_outcomes(attempts))
    spec = outcomes[0].spec
    outcomes[0] = campaign.AttemptOutcome(
        spec=spec,
        status="undefined",
        child_outcomes=(_child_ref(1101, suffix="undefined"),),
        released_quality=None,
        released_diversity=None,
    )
    decision = campaign.derive_decision(
        registry=registry,
        stage_id="A",
        attempts=attempts,
        outcomes=tuple(outcomes),
        prior_decisions=decisions,
        predecessor=None,
        source_revision="9" * 40,
        decided_at_utc="2026-09-07T00:00:00+00:00",
    )
    entry = next(
        item for item in decision["entries"] if item["config_id"] == spec.config_id
    )
    assert entry["status"] == "undefined"
    assert entry["rankable"] is False
    assert entry["released_quality"] is None


def test_eligible_attempt_aggregates_exact_two_seeds(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    decisions, _attempts = _complete_through(registry, "C")
    attempts = campaign.derive_attempts(registry, "eligible", decisions)
    evidence: list[campaign.ChildEvidence] = []
    for spec in attempts:
        for offset, seed in enumerate(spec.seeds):
            terminal = campaign.TerminalChild(
                spec=spec,
                seed=seed,
                expected=None,  # type: ignore[arg-type]
                failed=False,
                command=(),
            )
            evidence.append(
                campaign.ChildEvidence(
                    terminal=terminal,
                    reference=_child_ref(seed, suffix=spec.config_id),
                    released_quality=0.2 + 0.2 * offset,
                    released_diversity=0.4 + 0.2 * offset,
                    diagnostic_structural_passed=None,
                )
            )
    outcomes = campaign._attempt_outcomes(attempts, evidence)
    assert all(outcome.released_quality == pytest.approx(0.3) for outcome in outcomes)
    assert all(outcome.released_diversity == pytest.approx(0.5) for outcome in outcomes)
    assert all(len(outcome.child_outcomes) == 2 for outcome in outcomes)


def test_gpu_selection_enforces_three_uuid_cap_and_strict_9_10_boundary() -> None:
    active = ({"pid": 1, "process_name": "other", "used_memory_mib": 5},)
    states = [
        _gpu(0, utilization=9, processes=active),
        _gpu(1, utilization=10),
        _gpu(2, utilization=0),
        _gpu(3, utilization=1),
    ]
    selected = campaign._launchable_gpus(states, running_uuids=set(), free_slots=3)
    assert len(selected) == 3
    assert "GPU-1" not in {state.uuid for state in selected}
    assert "GPU-0" in {state.uuid for state in selected}
    assert all(state.uuid.startswith("GPU-") for state in selected)
    with pytest.raises(campaign.CampaignValidationError, match="free GPU slot"):
        campaign._launchable_gpus(states, running_uuids=set(), free_slots=4)


def test_dry_run_record_is_explicitly_pure_and_never_authorizes_final(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    record = campaign._dry_run_record(
        registry=registry,
        source_revision={"head": "9" * 40, "upstream": "9" * 40},
        decisions={},
        through_stage="D",
    )
    assert record["generation_lease_acquired"] is False
    assert record["gpu_query_performed"] is False
    assert record["artifact_mutation_performed"] is False
    assert record["tmux_operation_performed"] is False
    assert record["next_stage_id"] == "D"
    assert record["requested_through_stage"] == "D"
    assert record["authorized_stage_ids"] == ["D"]
    assert record["next_stage_child_count"] == 1
    assert record["final_stage_not_authorized_or_launched"] is True


def test_main_dry_run_never_touches_tmux_gpu_lease_or_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launch_benchmark, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        launch_benchmark,
        "_require_project_virtual_environment",
        lambda: tmp_path / ".venv/bin/python",
    )
    monkeypatch.setattr(campaign, "load_registry", lambda *_args, **_kwargs: registry)
    monkeypatch.setattr(
        campaign,
        "_require_exact_registry_revision",
        lambda _registry: {"head": "9" * 40, "upstream": "9" * 40},
    )
    monkeypatch.setattr(campaign, "replay_decisions", lambda _registry: ({}, {}))
    for name in (
        "_require_tmux_for_execution",
        "_acquire_generation_lease",
        "_snapshot",
    ):
        monkeypatch.setattr(
            launch_benchmark,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"dry run called {_name}"
            ),
        )
    monkeypatch.setattr(
        campaign,
        "run_campaign",
        lambda **_kwargs: pytest.fail("dry run entered campaign execution"),
    )
    assert (
        campaign.main(
            [
                "--registry",
                campaign.REGISTRY_RELATIVE_PATH,
                "--expected-registry-sha256",
                "e" * 64,
                "--expected-registry-canonical-sha256",
                "f" * 64,
                "--dry-run",
            ]
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    assert record["gpu_query_performed"] is False
    assert record["artifact_mutation_performed"] is False


def test_d_only_limit_stops_after_durable_decision_and_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = _registry(tmp_path)
    trace = _install_execution_harness(monkeypatch, initial_decisions={})
    source = {"head": "9" * 40, "upstream": "9" * 40}

    assert (
        campaign.run_campaign(
            registry=registry, source_revision=source, through_stage="D"
        )
        == 0
    )
    assert trace["fresh"] == ["D"]
    assert trace["reserved"] == ["D"]
    assert trace["executed"] == ["D"]
    assert trace["published"] == ["D"]
    assert len(trace["acquired"]) == 1
    assert len(trace["released"]) == 1
    assert len(trace["source_revalidated"]) == 1
    assert trace["events"] == [
        "children:D",
        "decision:D",
        "source:revalidated",
        "lease:released",
    ]
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[-1] == {
        "event": "candidate_campaign_stage_limit_reached",
        "final_stage_launched": False,
        "next_stage_id": "A",
        "through_stage": "D",
    }


@pytest.mark.parametrize(
    ("through_stage", "expected_stages"),
    [("A", ["A"]), ("B", ["A", "B"])],
)
def test_resume_starts_at_a_and_advances_sequentially_only_to_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    through_stage: str,
    expected_stages: list[str],
) -> None:
    registry = _registry(tmp_path)
    initial_decisions, _attempts = _complete_through(registry, "D")
    frozen_initial = json.dumps(initial_decisions, sort_keys=True)
    trace = _install_execution_harness(monkeypatch, initial_decisions=initial_decisions)

    assert (
        campaign.run_campaign(
            registry=registry,
            source_revision={"head": "9" * 40, "upstream": "9" * 40},
            through_stage=through_stage,
        )
        == 0
    )
    assert trace["executed"] == expected_stages
    assert trace["published"] == expected_stages
    assert len(trace["released"]) == 1
    assert json.dumps(initial_decisions, sort_keys=True) == frozen_initial


def test_stage_limit_rejects_gap_final_and_completed_frontier() -> None:
    decision = {"status": "completed"}
    with pytest.raises(campaign.CampaignValidationError, match="contiguous"):
        campaign._authorized_stage_ids({"A": decision}, "B")
    with pytest.raises(campaign.CampaignValidationError, match="final"):
        campaign._authorized_stage_ids({}, "final")
    with pytest.raises(campaign.CampaignValidationError, match="already terminal"):
        campaign._authorized_stage_ids({"D": decision}, "D")
    assert campaign._authorized_stage_ids({}, "B") == ("D", "A", "B")


def test_completed_limit_and_foreign_slot_are_never_reexecuted_or_clobbered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    decisions, _attempts = _complete_through(registry, "D")
    trace = _install_execution_harness(monkeypatch, initial_decisions=decisions)
    with pytest.raises(campaign.CampaignValidationError, match="already terminal"):
        campaign.run_campaign(
            registry=registry,
            source_revision={"head": "9" * 40, "upstream": "9" * 40},
            through_stage="D",
        )
    assert trace["fresh"] == []
    assert trace["acquired"] == []
    assert trace["executed"] == []
    assert trace["published"] == []

    collision_trace = _install_execution_harness(monkeypatch, initial_decisions={})
    monkeypatch.setattr(
        campaign,
        "_fresh_stage_paths",
        lambda _attempts: (_ for _ in ()).throw(FileExistsError("foreign slot")),
    )
    with pytest.raises(FileExistsError, match="foreign slot"):
        campaign.run_campaign(
            registry=registry,
            source_revision={"head": "9" * 40, "upstream": "9" * 40},
            through_stage="D",
        )
    assert collision_trace["acquired"] == []
    assert collision_trace["executed"] == []
    assert collision_trace["published"] == []


def test_attempt_root_reservation_rolls_back_owned_output_on_log_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    attempt = campaign.derive_attempts(registry, "D", {})[0]
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launch_benchmark, "REPOSITORY_ROOT", tmp_path)
    (tmp_path / campaign.CAMPAIGN_RELATIVE_ROOT / "attempts").mkdir(parents=True)
    log_parent = tmp_path / campaign.CAMPAIGN_LOG_RELATIVE_ROOT
    log_parent.mkdir(parents=True)
    (log_parent / attempt.attempt_id).mkdir()
    with pytest.raises(FileExistsError):
        campaign._reserve_attempt_roots((attempt,))
    assert not campaign._attempt_root(attempt).exists()


def test_decision_publication_is_completion_last_no_clobber_and_replay_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launch_benchmark, "REPOSITORY_ROOT", tmp_path)
    attempts = campaign.derive_attempts(registry, "D", {})
    envelope = {
        "schema_version": 2,
        "artifact_kind": "pilot_evaluation",
        "status": "completed",
        "attempt_id": attempts[0].attempt_id,
        "candidate_id": attempts[0].candidate_id,
        "pilot_seed": 1100,
        "final_seed_results_included": False,
    }
    evidence_path = tmp_path / "output/evidence/seed_1100.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_payload = campaign._json_bytes(envelope)
    evidence_path.write_bytes(evidence_payload)
    reference = _child_ref(1100)
    reference["sha256"] = hashlib.sha256(evidence_payload).hexdigest()
    outcome = campaign.AttemptOutcome(
        spec=attempts[0],
        status="completed",
        child_outcomes=(reference,),
        released_quality=None,
        released_diversity=None,
        diagnostic_structural_passed=True,
    )
    decision = campaign.derive_decision(
        registry=registry,
        stage_id="D",
        attempts=attempts,
        outcomes=(outcome,),
        prior_decisions={},
        predecessor=None,
        source_revision="9" * 40,
        decided_at_utc="2026-09-07T00:00:00+00:00",
    )
    completion = campaign.publish_decision(
        registry=registry, decision=decision, predecessor_completion=None
    )
    entries_path, decision_path = campaign._decision_paths("D")
    assert (tmp_path / entries_path).is_file()
    original = (tmp_path / decision_path).read_bytes()
    assert completion["sha256"] == hashlib.sha256(original).hexdigest()
    replayed, references = campaign.replay_decisions(registry)
    assert replayed["D"] == decision
    assert references["D"] == completion
    with pytest.raises(FileExistsError):
        campaign.publish_decision(
            registry=registry, decision=decision, predecessor_completion=None
        )
    assert (tmp_path / decision_path).read_bytes() == original


def test_live_child_envelope_replacement_invalidates_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launch_benchmark, "REPOSITORY_ROOT", tmp_path)
    attempts = campaign.derive_attempts(registry, "D", {})
    evidence_path = tmp_path / "output/evidence/seed_1100.json"
    evidence_path.parent.mkdir(parents=True)
    payload = campaign._json_bytes(
        {
            "schema_version": 2,
            "artifact_kind": "pilot_evaluation",
            "pilot_seed": 1100,
        }
    )
    evidence_path.write_bytes(payload)
    reference = _child_ref(1100)
    reference["sha256"] = hashlib.sha256(payload).hexdigest()
    outcome = campaign.AttemptOutcome(
        spec=attempts[0],
        status="completed",
        child_outcomes=(reference,),
        released_quality=None,
        released_diversity=None,
        diagnostic_structural_passed=True,
    )
    decision = campaign.derive_decision(
        registry=registry,
        stage_id="D",
        attempts=attempts,
        outcomes=(outcome,),
        prior_decisions={},
        predecessor=None,
        source_revision="9" * 40,
        decided_at_utc="2026-09-07T00:00:00+00:00",
    )
    campaign.publish_decision(
        registry=registry, decision=decision, predecessor_completion=None
    )
    evidence_path.write_bytes(b"{}\n")
    with pytest.raises(campaign.CampaignValidationError, match="binding"):
        campaign.replay_decisions(registry)


def test_generation_lease_revalidation_and_exact_release_preserve_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch_benchmark, "REPOSITORY_ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/artifact_io.py").write_text("# bound\n", encoding="utf-8")
    (tmp_path / "output").mkdir()
    lease = launch_benchmark._acquire_generation_lease(source_revision="9" * 40)
    launch_benchmark._revalidate_generation_lease(lease)
    lock_path = tmp_path / launch_benchmark.GENERATION_LEASE_RELATIVE_PATH
    original = lock_path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == lease.claim.sha256
    lock_path.unlink()
    lock_path.write_bytes(b"foreign\n")
    with pytest.raises(RuntimeError, match="identity or bytes changed"):
        launch_benchmark._release_generation_lease_exact(lease)
    assert lock_path.read_bytes() == b"foreign\n"


def test_artifact_claim_type_used_by_campaign_is_descriptor_bound() -> None:
    fields = set(artifact_io.FileClaim.__dataclass_fields__)
    assert {"relative_path", "device", "inode", "sha256"}.issubset(fields)
    assert stat.S_ISREG(stat.S_IFREG | 0o644)
