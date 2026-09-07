"""CPU-only capacity races, bounded waiting, and single-child controller evidence."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from scripts.udlm import launch_engineering_training as engine
from scripts.udlm.launch_train_pilot import GPUState


SOURCE = {"head": "a" * 40, "upstream": "a" * 40}
POLICY = {
    "gpu_availability_wait_seconds": 21600,
    "gpu_availability_poll_seconds": 30,
}


def gpu(index=2, utilization=9, free=30000):
    return GPUState(
        physical_index=index,
        uuid=f"GPU-fixture-{index}",
        name="CPU fixture",
        memory_used_mib=49140 - free,
        memory_total_mib=49140,
        utilization_percent=utilization,
        compute_mode="Default",
        compute_processes=({"pid": 123, "used_memory_mib": 19140},),
    )


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 30
        self.sleeps.append(seconds)
        self.now += seconds


def plan():
    return {"gpu_count": 2, "protocol": dict(POLICY), **POLICY}


@pytest.fixture
def harness(monkeypatch):
    clock, events, record = Clock(), [], {}
    monkeypatch.setattr(engine, "time", clock)
    monkeypatch.setattr(
        engine.benchmark, "_require_clean_pushed_source", lambda: SOURCE
    )
    monkeypatch.setattr(engine, "check_leases", lambda root, leases: None)
    monkeypatch.setattr(engine.audited, "probe_all_gpus", lambda: [gpu(2), gpu(6)])
    monkeypatch.setattr(
        engine.audited, "probe_gpu_uuid", lambda uuid: gpu(int(uuid.rsplit("-", 1)[1]))
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("capacity helper must never launch or signal a process")

    monkeypatch.setattr(engine.subprocess, "Popen", forbidden)
    monkeypatch.setattr(engine.os, "killpg", forbidden)
    return clock, events, record


def test_absent_policy_preserves_historical_immediate_launch():
    assert engine.gpu_availability_wait_policy({"protocol": {}}) is None
    assert engine.gpu_availability_wait_policy(plan()) == POLICY


@pytest.mark.parametrize("scope", ["plan", "protocol"])
@pytest.mark.parametrize("key", list(POLICY))
@pytest.mark.parametrize("value", [None, True, 0, 21600.0, 60, 21601])
def test_wait_requires_exact_complete_prospective_policy(scope, key, value):
    candidate = plan()
    target = candidate if scope == "plan" else candidate["protocol"]
    if value is None:
        del target[key]
    else:
        target[key] = value
    with pytest.raises(ValueError, match="matching 21600-second/30-second"):
        engine.gpu_availability_wait_policy(candidate)


def test_ready_capacity_records_full_and_final_probes_without_sleep(harness):
    clock, events, record = harness
    selected = engine.wait_for_gpu_capacity(plan(), SOURCE, [], events.append, record)
    assert {state.physical_index for state in selected} == {2, 6}
    assert all(state.compute_processes for state in selected)
    assert clock.sleeps == []
    assert [event["phase"] for event in events] == [
        "selection",
        "immediately_before_launch",
        "immediately_before_launch",
    ]
    assert events[0]["rejection_reasons"] == [
        {"uuid": gpu(index).uuid, "reasons": []} for index in (2, 6)
    ]
    assert record["rounds"] == 1 and record["outcome"] == "ready"
    assert record["elapsed_seconds"] == 0


def test_shortage_and_final_probe_race_both_wait_then_reselect(harness, monkeypatch):
    clock, events, record = harness
    inventories = iter(
        [
            [gpu(2), gpu(6, utilization=10)],
            [gpu(2), gpu(6)],
            [gpu(3), gpu(7)],
        ]
    )
    monkeypatch.setattr(engine.audited, "probe_all_gpus", lambda: next(inventories))
    probes = []

    def final(uuid):
        probes.append(uuid)
        index = int(uuid.rsplit("-", 1)[1])
        return gpu(index, free=29999 if index == 2 else 30000)

    monkeypatch.setattr(engine.audited, "probe_gpu_uuid", final)
    selected = engine.wait_for_gpu_capacity(plan(), SOURCE, [], events.append, record)
    assert {state.physical_index for state in selected} == {3, 7}
    assert clock.sleeps == [30, 30]
    assert len(probes) == 4
    waits = [event for event in events if event["phase"] == "availability_wait"]
    assert [event["availability_round"] for event in waits] == [1, 2]
    assert "only 1 eligible" in waits[0]["error"]
    assert "final GPU probe rejected" in waits[1]["error"]
    assert record["rounds"] == 3 and record["elapsed_seconds"] == 60


def test_never_ready_stops_at_six_hours_with_positive_bounded_sleeps(
    harness, monkeypatch
):
    clock, events, record = harness
    monkeypatch.setattr(engine.audited, "probe_all_gpus", lambda: [gpu(2)])
    with pytest.raises(TimeoutError, match="21600 seconds"):
        engine.wait_for_gpu_capacity(plan(), SOURCE, [], events.append, record)
    assert clock.now == sum(clock.sleeps) == 21600
    assert len(clock.sleeps) == record["rounds"] == 720
    assert record["outcome"] == "timed_out"
    assert not any(event["phase"] == "immediately_before_launch" for event in events)


def test_slow_probe_uses_remaining_sleep_and_cannot_launch_at_deadline(
    harness, monkeypatch
):
    clock, events, record = harness

    def slow_inventory():
        clock.now += 21595
        return [gpu(2)]

    monkeypatch.setattr(engine.audited, "probe_all_gpus", slow_inventory)
    with pytest.raises(TimeoutError):
        engine.wait_for_gpu_capacity(plan(), SOURCE, [], events.append, record)
    assert clock.sleeps == [5]
    assert record["rounds"] == 1 and clock.now == 21600


def test_successful_final_probe_after_deadline_still_cannot_launch(
    harness, monkeypatch
):
    clock, events, record = harness

    def late_final(uuid):
        clock.now += 10800
        return gpu(int(uuid.rsplit("-", 1)[1]))

    monkeypatch.setattr(engine.audited, "probe_gpu_uuid", late_final)
    with pytest.raises(TimeoutError):
        engine.wait_for_gpu_capacity(plan(), SOURCE, [], events.append, record)
    assert clock.sleeps == [] and record["outcome"] == "timed_out"


@pytest.mark.parametrize("defect", ["duplicate", "identity", "query", "policy"])
def test_invalid_gpu_evidence_is_a_defect_never_a_capacity_retry(
    harness, monkeypatch, defect
):
    clock, events, record = harness
    if defect == "duplicate":
        monkeypatch.setattr(engine.audited, "probe_all_gpus", lambda: [gpu(2), gpu(2)])
    elif defect == "identity":
        monkeypatch.setattr(
            engine.audited, "probe_gpu_uuid", lambda uuid: replace(gpu(2), uuid="wrong")
        )
    elif defect == "query":

        def broken(uuid):
            raise OSError("query failed")

        monkeypatch.setattr(engine.audited, "probe_gpu_uuid", broken)
    else:
        monkeypatch.setattr(engine.audited, "ACTIVE_COMPUTE_PROCESSES_ALLOWED", False)
    with pytest.raises((ValueError, RuntimeError)):
        engine.wait_for_gpu_capacity(plan(), SOURCE, [], events.append, record)
    assert clock.sleeps == [] and record["outcome"] == "failed"
    assert record["rounds"] == 1


@pytest.mark.parametrize("defect", ["source", "lease"])
def test_changed_source_or_lease_aborts_after_wait_before_next_probe(
    harness, monkeypatch, defect
):
    clock, events, record = harness
    inventories = []

    def inventory():
        inventories.append(True)
        return [gpu(2)]

    monkeypatch.setattr(engine.audited, "probe_all_gpus", inventory)
    if defect == "source":
        monkeypatch.setattr(
            engine.benchmark,
            "_require_clean_pushed_source",
            lambda: SOURCE if clock.now == 0 else {},
        )
    else:

        def leases(root, claims):
            if clock.now:
                raise RuntimeError("lease changed")

        monkeypatch.setattr(engine, "check_leases", leases)
    with pytest.raises(RuntimeError, match=f"{defect} changed"):
        engine.wait_for_gpu_capacity(plan(), SOURCE, [], events.append, record)
    assert len(inventories) == 1 and clock.sleeps == [30]
    assert record["outcome"] == "failed"


@pytest.mark.parametrize(
    "outcome", ["completed", "timeout", "child_failed", "lease_replaced"]
)
def test_execute_wait_is_prelaunch_only_and_preserves_terminal_evidence(
    tmp_path, monkeypatch, outcome
):
    candidate = deepcopy(engine.build_plan(2))
    candidate.update(POLICY)
    candidate["protocol"].update(POLICY)
    source, clock, calls = dict(SOURCE), Clock(), []
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "time", clock)
    monkeypatch.setattr(
        engine.benchmark, "_require_clean_pushed_source", lambda: source
    )

    def input_claim(value):
        calls.append("checkpoint_hash")
        return {"sha256": "b" * 64}

    def builder(count):
        calls.append("plan_validation")
        return candidate

    monkeypatch.setattr(engine, "verify_checkpoint_input", input_claim)
    directory = tmp_path / candidate["output_relative"]
    request_bytes, lease_bytes = [], []

    def inventory():
        calls.append("inventory")
        request_bytes.append((directory / "request_manifest.json").read_bytes())
        lease_bytes.append(
            [(tmp_path / path).read_bytes() for path in engine.LEASE_PATHS]
        )
        if outcome == "timeout":
            clock.now += 21595
            return [gpu(2)]
        if calls.count("inventory") == 1:
            if outcome == "lease_replaced":
                (tmp_path / engine.LEASE_PATHS[0]).write_bytes(b"foreign owner")
            return [gpu(2)]
        return [gpu(3), gpu(7)]

    def final(uuid):
        calls.append("final_uuid")
        return gpu(int(uuid.rsplit("-", 1)[1]))

    def launch(*args, **kwargs):
        calls.append("Popen")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-fixture-3,GPU-fixture-7"
        clock.now += 1
        code = 7 if outcome == "child_failed" else 0
        return SimpleNamespace(pid=12345, returncode=code, poll=lambda: code)

    monkeypatch.setattr(engine.audited, "probe_all_gpus", inventory)
    monkeypatch.setattr(engine.audited, "probe_gpu_uuid", final)
    monkeypatch.setattr(engine.subprocess, "Popen", launch)
    monkeypatch.setattr(engine, "process_group_exists", lambda pid: False)
    monkeypatch.setattr(
        engine,
        "validate_checkpoint_output",
        lambda *args, **kwargs: {"global_step": 20, "sha256": "c" * 64},
    )
    result = engine.execute(candidate, source, plan_builder=builder)
    terminal = json.loads((directory / "terminal_manifest.json").read_text())
    assert result == (0 if outcome == "completed" else 1)
    assert calls[:3] == ["checkpoint_hash", "plan_validation", "inventory"]
    assert calls.count("checkpoint_hash") == calls.count("plan_validation") == 1
    assert all(value == request_bytes[0] for value in request_bytes)
    assert all(value == lease_bytes[0] for value in lease_bytes)
    assert terminal["status"] == ("completed" if outcome == "completed" else "failed")
    waiting = terminal["gpu_availability_wait"]
    if outcome in {"completed", "child_failed"}:
        assert calls == [
            "checkpoint_hash",
            "plan_validation",
            "inventory",
            "inventory",
            "final_uuid",
            "final_uuid",
            "Popen",
        ]
        launch_record = json.loads((directory / "launch_manifest.json").read_text())
        assert waiting == launch_record["gpu_availability_wait"]
        assert waiting["outcome"] == "ready" and waiting["rounds"] == 2
        assert terminal["training_return_code"] == (
            7 if outcome == "child_failed" else 0
        )
        assert terminal["leases_release_authorized"] is True
        assert not any((tmp_path / path).exists() for path in engine.LEASE_PATHS)
    else:
        assert "Popen" not in calls
        assert not (directory / "launch_manifest.json").exists()
        assert "training_pid" not in terminal
        assert terminal["training_return_code"] is terminal["checkpoint"] is None
        assert waiting["outcome"] == ("timed_out" if outcome == "timeout" else "failed")
        assert terminal["leases_release_authorized"] is (outcome == "timeout")
        if outcome == "lease_replaced":
            assert (tmp_path / engine.LEASE_PATHS[0]).read_bytes() == b"foreign owner"
            assert all((tmp_path / path).exists() for path in engine.LEASE_PATHS)
    assert terminal["completed_example_exposures"] == (
        2560 if outcome == "completed" else None
    )
    if outcome != "completed":
        assert terminal["end_to_end_training_examples_per_second"] is None
    telemetry = [
        json.loads(line)
        for line in (directory / "gpu_telemetry.jsonl").read_text().splitlines()
    ]
    assert any(event["phase"] == "availability_wait" for event in telemetry)
    assert (
        sum(event["phase"] == "selection" for event in telemetry) == waiting["rounds"]
    )
    with pytest.raises(FileExistsError):
        engine.execute(candidate, source, plan_builder=builder)
