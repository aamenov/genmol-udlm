import copy
import json
from pathlib import Path

import pytest

from scripts.udlm import launch_exploration as screen


@pytest.fixture
def protocol():
    return json.loads(
        (screen.ROOT / "experiments/udlm/protocols/engineering_v5.json").read_text()
    )


def test_screen_commands_limit_gpus_and_reserve_final_seeds(protocol):
    assert len(protocol["entries"]) == 12
    assert len({e["attempt_id"] for e in protocol["entries"]}) == 12
    for entry in protocol["entries"]:
        command = screen.command_for(entry, protocol, Path("runs"), Path("logs"))
        assert command[command.index("--gpu-count") + 1] == "2"
        assert command[command.index("--max-utilization-percent") + 1] == "10"
        assert command[command.index("--min-free-memory-mib") + 1] == "30000"
        assert "--pilot" in command
        start, end = command.index("--seeds") + 1, command.index("--gpu-count")
        assert list(map(int, command[start:end])) == [1200, 1201]


@pytest.mark.parametrize(
    "change",
    [
        "third_gpu",
        "ten_percent_allowed",
        "final_seed",
        "duplicate_seed",
        "duplicate_attempt",
        "large_pilot",
        "path_escape",
    ],
)
def test_protocol_rejects_policy_and_identity_violations(protocol, tmp_path, change):
    value = copy.deepcopy(protocol)
    if change == "third_gpu":
        value["gpu_policy"]["max_gpus"] = 3
    elif change == "ten_percent_allowed":
        value["gpu_policy"]["max_utilization_percent"] = 11
    elif change == "final_seed":
        value["seeds"][0] = 0
    elif change == "duplicate_seed":
        value["seeds"][0] = value["seeds"][1]
    elif change == "duplicate_attempt":
        value["entries"].append(value["entries"][0])
    elif change == "large_pilot":
        value["num_samples"] = 1000
    elif change == "path_escape":
        value["entries"][0]["config"] = "../../outside.yaml"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        screen.load_protocol(path)


def test_output_digest_changes_if_raw_evidence_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    output = tmp_path / "output"
    entry = {"attempt_id": "example"}
    folder = output / "example/seed_1200"
    folder.mkdir(parents=True)
    raw = folder / "raw_samples.csv"
    raw.write_text("raw_model_text\nCCO\n")
    before = screen.output_hashes(output, entry, [1200])
    raw.write_text("raw_model_text\nCCC\n")
    assert before != screen.output_hashes(output, entry, [1200])


def test_write_once_preserves_prior_evidence(tmp_path):
    path = tmp_path / "receipt.json"
    screen.write_once(path, {"failed": True})
    with pytest.raises(FileExistsError):
        screen.write_once(path, {"failed": False})
    assert json.loads(path.read_text()) == {"failed": True}


def test_success_cannot_have_missing_or_failed_seed_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    output = tmp_path / "output"
    entry = {"attempt_id": "example"}
    with pytest.raises(RuntimeError, match="lacks"):
        screen.validate_success_artifacts(output, entry, [1200, 1201], {})
    artifacts = {
        f"output/example/seed_{seed}/{name}": "a" * 64
        for seed in (1200, 1201)
        for name in ("summary.json", "raw_samples.csv")
    }
    screen.validate_success_artifacts(output, entry, [1200, 1201], artifacts)
    artifacts["output/example/seed_1201/failure_receipt.json"] = "b" * 64
    with pytest.raises(RuntimeError, match="failure receipt"):
        screen.validate_success_artifacts(output, entry, [1200, 1201], artifacts)
