from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.udlm import report_exploration as report


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def screen(tmp_path: Path):
    protocol_path = tmp_path / "protocol.json"
    output = tmp_path / "output/engineering"
    entry = {
        "attempt_id": "v5-e-t050",
        "candidate_id": "e-1000u",
        "arm_id": "E",
        "config_id": "e_t050",
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    protocol = {
        "study_id": "engineering",
        "entries": [entry],
        "seeds": [1200, 1201],
        "num_samples": 2,
    }
    _write(protocol_path, protocol)

    def complete(seed: int) -> Path:
        directory = output / entry["attempt_id"] / f"seed_{seed}"
        summary = {
            "seed": seed,
            "num_samples": 2,
            "checkpoint": {"sha256": "a" * 64},
            "config": {"sha256": "b" * 64, "sampling": {"softmax_temp": 0.5}},
            "run": {
                "final_protocol_eligible": False,
                "generation_protocol": {"nfe": 128},
            },
            "git": {"commit": "c" * 40},
            "environment": {},
            "runtime_seconds": {"generation": 1.5},
            "metrics": {"producer_only_field": "must never be used as metric truth"},
        }
        _write(directory / "summary.json", summary)
        (directory / "raw_samples.csv").write_text(
            "sample_index,raw_model_text\n0,CC\n1,CO\n"
        )
        return directory

    def rescore(summary_path: Path, *, root: Path) -> dict:
        assert root == tmp_path
        summary = json.loads(summary_path.read_text())
        quality = 0.5 if summary["seed"] == 1200 else 1.0
        metrics = {
            branch: {
                "validity": 1.0,
                "uniqueness": 1.0,
                "quality": quality,
                "diversity": 0.8,
            }
            for branch in report.BRANCHES
        }
        return {
            "status": "exact_match",
            "seed": summary["seed"],
            "metrics": metrics,
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "raw_samples_sha256": hashlib.sha256(
                summary_path.with_name("raw_samples.csv").read_bytes()
            ).hexdigest(),
        }

    def terminal(return_code: int = 0) -> None:
        artifacts = {
            path.relative_to(tmp_path)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in output.glob(f"{entry['attempt_id']}/seed_*/*")
            if path.is_file()
        }
        _write(
            output / "controller_receipts" / f"{entry['attempt_id']}.json",
            {
                "identity": {
                    "source": {"head": "c" * 40, "upstream": "c" * 40},
                    "protocol_sha256": hashlib.sha256(
                        protocol_path.read_bytes()
                    ).hexdigest(),
                    "entry": entry,
                    "seeds": [1200, 1201],
                    "num_samples": 2,
                },
                "artifacts": artifacts,
                "return_code": return_code,
            },
        )

    return tmp_path, protocol_path, output, complete, rescore, terminal


def test_partial_then_complete_uses_independent_metrics_and_seed_mean(screen):
    root, protocol, output, complete, rescore, terminal = screen
    complete(1200)
    partial = report.build_report(protocol, output, root=root, rescore=rescore)
    assert partial["status"] == "incomplete"
    assert partial["accounting"]["status_counts"] == {"pending": 2}
    assert partial["aggregates"][0]["complete"] is False
    complete(1201)
    terminal()
    result = report.build_report(protocol, output, root=root, rescore=rescore)
    assert result["status"] == "complete"
    assert result["superiority_established"] is False
    assert result["aggregates"][0]["metrics"]["released_comparable"]["quality"] == {
        "mean": 0.75,
        "sample_sd": pytest.approx(0.3535533905932738),
        "defined_seed_count": 2,
    }
    assert result["accounting"]["independently_rescored_requests"] == 4


def test_failure_receipt_wins_over_summary_and_is_never_rescored(screen):
    root, protocol, output, complete, _rescore, terminal = screen
    directory = complete(1200)
    _write(
        directory / "failure_receipt.json",
        {"reason": "completion validation rejected", "stage": "completion_validation"},
    )
    terminal(2)

    def forbidden(*args, **kwargs):
        pytest.fail("failed run was promoted to rescore")

    result = report.build_report(protocol, output, root=root, rescore=forbidden)
    assert result["runs"][0]["status"] == "failed"
    assert result["runs"][0]["metrics"] is None
    assert "raw_samples.csv" in result["runs"][0]["artifacts"]
    assert result["aggregates"][0]["metrics"]["strict"]["quality"]["mean"] is None


def test_protocol_checkpoint_binding_rejects_wrong_model_before_rescore(screen):
    root, protocol, output, complete, _rescore, terminal = screen
    directory = complete(1200)
    document = json.loads((directory / "summary.json").read_text())
    document["checkpoint"]["sha256"] = "d" * 64
    _write(directory / "summary.json", document)
    terminal()

    def forbidden(*args, **kwargs):
        pytest.fail("wrong checkpoint should be rejected before worker")

    result = report.build_report(protocol, output, root=root, rescore=forbidden)
    assert result["runs"][0]["status"] == "invalid"
    assert "checkpoint digest differs" in result["runs"][0]["failure_reason"]


def test_worker_validation_failure_disclosed_with_no_aggregate(screen):
    root, protocol, output, complete, _rescore, terminal = screen
    complete(1200)
    terminal()

    def failed(*args, **kwargs):
        raise ValueError("raw QED values differ from independently recomputed values")

    result = report.build_report(protocol, output, root=root, rescore=failed)
    assert result["runs"][0]["status"] == "invalid"
    assert result["runs"][0]["metrics"] is None
    assert result["accounting"]["independently_rescored_requests"] == 0


def test_extra_runs_make_complete_screen_incomplete(screen):
    root, protocol, output, complete, rescore, terminal = screen
    complete(1200)
    complete(1201)
    terminal()
    _write(output / "undeclared" / "seed_1200" / "summary.json", {})
    result = report.build_report(protocol, output, root=root, rescore=rescore)
    assert result["status"] == "incomplete"
    assert result["unexpected_run_directories"] == [
        "output/engineering/undeclared/seed_1200"
    ]


def test_report_bundle_contains_readable_pdf_csv_and_preserves_previous_snapshot(
    screen,
):
    root, protocol, output, complete, rescore, terminal = screen
    complete(1200)
    complete(1201)
    terminal()
    result = report.build_report(protocol, output, root=root, rescore=rescore)
    paths = report.write_report(result, root / "reports/partial", root=root)
    assert Path(paths["pdf"]).read_bytes().startswith(b"%PDF-")
    with Path(paths["csv"]).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["released_comparable_quality"] == "0.5"
    assert rows[1]["status"] == "completed"
    before = {key: Path(path).read_bytes() for key, path in paths.items()}
    with pytest.raises((FileExistsError, ValueError)):
        report.write_report(result, root / "reports/partial", root=root)
    assert before == {key: Path(path).read_bytes() for key, path in paths.items()}


def test_report_paths_and_reserved_seeds_are_rejected(screen):
    root, protocol, output, _complete, rescore, _terminal = screen
    with pytest.raises(ValueError, match="outside repository"):
        report.build_report(
            protocol, root.parent / "outside", root=root, rescore=rescore
        )
    document = json.loads(protocol.read_text())
    document["seeds"] = [0]
    _write(protocol, document)
    with pytest.raises(ValueError, match="seeds >=1000"):
        report.build_report(protocol, output, root=root, rescore=rescore)


def test_nonzero_controller_rejects_even_successful_child_summaries(screen):
    root, protocol, output, complete, _rescore, terminal = screen
    complete(1200)
    complete(1201)
    terminal(1)

    def forbidden(*args, **kwargs):
        pytest.fail("nonzero attempt must not be promoted")

    result = report.build_report(protocol, output, root=root, rescore=forbidden)
    assert result["accounting"]["status_counts"] == {"failed": 2}
    assert result["accounting"]["independently_rescored_requests"] == 0


def test_output_changed_after_controller_receipt_is_invalid(screen):
    root, protocol, output, complete, rescore, terminal = screen
    directory = complete(1200)
    complete(1201)
    terminal()
    with (directory / "raw_samples.csv").open("a") as handle:
        handle.write("2,CCC\n")
    result = report.build_report(protocol, output, root=root, rescore=rescore)
    assert result["accounting"]["status_counts"] == {"invalid": 2}
    assert "artifact hashes differ" in result["runs"][0]["failure_reason"]


def test_wrong_controller_protocol_digest_is_invalid(screen):
    root, protocol, output, complete, rescore, terminal = screen
    complete(1200)
    complete(1201)
    terminal()
    path = next((output / "controller_receipts").glob("*.json"))
    receipt = json.loads(path.read_text())
    receipt["identity"]["protocol_sha256"] = "e" * 64
    _write(path, receipt)
    result = report.build_report(protocol, output, root=root, rescore=rescore)
    assert result["accounting"]["status_counts"] == {"invalid": 2}
