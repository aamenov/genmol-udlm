from __future__ import annotations

import csv
import copy
import hashlib
import io
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


def _gibbs_export_fixture(screen):
    root, protocol, output, complete, rescore, terminal = screen
    complete(1200)
    complete(1201)
    terminal()
    result = report.build_report(protocol, output, root=root, rescore=rescore)
    specification = result["protocol"]["configuration"]
    specification["design"] = {
        "temperature": 0.5,
        "predictor_control": {"predictor_transitions": 128, "corrector_updates": 0},
        "gibbs_treatment": {"predictor_transitions": 64, "corrector_updates": 64},
        "claim_boundary": "Learned own-token dependence removes exact stationarity.",
    }
    specification["limitations"] = [
        "Temperature 0.5 conditionals are approximate; no molecular win is established."
    ]
    for run, corrected in zip(result["runs"], (False, True)):
        run["config_id"] = "e_t050_gibbs" if corrected else "e_t050_predictor"
        run["generation_protocol"] = {
            "diffusion_type": "udlm",
            "nfe": 128,
            "num_steps": 128,
            "nfe_definition": (
                "one fresh backbone evaluation per predictor or corrector"
                if corrected
                else "one full backbone forward evaluation per reverse step"
            ),
        }
        if corrected:
            run["generation_protocol"].update(
                gibbs_corrector=True,
                predictor_transitions_per_molecule=64,
                corrector_steps_per_molecule=64,
            )
            run["independent_rescore"]["identity"] = {
                "source": {"corrector_source_sha256": "9" * 64}
            }
    return result


def test_gibbs_csv_exports_certified_budget_provenance_and_design_caveats(screen):
    result = _gibbs_export_fixture(screen)
    rows = list(csv.DictReader(io.StringIO(report._csv_bytes(result).decode())))
    control, corrected = rows
    assert control["gibbs_corrector"] == "False"
    assert corrected["gibbs_corrector"] == "True"
    assert control["nfe"] == corrected["nfe"] == "128"
    assert control["predictor_transitions_per_molecule"] == "128"
    assert control["corrector_steps_per_molecule"] == "0"
    assert corrected["predictor_transitions_per_molecule"] == "64"
    assert corrected["corrector_steps_per_molecule"] == "64"
    assert control["corrector_source_sha256"] == ""
    assert corrected["corrector_source_sha256"] == "9" * 64
    for row, run in zip(rows, result["runs"]):
        assert json.loads(row["generation_protocol"]) == run["generation_protocol"]
        specification = result["protocol"]["configuration"]
        assert json.loads(row["protocol_design"]) == specification["design"]
        assert json.loads(row["protocol_limitations"]) == specification["limitations"]


def test_gibbs_pdf_discloses_allocation_approximation_and_source_hash(screen):
    from pypdf import PdfReader

    result = _gibbs_export_fixture(screen)
    before = copy.deepcopy(result)
    pdf = report._pdf_bytes(result)
    text = " ".join(
        " ".join(page.extract_text().split())
        for page in PdfReader(io.BytesIO(pdf)).pages
    )
    assert "Prospective sampling design" in text
    assert "Sampling evaluation budget" in text
    assert "e_t050_predictor 1200 128 128 0" in text
    assert "e_t050_gibbs 1201 128 64 64" in text
    assert "Learned own-token dependence removes exact stationarity." in text
    assert "Temperature 0.5 conditionals are approximate" in text
    assert "generation_protocol" in text
    assert "corrector_steps_per_molecule" in text
    assert "corrector_source_sha256" in text
    assert "9" * 64 in text.replace(" ", "")
    assert result == before


def test_gibbs_pending_rows_retain_design_without_inventing_observed_budget(screen):
    result = _gibbs_export_fixture(screen)
    for run in result["runs"]:
        run["status"] = "pending"
        del run["generation_protocol"]
        del run["independent_rescore"]
    rows = list(csv.DictReader(io.StringIO(report._csv_bytes(result).decode())))
    for row in rows:
        assert row["nfe"] == row["gibbs_corrector"] == ""
        assert row["predictor_transitions_per_molecule"] == ""
        assert row["corrector_steps_per_molecule"] == ""
        assert row["corrector_source_sha256"] == ""
        assert json.loads(row["generation_protocol"]) == {}
        assert "gibbs_treatment" in json.loads(row["protocol_design"])


def test_historical_temperature_csv_preserves_columns_without_gibbs_fields(screen):
    root, protocol, output, complete, rescore, terminal = screen
    complete(1200)
    complete(1201)
    terminal()
    result = report.build_report(protocol, output, root=root, rescore=rescore)
    rows = list(csv.DictReader(io.StringIO(report._csv_bytes(result).decode())))
    assert "nfe" not in rows[0]
    assert "gibbs_corrector" not in rows[0]
    assert "generation_protocol" not in rows[0]
    assert "protocol_design" not in rows[0]
    assert report._has_gibbs_design(result) is False
