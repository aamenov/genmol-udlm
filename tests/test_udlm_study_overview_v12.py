"""Synthetic evidence only; never invokes chemistry, model loading or GPU discovery."""

import base64
from collections import Counter
import copy
import csv
import io
import json
import statistics
import struct

import pytest
from pypdf import PdfReader

from scripts.udlm import generate_study_overview_v12 as overview


def audit_fixture(rows=100):
    ids = [1, 4, 5, 2] * rows
    raw = struct.pack(f"<{len(ids)}H", *ids)
    bits = bytes([0b01100110] * (rows // 2))
    counts = {"unk": 0, "bos": 0, "eos": 0, "pad": 0, "mask": rows}
    audit = {
        "rows": rows,
        "columns": 4,
        "control_token_ids": {"unk": 0, "bos": 1, "eos": 2, "pad": 3, "mask": 4},
        "final_sampled_ids": {
            "data_base64": base64.b64encode(raw).decode(),
            "decoded_sha256": overview.old.digest(raw),
            "decoded_byte_count": len(raw),
            "dtype": "uint16",
            "byte_order": "little",
        },
        "editable_mask": {
            "data_base64": base64.b64encode(bits).decode(),
            "decoded_sha256": overview.old.digest(bits),
            "decoded_byte_count": len(bits),
            "bit_order": "msb0",
            "logical_bit_count": len(ids),
        },
        "control_token_counts": {"final_sampled_editable_positions": counts},
    }
    identity = {
        "control_token_counts": audit["control_token_counts"],
        "final_sampled_ids_sha256": audit["final_sampled_ids"]["decoded_sha256"],
        "editable_mask_sha256": audit["editable_mask"]["decoded_sha256"],
        "exact_batch_decode_match": True,
    }
    return audit, identity


def write_json(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = overview.old.json_bytes(value)
    path.write_bytes(payload)
    return {
        "relative_path": relative,
        "sha256": overview.old.digest(payload),
        "size_bytes": len(payload),
    }


def report_fixture(root, *, failed_attempt=False, invalid_seed=False):
    """Create a complete, self-bound tiny synthetic report with real byte hashes."""
    source = {"head": "a" * 40, "upstream": "a" * 40}
    protocol = {
        "study_id": "engineering-v12-ce-mask-prior",
        "seeds": [2000, 2001],
        "num_samples": 100,
        "nfe": 128,
        "output_root": "output/udlm/engineering_v12",
        "entries": [],
        "design": {"prior_comparison": {}},
        "training": {"arms": {}},
    }
    for arm, variant in (
        ("E_CE", "empirical_frequency"),
        ("MASK_CE", "mask_rich_empirical"),
    ):
        protocol["training"]["arms"][arm] = {
            "udlm_prior_metadata": {"variant": variant},
            "udlm_denoiser_metadata": {"parameterization": "x0_denoiser"},
        }
    configs = {}
    for suffix, temperature, label in (
        ("100", 1.0, "primary"),
        ("050", 0.5, "secondary"),
    ):
        protocol["design"]["prior_comparison"][label] = {
            "control_config": f"e_ce_t{suffix}",
            "treatment_config": f"mask_ce_t{suffix}",
            "temperature": temperature,
        }
        for prefix, arm, variant in (
            ("e", "E_CE", "empirical_frequency"),
            ("mask", "MASK_CE", "mask_rich_empirical"),
        ):
            config_id = f"{prefix}_ce_t{suffix}"
            config = {
                "model_path": f"output/{arm}.ckpt",
                "num_samples": 100,
                "diffusion_type": "udlm",
                "parameterization": "x0_denoiser",
                "softmax_temp": temperature,
                "raw_loo_top_p": 1.0,
                "randomness": 0.0,
                "min_add_len": 40,
                "num_steps": 128,
                "inference_eps": 1e-5,
                "exclude_special_tokens": False,
                "prior_variant": variant,
                "prior_metadata_sha256": ("b" if arm == "E_CE" else "c") * 64,
            }
            import yaml

            config_path = root / f"configs/{config_id}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_payload = yaml.safe_dump(config).encode()
            config_path.write_bytes(config_payload)
            config_ref = {
                "relative_path": str(config_path.relative_to(root)),
                "sha256": overview.old.digest(config_payload),
            }
            entry = {
                "config_id": config_id,
                "arm_id": arm,
                "attempt_id": f"v12-{config_id}",
                "candidate_id": f"synthetic-{arm}",
                "checkpoint_sha256": "d" * 64,
                "config": config_ref["relative_path"],
                "config_sha256": config_ref["sha256"],
                "parameterization": "x0_denoiser",
                "prior_variant": variant,
                "prior_metadata_sha256": config["prior_metadata_sha256"],
            }
            protocol["entries"].append(entry)
            configs[config_id] = config
    protocol["prospective_design"] = write_json(
        root, "design.json", {"fixture": "synthetic"}
    )
    protocol_ref = write_json(root, "protocol.json", protocol)
    runs, aggregates = [], []
    for entry in protocol["entries"]:
        pair, artifacts = [], {}
        failed = failed_attempt and entry["config_id"] == "mask_ce_t100"
        for index, seed in enumerate(protocol["seeds"]):
            directory = f"{protocol['output_root']}/{entry['attempt_id']}/seed_{seed}"
            config = configs[entry["config_id"]]
            sampling = {
                key: value
                for key, value in config.items()
                if key not in {"model_path", "num_samples"}
            }
            record_config = {
                "sha256": entry["config_sha256"],
                "source": config,
                "sampling": sampling,
                "effective": config,
            }
            checkpoint = {
                "sha256": entry["checkpoint_sha256"],
                **protocol["training"]["arms"][entry["arm_id"]],
            }
            gp = {
                "nfe": 128,
                "inference_weights": {"source": "ema", "ema_applied": True},
            }
            audit, token_identity = audit_fixture()
            summary = {
                "seed": seed,
                "num_samples": 100,
                "git": {"commit": source["head"]},
                "config": record_config,
                "checkpoint": checkpoint,
                "runtime_seconds": {"generation": 1.0},
                "run": {"generation_protocol": gp},
                "sampled_token_control_audit": audit,
            }
            ref = write_json(root, directory + "/summary.json", summary)
            raw = "raw_model_text\n" + "C\n" * 100
            raw_path = root / directory / "raw_samples.csv"
            raw_path.write_text(raw)
            raw_ref = {
                "relative_path": directory + "/raw_samples.csv",
                "sha256": overview.old.digest(raw.encode()),
            }
            artifacts.update(
                {
                    ref["relative_path"]: ref["sha256"],
                    raw_ref["relative_path"]: raw_ref["sha256"],
                }
            )
            count = 40 + (5 if entry["arm_id"] == "MASK_CE" else 0) + 2 * index
            metrics = {
                branch: {
                    "validity": 0.98,
                    "uniqueness": 1.0,
                    "quality": count / 100,
                    "quality_count": count,
                    "quality_denominator": 100,
                    "diversity": 0.87 + 0.01 * index,
                }
                for branch in overview.old.BRANCHES
            }
            run = {
                key: entry[key]
                for key in ("config_id", "candidate_id", "arm_id", "attempt_id")
            }
            run.update(
                seed=seed,
                requested_samples=100,
                run_directory=directory,
                status="completed",
                raw_row_count=100,
                config=record_config,
                checkpoint=checkpoint,
                source=summary["git"],
                runtime_seconds=summary["runtime_seconds"],
                generation_protocol=gp,
                metrics=metrics,
                artifacts={"summary.json": ref, "raw_samples.csv": raw_ref},
                independent_rescore={
                    "status": "exact_match",
                    "seed": seed,
                    "summary_sha256": ref["sha256"],
                    "raw_samples_sha256": raw_ref["sha256"],
                    "metrics": metrics,
                    "identity": {"sampled_token_control_audit": token_identity},
                },
            )
            if failed or (
                invalid_seed and entry["config_id"] == "mask_ce_t100" and seed == 2001
            ):
                run.update(
                    status="failed" if failed else "invalid",
                    metrics=None,
                    failure_reason="synthetic terminal outcome",
                )
                del run["raw_row_count"]
                del run["independent_rescore"]
            pair.append(run)
        receipt = {
            "return_code": 7 if failed else 0,
            "identity": {
                "entry": entry,
                "seeds": protocol["seeds"],
                "num_samples": 100,
                "protocol_sha256": protocol_ref["sha256"],
                "source": source,
            },
            "artifacts": artifacts,
        }
        ref = write_json(
            root,
            f"{protocol['output_root']}/controller_receipts/{entry['attempt_id']}.json",
            receipt,
        )
        controller = {
            "status": "failed" if failed else "completed",
            "receipt": receipt,
            "artifact": ref,
        }
        for run in pair:
            run["controller"] = controller
        successful = [run for run in pair if run["status"] == "completed"]
        aggregate = {
            key: entry[key]
            for key in ("config_id", "candidate_id", "arm_id", "attempt_id")
        }
        aggregate.update(
            complete=len(successful) == 2,
            expected_seeds=protocol["seeds"],
            completed_seeds=[run["seed"] for run in successful],
            metrics={},
        )
        for branch in overview.old.BRANCHES:
            aggregate["metrics"][branch] = {}
            for metric in overview.old.METRICS:
                values = [run["metrics"][branch][metric] for run in successful]
                aggregate["metrics"][branch][metric] = {
                    "mean": statistics.mean(values) if values else None,
                    "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                    "defined_seed_count": len(values),
                }
        aggregates.append(aggregate)
        runs += pair
    directory = root / "report"
    directory.mkdir()
    hashes = {}
    for name in ("report.pdf", "report.csv", "paired_contrasts.csv"):
        payload = b"SYNTHETIC TEST ARTIFACT\n"
        (directory / name).write_bytes(payload)
        hashes[
            {
                "report.pdf": "pdf_sha256",
                "report.csv": "csv_sha256",
                "paired_contrasts.csv": "paired_contrasts_csv_sha256",
            }[name]
        ] = overview.old.digest(payload)
    statuses = dict(Counter(run["status"] for run in runs))
    report = {
        "status": "complete" if statuses.get("completed") == 8 else "incomplete",
        "superiority_established": False,
        "unexpected_run_directories": [],
        "protocol": dict(protocol_ref, configuration=protocol),
        "baselines": overview.reporter.BASELINES,
        "report_artifacts": hashes,
        "runs": runs,
        "aggregates": aggregates,
        "paired_contrasts": overview.reporter._paired_contrasts(protocol, runs),
        "accounting": {
            "scheduled_runs": 8,
            "scheduled_requests": 800,
            "status_counts": statuses,
            "independently_rescored_requests": 100 * statuses.get("completed", 0),
        },
    }
    ref = write_json(root, "report/report.json", report)
    return report, directory, ref["sha256"]


@pytest.mark.parametrize(
    "failed,invalid,accepted",
    [(False, False, 800), (True, False, 600), (False, True, 700)],
)
def test_terminal_reports_derive_counts_and_withhold_incomplete_means(
    tmp_path, failed, invalid, accepted
):
    report, directory, digest = report_fixture(
        tmp_path, failed_attempt=failed, invalid_seed=invalid
    )
    before = copy.deepcopy(report)
    loaded, protocol, records, masks = overview.load_v12(
        overview.old.Inputs(tmp_path), tmp_path, directory, digest
    )
    assert report == before
    assert loaded["accounting"]["independently_rescored_requests"] == accepted
    assert sum(record["accepted_samples"] for record in records) == accepted
    assert sum(row["rows"] for row in masks) == accepted
    assert len(records) == 4
    selected = next(row for row in records if row["config_id"] == "mask_ce_t100")
    contrast = next(
        row for row in loaded["paired_contrasts"] if row["contrast_id"] == "primary"
    )
    assert (selected["metrics"]["strict"]["quality"]["mean"] is None) == (
        failed or invalid
    )
    assert (contrast["metrics"]["strict"]["quality"]["mean_difference"] is None) == (
        failed or invalid
    )
    assert protocol["seeds"] == [2000, 2001]


@pytest.mark.parametrize(
    "mutation",
    [
        "pending",
        "raw_bytes",
        "seed",
        "accepted_count",
        "contrast_sign",
        "raw_count",
        "prior",
        "namespace",
    ],
)
def test_rejects_unready_or_misbound_v12_evidence(tmp_path, mutation):
    report, directory, digest = report_fixture(tmp_path)
    first = report["runs"][0]
    if mutation == "pending":
        first["controller"]["status"] = "pending"
    elif mutation == "raw_bytes":
        (tmp_path / first["artifacts"]["raw_samples.csv"]["relative_path"]).write_bytes(
            b"changed"
        )
    elif mutation == "seed":
        first["seed"] = 0
    elif mutation == "accepted_count":
        report["accounting"]["independently_rescored_requests"] += 32
    elif mutation == "contrast_sign":
        report["paired_contrasts"][0]["metrics"]["strict"]["quality"][
            "mean_difference"
        ] *= -1
    elif mutation == "raw_count":
        first["raw_row_count"] = 99
    elif mutation == "prior":
        first["config"]["sampling"]["prior_variant"] = "other"
    else:
        first["run_directory"] = "another/study/seed_2000"
    digest = write_json(tmp_path, "report/report.json", report)["sha256"]
    with pytest.raises(ValueError):
        overview.load_v12(overview.old.Inputs(tmp_path), tmp_path, directory, digest)


def test_final_mask_count_uses_editable_bitmask_not_all_positions():
    audit, identity = audit_fixture(rows=2)
    value = overview.token_counts({"sampled_token_control_audit": audit}, identity)
    assert value == {
        "editable_positions": 4,
        "final_control_counts": {"unk": 0, "bos": 0, "eos": 0, "pad": 0, "mask": 2},
        "rows": 2,
        "rows_with_mask": 2,
    }
    audit["final_sampled_ids"]["decoded_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash/length"):
        overview.token_counts({"sampled_token_control_audit": audit}, identity)


def synthetic_cover_summary():
    records = []
    for study, count, per_seed in (
        ("V5", 12, 64),
        ("V6", 6, 64),
        ("V9", 4, 100),
        ("V10", 4, 100),
        ("V12", 4, 100),
    ):
        for index in range(count):
            records.append(
                {
                    "study": study,
                    "config_id": f"SYNTHETIC_{study}_{index:02d}",
                    "arm_id": "CE",
                    "complete": True,
                    "samples": 2 * per_seed,
                    "accepted_samples": 2 * per_seed,
                    "seeds": [2000, 2001],
                    "temperature": 1.0,
                    "kernel": "128P",
                    "metrics": {
                        branch: {
                            metric: {
                                "mean": 0.35 + index * 0.02,
                                "sample_sd": 0.02,
                                "defined_seed_count": 2,
                            }
                            for metric in overview.old.METRICS
                        }
                        for branch in overview.old.BRANCHES
                    },
                }
            )
    totals = overview.accounting(records)
    return {
        "synthetic_test_only": True,
        "status": "SYNTHETIC_TEST_DATA_NOT_MOLECULAR_RESULTS",
        "accounting": totals,
        "configurations": records,
        "training_rows": [
            [
                "SYNTHETIC training fixture",
                "failed",
                "unavailable",
                "Original failure retained; test data only",
            ]
        ],
        "v12_prior_contrasts": [],
        "v12_runs": [],
        "v12_token_counts": [],
    }


def test_combined_accounting_covers_all_thirty_configs_without_baseline_or_failed_v4():
    summary = synthetic_cover_summary()
    assert summary["accounting"]["total"] == {
        "configurations": 30,
        "scheduled_runs": 60,
        "scheduled_requests": 4704,
        "accepted_requests": 4704,
        "status_counts": {"completed": 60},
    }
    summary["configurations"][-1].update(accepted_samples=0, statuses={"failed": 2})
    total = overview.accounting(summary["configurations"])["total"]
    assert total["accepted_requests"] == 4504 and total["scheduled_requests"] == 4704
    assert total["status_counts"] == {"completed": 58, "failed": 2}


def test_synthetic_cover_contains_every_setting_and_sd_caveat():
    summary = synthetic_cover_summary()
    image = overview.comparison_figure(summary["configurations"], synthetic=True)
    pdf = overview.make_cover(summary, image)
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    for record in summary["configurations"]:
        assert record["config_id"] in text
    assert "not a confidence interval" in text
    assert "SYNTHETIC_TEST_DATA_NOT_MOLECULAR_RESULTS" in text
    assert "4704" in text


def test_missing_future_report_publishes_nothing(tmp_path):
    output = tmp_path / "fresh"
    assert (
        overview.main(
            [
                "--input-root",
                str(tmp_path),
                "--v12-report-directory",
                str(tmp_path / "missing"),
                "--v12-report-sha256",
                "0" * 64,
                "--output-directory",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_preserved_bundle_hash_closure_and_no_overwrite(tmp_path):
    payload = b"unchanged evidence"
    upstream = tmp_path / "upstream.txt"
    upstream.write_bytes(payload)
    directory = tmp_path / "old"
    directory.mkdir()
    stream = io.BytesIO()
    writer = overview.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(stream)
    pdf = stream.getvalue()
    (directory / "report.pdf").write_bytes(pdf)
    manifest = {
        "outputs": {
            "report.pdf": {"sha256": overview.old.digest(pdf), "size_bytes": len(pdf)}
        },
        "inputs": [
            {
                "workspace_relative_path": "upstream.txt",
                "sha256": overview.old.digest(payload),
                "size_bytes": len(payload),
            }
        ],
    }
    ref = write_json(tmp_path, "old/input_hash_manifest.json", manifest)
    inputs = overview.old.Inputs(tmp_path)
    assert (
        overview.load_preserved_bundle(
            inputs, directory, ref["sha256"], "report.pdf", 1
        )
        == pdf
    )
    upstream.write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="Input changed"):
        inputs.manifest()
    with pytest.raises(ValueError, match="fresh"):
        overview.old.publish_bundle(
            directory, {"report.pdf": b"replacement"}, workspace=tmp_path
        )
    assert (directory / "report.pdf").read_bytes() == pdf


def test_configuration_csv_retains_failed_status_and_planned_vs_accepted():
    summary = synthetic_cover_summary()
    final = summary["configurations"][-1]
    final.update(
        complete=False, accepted_samples=0, statuses={"failed": 2}, completed_seeds=[]
    )
    for metrics in final["metrics"].values():
        for value in metrics.values():
            value.update(mean=None, sample_sd=None)
    rows = list(
        csv.DictReader(
            io.StringIO(overview.configuration_csv(summary["configurations"]).decode())
        )
    )
    assert len(rows) == 60
    assert len({(row["study"], row["config_id"]) for row in rows}) == 30
    for row in rows[-2:]:
        assert row["scheduled_samples"] == "200" and row["accepted_samples"] == "0"
        assert row["quality_mean"] == "" and row["quality_sample_sd"] == ""
        assert json.loads(row["status_counts"]) == {"failed": 2}
        assert json.loads(row["completed_seeds"]) == []
