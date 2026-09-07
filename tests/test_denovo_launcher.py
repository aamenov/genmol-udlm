from __future__ import annotations

import csv
import json
import stat
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from scripts.exps.denovo import launch_benchmark as launcher


def _gpu(
    *,
    index: int = 2,
    uuid: str = "GPU-test-uuid",
    memory_used_mib: int = 20,
    memory_total_mib: int = 49_140,
    utilization_percent: int = 1,
    compute_mode: str = "Default",
    processes: tuple[dict[str, object], ...] = (),
) -> launcher.GPUState:
    return launcher.GPUState(
        index=index,
        uuid=uuid,
        name="NVIDIA RTX A6000",
        memory_used_mib=memory_used_mib,
        memory_total_mib=memory_total_mib,
        utilization_percent=utilization_percent,
        compute_mode=compute_mode,
        compute_processes=processes,
    )


def _file_claim(relative_path: str, *, sha256: str = "f" * 64) -> object:
    return launcher.artifact_io.FileClaim(
        relative_path=relative_path,
        device=11,
        inode=22,
        mode=stat.S_IFREG | 0o644,
        link_count=1,
        size_bytes=1,
        mtime_ns=33,
        ctime_ns=44,
        sha256=sha256,
    )


def _lease() -> launcher.GenerationLease:
    return launcher.GenerationLease(
        claim=_file_claim(launcher.GENERATION_LEASE_RELATIVE_PATH),
        owner_token="a" * 64,
        payload=b"{}\n",
        artifact_io_source=_file_claim(
            launcher.ARTIFACT_IO_RELATIVE_PATH, sha256="e" * 64
        ),
    )


def _fake_final_authority(
    expected: launcher.ExpectedRunIdentity, output_root: Path
) -> launcher.FinalCandidateAuthority:
    return launcher.FinalCandidateAuthority(
        claim=_file_claim(launcher.CANDIDATE_LOCK_RELATIVE_PATH),
        payload=b"{}\n",
        source_revision=str(expected.source_revision),
        normalized_lock={},
        candidate_id="r-w1-1000u-dcb271453411",
        config_id="r_t050_p100",
        output_root=output_root,
    )


def _expected(tmp_path: Path, *, num_samples: int = 3) -> launcher.ExpectedRunIdentity:
    checkpoint = (tmp_path / "model.ckpt").resolve()
    checkpoint.write_bytes(b"checkpoint fixture")
    config = (tmp_path / "config.yaml").resolve()
    config.write_text(
        "model_path: ignored.ckpt\n"
        "num_samples: 17\n"
        "softmax_temp: 0.5\n"
        "randomness: 0.5\n"
        "min_add_len: 40\n",
        encoding="utf-8",
    )
    source = {
        "model_path": "ignored.ckpt",
        "num_samples": 17,
        "softmax_temp": 0.5,
        "randomness": 0.5,
        "min_add_len": 40,
    }
    source_revision = "e" * 40
    source_config_sha256 = launcher._sha256_file(config)
    config_git_tracking = {
        "path": str(config),
        "relative_path": "config.yaml",
        "source_revision": source_revision,
        "sha256": source_config_sha256,
        "tracked_at_source_revision": True,
    }
    sampling = {
        "diffusion_type": "mdlm",
        "softmax_temp": 0.5,
        "randomness": 0.5,
        "min_add_len": 40,
        "num_steps": None,
        "inference_eps": None,
        "exclude_special_tokens": None,
        "prior_variant": None,
        "prior_metadata_sha256": None,
    }
    effective = dict(source)
    effective.update(
        {
            "model_path": str(checkpoint),
            "num_samples": num_samples,
            "device": "cuda:0",
        }
    )
    implementation_inputs = {
        "sampler_source": {
            "path": str((tmp_path / "src/genmol/sampler.py").resolve()),
            "sha256": "b" * 64,
            "size_bytes": 123,
        },
        "length_distribution": {
            "path": str((tmp_path / "data/len.pk").resolve()),
            "sha256": "c" * 64,
            "size_bytes": 456,
            "count": 10,
            "minimum": 1,
            "median": 5.5,
            "maximum": 10,
        },
    }
    metric_inputs = {
        "schema_version": 1,
        "sa_fragment_scores": {
            "path": str((tmp_path / "oracle/fpscores.pkl").resolve()),
            "sha256": "d" * 64,
            "size_bytes": 789,
        },
    }
    return launcher.ExpectedRunIdentity(
        checkpoint_path=checkpoint,
        checkpoint_sha256=launcher._sha256_file(checkpoint),
        checkpoint_global_step=50_000,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        checkpoint_diffusion_type="mdlm",
        checkpoint_udlm_inference_eps=None,
        checkpoint_udlm_exclude_special_tokens=None,
        checkpoint_udlm_prior_variant=None,
        checkpoint_udlm_prior_metadata=None,
        checkpoint_udlm_prior_metadata_sha256=None,
        config_path=config,
        source_config=source,
        source_config_sha256=source_config_sha256,
        config_git_tracking=config_git_tracking,
        sampling_config=sampling,
        sampling_config_sha256=launcher._canonical_json_sha256(sampling),
        effective_config=effective,
        effective_config_sha256=launcher._canonical_json_sha256(effective),
        benchmark_runner_sha256="a" * 64,
        implementation_inputs=implementation_inputs,
        metric_inputs=metric_inputs,
        num_samples=num_samples,
        source_revision=source_revision,
    )


def _final_expected(tmp_path: Path) -> launcher.ExpectedRunIdentity:
    expected = _expected(tmp_path, num_samples=1_000)
    sampling = {
        "diffusion_type": "udlm",
        "softmax_temp": 0.5,
        "raw_loo_top_p": 1.0,
        "randomness": 0.0,
        "min_add_len": 40,
        "num_steps": 128,
        "inference_eps": 1e-05,
        "exclude_special_tokens": True,
        "prior_variant": "release_uniform",
        "prior_metadata_sha256": None,
    }
    implementation_inputs = {
        **expected.implementation_inputs,
        "sampler_source": {
            "path": str(tmp_path / launcher.SAMPLER_SOURCE_RELATIVE_PATH),
            "sha256": "b" * 64,
            "size_bytes": 123,
        },
        "artifact_io_source": {
            "path": str(tmp_path / launcher.ARTIFACT_IO_RELATIVE_PATH),
            "sha256": "e" * 64,
            "size_bytes": 456,
        },
    }
    return replace(
        expected,
        checkpoint_diffusion_type="udlm",
        checkpoint_udlm_inference_eps=1e-05,
        checkpoint_udlm_exclude_special_tokens=True,
        checkpoint_udlm_prior_variant="release_uniform",
        sampling_config=sampling,
        sampling_config_sha256=launcher._canonical_json_sha256(sampling),
        implementation_inputs=implementation_inputs,
    )


def _normalized_final_lock(
    expected: launcher.ExpectedRunIdentity,
    *,
    candidate_id: str = "r-w1-1000u-dcb271453411",
    config_id: str = "r_t050_p100",
) -> dict[str, object]:
    root = Path("output/udlm/final") / candidate_id / config_id
    return {
        "candidate_id": candidate_id,
        "checkpoint": {
            "relative_path": expected.checkpoint_path.relative_to(
                launcher.REPOSITORY_ROOT
            ),
            "sha256": expected.checkpoint_sha256,
            "size_bytes": expected.checkpoint_size_bytes,
            "global_step": expected.checkpoint_global_step,
        },
        "evaluation_config_relative_path": expected.config_path.relative_to(
            launcher.REPOSITORY_ROOT
        ),
        "evaluation_config_sha256": expected.source_config_sha256,
        "sampling_config": dict(expected.sampling_config),
        "sampling_sha256": expected.sampling_config_sha256,
        "inference_weights": dict(launcher.FINAL_INFERENCE_WEIGHTS),
        "benchmark_runner_sha256": expected.benchmark_runner_sha256,
        "sampler_source_sha256": expected.implementation_inputs["sampler_source"][
            "sha256"
        ],
        "implementation_inputs_sha256": launcher._canonical_json_sha256(
            expected.implementation_inputs
        ),
        "metric_inputs_sha256": launcher._canonical_json_sha256(expected.metric_inputs),
        "final_run_directories": {seed: root / f"seed_{seed}" for seed in (0, 1, 2)},
        "gate_source_sha256": "1" * 64,
        "report_source_sha256": "2" * 64,
        "rescore_source_sha256": "3" * 64,
        "rescore_dependency_sha256": "4" * 64,
        "benchmark_launcher_source_sha256": "5" * 64,
        "pilot_evidence_writer_source_sha256": "6" * 64,
    }


def _write_raw_csv(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=launcher.benchmark_runner.RAW_SAMPLE_FIELDS
        )
        writer.writeheader()
        for index in range(count):
            row = {field: "" for field in launcher.benchmark_runner.RAW_SAMPLE_FIELDS}
            row["sample_index"] = index
            row["raw_model_text"] = f"SAFE-{index}"
            writer.writerow(row)


def _write_matching_artifacts(
    output_root: Path,
    seed: int,
    expected: launcher.ExpectedRunIdentity,
) -> tuple[Path, Path]:
    run_dir = output_root / f"seed_{seed}"
    raw_path = run_dir / launcher.benchmark_runner.RAW_SAMPLES_FILENAME
    summary_path = run_dir / launcher.benchmark_runner.SUMMARY_FILENAME
    _write_raw_csv(raw_path, expected.num_samples)
    command = launcher._legacy_schema7_command(
        checkpoint=expected.checkpoint_path,
        expected_checkpoint_sha256=expected.checkpoint_sha256,
        expected_source_revision=expected.source_revision,
        config=expected.config_path,
        expected_config_sha256=expected.source_config_sha256,
        num_samples=expected.num_samples,
        seed=seed,
        output_dir=run_dir.resolve(),
    )
    selection = {
        "event": "launch",
        "source_revision": {
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
        "command": command,
    }
    summary = {
        "schema_version": 7,
        "status": "completed",
        "seed": seed,
        "num_samples": expected.num_samples,
        "run": {
            "seed": seed,
            "requested_sample_count": expected.num_samples,
            "evaluation_tier": ("final" if expected.num_samples == 1_000 else "pilot"),
            "final_protocol_eligible": expected.num_samples == 1_000,
            "started_at_utc": "2026-09-05T00:00:00+00:00",
            "completed_at_utc": "2026-09-05T00:01:00+00:00",
            "one_seed_per_invocation": True,
            "single_generation_batch": True,
            "command": command,
            "seed_configuration": {
                "seed": seed,
                "seed_applied_immediately_before_generation": True,
                "python_random": True,
                "numpy": True,
                "torch_cpu": True,
                "torch_cuda_all": True,
                "python_hash_seed": str(seed),
            },
            "generation_protocol": {
                "diffusion_type": expected.sampling_config["diffusion_type"],
                "nfe": expected.sampling_config["num_steps"] or 2,
                "num_steps": expected.sampling_config["num_steps"],
                "inference_eps": expected.sampling_config["inference_eps"],
                "exclude_special_tokens": expected.sampling_config[
                    "exclude_special_tokens"
                ],
                "prior_variant": expected.sampling_config["prior_variant"],
                "prior_metadata_sha256": expected.sampling_config[
                    "prior_metadata_sha256"
                ],
                "temperature": expected.sampling_config["softmax_temp"],
                "randomness": expected.sampling_config["randomness"],
                "randomness_used_by_sampler": (
                    expected.sampling_config["diffusion_type"] == "mdlm"
                ),
            },
        },
        "checkpoint": {
            "path": str(expected.checkpoint_path),
            "sha256": expected.checkpoint_sha256,
            "global_step": expected.checkpoint_global_step,
            "size_bytes": expected.checkpoint_size_bytes,
            "byte_identity_verified_before_and_after_load": True,
            "diffusion_type": expected.checkpoint_diffusion_type,
            "udlm_inference_eps": expected.checkpoint_udlm_inference_eps,
            "udlm_exclude_special_tokens": (
                expected.checkpoint_udlm_exclude_special_tokens
            ),
            "udlm_prior_variant": expected.checkpoint_udlm_prior_variant,
            "udlm_prior_metadata": expected.checkpoint_udlm_prior_metadata,
            "udlm_prior_metadata_sha256": (
                expected.checkpoint_udlm_prior_metadata_sha256
            ),
        },
        "config": {
            "path": str(expected.config_path),
            "sha256": expected.source_config_sha256,
            "git_tracking": expected.config_git_tracking,
            "source": expected.source_config,
            "sampling": expected.sampling_config,
            "sampling_sha256": expected.sampling_config_sha256,
            "effective": expected.effective_config,
            "effective_sha256": expected.effective_config_sha256,
        },
        "metrics": {
            "released_comparable": {"validity": 1.0},
            "strict": {"validity": 1.0},
        },
        "failure_counts": {
            "raw_safe_conversion_failed": 0,
            "strict_decode_failed": 0,
            "released_decode_failed": 0,
            "released_recovered_strict_failure": 0,
            "strict_valid_but_released_failed": 0,
            "released_largest_component_applied": 0,
            "strict_duplicates": 0,
            "released_duplicates": 0,
        },
        "runtime_seconds": {
            "model_load_and_device_move": 1.0,
            "model_sampling_and_tokenizer": 1.0,
            "released_postprocessing": 0.0,
            "generation": 1.0,
            "decode_and_metrics": 1.0,
            "total_before_summary_write": 3.0,
        },
        "environment": {
            "requested_device": "cuda:0",
            "resolved_model_device": "cuda:0",
            "torch_cuda_available": True,
            "launch_environment": {
                "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT": json.dumps(selection)
            },
        },
        "git": {
            "commit": expected.source_revision,
            "upstream": expected.source_revision,
            "expected_source_revision": expected.source_revision,
            "dirty": False,
            "clean_pushed_source_verified_before_and_after_run": True,
            "runner_sha256": expected.benchmark_runner_sha256,
        },
        "implementation_inputs": expected.implementation_inputs,
        "metric_inputs": expected.metric_inputs,
        "tokenizer": {"effective_size": 1_880},
        "artifacts": {
            "raw_samples_csv": {
                "path": str(raw_path.resolve()),
                "sha256": launcher._sha256_file(raw_path),
                "row_count": expected.num_samples,
                "fields": list(launcher.benchmark_runner.RAW_SAMPLE_FIELDS),
            },
            "summary_json": {"path": str(summary_path.resolve())},
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return raw_path, summary_path


def _read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _failure_job(
    output_root: Path,
    log_root: Path,
    expected: launcher.ExpectedRunIdentity,
    *,
    seed: int,
    started_at_utc: str = "2026-09-06T01:00:00+00:00",
) -> launcher.RunningJob:
    run_dir = output_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_state = run_dir.stat(follow_symlinks=False)
    output_owner = launcher.artifact_io.OwnedDirectory(
        relative_path=launcher._repository_relative(
            run_dir, label="test output directory"
        ),
        device=int(run_state.st_dev),
        inode=int(run_state.st_ino),
        mode=int(run_state.st_mode),
    )
    command = launcher._command(
        checkpoint=expected.checkpoint_path,
        expected_checkpoint_sha256=expected.checkpoint_sha256,
        expected_source_revision=str(expected.source_revision),
        config=expected.config_path,
        expected_config_sha256=expected.source_config_sha256,
        num_samples=expected.num_samples,
        seed=seed,
        output_dir=run_dir,
        expected_output_directory_device=output_owner.device,
        expected_output_directory_inode=output_owner.inode,
    )
    log_path = log_root / f"seed_{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(f"synthetic failure log for seed {seed}\n".encode())
    return launcher.RunningJob(
        seed=seed,
        gpu=_gpu(),
        process=mock.Mock(),
        log_handle=mock.Mock(closed=True),
        log_path=log_path,
        command=tuple(command),
        started_at_utc=started_at_utc,
        output_directory_owner=output_owner,
    )


def test_gpu_request_accepts_only_a_count_capped_at_three() -> None:
    required = [
        "--checkpoint",
        "model.ckpt",
        "--config",
        "config.yaml",
        "--seeds",
        "1",
        "--output-root",
        "runs",
        "--gpu-count",
        "2",
    ]
    parsed = launcher._parse_args(required)
    assert parsed.gpu_count == 2
    assert not hasattr(parsed, "gpu_indices")
    assert parsed.max_utilization_percent == 10
    assert parsed.min_free_memory_mib == 30_000
    assert launcher._validate_gpu_count(2) == 2
    assert launcher._validate_gpu_count(3) == 3

    with pytest.raises(ValueError, match="between 1 and 3"):
        launcher._validate_gpu_count(4)
    with pytest.raises(ValueError, match="between 1 and 3"):
        launcher._validate_gpu_count(True)
    with pytest.raises(SystemExit):
        launcher._parse_args([*required, "--gpu-indices", "4", "2"])


def test_sample_tier_requires_explicit_bounded_pilot() -> None:
    assert launcher._validate_sample_tier(1_000, pilot=False) == "final"
    assert launcher._validate_sample_tier(32, pilot=True) == "pilot"
    assert (
        launcher._validate_sample_tier(
            256,
            pilot=False,
            selection_pilot=True,
        )
        == "pilot"
    )
    with pytest.raises(ValueError, match="exactly 1000"):
        launcher._validate_sample_tier(32, pilot=False)
    with pytest.raises(ValueError, match="capped at 100"):
        launcher._validate_sample_tier(101, pilot=True)
    with pytest.raises(ValueError, match="exactly 256"):
        launcher._validate_sample_tier(
            255,
            pilot=False,
            selection_pilot=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        launcher._validate_sample_tier(
            32,
            pilot=True,
            selection_pilot=True,
        )
    with pytest.raises(ValueError, match="must be an integer"):
        launcher._validate_sample_tier(True, pilot=True)


def test_candidate_lock_scope_is_mandatory_only_for_final_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    fixed = Path(launcher.CANDIDATE_LOCK_RELATIVE_PATH)
    assert launcher._validate_candidate_lock_scope(
        fixed, pilot=False, selection_pilot=False
    ) == (tmp_path / fixed)
    assert launcher._validate_candidate_lock_scope(
        tmp_path / fixed, pilot=False, selection_pilot=False
    ) == (tmp_path / fixed)
    with pytest.raises(ValueError, match="require --candidate-lock"):
        launcher._validate_candidate_lock_scope(
            None, pilot=False, selection_pilot=False
        )
    with pytest.raises(ValueError, match="fixed canonical path exactly"):
        launcher._validate_candidate_lock_scope(
            Path("experiments/udlm/candidates/../candidates/candidate_lock.json"),
            pilot=False,
            selection_pilot=False,
        )
    for pilot, selection_pilot in ((True, False), (False, True)):
        with pytest.raises(ValueError, match="forbidden"):
            launcher._validate_candidate_lock_scope(
                fixed,
                pilot=pilot,
                selection_pilot=selection_pilot,
            )
        assert (
            launcher._validate_candidate_lock_scope(
                None,
                pilot=pilot,
                selection_pilot=selection_pilot,
            )
            is None
        )


def test_committed_regular_snapshot_rejects_dirty_bytes_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    relative = launcher.CANDIDATE_LOCK_RELATIVE_PATH
    lock_path = tmp_path / relative
    lock_path.parent.mkdir(parents=True)
    payload = b'{"schema_version":2}\n'
    lock_path.write_bytes(payload)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", relative], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "lock"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    claim, observed = launcher._committed_regular_file_snapshot(
        relative,
        source_revision=revision,
        label="candidate lock",
        expected_sha256=launcher.hashlib.sha256(payload).hexdigest(),
    )
    assert observed == payload
    assert claim.size_bytes == len(payload)

    lock_path.write_bytes(b'{"schema_version":3}\n')
    with pytest.raises(launcher.FinalCandidateLockError, match="differ from the exact"):
        launcher._committed_regular_file_snapshot(
            relative, source_revision=revision, label="candidate lock"
        )

    lock_path.unlink()
    target = tmp_path / "foreign-lock.json"
    target.write_bytes(payload)
    lock_path.symlink_to(target)
    with pytest.raises(launcher.FinalCandidateLockError, match="stable candidate lock"):
        launcher._committed_regular_file_snapshot(
            relative, source_revision=revision, label="candidate lock"
        )


def test_final_lock_commit_must_be_the_exact_lock_only_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True
    )
    (tmp_path / "base.txt").write_text("base\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "base.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    parent = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock_path = tmp_path / launcher.CANDIDATE_LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True)
    lock_payload = b'{"schema_version":2}\n'
    lock_path.write_bytes(lock_payload)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "add",
            launcher.CANDIDATE_LOCK_RELATIVE_PATH,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "lock only"], check=True
    )
    lock_revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert (
        launcher._require_exact_final_lock_commit(
            source_revision=lock_revision, lock_payload=lock_payload
        )
        == parent
    )

    (tmp_path / "later.txt").write_text("later\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "later.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "later descendant"],
        check=True,
    )
    descendant = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(launcher.FinalCandidateLockError, match="add only"):
        launcher._require_exact_final_lock_commit(
            source_revision=descendant, lock_payload=lock_payload
        )


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("seeds", "ordered list"),
        ("samples", "exactly 1000"),
        ("checkpoint", "checkpoint bytes"),
        ("config", "evaluation config bytes"),
        ("ema", "locked EMA UDLM"),
        ("sampler", "sampler source"),
        ("implementation", "implementation-input map"),
        ("output", "output root"),
        ("layout", "fixed layout"),
    ],
)
def test_final_candidate_binding_rejects_every_launch_identity_mismatch(
    mismatch: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher.benchmark_runner,
        "__file__",
        str(tmp_path / launcher.BENCHMARK_RUNNER_RELATIVE_PATH),
    )
    expected = _final_expected(tmp_path)
    normalized = _normalized_final_lock(expected)
    seeds = [0, 1, 2]
    samples = 1_000
    output_root = tmp_path / "output/udlm/final/r-w1-1000u-dcb271453411/r_t050_p100"
    if mismatch == "seeds":
        seeds = [0, 2, 1]
    elif mismatch == "samples":
        samples = 999
    elif mismatch == "checkpoint":
        normalized["checkpoint"] = {
            **normalized["checkpoint"],
            "sha256": "0" * 64,
        }
    elif mismatch == "config":
        normalized["evaluation_config_sha256"] = "0" * 64
    elif mismatch == "ema":
        normalized["inference_weights"] = {
            **launcher.FINAL_INFERENCE_WEIGHTS,
            "source": "raw",
        }
    elif mismatch == "sampler":
        normalized["sampler_source_sha256"] = "0" * 64
    elif mismatch == "implementation":
        normalized["implementation_inputs_sha256"] = "0" * 64
    elif mismatch == "output":
        output_root = tmp_path / "output/wrong"
    elif mismatch == "layout":
        directories = dict(normalized["final_run_directories"])
        directories[2] = directories[2].with_name("seed_1")
        normalized["final_run_directories"] = directories
    monkeypatch.setattr(
        launcher,
        "_implementation_source_digest",
        mock.Mock(side_effect=["b" * 64, "e" * 64]),
    )
    monkeypatch.setattr(
        launcher,
        "_committed_regular_file_snapshot",
        mock.Mock(return_value=(_file_claim("source"), b"source")),
    )

    with pytest.raises(launcher.FinalCandidateLockError, match=message):
        launcher._bind_final_candidate_authority(
            normalized_lock=normalized,
            expected=expected,
            checkpoint=expected.checkpoint_path,
            config=expected.config_path,
            seeds=seeds,
            num_samples=samples,
            output_root=output_root,
            source_revision=str(expected.source_revision),
        )


def test_final_candidate_binding_accepts_exact_builder_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher.benchmark_runner,
        "__file__",
        str(tmp_path / launcher.BENCHMARK_RUNNER_RELATIVE_PATH),
    )
    expected = _final_expected(tmp_path)
    normalized = _normalized_final_lock(expected)
    monkeypatch.setattr(
        launcher,
        "_implementation_source_digest",
        mock.Mock(side_effect=["b" * 64, "e" * 64]),
    )
    monkeypatch.setattr(
        launcher,
        "_committed_regular_file_snapshot",
        mock.Mock(return_value=(_file_claim("source"), b"source")),
    )
    output_root = tmp_path / "output/udlm/final/r-w1-1000u-dcb271453411/r_t050_p100"

    assert launcher._bind_final_candidate_authority(
        normalized_lock=normalized,
        expected=expected,
        checkpoint=expected.checkpoint_path,
        config=expected.config_path,
        seeds=[0, 1, 2],
        num_samples=1_000,
        output_root=output_root,
        source_revision=str(expected.source_revision),
    ) == ("r-w1-1000u-dcb271453411", "r_t050_p100", output_root)


def test_final_preflight_invokes_protocol_schema_and_runtime_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.udlm import superiority_gate

    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    expected = _final_expected(tmp_path)
    output_root = tmp_path / "output/udlm/final/r-w1-1000u-dcb271453411/r_t050_p100"
    normalized = _normalized_final_lock(expected)
    normalized["scipy_version"] = "test"
    snapshots: list[str] = []

    def committed_snapshot(
        relative_path: str, **_kwargs: object
    ) -> tuple[object, bytes]:
        snapshots.append(relative_path)
        return _file_claim(relative_path), b"{}\n"

    monkeypatch.setattr(
        launcher, "_committed_regular_file_snapshot", committed_snapshot
    )
    monkeypatch.setattr(launcher, "_require_exact_final_lock_commit", mock.Mock())
    bind = mock.Mock(
        return_value=(
            "r-w1-1000u-dcb271453411",
            "r_t050_p100",
            output_root,
        )
    )
    monkeypatch.setattr(launcher, "_bind_final_candidate_authority", bind)
    validate_protocol = mock.Mock()
    validate_lock = mock.Mock(return_value=normalized)
    validate_runtime = mock.Mock(return_value={"sources_match_prelocked_bytes": True})
    monkeypatch.setattr(
        superiority_gate, "strict_json_loads", mock.Mock(side_effect=[{}, {}])
    )
    monkeypatch.setattr(superiority_gate, "validate_protocol", validate_protocol)
    monkeypatch.setattr(superiority_gate, "validate_candidate_lock", validate_lock)
    monkeypatch.setattr(superiority_gate, "validate_analysis_runtime", validate_runtime)

    authority = launcher._preflight_final_candidate_lock(
        candidate_lock=tmp_path / launcher.CANDIDATE_LOCK_RELATIVE_PATH,
        expected=expected,
        checkpoint=expected.checkpoint_path,
        config=expected.config_path,
        seeds=[0, 1, 2],
        num_samples=1_000,
        output_root=output_root,
        source_revision=str(expected.source_revision),
    )

    assert authority.candidate_id == "r-w1-1000u-dcb271453411"
    assert snapshots[:2] == [
        launcher.CANDIDATE_LOCK_RELATIVE_PATH,
        superiority_gate.PROTOCOL_RELATIVE_PATH.as_posix(),
    ]
    assert set(snapshots[2:]) == {
        "scripts/udlm/superiority_gate.py",
        "scripts/exps/denovo/report.py",
        "scripts/udlm/rescore_denovo_run.py",
        "scripts/udlm/rescore_mdlm_baseline.py",
        "scripts/exps/denovo/launch_benchmark.py",
        "scripts/udlm/write_pilot_evidence.py",
    }
    validate_protocol.assert_called_once_with({})
    validate_lock.assert_called_once_with({}, {})
    validate_runtime.assert_called_once_with(
        normalized,
        expected_artifact_io_source_sha256="e" * 64,
    )
    bind.assert_called_once()


def test_final_preflight_rejects_schema_failure_before_input_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.udlm import superiority_gate

    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    expected = _final_expected(tmp_path)
    monkeypatch.setattr(
        launcher,
        "_committed_regular_file_snapshot",
        lambda relative_path, **_kwargs: (_file_claim(relative_path), b"{}\n"),
    )
    monkeypatch.setattr(launcher, "_require_exact_final_lock_commit", mock.Mock())
    monkeypatch.setattr(
        superiority_gate, "strict_json_loads", mock.Mock(side_effect=[{}, {}])
    )
    monkeypatch.setattr(superiority_gate, "validate_protocol", mock.Mock())
    monkeypatch.setattr(
        superiority_gate,
        "validate_candidate_lock",
        mock.Mock(side_effect=superiority_gate.GateValidationError("bad lock")),
    )
    bind = mock.Mock(side_effect=AssertionError("must not bind rejected lock"))
    monkeypatch.setattr(launcher, "_bind_final_candidate_authority", bind)

    with pytest.raises(launcher.FinalCandidateLockError, match="schema-2 contract"):
        launcher._preflight_final_candidate_lock(
            candidate_lock=tmp_path / launcher.CANDIDATE_LOCK_RELATIVE_PATH,
            expected=expected,
            checkpoint=expected.checkpoint_path,
            config=expected.config_path,
            seeds=[0, 1, 2],
            num_samples=1_000,
            output_root=tmp_path / "output/udlm/final/r/x",
            source_revision=str(expected.source_revision),
        )
    bind.assert_not_called()


def test_final_lock_revalidation_rejects_same_bytes_under_replaced_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _expected(tmp_path, num_samples=1_000)
    authority = _fake_final_authority(expected, tmp_path / "output/final")
    replacement = replace(authority.claim, inode=authority.claim.inode + 1)
    monkeypatch.setattr(
        launcher,
        "_committed_regular_file_snapshot",
        mock.Mock(return_value=(replacement, authority.payload)),
    )

    with pytest.raises(launcher.FinalCandidateLockError, match="identity or bytes"):
        launcher._revalidate_final_candidate_lock(authority)


def test_selection_pilot_cli_modes_are_mutually_exclusive() -> None:
    required = [
        "--checkpoint",
        "model.ckpt",
        "--config",
        "config.yaml",
        "--num-samples",
        "256",
        "--seeds",
        "1000",
        "1001",
        "--output-root",
        "runs",
        "--gpu-count",
        "1",
    ]
    parsed = launcher._parse_args(
        [
            *required,
            "--selection-pilot",
            "--attempt-id",
            "schedule-l1-a1",
            "--candidate-id",
            "schedule-uniform",
        ]
    )
    assert parsed.selection_pilot is True
    assert parsed.pilot is False
    assert parsed.attempt_id == "schedule-l1-a1"
    assert parsed.candidate_id == "schedule-uniform"

    with pytest.raises(SystemExit):
        launcher._parse_args(
            [
                *required,
                "--pilot",
                "--selection-pilot",
                "--attempt-id",
                "schedule-l1-a1",
                "--candidate-id",
                "schedule-uniform",
            ]
        )


def test_selection_pilot_requires_exact_seeds_and_normalized_attempt_id() -> None:
    assert (
        launcher._validate_attempt_scope(
            selection_pilot=True,
            attempt_id="schedule-l1-a1",
            seeds=[1000, 1001],
        )
        == "schedule-l1-a1"
    )
    for seeds in ([1001, 1000], [1000], [1000, 1001, 1002]):
        with pytest.raises(ValueError, match="ordered seeds"):
            launcher._validate_attempt_scope(
                selection_pilot=True,
                attempt_id="schedule-l1-a1",
                seeds=seeds,
            )
    for attempt_id in (None, " Schedule-l1", "Schedule-l1", "a/b", "a" * 97):
        with pytest.raises(ValueError, match="normalized"):
            launcher._validate_attempt_scope(
                selection_pilot=True,
                attempt_id=attempt_id,
                seeds=[1000, 1001],
            )
    assert (
        launcher._validate_attempt_scope(
            pilot=True,
            selection_pilot=False,
            attempt_id="engineering-a",
            seeds=[1002, 1007],
        )
        == "engineering-a"
    )
    with pytest.raises(ValueError, match=">=1000"):
        launcher._validate_attempt_scope(
            pilot=True,
            selection_pilot=False,
            attempt_id="engineering-a",
            seeds=[999],
        )
    with pytest.raises(ValueError, match="only with a pilot mode"):
        launcher._validate_attempt_scope(
            selection_pilot=False,
            attempt_id="unregistered",
            seeds=[7],
        )
    assert (
        launcher._validate_attempt_scope(
            selection_pilot=False,
            attempt_id=None,
            seeds=[7],
        )
        is None
    )
    assert (
        launcher._validate_candidate_scope(
            selection_pilot=True,
            candidate_id="schedule-uniform",
        )
        == "schedule-uniform"
    )
    assert (
        launcher._validate_candidate_scope(
            pilot=True,
            selection_pilot=False,
            candidate_id="engineering-candidate",
        )
        == "engineering-candidate"
    )
    for candidate_id in (None, "ab", "Schedule-uniform", "candidate/path"):
        with pytest.raises(ValueError, match="candidate-id must already be normalized"):
            launcher._validate_candidate_scope(
                selection_pilot=True,
                candidate_id=candidate_id,
            )
    with pytest.raises(ValueError, match="only with a pilot mode"):
        launcher._validate_candidate_scope(
            selection_pilot=False,
            candidate_id="schedule-uniform",
        )


def test_selection_attempt_roots_are_keyed_and_never_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    output_root, log_root = launcher._resolve_attempt_keyed_roots(
        Path("output/pilots"),
        Path("output/logs/pilots"),
        attempt_id="schedule-l1-a1",
    )
    assert output_root == tmp_path / "output/pilots/schedule-l1-a1"
    assert log_root == tmp_path / "output/logs/pilots/schedule-l1-a1"
    launcher._require_fresh_pilot_attempt_paths(output_root, log_root)
    launcher._reserve_pilot_attempt_paths(output_root, log_root)
    assert output_root.is_dir()
    assert log_root.is_dir()
    with pytest.raises(FileExistsError, match="retries require a new attempt-id"):
        launcher._require_fresh_pilot_attempt_paths(output_root, log_root)
    with pytest.raises(FileExistsError, match="concurrently claimed"):
        launcher._reserve_pilot_attempt_paths(output_root, log_root)

    plain_output, plain_logs = launcher._resolve_attempt_keyed_roots(
        Path("plain-runs"),
        Path("plain-logs"),
        attempt_id=None,
    )
    assert plain_output == tmp_path / "plain-runs"
    assert plain_logs == tmp_path / "plain-logs"


def test_selection_pilot_completion_requires_schema7_pilot_ineligible_markers(
    tmp_path: Path,
) -> None:
    from scripts.exps.denovo import report

    expected = _expected(tmp_path, num_samples=256)
    output_root = tmp_path / "selection-runs/schedule-l1-a1"
    _, summary_path = _write_matching_artifacts(output_root, 1000, expected)
    summary = _read_summary(summary_path)
    assert summary["schema_version"] == 7
    assert summary["run"]["evaluation_tier"] == "pilot"
    assert summary["run"]["final_protocol_eligible"] is False
    with (
        mock.patch.object(
            launcher,
            "_require_clean_pushed_source",
            return_value={
                "head": expected.source_revision,
                "upstream": expected.source_revision,
            },
        ),
        mock.patch.object(
            report,
            "validate_run_evidence",
            return_value={"validated": True},
        ) as validate,
    ):
        assert launcher._completed(output_root, 1000, expected) is True
    validate.assert_called_once_with(
        output_root / "seed_1000",
        1000,
        expected_samples=256,
        expected_tier="pilot",
        final_protocol_eligible=False,
    )

    summary["run"]["final_protocol_eligible"] = True
    _write_summary(summary_path, summary)
    with pytest.raises(
        launcher.CompletionArtifactError,
        match="run.final_protocol_eligible",
    ):
        launcher._completed(output_root, 1000, expected)


def test_schema7_udlm_final_is_never_recovered_as_locked_final_evidence(
    tmp_path: Path,
) -> None:
    expected = _final_expected(tmp_path)
    output_root = tmp_path / "final-runs"
    _write_matching_artifacts(output_root, 0, expected)

    with pytest.raises(
        launcher.CompletionArtifactError,
        match="locked UDLM final evidence must use current benchmark schema 8",
    ):
        launcher._completed(output_root, 0, expected)


def test_selection_failure_receipt_binds_inputs_partial_artifacts_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    expected = _expected(tmp_path, num_samples=256)
    output_root = tmp_path / "output/selection/attempt-a"
    log_root = tmp_path / "output/logs/selection/attempt-a"
    job = _failure_job(output_root, log_root, expected, seed=1000)
    run_dir = output_root / "seed_1000"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / launcher.benchmark_runner.SUMMARY_FILENAME
    raw_path = run_dir / launcher.benchmark_runner.RAW_SAMPLES_FILENAME
    summary_path.write_bytes(b'{"status":"partial"}\n')
    raw_path.write_bytes(b"sample_index,raw_model_text\n0,C\n")
    source_revision = {
        "head": expected.source_revision,
        "upstream": expected.source_revision,
    }

    receipt_path = launcher._write_pilot_failure_receipt(
        output_root=output_root,
        job=job,
        expected=expected,
        attempt_id="attempt-a",
        candidate_id="schedule-uniform",
        pilot_mode="registered_selection",
        source_revision=source_revision,
        stage="benchmark_child_process",
        reason="benchmark child exited with status 17",
        process_exit_status=17,
        failed_at_utc="2026-09-06T01:05:00+00:00",
    )
    original = receipt_path.read_bytes()
    receipt = json.loads(original)
    assert set(receipt) == {
        "schema_version",
        "artifact_kind",
        "status",
        "attempt_id",
        "candidate_id",
        "pilot_seed",
        "pilot_mode",
        "requested_samples",
        "stage",
        "reason",
        "started_at_utc",
        "failed_at_utc",
        "process_exit_status",
        "checkpoint",
        "config",
        "command",
        "source_revision",
        "launcher_source",
        "log",
        "partial_artifacts",
    }
    assert receipt["schema_version"] == 1
    assert receipt["artifact_kind"] == "pilot_failure"
    assert receipt["status"] == "failed"
    assert receipt["attempt_id"] == "attempt-a"
    assert receipt["candidate_id"] == "schedule-uniform"
    assert receipt["pilot_seed"] == 1000
    assert receipt["pilot_mode"] == "registered_selection"
    assert receipt["requested_samples"] == 256
    assert receipt["process_exit_status"] == 17
    assert receipt["checkpoint"] == {
        "path": str(expected.checkpoint_path),
        "sha256": expected.checkpoint_sha256,
        "size_bytes": expected.checkpoint_size_bytes,
        "global_step": expected.checkpoint_global_step,
    }
    assert receipt["config"] == {
        "path": str(expected.config_path),
        "sha256": expected.source_config_sha256,
        "sampling": expected.sampling_config,
        "sampling_sha256": expected.sampling_config_sha256,
    }
    assert receipt["command"] == list(job.command)
    assert receipt["source_revision"] == source_revision
    assert receipt["launcher_source"] == {
        "path": str(Path(launcher.__file__).resolve()),
        "sha256": launcher._sha256_file(Path(launcher.__file__).resolve()),
        "size_bytes": Path(launcher.__file__).stat().st_size,
    }
    assert receipt["log"] == {
        "path": str(job.log_path),
        "sha256": launcher._sha256_file(job.log_path),
        "size_bytes": job.log_path.stat().st_size,
    }
    assert receipt["partial_artifacts"] == {
        "summary_json": {
            "path": str(summary_path),
            "sha256": launcher._sha256_file(summary_path),
            "size_bytes": summary_path.stat().st_size,
        },
        "raw_samples_csv": {
            "path": str(raw_path),
            "sha256": launcher._sha256_file(raw_path),
            "size_bytes": raw_path.stat().st_size,
        },
    }

    with pytest.raises(FileExistsError, match="refusing to replace"):
        launcher._write_pilot_failure_receipt(
            output_root=output_root,
            job=job,
            expected=expected,
            attempt_id="attempt-a",
            candidate_id="schedule-uniform",
            pilot_mode="registered_selection",
            source_revision=source_revision,
            stage="benchmark_child_process",
            reason="different reason must not replace the original",
            process_exit_status=9,
            failed_at_utc="2026-09-06T01:06:00+00:00",
        )
    assert receipt_path.read_bytes() == original


def test_selection_failure_receipt_enforces_stage_exit_status_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    expected = _expected(tmp_path, num_samples=256)
    output_root = tmp_path / "output/selection/attempt-a"
    job = _failure_job(
        output_root,
        tmp_path / "output/logs/selection/attempt-a",
        expected,
        seed=1000,
    )
    common = {
        "output_root": output_root,
        "job": job,
        "expected": expected,
        "attempt_id": "attempt-a",
        "candidate_id": "schedule-uniform",
        "source_revision": {
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
        "reason": "synthetic failure",
        "failed_at_utc": "2026-09-06T01:05:00+00:00",
    }
    with pytest.raises(ValueError, match="nonzero status"):
        launcher._build_pilot_failure_receipt(
            **common,
            pilot_mode="registered_selection",
            stage="benchmark_child_process",
            process_exit_status=None,
        )
    with pytest.raises(ValueError, match="null process status"):
        launcher._build_pilot_failure_receipt(
            **common,
            pilot_mode="registered_selection",
            stage="completion_validation",
            process_exit_status=1,
        )


def test_finished_selection_job_emits_failure_receipts_for_both_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    expected = _expected(tmp_path, num_samples=256)
    source_revision = {
        "head": expected.source_revision,
        "upstream": expected.source_revision,
    }

    child_output = tmp_path / "output/selection/child-failure"
    child_job = _failure_job(
        child_output,
        tmp_path / "output/logs/selection/child-failure",
        expected,
        seed=1000,
    )
    validation_output = tmp_path / "output/selection/validation-failure"
    validation_job = _failure_job(
        validation_output,
        tmp_path / "output/logs/selection/validation-failure",
        expected,
        seed=1001,
    )
    engineering_fixture = tmp_path / "engineering"
    engineering_fixture.mkdir()
    engineering_expected = _expected(engineering_fixture, num_samples=32)
    engineering_output = tmp_path / "output/selection/engineering-failure"
    engineering_job = _failure_job(
        engineering_output,
        tmp_path / "output/logs/selection/engineering-failure",
        engineering_expected,
        seed=1007,
    )
    engineering_source_revision = {
        "head": engineering_expected.source_revision,
        "upstream": engineering_expected.source_revision,
    }
    with (
        mock.patch.object(
            launcher,
            "_snapshot",
            side_effect=AssertionError("failure finalization must not inventory GPUs"),
        ),
        mock.patch.object(
            launcher,
            "_probe_gpu",
            side_effect=AssertionError("failure finalization must not probe a GPU"),
        ),
    ):
        failed, details = launcher._finalize_finished_job(
            job=child_job,
            return_code=17,
            output_root=child_output,
            expected=expected,
            pilot_mode="registered_selection",
            attempt_id="child-failure",
            candidate_id="schedule-uniform",
            source_revision=source_revision,
        )
        assert failed is True
        assert any("failure receipt published" in detail for detail in details)

        failed, details = launcher._finalize_finished_job(
            job=engineering_job,
            return_code=9,
            output_root=engineering_output,
            expected=engineering_expected,
            pilot_mode="engineering",
            attempt_id="engineering-failure",
            candidate_id="engineering-candidate",
            source_revision=engineering_source_revision,
        )
        assert failed is True
        assert any("failure receipt published" in detail for detail in details)

        with mock.patch.object(
            launcher,
            "_completed",
            side_effect=launcher.CompletionArtifactError("synthetic invalid summary"),
        ):
            failed, details = launcher._finalize_finished_job(
                job=validation_job,
                return_code=0,
                output_root=validation_output,
                expected=expected,
                pilot_mode="registered_selection",
                attempt_id="validation-failure",
                candidate_id="schedule-uniform",
                source_revision=source_revision,
            )
        assert failed is True
        assert any("failure receipt published" in detail for detail in details)

    child_receipt = json.loads(
        (child_output / "seed_1000/failure_receipt.json").read_text()
    )
    validation_receipt = json.loads(
        (validation_output / "seed_1001/failure_receipt.json").read_text()
    )
    engineering_receipt = json.loads(
        (engineering_output / "seed_1007/failure_receipt.json").read_text()
    )
    assert child_receipt["stage"] == "benchmark_child_process"
    assert child_receipt["process_exit_status"] == 17
    assert child_receipt["partial_artifacts"] == {
        "summary_json": None,
        "raw_samples_csv": None,
    }
    assert validation_receipt["stage"] == "completion_validation"
    assert validation_receipt["process_exit_status"] is None
    assert "synthetic invalid summary" in validation_receipt["reason"]
    assert engineering_receipt["pilot_mode"] == "engineering"
    assert engineering_receipt["requested_samples"] == 32
    assert engineering_receipt["pilot_seed"] == 1007


def test_finished_nonselection_job_never_emits_selection_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    expected = _expected(tmp_path, num_samples=3)
    output_root = tmp_path / "output/generic-pilot"
    job = _failure_job(
        output_root,
        tmp_path / "output/logs/generic-pilot",
        expected,
        seed=7,
    )
    failed, _ = launcher._finalize_finished_job(
        job=job,
        return_code=4,
        output_root=output_root,
        expected=expected,
        pilot_mode=None,
        attempt_id=None,
        candidate_id=None,
        source_revision={
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
    )
    assert failed is True
    assert not (output_root / "seed_7/failure_receipt.json").exists()


def test_checkpoint_may_be_shared_from_project_but_not_escape_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    worktree = project / "run_sources" / "udlm"
    worktree.mkdir(parents=True)
    checkpoint = project / "outputs" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", worktree)

    assert launcher._resolve_checkpoint(checkpoint) == checkpoint
    with pytest.raises(ValueError, match="escapes project root"):
        launcher._resolve_checkpoint(tmp_path / "outside.ckpt")


def test_source_revision_allows_only_output_and_requires_pushed_head() -> None:
    with mock.patch.object(
        launcher,
        "_git_text",
        side_effect=["?? output/run.json", "abc", "abc"],
    ):
        assert launcher._require_clean_pushed_source() == {
            "head": "abc",
            "upstream": "abc",
        }
    with mock.patch.object(
        launcher,
        "_git_text",
        return_value=" M src/genmol/model.py",
    ):
        with pytest.raises(RuntimeError, match="dirty outside output"):
            launcher._require_clean_pushed_source()
    with mock.patch.object(
        launcher,
        "_git_text",
        side_effect=["", "local", "remote"],
    ):
        with pytest.raises(RuntimeError, match="not the pushed upstream"):
            launcher._require_clean_pushed_source()


def test_snapshot_enumerates_and_probes_every_physical_gpu() -> None:
    with (
        mock.patch.object(launcher, "_physical_gpu_indices", return_value=[0, 2, 4]),
        mock.patch.object(
            launcher,
            "_probe_gpu",
            side_effect=lambda index: _gpu(index=index, uuid=f"GPU-{index}"),
        ) as probe,
    ):
        states = launcher._snapshot()

    assert [state.index for state in states] == [0, 2, 4]
    assert probe.call_args_list == [mock.call(0), mock.call(2), mock.call(4)]


def test_snapshot_refuses_partial_unverifiable_inventory() -> None:
    with (
        mock.patch.object(launcher, "_physical_gpu_indices", return_value=[0, 1]),
        mock.patch.object(
            launcher,
            "_probe_gpu",
            side_effect=[_gpu(index=0, uuid="GPU-0"), RuntimeError("query failed")],
        ),
        pytest.raises(RuntimeError, match="complete NVIDIA GPU inventory"),
    ):
        launcher._snapshot()


def test_selection_policy_schema_records_dynamic_full_inventory_semantics() -> None:
    assert launcher._selection_policy(
        requested_gpu_count=2,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    ) == {
        "selection_method": "dynamic_idle_discovery",
        "inventory_scope": "all_nvidia_gpus",
        "requested_gpu_count": 2,
        "max_utilization_percent": 10,
        "utilization_comparison": "strictly_less_than",
        "min_free_memory_mib": 30_000,
        "active_compute_processes_allowed": True,
    }
    with pytest.raises(ValueError, match="between 1 and 3"):
        launcher._selection_policy(
            requested_gpu_count=True,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
        )


def test_probe_gpu_parses_csv_and_allows_fully_recorded_processes_under_policy() -> (
    None
):
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            '2, GPU-test-uuid, "NVIDIA RTX A6000", 23, 49140, 2, Default\n',
            "",
        ),
        subprocess.CompletedProcess(
            [],
            0,
            "GPU-test-uuid, 1234, /other/user/python, 1500\n",
            "",
        ),
    ]
    with mock.patch.object(launcher, "_run_nvidia_smi", side_effect=responses):
        state = launcher._probe_gpu(2)

    assert state.index == 2
    assert state.uuid == "GPU-test-uuid"
    assert state.name == "NVIDIA RTX A6000"
    assert state.memory_total_mib == 49_140
    assert state.compute_processes == (
        {
            "pid": 1234,
            "process_name": "/other/user/python",
            "used_memory_mib": 1500,
        },
    )
    assert launcher._eligible(
        state,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )
    assert (
        state.rejection_reasons(
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
        )
        == []
    )


@pytest.mark.parametrize(
    "process_output",
    [
        (
            "GPU-test-uuid, 1234, /other/user/python, 1500\n"
            "GPU-test-uuid, 1234, /other/user/python, 1500\n"
        ),
        (
            "No running processes found\n"
            "GPU-test-uuid, 1234, /other/user/python, 1500\n"
        ),
        "GPU-different, 1234, /other/user/python, 1500\n",
    ],
)
def test_probe_gpu_rejects_duplicate_or_ambiguous_process_telemetry(
    process_output: str,
) -> None:
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            '2, GPU-test-uuid, "NVIDIA RTX A6000", 23, 49140, 2, Default\n',
            "",
        ),
        subprocess.CompletedProcess([], 0, process_output, ""),
    ]
    with (
        mock.patch.object(launcher, "_run_nvidia_smi", side_effect=responses),
        pytest.raises(RuntimeError, match="ambiguous|invalid"),
    ):
        launcher._probe_gpu("GPU-test-uuid")


def test_snapshot_rejects_duplicate_uuids_across_inventory() -> None:
    with (
        mock.patch.object(launcher, "_physical_gpu_indices", return_value=[0, 1]),
        mock.patch.object(
            launcher,
            "_probe_gpu",
            side_effect=[
                _gpu(index=0, uuid="GPU-same"),
                _gpu(index=1, uuid="GPU-same"),
            ],
        ),
        pytest.raises(RuntimeError, match="duplicate UUIDs"),
    ):
        launcher._snapshot()


def test_gpu_eligibility_enforces_free_memory_and_exclusive_utilization() -> None:
    assert launcher._eligible(
        _gpu(memory_used_mib=19_140, utilization_percent=14),
        max_utilization_percent=15,
        min_free_memory_mib=30_000,
    )
    for state in (
        _gpu(memory_used_mib=19_141),
        _gpu(utilization_percent=15),
        _gpu(compute_mode="Prohibited"),
    ):
        assert not launcher._eligible(
            state,
            max_utilization_percent=15,
            min_free_memory_mib=30_000,
        )


def test_final_probe_rechecks_policy_uuid_and_allows_recorded_processes() -> None:
    candidate = _gpu()
    reached_threshold = _gpu(utilization_percent=10)
    with mock.patch.object(
        launcher, "_probe_gpu", return_value=reached_threshold
    ) as probe:
        selected, reasons = launcher._recheck_gpu_for_launch(
            candidate,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
        )
    assert selected is None
    assert any("not strictly below" in reason for reason in reasons)
    probe.assert_called_once_with(candidate.uuid)

    shared_but_below_threshold = _gpu(
        utilization_percent=9,
        processes=({"pid": 9, "process_name": "/other/python", "used_memory_mib": 20},),
    )
    with mock.patch.object(
        launcher,
        "_probe_gpu",
        return_value=shared_but_below_threshold,
    ):
        selected, reasons = launcher._recheck_gpu_for_launch(
            candidate,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
        )
    assert selected == shared_but_below_threshold
    assert reasons == []

    with mock.patch.object(
        launcher,
        "_probe_gpu",
        return_value=_gpu(uuid="GPU-replaced-at-same-index"),
    ):
        selected, reasons = launcher._recheck_gpu_for_launch(
            candidate,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
        )
    assert selected is None
    assert any("identity changed" in reason for reason in reasons)


def test_child_environment_maps_uuid_and_drops_inherited_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setenv("UNRELATED", "preserved")
    hostile_main = "/hostile/main-checkout/src"
    monkeypatch.setenv("PYTHONPATH", hostile_main)
    monkeypatch.setenv("PYTHONHOME", "/hostile/python-home")
    monkeypatch.setenv("PYTHONUSERBASE", "/hostile/user-base")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/hostile/pycache")
    monkeypatch.setenv("PYTHONWARNINGS", "error")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    monkeypatch.setenv("GENMOL_BENCHMARK_HOSTILE", "must-be-removed")
    monkeypatch.setattr(launcher, "_revalidate_generation_lease", lambda _lease: None)
    selection = {"event": "launch", "physical_gpu": _gpu().as_dict()}
    lease = _lease()
    authority = {"schema_version": launcher.LAUNCH_AUTHORITY_SCHEMA_VERSION}
    environment = launcher._child_environment(
        seed=17,
        gpu=_gpu(),
        selection=selection,
        run_label="denovo-test",
        generation_lease=lease,
        launch_authority=authority,
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test-uuid"
    assert environment["GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX"] == "2"
    assert environment["GENMOL_BENCHMARK_GPU_UUID"] == "GPU-test-uuid"
    assert environment["PYTHONHASHSEED"] == "17"
    assert environment["UNRELATED"] == "preserved"
    assert environment["PYTHONPATH"].split(launcher.os.pathsep) == [
        str(tmp_path / "src"),
        str(tmp_path),
    ]
    assert hostile_main not in environment["PYTHONPATH"]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONOPTIMIZE"] == "0"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert "PYTHONHOME" not in environment
    assert "PYTHONUSERBASE" not in environment
    assert "PYTHONPYCACHEPREFIX" not in environment
    assert "PYTHONWARNINGS" not in environment
    assert "GENMOL_BENCHMARK_HOSTILE" not in environment
    assert (
        json.loads(environment["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"]) == selection
    )
    command = launcher._command(
        checkpoint=tmp_path / "model.ckpt",
        expected_checkpoint_sha256="a" * 64,
        expected_source_revision="b" * 40,
        config=tmp_path / "config.yaml",
        expected_config_sha256="c" * 64,
        num_samples=1_000,
        seed=17,
        output_dir=tmp_path / "seed_17",
        expected_output_directory_device=123,
        expected_output_directory_inode=456,
    )
    assert len(command) == 24
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--expected-checkpoint-sha256") + 1] == "a" * 64
    assert command[command.index("--expected-source-revision") + 1] == "b" * 40
    assert command[command.index("--expected-config-sha256") + 1] == "c" * 64
    assert command[-4:] == [
        "--expected-output-directory-device",
        "123",
        "--expected-output-directory-inode",
        "456",
    ]
    assert "GPU-test-uuid" not in command


def test_child_environment_sets_worktree_pythonpath_when_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(launcher, "_revalidate_generation_lease", lambda _lease: None)
    environment = launcher._child_environment(
        seed=1,
        gpu=_gpu(),
        selection={"event": "launch"},
        run_label="denovo-test",
        generation_lease=_lease(),
        launch_authority={"schema_version": launcher.LAUNCH_AUTHORITY_SCHEMA_VERSION},
    )
    assert environment["PYTHONPATH"].split(launcher.os.pathsep) == [
        str(tmp_path / "src"),
        str(tmp_path),
    ]


def test_matching_artifacts_are_the_only_skippable_state(tmp_path: Path) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    assert launcher._completed(output_root, 7, expected) is False
    _write_matching_artifacts(output_root, 7, expected)
    assert launcher._completed(output_root, 7, expected) is True


def test_skeletal_current_schema_summary_is_not_skippable(tmp_path: Path) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    _, summary_path = _write_matching_artifacts(output_root, 7, expected)
    summary = _read_summary(summary_path)
    for field in ("metrics", "runtime_seconds", "environment", "tokenizer"):
        summary.pop(field)
    _write_summary(summary_path, summary)

    with pytest.raises(
        launcher.CompletionArtifactError,
        match="top-level fields differ from the current child contract",
    ):
        launcher._completed(output_root, 7, expected)


def test_final_completion_delegates_to_report_consumability_validator(
    tmp_path: Path,
) -> None:
    from scripts.exps.denovo import report

    expected = _expected(tmp_path, num_samples=1_000)
    output_root = tmp_path / "runs"
    _write_matching_artifacts(output_root, 7, expected)

    with mock.patch.object(
        launcher,
        "_require_clean_pushed_source",
        return_value={
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
    ):
        with mock.patch.object(
            report,
            "validate_run_evidence",
            return_value={"validated": True},
        ) as validate:
            assert launcher._completed(output_root, 7, expected) is True
        validate.assert_called_once_with(
            output_root / "seed_7",
            7,
            expected_samples=1_000,
            expected_tier="final",
            final_protocol_eligible=True,
        )

        with mock.patch.object(
            report,
            "validate_run_evidence",
            side_effect=report.ReportValidationError("synthetic report rejection"),
        ):
            with pytest.raises(
                launcher.CompletionArtifactError,
                match="fail registered report validation.*synthetic report rejection",
            ):
                launcher._completed(output_root, 7, expected)


def test_expected_identity_uses_checkpoint_metadata_and_normalized_sampling(
    tmp_path: Path,
) -> None:
    expected_fixture = _expected(tmp_path)
    preflight_order = []

    def metric_preflight():
        preflight_order.append("metric_input")
        return expected_fixture.metric_inputs

    def checkpoint_preflight(_path):
        preflight_order.append("checkpoint")
        return {
            "sha256": expected_fixture.checkpoint_sha256,
            "global_step": 50_000,
            "size_bytes": expected_fixture.checkpoint_size_bytes,
        }

    with (
        mock.patch.object(
            launcher.benchmark_runner,
            "checkpoint_metadata",
            side_effect=checkpoint_preflight,
        ),
        mock.patch.object(
            launcher.benchmark_runner,
            "implementation_input_provenance",
            return_value=expected_fixture.implementation_inputs,
        ),
        mock.patch.object(
            launcher.benchmark_runner,
            "metric_input_provenance",
            side_effect=metric_preflight,
        ),
    ):
        expected = launcher._build_expected_run_identity(
            expected_fixture.checkpoint_path,
            expected_fixture.config_path,
            expected_fixture.num_samples,
        )

    assert expected.checkpoint_global_step == 50_000
    assert expected.checkpoint_sha256 == expected_fixture.checkpoint_sha256
    assert expected.sampling_config == {
        "diffusion_type": "mdlm",
        "softmax_temp": 0.5,
        "randomness": 0.5,
        "min_add_len": 40,
        "num_steps": None,
        "inference_eps": None,
        "exclude_special_tokens": None,
        "prior_variant": None,
        "prior_metadata_sha256": None,
    }
    assert expected.sampling_config_sha256 == launcher._canonical_json_sha256(
        expected.sampling_config
    )
    assert expected.effective_config["num_samples"] == expected_fixture.num_samples
    assert expected.effective_config["device"] == "cuda:0"
    # The historical MDLM effective config and its digest must stay unchanged.
    assert expected.effective_config == expected_fixture.effective_config
    assert expected.effective_config_sha256 == expected_fixture.effective_config_sha256
    assert "raw_loo_top_p" not in expected.effective_config
    assert expected.benchmark_runner_sha256 == launcher._sha256_file(
        Path(launcher.benchmark_runner.__file__).resolve()
    )
    assert expected.implementation_inputs == expected_fixture.implementation_inputs
    assert expected.metric_inputs == expected_fixture.metric_inputs
    assert preflight_order[:2] == ["metric_input", "checkpoint"]


@pytest.mark.parametrize("top_p", [None, 1, 0.95])
def test_expected_identity_normalizes_udlm_top_p_and_rejects_endpoint_mismatch(
    tmp_path: Path,
    top_p: float | None,
) -> None:
    expected_fixture = _expected(tmp_path)
    expected_fixture.config_path.write_text(
        "diffusion_type: udlm\n"
        "softmax_temp: 1.0\n"
        "randomness: 0.0\n"
        "min_add_len: 40\n"
        "num_steps: 32\n"
        "inference_eps: 1.0e-5\n"
        "exclude_special_tokens: false\n"
        + ("" if top_p is None else f"raw_loo_top_p: {top_p}\n"),
        encoding="utf-8",
    )
    source_bytes = expected_fixture.config_path.read_bytes()
    metadata = {
        "sha256": expected_fixture.checkpoint_sha256,
        "global_step": 100,
        "size_bytes": expected_fixture.checkpoint_size_bytes,
        "diffusion_type": "udlm",
        "udlm_inference_eps": 1e-5,
        "udlm_exclude_special_tokens": False,
    }
    with (
        mock.patch.object(
            launcher.benchmark_runner,
            "checkpoint_metadata",
            return_value=metadata,
        ),
        mock.patch.object(
            launcher.benchmark_runner,
            "implementation_input_provenance",
            return_value=expected_fixture.implementation_inputs,
        ),
        mock.patch.object(
            launcher.benchmark_runner,
            "metric_input_provenance",
            return_value=expected_fixture.metric_inputs,
        ),
    ):
        expected = launcher._build_expected_run_identity(
            expected_fixture.checkpoint_path,
            expected_fixture.config_path,
            32,
        )
        assert expected.checkpoint_diffusion_type == "udlm"
        assert expected.sampling_config["num_steps"] == 32
        # The child records the normalized default even when the historical
        # YAML omits it; hash exactly the same effective configuration.
        effective = {
            "diffusion_type": "udlm",
            "softmax_temp": 1.0,
            "randomness": 0.0,
            "min_add_len": 40,
            "num_steps": 32,
            "inference_eps": 1e-5,
            "exclude_special_tokens": False,
            "model_path": str(expected_fixture.checkpoint_path),
            "num_samples": 32,
            "device": "cuda:0",
            "raw_loo_top_p": 1.0 if top_p is None else float(top_p),
        }
        assert expected.effective_config == effective
        assert isinstance(expected.effective_config["raw_loo_top_p"], float)
        assert expected.effective_config_sha256 == launcher._canonical_json_sha256(
            effective
        )
        assert ("raw_loo_top_p" in expected.source_config) is (top_p is not None)
        assert expected_fixture.config_path.read_bytes() == source_bytes
        assert expected.source_config_sha256 == launcher._sha256_file(
            expected_fixture.config_path
        )

        metadata["udlm_inference_eps"] = 2e-5
        with pytest.raises(ValueError, match="inference_eps"):
            launcher._build_expected_run_identity(
                expected_fixture.checkpoint_path,
                expected_fixture.config_path,
                32,
            )


@pytest.mark.parametrize("prior_variant", ["schedule_uniform", "empirical_frequency"])
def test_expected_identity_pins_full_categorical_prior_metadata(
    tmp_path: Path,
    prior_variant: str,
) -> None:
    expected_fixture = _expected(tmp_path)
    prior_metadata = {"variant": prior_variant, "immutable": True}
    prior_digest = "a" * 64
    expected_fixture.config_path.write_text(
        "diffusion_type: udlm\n"
        "softmax_temp: 1.0\n"
        "randomness: 0.0\n"
        "min_add_len: 40\n"
        "num_steps: 32\n"
        "inference_eps: 1.0e-5\n"
        "exclude_special_tokens: false\n"
        f"prior_variant: {prior_variant}\n"
        f"prior_metadata_sha256: {prior_digest}\n",
        encoding="utf-8",
    )
    metadata = {
        "sha256": expected_fixture.checkpoint_sha256,
        "global_step": 100,
        "size_bytes": expected_fixture.checkpoint_size_bytes,
        "diffusion_type": "udlm",
        "udlm_inference_eps": 1e-5,
        "udlm_exclude_special_tokens": False,
        "udlm_prior_variant": prior_variant,
        "udlm_prior_metadata": prior_metadata,
        "udlm_prior_metadata_sha256": prior_digest,
    }
    with (
        mock.patch.object(
            launcher.benchmark_runner,
            "checkpoint_metadata",
            return_value=metadata,
        ),
        mock.patch.object(
            launcher.benchmark_runner,
            "implementation_input_provenance",
            return_value=expected_fixture.implementation_inputs,
        ),
        mock.patch.object(
            launcher.benchmark_runner,
            "metric_input_provenance",
            return_value=expected_fixture.metric_inputs,
        ),
    ):
        expected = launcher._build_expected_run_identity(
            expected_fixture.checkpoint_path,
            expected_fixture.config_path,
            32,
        )
        assert expected.checkpoint_udlm_prior_metadata == prior_metadata
        assert expected.checkpoint_udlm_prior_metadata_sha256 == prior_digest
        assert expected.effective_config["raw_loo_top_p"] == 1.0
        assert "raw_loo_top_p" not in expected.source_config

        metadata["udlm_prior_metadata_sha256"] = "b" * 64
        with pytest.raises(ValueError, match="prior_metadata_sha256"):
            launcher._build_expected_run_identity(
                expected_fixture.checkpoint_path,
                expected_fixture.config_path,
                32,
            )


def test_completed_artifacts_reject_checkpoint_and_sampling_mismatches(
    tmp_path: Path,
) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    _, summary_path = _write_matching_artifacts(output_root, 3, expected)
    summary = _read_summary(summary_path)
    summary["checkpoint"]["sha256"] = "0" * 64
    summary["checkpoint"]["global_step"] = 40_000
    summary["config"]["sampling"]["randomness"] = 2.0
    summary["checkpoint"]["udlm_prior_metadata"] = {"forged": True}
    summary["checkpoint"]["udlm_prior_metadata_sha256"] = "f" * 64
    summary["config"]["sampling_sha256"] = launcher._canonical_json_sha256(
        summary["config"]["sampling"]
    )
    _write_summary(summary_path, summary)

    with pytest.raises(launcher.CompletionArtifactError) as caught:
        launcher._completed(output_root, 3, expected)
    message = str(caught.value)
    assert "checkpoint.sha256" in message
    assert "checkpoint.global_step" in message
    assert "config.sampling" in message
    assert "config.sampling_sha256" in message
    assert "checkpoint.udlm_prior_metadata" in message
    assert "checkpoint.udlm_prior_metadata_sha256" in message
    assert "Refusing to skip or relaunch" in message


def test_completed_artifacts_reject_runner_and_implementation_input_mismatches(
    tmp_path: Path,
) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    _, summary_path = _write_matching_artifacts(output_root, 12, expected)
    summary = _read_summary(summary_path)
    summary["git"]["runner_sha256"] = "0" * 64
    summary["implementation_inputs"]["sampler_source"]["sha256"] = "1" * 64
    summary["metric_inputs"]["sa_fragment_scores"]["sha256"] = "2" * 64
    _write_summary(summary_path, summary)

    with pytest.raises(launcher.CompletionArtifactError) as caught:
        launcher._completed(output_root, 12, expected)
    message = str(caught.value)
    assert "git.runner_sha256" in message
    assert "implementation_inputs" in message
    assert "metric_inputs" in message


@pytest.mark.parametrize("missing_name", ["raw_samples.csv", "summary.json"])
def test_partial_artifacts_fail_before_launch(
    tmp_path: Path,
    missing_name: str,
) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    raw_path, summary_path = _write_matching_artifacts(output_root, 1, expected)
    (raw_path if missing_name == "raw_samples.csv" else summary_path).unlink()

    with pytest.raises(
        launcher.CompletionArtifactError, match="partial benchmark artifacts"
    ):
        launcher._completed(output_root, 1, expected)


def test_raw_csv_digest_row_count_and_schema_are_verified(tmp_path: Path) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    raw_path, _ = _write_matching_artifacts(output_root, 5, expected)

    rows = list(csv.reader(raw_path.open("r", encoding="utf-8", newline="")))
    rows[0][1] = "unexpected_field"
    rows.pop()
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    with pytest.raises(launcher.CompletionArtifactError) as caught:
        launcher._completed(output_root, 5, expected)
    message = str(caught.value)
    assert "header/schema" in message
    assert "data rows" in message
    assert "artifacts.raw_samples_csv.sha256" in message


def test_summary_artifact_path_identity_is_verified(tmp_path: Path) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    _, summary_path = _write_matching_artifacts(output_root, 9, expected)
    summary = _read_summary(summary_path)
    summary["artifacts"]["summary_json"]["path"] = str(
        (tmp_path / "different-summary.json").resolve()
    )
    _write_summary(summary_path, summary)

    with pytest.raises(
        launcher.CompletionArtifactError,
        match="artifacts.summary_json.path",
    ):
        launcher._completed(output_root, 9, expected)


def test_main_skips_matching_run_without_probing_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.exps.denovo import report

    expected = _expected(tmp_path, num_samples=1_000)
    output_root = tmp_path / "runs"
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_require_project_virtual_environment",
        lambda: tmp_path / ".venv/bin/python",
    )
    for seed in (0, 1, 2):
        _write_matching_artifacts(output_root, seed, expected)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples, **_kwargs: expected,
    )
    preflight = mock.Mock(return_value=_fake_final_authority(expected, output_root))
    monkeypatch.setattr(launcher, "_preflight_final_candidate_lock", preflight)
    monkeypatch.setattr(launcher, "_revalidate_final_candidate_lock", mock.Mock())
    monkeypatch.setattr(
        launcher,
        "_snapshot",
        lambda *_: pytest.fail("matching completed runs must not probe GPUs"),
    )
    monkeypatch.setattr(
        report,
        "validate_run_evidence",
        lambda *_args, **_kwargs: {"validated": True},
    )

    launcher.main(
        [
            "--checkpoint",
            expected.checkpoint_path.name,
            "--config",
            expected.config_path.name,
            "--num-samples",
            str(expected.num_samples),
            "--seeds",
            "0",
            "1",
            "2",
            "--output-root",
            output_root.name,
            "--candidate-lock",
            launcher.CANDIDATE_LOCK_RELATIVE_PATH,
            "--gpu-count",
            "1",
            "--log-root",
            "logs",
        ]
    )

    assert (
        "already have matching, integrity-checked artifacts" in capsys.readouterr().out
    )


def test_final_dry_run_preflights_lock_before_completion_without_mutation_or_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _expected(tmp_path, num_samples=1_000)
    output_root = tmp_path / "final-runs"
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_require_project_virtual_environment",
        lambda: tmp_path / ".venv/bin/python",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples, **_kwargs: expected,
    )
    order: list[str] = []

    def preflight(**_kwargs: object) -> launcher.FinalCandidateAuthority:
        order.append("preflight")
        return _fake_final_authority(expected, output_root)

    def completed(_root: Path, seed: int, _expected: object) -> bool:
        order.append(f"completed-{seed}")
        return False

    monkeypatch.setattr(launcher, "_preflight_final_candidate_lock", preflight)
    monkeypatch.setattr(launcher, "_completed", completed)
    monkeypatch.setattr(launcher, "_revalidate_final_candidate_lock", mock.Mock())
    monkeypatch.setattr(
        launcher,
        "_snapshot",
        lambda: pytest.fail("final dry-run must not inventory GPUs"),
    )
    monkeypatch.setattr(
        launcher,
        "_acquire_generation_lease",
        lambda **_kwargs: pytest.fail("final dry-run must not acquire the lease"),
    )
    monkeypatch.setattr(
        launcher,
        "_ensure_repository_directory",
        lambda *_args, **_kwargs: pytest.fail("final dry-run must not create paths"),
    )

    launcher.main(
        [
            "--checkpoint",
            expected.checkpoint_path.name,
            "--config",
            expected.config_path.name,
            "--num-samples",
            "1000",
            "--seeds",
            "0",
            "1",
            "2",
            "--output-root",
            output_root.name,
            "--candidate-lock",
            launcher.CANDIDATE_LOCK_RELATIVE_PATH,
            "--gpu-count",
            "1",
            "--log-root",
            "logs",
            "--dry-run",
        ]
    )

    assert order == ["preflight", "completed-0", "completed-1", "completed-2"]
    assert '"gpu_query_performed": false' in capsys.readouterr().out
    assert not output_root.exists()
    assert not (tmp_path / "logs").exists()


def test_main_requires_tmux_only_for_real_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _expected(tmp_path)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_require_project_virtual_environment",
        lambda: tmp_path / ".venv/bin/python",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples, **_kwargs: expected,
    )
    monkeypatch.setattr(launcher, "_snapshot", lambda: [_gpu()])
    common = [
        "--checkpoint",
        expected.checkpoint_path.name,
        "--config",
        expected.config_path.name,
        "--num-samples",
        str(expected.num_samples),
        "--pilot",
        "--attempt-id",
        "engineering-tmux-a",
        "--candidate-id",
        "engineering-candidate",
        "--seeds",
        "1004",
        "--gpu-count",
        "1",
        "--log-root",
        "logs",
    ]

    launcher.main([*common, "--output-root", "dry-runs", "--dry-run"])
    dry_output = capsys.readouterr().out
    assert "DRY RUN" in dry_output
    assert '"pilot_mode": "engineering"' in dry_output
    assert str(tmp_path / "dry-runs/engineering-tmux-a/seed_1004") in dry_output
    assert not (tmp_path / "dry-runs").exists()
    assert not (tmp_path / "logs").exists()

    monkeypatch.setattr(
        launcher,
        "_snapshot",
        lambda: pytest.fail("tmux guard must run before a real GPU probe"),
    )
    with pytest.raises(RuntimeError, match="must run inside tmux"):
        launcher.main([*common, "--output-root", "real-runs"])
    assert not (tmp_path / "real-runs").exists()
    assert not (tmp_path / "logs").exists()


def test_selection_pilot_dry_run_uses_attempt_keyed_schema7_pilot_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _expected(tmp_path, num_samples=256)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_require_project_virtual_environment",
        lambda: tmp_path / ".venv/bin/python",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples, **_kwargs: expected,
    )
    monkeypatch.setattr(launcher, "_snapshot", lambda: [_gpu()])

    launcher.main(
        [
            "--checkpoint",
            expected.checkpoint_path.name,
            "--config",
            expected.config_path.name,
            "--num-samples",
            "256",
            "--selection-pilot",
            "--attempt-id",
            "schedule-l1-a1",
            "--candidate-id",
            "schedule-uniform",
            "--seeds",
            "1000",
            "1001",
            "--output-root",
            "selection-runs",
            "--gpu-count",
            "1",
            "--log-root",
            "selection-logs",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    controller_event = json.loads(output.splitlines()[0])
    assert controller_event["benchmark_mode"] == "selection_pilot"
    assert controller_event["evaluation_tier"] == "pilot"
    assert controller_event["attempt_id"] == "schedule-l1-a1"
    assert controller_event["candidate_id"] == "schedule-uniform"
    assert controller_event["pilot_mode"] == "registered_selection"
    assert controller_event["seeds"] == [1000, 1001]
    assert str(tmp_path / "selection-runs/schedule-l1-a1/seed_1000") in output
    assert str(tmp_path / "selection-runs/schedule-l1-a1/seed_1001") in output
    assert not (tmp_path / "selection-runs").exists()
    assert not (tmp_path / "selection-logs").exists()


def test_main_rejects_partial_output_before_probing_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _expected(tmp_path, num_samples=1_000)
    output_root = tmp_path / "runs"
    raw_path, summary_path = _write_matching_artifacts(output_root, 0, expected)
    summary_path.unlink()
    assert raw_path.exists()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_require_project_virtual_environment",
        lambda: tmp_path / ".venv/bin/python",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples, **_kwargs: expected,
    )
    monkeypatch.setattr(
        launcher,
        "_preflight_final_candidate_lock",
        mock.Mock(return_value=_fake_final_authority(expected, output_root)),
    )
    monkeypatch.setattr(launcher, "_revalidate_final_candidate_lock", mock.Mock())
    monkeypatch.setattr(
        launcher,
        "_snapshot",
        lambda *_: pytest.fail("partial artifacts must fail before a GPU probe"),
    )

    with pytest.raises(
        launcher.CompletionArtifactError, match="partial benchmark artifacts"
    ):
        launcher.main(
            [
                "--checkpoint",
                expected.checkpoint_path.name,
                "--config",
                expected.config_path.name,
                "--num-samples",
                str(expected.num_samples),
                "--seeds",
                "0",
                "1",
                "2",
                "--output-root",
                output_root.name,
                "--candidate-lock",
                launcher.CANDIDATE_LOCK_RELATIVE_PATH,
                "--gpu-count",
                "1",
                "--log-root",
                "logs",
            ]
        )


def test_main_rejects_mismatched_completion_before_probing_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _expected(tmp_path, num_samples=1_000)
    output_root = tmp_path / "runs"
    _, summary_path = _write_matching_artifacts(output_root, 0, expected)
    summary = _read_summary(summary_path)
    summary["checkpoint"]["global_step"] = 45_000
    _write_summary(summary_path, summary)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_require_project_virtual_environment",
        lambda: tmp_path / ".venv/bin/python",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {
            "head": expected.source_revision,
            "upstream": expected.source_revision,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples, **_kwargs: expected,
    )
    monkeypatch.setattr(
        launcher,
        "_preflight_final_candidate_lock",
        mock.Mock(return_value=_fake_final_authority(expected, output_root)),
    )
    monkeypatch.setattr(launcher, "_revalidate_final_candidate_lock", mock.Mock())
    monkeypatch.setattr(
        launcher,
        "_snapshot",
        lambda *_: pytest.fail("mismatched artifacts must fail before a GPU probe"),
    )

    with pytest.raises(
        launcher.CompletionArtifactError, match="checkpoint.global_step"
    ):
        launcher.main(
            [
                "--checkpoint",
                expected.checkpoint_path.name,
                "--config",
                expected.config_path.name,
                "--num-samples",
                str(expected.num_samples),
                "--seeds",
                "0",
                "1",
                "2",
                "--output-root",
                output_root.name,
                "--candidate-lock",
                launcher.CANDIDATE_LOCK_RELATIVE_PATH,
                "--gpu-count",
                "1",
                "--log-root",
                "logs",
            ]
        )
