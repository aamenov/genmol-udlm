"""Launch the separately recorded V8b CE arm after the original V8 failure.

The original CT controller/campaign remain failed. A separate CPU audit
validated its checkpoint. This follow-up starts fresh from MDLM50k EMA and
never resumes CT or declares the original V8 campaign completed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "experiments/udlm/protocols/engineering_v8b_ce_followup.json"
CONFIG = "configs/udlm_e_objective_ce_followup.yaml"
OUTPUT = "output/udlm/engineering_v8b/ce_e_1000_b128_w2"
ORIGINAL_SOURCE = "c8434dda105c5fb2a15bb784d24e0387062c733e"
ORIGINAL_PROTOCOL_SHA256 = (
    "a27875752d25a4a7928401434f9bd90ee6e9d01539586bab70ce3ba75499026e"
)
COMMON_CONFIG_SHA256 = (
    "f373121441fca1aaa2f2657a76d7f7ce66be199cd35637597d0df240e257e89a"
)
ORIGINAL_OUTPUT = "output/udlm/engineering_v8/ct_ce_e_1000_b128_w2"
EVIDENCE = {
    "original_ct_terminal": {
        "relative_path": f"{ORIGINAL_OUTPUT}/ct/terminal_manifest.json",
        "sha256": "1021ba513564a25ca26ea3c2d6c148d271e1cd98d27078056285e848c9c69b5e",
    },
    "original_campaign_terminal": {
        "relative_path": f"{ORIGINAL_OUTPUT}/campaign_terminal.json",
        "sha256": "98075a11c7340dafcc855887348201477981bd2a1a8f5d39053da24866cc663b",
    },
    "post_exit_audit": {
        "relative_path": f"{ORIGINAL_OUTPUT}/post_exit_audit/post_exit_audit.json",
        "sha256": "959151c37072431be23b1f5696d7215da5291e0bace40d6f307864a7262f93c2",
    },
}
for import_root in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(import_root))

from scripts import artifact_io  # noqa: E402
from scripts.udlm import launch_engineering_training as engine  # noqa: E402
from scripts.udlm import launch_objective_training as original  # noqa: E402


def verify_prior_evidence(root, protocol, original_plan):
    """Bind the disclosed failure, separate CPU audit and unchanged training code."""
    if protocol["evidence"] != EVIDENCE:
        raise ValueError("follow-up evidence differs from the fixed V8 incident")
    documents = {}
    for name, reference in EVIDENCE.items():
        claim, payload = artifact_io.snapshot_file(
            root, reference["relative_path"], capture_bytes=True
        )
        if claim.sha256 != reference["sha256"]:
            raise ValueError(f"follow-up {name} digest mismatch")
        documents[name] = json.loads(payload)
    ct, campaign, audit = (
        documents[key]
        for key in (
            "original_ct_terminal",
            "original_campaign_terminal",
            "post_exit_audit",
        )
    )
    source = {"head": ORIGINAL_SOURCE, "upstream": ORIGINAL_SOURCE}
    if (
        ct["status"] != "failed"
        or ct["source"] != source
        or ct["training_return_code"] != 0
        or ct["completed_example_exposures"] is not None
        or campaign["status"] != "failed"
        or campaign["source"] != source
        or campaign["arms"][1]
        != {"arm_id": "ce", "status": "not_completed_after_campaign_stop"}
        or original.common_config(ct["plan"]["config"])
        != original.common_config(original_plan["config"])
        or audit["checkpoint_status"] != "separately_validated_after_controller_failure"
        or audit["original_campaign_status"] != "failed"
        or audit["original_training_return_code"] != 0
        or audit["source"] != source
        or audit["leases_released"] is not True
        or audit["original_terminal_sha256"]
        != EVIDENCE["original_ct_terminal"]["sha256"]
        or audit["original_campaign_terminal_sha256"]
        != EVIDENCE["original_campaign_terminal"]["sha256"]
        or audit["checkpoint"]["sha256"]
        != protocol["separately_validated_ct_checkpoint_sha256"]
        or audit["checkpoint"]["global_step"] != 1000
        or audit["configured_example_exposures_supported_by_step_and_batch"] != 128000
    ):
        raise ValueError("original failure or separate checkpoint audit does not match")
    checkpoint = audit["checkpoint"]
    if checkpoint["relative_path"] != f"{ORIGINAL_OUTPUT}/ct/checkpoints/1000.ckpt":
        raise ValueError("audited CT checkpoint path differs from original V8")
    checkpoint_claim, _ = artifact_io.snapshot_file(root, checkpoint["relative_path"])
    if (
        checkpoint_claim.sha256 != checkpoint["sha256"]
        or checkpoint_claim.size_bytes != checkpoint["size_bytes"]
    ):
        raise ValueError("audited CT checkpoint bytes changed before CE follow-up")
    relative_sources = sorted(
        [
            "scripts/train.py",
            *[str(path.relative_to(root)) for path in (root / "src").rglob("*.py")],
        ]
    )
    actual = {
        relative: artifact_io.snapshot_file(root, relative)[0].sha256
        for relative in relative_sources
    }
    if actual != audit["training_implementation_sha256"]:
        raise ValueError("training implementation differs from the audited V8 CT run")


def build_plan(gpu_count, *, root=ROOT):
    """Compose only the fixed W2 CE arm, with a new output namespace."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    if type(gpu_count) is not int or gpu_count != 2:
        raise ValueError("V8b requires the same two-GPU count as original V8")
    baseline = original.build_plans(2, root=root)[1]
    claim, payload = artifact_io.snapshot_file(root, PROTOCOL, capture_bytes=True)
    protocol = json.loads(payload)
    fixed = {
        "schema_version": 1,
        "study_id": "engineering-v8b-ce-e-1000-b128-w2",
        "claim": "separate_ce_followup_original_v8_campaign_remains_failed",
        "original_source_revision": ORIGINAL_SOURCE,
        "original_protocol": original.PROTOCOL,
        "original_protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
        "matched_common_config_sha256": COMMON_CONFIG_SHA256,
        "config": CONFIG,
        "gpu_count": 2,
        "seed": 1500,
        "optimizer_updates": 1000,
        "global_batch_size": 128,
        "micro_batch_size": 16,
        "gpu_policy": engine.POLICY,
        "checkpoint": baseline["protocol"]["checkpoint"],
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "example_exposures_per_arm": 128000,
        "memory_measurement": baseline["protocol"]["memory_measurement"],
        "separately_validated_ct_checkpoint_sha256": "48986899c401c09cdc1e9e865e773cadc40899b62129420689f89a8e622a9c99",
    }
    if (
        any(protocol.get(key) != value for key, value in fixed.items())
        or baseline["protocol_sha256"] != ORIGINAL_PROTOCOL_SHA256
        or baseline["matched_common_config_sha256"] != COMMON_CONFIG_SHA256
    ):
        raise ValueError("fixed V8b follow-up or original V8 training design changed")
    verify_prior_evidence(root, protocol, baseline)
    config_claim, _ = artifact_io.snapshot_file(root, CONFIG)
    if config_claim.sha256 != protocol["config_sha256"]:
        raise ValueError("follow-up config digest mismatch")
    overrides = [
        argument
        for argument in baseline["training_argv"][5:]
        if not argument.startswith(("callback.dirpath=", "hydra.run.dir="))
    ]
    overrides.extend(
        [
            f"callback.dirpath={root / OUTPUT / 'checkpoints'}",
            f"hydra.run.dir={root / OUTPUT / 'hydra'}",
        ]
    )
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        config = OmegaConf.to_container(
            compose(config_name=Path(CONFIG).stem, overrides=overrides), resolve=True
        )
    expected = copy.deepcopy(baseline["config"])
    expected["callback"]["dirpath"] = str(root / OUTPUT / "checkpoints")
    if config != expected:
        raise ValueError("follow-up differs from V8 CE beyond its output directory")
    command = [*baseline["training_argv"][:4], Path(CONFIG).stem, *overrides]
    plan = {
        **baseline,
        "protocol": protocol,
        "protocol_sha256": claim.sha256,
        "config_source_sha256": config_claim.sha256,
        "config": config,
        "config_sha256": engine.canonical_digest(config),
        "training_argv": command,
        "argv_sha256": engine.canonical_digest(command),
        "output_relative": OUTPUT,
        "log_relative": "output/logs/engineering_v8b/ce_e_1000_b128_w2.training.log",
    }
    del plan["campaign_relative"]
    return plan


def require_cleanup_grace():
    if (
        getattr(engine, "PROCESS_GROUP_EXIT_GRACE_SECONDS", None) != 15.0
        or getattr(engine, "PROCESS_GROUP_EXIT_POLL_SECONDS", None) != 0.25
        or not callable(getattr(engine, "wait_for_process_group_exit", None))
    ):
        raise RuntimeError(
            "V8b requires the reviewed bounded process-group cleanup grace"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, choices=(2,), default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    engine.benchmark._require_project_virtual_environment()
    if Path(sys.prefix).resolve() != (engine.PROJECT_ROOT / ".venv").resolve():
        raise RuntimeError("controller requires the project virtual environment")
    source = engine.benchmark._require_clean_pushed_source()
    plan = build_plan(args.gpu_count)
    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            PROTOCOL,
            CONFIG,
            original.PROTOCOL,
            *[reference["relative_path"] for reference in EVIDENCE.values()],
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    engine.verify_checkpoint_input(plan)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "source": source,
                    "plan": plan,
                    "gpu_queries": 0,
                    "artifact_mutations": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if ROOT != engine.CANONICAL_ROOT:
        raise RuntimeError(
            "merge into the canonical artifact worktree before execution"
        )
    require_cleanup_grace()
    engine.benchmark._require_tmux_for_execution()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES on the controller")
    return engine.execute(plan, source, plan_builder=build_plan)


if __name__ == "__main__":
    raise SystemExit(main())
