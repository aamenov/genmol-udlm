"""Train only the prospective V11 MASK-rich CE arm from fresh MDLM EMA.

Reuse the completed V8b CE training settings with a distinct prior identity
and output namespace. No resume, retry, molecular sampling or automatic next run.
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
PROTOCOL = "experiments/udlm/protocols/engineering_v11_mask_prior.json"
CONFIG = "configs/udlm_mask_objective_ce.yaml"
OUTPUT = "output/udlm/engineering_v11/mask_ce_1000_b128_w2"
CONTROL_SOURCE = "b84f21ba1f30ceb62d2626daf2caad8adb6534cf"
CONTROL_CHECKPOINT_SHA256 = (
    "b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1"
)
ORIGINAL_PROTOCOL_SHA256 = (
    "a27875752d25a4a7928401434f9bd90ee6e9d01539586bab70ce3ba75499026e"
)
COMMON_CONFIG_SHA256 = (
    "f373121441fca1aaa2f2657a76d7f7ce66be199cd35637597d0df240e257e89a"
)
PRIOR = {
    "variant": "mask_rich_empirical",
    "mask_mixture_weight": 0.9,
    "mask_token_id": 4,
    "uniform_mixture_weight": 0.0002,
    "active_vocab_size": 1880,
    "metadata_sha256": "4e1febe2684beebdbf3d4c86aa01bc24ed1d3941af67b63473979e2ae810ee45",
    "stationary_probs_sha256": "07d011c07bdac8fdbe096f514ded6e2d1b6850268c48bec8dd0e3f3853d013eb",
    "base_stationary_probs_sha256": "3b6ce79449e0abc083bc860b3543be554667435ce9f7f1b92c0f960ff27400c3",
    "base_prior_metadata_sha256": "f738b8b17de5c4704058018bbddacd7fed779c85248e33d199b68a648151e612",
    "frequency_artifact": "experiments/udlm/token_frequency/train_first_10000.json",
    "frequency_artifact_sha256": "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed",
}
EVIDENCE = {
    "v8b_ce_terminal": {
        "relative_path": "output/udlm/engineering_v8b/ce_e_1000_b128_w2/terminal_manifest.json",
        "sha256": "2dd0423257e8908548b69a28b8aa24b3cc956507944c343e24ef615253af2205",
    },
    "v8b_protocol": {
        "relative_path": "experiments/udlm/protocols/engineering_v8b_ce_followup.json",
        "sha256": "acb286ac4b938aa049327740affd2dea5beb0ded745cd51dca627f5e8fd9de8b",
    },
    "original_failed_campaign": {
        "relative_path": "output/udlm/engineering_v8/ct_ce_e_1000_b128_w2/campaign_terminal.json",
        "sha256": "98075a11c7340dafcc855887348201477981bd2a1a8f5d39053da24866cc663b",
    },
    "ct_post_exit_audit": {
        "relative_path": "output/udlm/engineering_v8/ct_ce_e_1000_b128_w2/post_exit_audit/post_exit_audit.json",
        "sha256": "959151c37072431be23b1f5696d7215da5291e0bace40d6f307864a7262f93c2",
    },
}
for import_root in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(import_root))

from scripts import artifact_io  # noqa: E402
from scripts.udlm import launch_engineering_training as engine  # noqa: E402
from scripts.udlm import launch_objective_training as original  # noqa: E402


def comparable_config(config):
    result = copy.deepcopy(config)
    result["callback"]["dirpath"] = "<arm-output>/checkpoints"
    return result


def verify_prior_evidence(root, protocol, baseline):
    """Bind the completed historical CE settings and retain the original failure."""
    if protocol["evidence"] != EVIDENCE:
        raise ValueError("V11 evidence differs from the fixed historical records")
    documents = {}
    for name, reference in EVIDENCE.items():
        claim, payload = artifact_io.snapshot_file(
            root, reference["relative_path"], capture_bytes=True
        )
        if claim.sha256 != reference["sha256"]:
            raise ValueError(f"V11 {name} evidence digest mismatch")
        documents[name] = json.loads(payload)
    terminal = documents["v8b_ce_terminal"]
    control = protocol["historical_control"]
    source = {"head": CONTROL_SOURCE, "upstream": CONTROL_SOURCE}
    if (
        terminal["status"] != control["status"]
        or terminal["status"] != "completed"
        or terminal["source"] != source
        or control["source_revision"] != CONTROL_SOURCE
        or terminal["training_return_code"] != 0
        or terminal["leases_release_authorized"] is not True
        or terminal["process_group_exit_grace"]["group_present_at_end"] is not False
        or terminal["completed_example_exposures"] != control["example_exposures"]
        or control["example_exposures"] != 128000
        or terminal["checkpoint"]["global_step"] != control["optimizer_updates"]
        or control["optimizer_updates"] != 1000
        or terminal["checkpoint"]["sha256"] != CONTROL_CHECKPOINT_SHA256
        or any(
            terminal["checkpoint"][key] != control["checkpoint"][key]
            for key in ("relative_path", "sha256", "size_bytes")
        )
        or control["prior_variant"] != "empirical_frequency"
        or control["prior_metadata_sha256"] != PRIOR["base_prior_metadata_sha256"]
        or terminal["plan"]["protocol"]["study_id"] != control["study_id"]
        or terminal["plan"]["protocol_sha256"] != EVIDENCE["v8b_protocol"]["sha256"]
        or terminal["plan"]["config_sha256"]
        != engine.canonical_digest(terminal["plan"]["config"])
        or comparable_config(terminal["plan"]["config"])
        != comparable_config(baseline["config"])
        or documents["original_failed_campaign"]["status"] != "failed"
        or documents["ct_post_exit_audit"]["original_campaign_status"] != "failed"
        or documents["ct_post_exit_audit"]["checkpoint_status"]
        != "separately_validated_after_controller_failure"
    ):
        raise ValueError("V11 historical control or original failure identity changed")
    frequency, _ = artifact_io.snapshot_file(root, PRIOR["frequency_artifact"])
    if frequency.sha256 != PRIOR["frequency_artifact_sha256"]:
        raise ValueError("V11 pinned frequency artifact changed")


def build_plan(gpu_count, *, root=ROOT):
    """Compose the one fixed W2 arm on CPU and revalidate its comparison contract."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    if type(gpu_count) is not int or gpu_count != 2:
        raise ValueError("V11 requires the same two-GPU count as completed V8b CE")
    baseline = original.build_plans(2, root=root)[1]
    claim, payload = artifact_io.snapshot_file(root, PROTOCOL, capture_bytes=True)
    protocol = json.loads(payload)
    fixed = {
        "schema_version": 1,
        "study_id": "engineering-v11-mask-ce-1000-b128-w2",
        "claim": "single_exploratory_prior_training_hypothesis_no_superiority",
        "config": CONFIG,
        "gpu_count": 2,
        "seed": 1500,
        "optimizer_updates": 1000,
        "global_batch_size": 128,
        "micro_batch_size": 16,
        "accumulate_grad_batches": 4,
        "gpu_policy": engine.POLICY,
        "checkpoint": baseline["protocol"]["checkpoint"],
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "example_exposures_per_arm": 128000,
        "memory_measurement": baseline["protocol"]["memory_measurement"],
        "matched_common_config_sha256": COMMON_CONFIG_SHA256,
        "parameterization": "x0_denoiser",
        "exclude_special_tokens": False,
        "mask_all_special_tokens": True,
        "conditioning_variant": "film_adaln",
        "peak_learning_rate": 0.0003,
        "scheduler": baseline["config"]["optim"]["scheduler"],
        "prior": PRIOR,
        "final_benchmark_seeds_reserved": [0, 1, 2],
    }
    if (
        any(protocol.get(key) != value for key, value in fixed.items())
        or baseline["protocol_sha256"] != ORIGINAL_PROTOCOL_SHA256
        or baseline["matched_common_config_sha256"] != COMMON_CONFIG_SHA256
    ):
        raise ValueError("the fixed V11 prior treatment or original CE design changed")
    verify_prior_evidence(root, protocol, baseline)
    config_claim, _ = artifact_io.snapshot_file(root, CONFIG)
    if config_claim.sha256 != protocol["config_sha256"]:
        raise ValueError("V11 config digest mismatch")
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
    expected["training"]["udlm"].update(
        prior_variant=PRIOR["variant"], mask_mixture_weight=PRIOR["mask_mixture_weight"]
    )
    expected["callback"]["dirpath"] = str(root / OUTPUT / "checkpoints")
    if config != expected:
        raise ValueError("V11 differs from completed CE beyond its prior and output")
    command = [*baseline["training_argv"][:4], Path(CONFIG).stem, *overrides]
    plan = {
        **baseline,
        "attempt_id": "mask_ce_1000_b128_w2",
        "arm_id": "mask_ce",
        "protocol": protocol,
        "protocol_sha256": claim.sha256,
        "config_source_sha256": config_claim.sha256,
        "config": config,
        "config_sha256": engine.canonical_digest(config),
        "training_argv": command,
        "argv_sha256": engine.canonical_digest(command),
        "output_relative": OUTPUT,
        "log_relative": "output/logs/engineering_v11/mask_ce_1000_b128_w2.training.log",
        "expected_prior_metadata_sha256": PRIOR["metadata_sha256"],
    }
    del plan["campaign_relative"]
    return plan


def require_cleanup_grace():
    if (
        getattr(engine, "PROCESS_GROUP_EXIT_GRACE_SECONDS", None) != 15.0
        or getattr(engine, "PROCESS_GROUP_EXIT_POLL_SECONDS", None) != 0.25
        or not callable(getattr(engine, "wait_for_process_group_exit", None))
    ):
        raise RuntimeError("V11 requires the reviewed bounded cleanup grace")


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
            PRIOR["frequency_artifact"],
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
