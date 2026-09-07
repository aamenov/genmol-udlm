"""Launch one fresh 4k adaptation arm after the audited resolution study.

Uses the existing two-GPU selection, leases, telemetry and terminal checkpoint
validation. A new isolated worktree owns this experiment. No resume or retry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts import artifact_io
from scripts.udlm import launch_engineering_training as engine

PROTOCOL = "experiments/udlm/protocols/followthrough_learning.json"


def build_plan(arm, gpu_count, *, root=ROOT):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    if arm not in ("ct", "ce", "mdlm"):
        raise ValueError("Unknown learning-curve arm")
    engine.validate_count(gpu_count)
    claim, payload = artifact_io.snapshot_file(root, PROTOCOL, capture_bytes=True)
    protocol = json.loads(payload)
    if (protocol["optimizer_updates"] != 4000 or protocol["global_batch_size"] != 128
            or protocol["micro_batch_size"] != 16 or protocol["seed"] != 17400
            or protocol["gpu_policy"] != engine.POLICY):
        raise ValueError("Learning-curve protocol differs from the fixed design")
    accumulation = engine.audited.exact_accumulation_steps(128, 16, gpu_count)
    name = f"followthrough_learning_{arm}"
    config_claim, _ = artifact_io.snapshot_file(root, f"configs/{name}.yaml")
    output = f"output/udlm/followthrough_learning/{arm}_4000_b128_w{gpu_count}"
    checkpoint = engine.PROJECT_ROOT / protocol["checkpoint"]
    overrides = [f"trainer.devices={gpu_count}",
                 f"trainer.accumulate_grad_batches={accumulation}",
                 f"training.init_from_mdlm_checkpoint={checkpoint}",
                 f"training.init_from_mdlm_checkpoint_sha256={protocol['checkpoint_sha256']}",
                 f"callback.dirpath={root / output / 'checkpoints'}",
                 f"hydra.run.dir={root / output / 'hydra'}"]
    for key, resolver in {"cwd": lambda: str(root), "device_count": lambda: gpu_count,
                          "eval": lambda value: eval(value, {"__builtins__": {}}, {}),
                          "div_up": lambda x, y: (x + y - 1) // y}.items():
        OmegaConf.register_new_resolver(key, resolver, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        config = OmegaConf.to_container(compose(config_name=name, overrides=overrides), resolve=True)
    training = config["training"]
    if (config["seed"] != 17400 or config["trainer"]["max_steps"] != 4000
            or config["trainer"]["devices"] != gpu_count
            or config["trainer"]["num_nodes"] != 1
            or config["loader"]["batch_size"] * gpu_count * config["trainer"]["accumulate_grad_batches"] != 128
            or config["loader"]["global_batch_size"] != 128
            or config["optim"]["scheduler"]["horizon_updates"] != 4000
            or config["callback"]["every_n_train_steps"] != 1000
            or training["diffusion"] != ("mdlm" if arm == "mdlm" else "udlm")
            or training["udlm"]["parameterization"] != ("x0_denoiser" if arm == "ce" else "raw_loo")
            or training["init_from_mdlm_ema"] is not True
            or training["reseed_after_model_initialization"] is not True
            or training["pilot_fail_on_nonfinite_loss"] is not True):
        raise ValueError("Resolved training configuration violates the learning-curve design")
    command = [str(engine.PROJECT_ROOT / ".venv/bin/python"), "-u",
               str(root / "scripts/train.py"), "--config-name", name, *overrides]
    return {"schema_version": 1, "kind": "manual_engineering_training", "arm_id": arm,
            "attempt_id": f"{arm}_4000_b128_w{gpu_count}", "protocol": protocol,
            "protocol_sha256": claim.sha256, "config_source_sha256": config_claim.sha256,
            "config": config, "config_sha256": engine.canonical_digest(config),
            "training_argv": command, "argv_sha256": engine.canonical_digest(command),
            "gpu_count": gpu_count, "output_relative": output,
            "log_relative": f"output/logs/followthrough_learning/{arm}_4000_b128_w{gpu_count}.training.log",
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": protocol["checkpoint_sha256"],
            "example_exposures": 512000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("ct", "ce", "mdlm"))
    parser.add_argument("--gpu-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    engine.benchmark._require_project_virtual_environment()
    plan = build_plan(args.arm, args.gpu_count)
    source = engine.benchmark._require_clean_pushed_source()
    engine.verify_checkpoint_input(plan)
    if args.dry_run:
        print(json.dumps({"source": source, "plan": plan, "gpu_queries": 0}, sort_keys=True))
        return 0
    engine.benchmark._require_tmux_for_execution()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Controller GPU selection must be dynamic")
    report_path = ROOT.parent / "udlm_diagnostics_followthrough/output/udlm/followthrough_resolution_r1_reports/complete/report.json"
    report = json.loads(report_path.read_text())
    if (report["status"] != "complete"
            or report["accounting"]["independently_rescored_requests"] != 800
            or len(report.get("resolution_contrasts", [])) != 16):
        raise ValueError("The complete independently rescored resolution comparison is required")
    return engine.execute(plan, source,
                          plan_builder=lambda count: build_plan(args.arm, count))


if __name__ == "__main__":
    raise SystemExit(main())
