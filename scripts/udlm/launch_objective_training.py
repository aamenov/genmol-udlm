"""Run the fixed prospective V8 CT-E then CE-E training comparison in tmux.

Both arms start fresh from the common MDLM EMA. A failed arm terminates this
campaign; existing outputs are never resumed or retried. There is no automatic
generation, selection, or further training after these two 1,000-update arms.
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
PROTOCOL = "experiments/udlm/protocols/engineering_v8_objectives.json"
for import_root in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(import_root))

from scripts import artifact_io  # noqa: E402
from scripts.udlm import launch_engineering_training as engine  # noqa: E402


def common_config(config):
    result = copy.deepcopy(config)
    result["training"]["udlm"].pop("parameterization")
    result["callback"]["dirpath"] = "<arm-output>/checkpoints"
    return result


def build_plans(gpu_count, *, root=ROOT):
    """Read and compose both frozen arms on CPU without any artifact mutation."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    engine.validate_count(gpu_count)
    protocol_claim, payload = artifact_io.snapshot_file(
        root, PROTOCOL, capture_bytes=True
    )
    protocol = json.loads(payload)
    fixed = {
        "schema_version": 1,
        "study_id": "engineering-v8-ct-ce-e-1000-b128",
        "seed": 1500,
        "optimizer_updates": 1000,
        "global_batch_size": 128,
        "micro_batch_size": 16,
        "checkpoint": "outputs/paper_v1/checkpoints/50000.ckpt",
        "checkpoint_sha256": "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6",
        "gpu_policy": engine.POLICY,
        "example_exposures_per_arm": 128000,
    }
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise ValueError("the frozen V8 objective protocol changed")
    if [(arm["arm_id"], arm["parameterization"]) for arm in protocol["arms"]] != [
        ("ct", "raw_loo"),
        ("ce", "x0_denoiser"),
    ]:
        raise ValueError("V8 requires exactly the CT then CE arms")
    checkpoint = engine.PROJECT_ROOT / protocol["checkpoint"]
    accumulation = engine.audited.exact_accumulation_steps(128, 16, gpu_count)
    resolvers = {
        "cwd": lambda: str(root),
        "device_count": lambda: gpu_count,
        "eval": lambda value: eval(value, {"__builtins__": {}}, {}),
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver, replace=True)
    campaign = f"output/udlm/engineering_v8/ct_ce_e_1000_b128_w{gpu_count}"
    plans = []
    for arm in protocol["arms"]:
        arm_id = arm["arm_id"]
        config_name = f"udlm_e_objective_{arm_id}"
        if arm["config"] != f"configs/{config_name}.yaml":
            raise ValueError("V8 arm config path differs from its identity")
        config_claim, _ = artifact_io.snapshot_file(root, arm["config"])
        if config_claim.sha256 != arm["config_sha256"]:
            raise ValueError("V8 arm config digest mismatch")
        output = f"{campaign}/{arm_id}"
        overrides = [
            f"trainer.devices={gpu_count}",
            f"trainer.accumulate_grad_batches={accumulation}",
            f"training.init_from_mdlm_checkpoint={checkpoint}",
            f"training.init_from_mdlm_checkpoint_sha256={protocol['checkpoint_sha256']}",
            f"callback.dirpath={root / output / 'checkpoints'}",
            f"hydra.run.dir={root / output / 'hydra'}",
        ]
        with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
            config = OmegaConf.to_container(
                compose(config_name=config_name, overrides=overrides), resolve=True
            )
        udlm = config["training"]["udlm"]
        if (
            config["seed"] != 1500
            or config["data"] != "safe"
            or config["trainer"]["devices"] != gpu_count
            or config["trainer"]["num_nodes"] != 1
            or config["trainer"]["max_steps"] != 1000
            or config["loader"]["global_batch_size"] != 128
            or config["loader"]["batch_size"] != 16
            or config["trainer"]["accumulate_grad_batches"] != accumulation
            or config["callback"]["every_n_train_steps"] != 1000
            or config["training"]["diffusion"] != "udlm"
            or config["training"]["init_from_mdlm_ema"] is not True
            or config["training"]["reseed_after_model_initialization"] is not False
            or udlm["parameterization"] != arm["parameterization"]
            or udlm["mask_all_special_tokens"] is not True
            or udlm["exclude_special_tokens"] is not False
            or udlm["prior_variant"] != "empirical_frequency"
            or udlm["empirical_uniform_mix"] != 0.0002
            or udlm["conditioning_variant"] != "film_adaln"
            or udlm["zero_init_conditioning"] is not False
            or config["optim"]["lr"] != 0.0003
            or config["optim"]["scheduler"]
            != {
                "name": "half_cosine_with_linear_warmup_and_floor",
                "warmup_updates": 50,
                "horizon_updates": 1000,
                "decay_floor_lr": 0.000003,
            }
        ):
            raise ValueError(
                "resolved V8 arm violates the fixed matched training design"
            )
        command = [
            str(engine.PROJECT_ROOT / ".venv/bin/python"),
            "-u",
            str(root / "scripts/train.py"),
            "--config-name",
            config_name,
            *overrides,
        ]
        plans.append(
            {
                "schema_version": 1,
                "kind": "manual_engineering_training",
                "attempt_id": f"{arm_id}_e_1000_b128_w{gpu_count}",
                "arm_id": arm_id,
                "protocol": protocol,
                "protocol_sha256": protocol_claim.sha256,
                "config_source_sha256": config_claim.sha256,
                "config": config,
                "config_sha256": engine.canonical_digest(config),
                "training_argv": command,
                "argv_sha256": engine.canonical_digest(command),
                "gpu_count": gpu_count,
                "campaign_relative": campaign,
                "output_relative": output,
                "log_relative": f"output/logs/engineering_v8/{arm_id}_e_1000_b128_w{gpu_count}.training.log",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": protocol["checkpoint_sha256"],
                "example_exposures": 128000,
            }
        )
    common = common_config(plans[0]["config"])
    if common != common_config(plans[1]["config"]):
        raise ValueError("CT/CE configs differ beyond parameterization and output path")
    for plan in plans:
        plan["matched_common_config_sha256"] = engine.canonical_digest(common)
    return plans


def execute_campaign(plans, source):
    """Exclusive campaign identity; execute both arms sequentially or stop disclosed."""
    campaign = plans[0]["campaign_relative"]
    engine.ensure_directory(ROOT, str(Path(campaign).parent))
    artifact_io.create_directory_exclusive(ROOT, campaign)

    def publish(name, value):
        return artifact_io.publish_bytes_exclusive(
            ROOT, f"{campaign}/{name}", engine.encode(value)
        )

    request = {
        "schema_version": 1,
        "source": source,
        "plans": plans,
        "created_at": engine.stamp(),
    }
    request_claim = publish("campaign_manifest.json", request)
    terminal = {
        "schema_version": 1,
        "status": "failed",
        "campaign_sha256": request_claim.sha256,
        "source": source,
        "arms": [],
        "started_at": engine.stamp(),
    }
    try:
        for index, plan in enumerate(plans):
            if engine.benchmark._require_clean_pushed_source() != source:
                raise RuntimeError("source changed between objective arms")

            # This builder revalidates both config hashes and their matching contract.
            def plan_builder(count, arm_index=index):
                return build_plans(count)[arm_index]

            result = engine.execute(plan, source, plan_builder=plan_builder)
            receipt_path = f"{plan['output_relative']}/terminal_manifest.json"
            receipt_claim, payload = artifact_io.snapshot_file(
                ROOT, receipt_path, capture_bytes=True
            )
            receipt = json.loads(payload)
            accepted = (
                result == 0
                and receipt.get("status") == "completed"
                and receipt.get("plan") == plan
                and receipt.get("source") == source
                and receipt.get("completed_example_exposures") == 128000
                and receipt.get("training_return_code") == 0
                and receipt.get("leases_release_authorized") is True
                and not any(
                    (ROOT / relative).exists() for relative in engine.LEASE_PATHS
                )
            )
            terminal["arms"].append(
                {
                    "arm_id": plan["arm_id"],
                    "status": "completed" if accepted else "failed",
                    "controller_return_code": result,
                    "terminal_relative": receipt_path,
                    "terminal_sha256": receipt_claim.sha256,
                }
            )
            if not accepted:
                terminal["stop_reason"] = (
                    "failed arm; later arms not started; no automatic retry"
                )
                break
        else:
            terminal["status"] = "completed"
    except BaseException as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
    finally:
        attempted = {arm["arm_id"] for arm in terminal["arms"]}
        for plan in plans:
            if plan["arm_id"] not in attempted:
                terminal["arms"].append(
                    {
                        "arm_id": plan["arm_id"],
                        "status": "not_completed_after_campaign_stop",
                    }
                )
        terminal["finished_at"] = engine.stamp()
        publish("campaign_terminal.json", terminal)
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "campaign_terminal": f"{campaign}/campaign_terminal.json",
            }
        ),
        flush=True,
    )
    return 0 if terminal["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    engine.benchmark._require_project_virtual_environment()
    if Path(sys.prefix).resolve() != (engine.PROJECT_ROOT / ".venv").resolve():
        raise RuntimeError("controller requires the project virtual environment")
    source = engine.benchmark._require_clean_pushed_source()
    plans = build_plans(args.gpu_count)
    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            PROTOCOL,
            *[p["protocol"]["arms"][i]["config"] for i, p in enumerate(plans)],
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    engine.verify_checkpoint_input(plans[0])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "source": source,
                    "plans": plans,
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
    engine.benchmark._require_tmux_for_execution()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES on the controller")
    return execute_campaign(plans, source)


if __name__ == "__main__":
    raise SystemExit(main())
