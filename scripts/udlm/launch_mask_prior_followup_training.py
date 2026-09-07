"""Start the fixed V11b arm after V11 failed before any training child launch.

Training settings and fresh MDLM EMA initialization remain unchanged. Only
the namespace and bounded prelaunch capacity wait differ; no child retries.
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
PROTOCOL = "experiments/udlm/protocols/engineering_v11b_mask_prior.json"
CONFIG = "configs/udlm_mask_objective_ce_followup.yaml"
OUTPUT = "output/udlm/engineering_v11b/mask_ce_1000_b128_w2"
ORIGINAL_SOURCE = "46ec3b0bc0c54208fd546af4b71108a3f1e31015"
ORIGINAL_PROTOCOL_SHA256 = (
    "29f46ec0f925eca992c3b9b2ed27386dcfacd6789076efe1840b14319d9d9198"
)
FAILURE_EVIDENCE = {
    "terminal": {
        "relative_path": "output/udlm/engineering_v11/mask_ce_1000_b128_w2/terminal_manifest.json",
        "sha256": "6a4aa47e1f7b76ad5efc3ce03cd3e4a55c0db4d95778b0cf60cc55cab7207408",
    },
    "request": {
        "relative_path": "output/udlm/engineering_v11/mask_ce_1000_b128_w2/request_manifest.json",
        "sha256": "1cf5a1c0b25a63ee6189aa77b94aad3e14ec25a666acd7eab358ef117f6b5110",
    },
}
WAIT_POLICY = {
    "gpu_availability_wait_seconds": 21600,
    "gpu_availability_poll_seconds": 30,
}
for import_root in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(import_root))

from scripts import artifact_io  # noqa: E402
from scripts.udlm import launch_mask_prior_training as original  # noqa: E402

engine = original.engine


def verify_prelaunch_failure(root, protocol, baseline):
    """Require the pinned V11 failure to contain no attempted training child."""
    if protocol["prelaunch_failure_evidence"] != FAILURE_EVIDENCE:
        raise ValueError("V11b original failure evidence changed")
    records = {}
    for name, reference in FAILURE_EVIDENCE.items():
        claim, payload = artifact_io.snapshot_file(
            root, reference["relative_path"], capture_bytes=True
        )
        if claim.sha256 != reference["sha256"]:
            raise ValueError(f"V11b {name} evidence digest mismatch")
        records[name] = json.loads(payload)
    terminal, request = records["terminal"], records["request"]
    source = {"head": ORIGINAL_SOURCE, "upstream": ORIGINAL_SOURCE}
    if (
        terminal["status"] != "failed"
        or terminal["source"] != source
        or request["source"] != source
        or terminal["request_sha256"] != FAILURE_EVIDENCE["request"]["sha256"]
        or terminal["leases_release_authorized"] is not True
        or any(
            terminal.get(field) is not None
            for field in (
                "training_pid",
                "launch_sha256",
                "training_return_code",
                "checkpoint",
                "completed_example_exposures",
                "end_to_end_training_examples_per_second",
                "process_group_exit_grace",
            )
        )
        or not terminal["error"].startswith(
            "RuntimeError: requested 2 GPU(s), but only 1 are genuinely idle;"
        )
        or terminal["plan"] != request["plan"]
        or request["plan"]["protocol_sha256"] != ORIGINAL_PROTOCOL_SHA256
        or request["plan"]["protocol"] != baseline["protocol"]
        or request["plan"]["config_sha256"]
        != engine.canonical_digest(request["plan"]["config"])
        or original.comparable_config(request["plan"]["config"])
        != original.comparable_config(baseline["config"])
        or request["plan"]["checkpoint_sha256"] != baseline["checkpoint_sha256"]
        or request["plan"]["expected_prior_metadata_sha256"]
        != baseline["expected_prior_metadata_sha256"]
    ):
        raise ValueError("V11b requires the unchanged V11 prelaunch-only failure")


def build_plan(gpu_count, *, root=ROOT):
    """Reuse the reviewed V11 plan and change only namespace and capacity policy."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    baseline = original.build_plan(gpu_count, root=root)
    claim, payload = artifact_io.snapshot_file(root, PROTOCOL, capture_bytes=True)
    protocol = json.loads(payload)
    fixed = {
        "schema_version": 1,
        "study_id": "engineering-v11b-mask-ce-1000-b128-w2",
        "claim": "separate_followup_after_capacity_only_prelaunch_failure",
        "config": CONFIG,
        "original_protocol": original.PROTOCOL,
        "original_protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
        "original_source_revision": ORIGINAL_SOURCE,
        **WAIT_POLICY,
    }
    if baseline["protocol_sha256"] != ORIGINAL_PROTOCOL_SHA256 or any(
        protocol.get(key) != value for key, value in fixed.items()
    ):
        raise ValueError("the fixed V11b follow-up identity changed")
    changed = {
        "study_id",
        "claim",
        "config",
        "config_sha256",
        "purpose",
        "execution_policy",
        "reporting_policy",
        "follow_up",
    }
    if any(
        protocol.get(key) != value
        for key, value in baseline["protocol"].items()
        if key not in changed
    ):
        raise ValueError("V11b changed the original training protocol")
    verify_prelaunch_failure(root, protocol, baseline)
    config_claim, _ = artifact_io.snapshot_file(root, CONFIG)
    if config_claim.sha256 != protocol["config_sha256"]:
        raise ValueError("V11b config digest mismatch")
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
        raise ValueError("V11b differs from V11 beyond its output directory")
    command = [*baseline["training_argv"][:4], Path(CONFIG).stem, *overrides]
    plan = {
        **baseline,
        "attempt_id": "mask_ce_1000_b128_w2_followup",
        "protocol": protocol,
        "protocol_sha256": claim.sha256,
        "config_source_sha256": config_claim.sha256,
        "config": config,
        "config_sha256": engine.canonical_digest(config),
        "training_argv": command,
        "argv_sha256": engine.canonical_digest(command),
        "output_relative": OUTPUT,
        "log_relative": "output/logs/engineering_v11b/mask_ce_1000_b128_w2.training.log",
        **WAIT_POLICY,
    }
    engine.gpu_availability_wait_policy(plan)
    return plan


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
            *[entry["relative_path"] for entry in FAILURE_EVIDENCE.values()],
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    original.require_cleanup_grace()
    if args.dry_run:
        engine.verify_checkpoint_input(plan)
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
    engine.benchmark._require_tmux_for_execution()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES on the controller")
    # The engine verifies input bytes once before the capacity loop, then probes
    # immediately before the only possible child launch. No repeated large hash.
    return engine.execute(plan, source, plan_builder=build_plan)


if __name__ == "__main__":
    raise SystemExit(main())
