"""Publish only the frozen V12 protocol after both CE checkpoints are accepted.

This CPU preparer never trains or generates. V11b's completed terminal digest
must be supplied after completion; no checkpoint hash is inferred in advance.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
CANONICAL_ROOT = PROJECT / "run_sources/udlm_genmol_worktree"
OUTPUT = "experiments/udlm/protocols/engineering_v12_mask_prior.json"
DESIGN = "experiments/udlm/designs/engineering_v12_mask_prior_evaluation.md"
DESIGN_SHA = "e1079ee3e8e63fd1328917a841483080d09f49859d0035d7ff1225ba33f5edcc"
TEMPLATE = "experiments/udlm/protocols/engineering_v9_objectives.json"
TEMPLATE_SHA = "718893198b0e08a2405e33177c64989499914207ff39b6880edbc5eff2b35c7d"
MDLM_SHA = "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
CONTROL_CP_SHA = "b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1"
CONTROL_TERMINAL_SHA = (
    "2dd0423257e8908548b69a28b8aa24b3cc956507944c343e24ef615253af2205"
)
CONFIG_SHAS = {
    "e_ce_t100": "b5795d93570490d52948fae8cbb3732d461d6a4bcb29703638634e22c277fbca",
    "mask_ce_t100": "1111875ba56c8fde4a3cbbfe92ffed1fec907a146456af2b38e12d40bd479439",
    "e_ce_t050": "3c37178b971c7927bf19a8aa482e1713542426a151dcf581a2eb4ea09ae43e78",
    "mask_ce_t050": "2cac4cc6569868c493a5d835b0a5ab5020583afa6c4e2463ef1354c7f88999c9",
}
ARMS = {
    "E_CE": {
        "directory": "output/udlm/engineering_v8b/ce_e_1000_b128_w2",
        "source": "b84f21ba1f30ceb62d2626daf2caad8adb6534cf",
        "protocol": "experiments/udlm/protocols/engineering_v8b_ce_followup.json",
        "protocol_sha256": "acb286ac4b938aa049327740affd2dea5beb0ded745cd51dca627f5e8fd9de8b",
        "prior_variant": "empirical_frequency",
        "prior_metadata_sha256": "f738b8b17de5c4704058018bbddacd7fed779c85248e33d199b68a648151e612",
    },
    "MASK_CE": {
        "directory": "output/udlm/engineering_v11b/mask_ce_1000_b128_w2",
        "source": "be18a3244a717978e027d6e43ba0ac20b8137e4a",
        "protocol": "experiments/udlm/protocols/engineering_v11b_mask_prior.json",
        "protocol_sha256": "5ecf638fa8fc7a3707a0497c8a358697610f52c8af7f2d6468458890e26368eb",
        "prior_variant": "mask_rich_empirical",
        "prior_metadata_sha256": "4e1febe2684beebdbf3d4c86aa01bc24ed1d3941af67b63473979e2ae810ee45",
    },
}
FAILURE = "output/udlm/engineering_v11/mask_ce_1000_b128_w2/terminal_manifest.json"
FAILURE_SHA = "6a4aa47e1f7b76ad5efc3ce03cd3e4a55c0db4d95778b0cf60cc55cab7207408"
for directory in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(directory))

from scripts import artifact_io  # noqa: E402
from scripts.exps.denovo import benchmark  # noqa: E402
from scripts.udlm import launch_engineering_training as engine  # noqa: E402
from scripts.udlm import report_exploration as reporter  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return engine.canonical_digest(value)


def read_record(root, relative, expected):
    require(
        isinstance(expected, str)
        and len(expected) == 64
        and all(c in "0123456789abcdef" for c in expected),
        "a concrete lowercase SHA-256 is required",
    )
    claim, payload = artifact_io.snapshot_file(root, relative, capture_bytes=True)
    require(claim.sha256 == expected, f"digest mismatch: {relative}")
    return payload, {
        "relative_path": relative,
        "sha256": claim.sha256,
        "size_bytes": claim.size_bytes,
    }


def completion_record(root, arm, terminal_sha):
    spec = ARMS[arm]
    payload, terminal_ref = read_record(
        root, spec["directory"] + "/terminal_manifest.json", terminal_sha
    )
    terminal = json.loads(payload)
    require(
        terminal.get("status") == "completed"
        and type(terminal.get("training_return_code")) is int
        and terminal["training_return_code"] == 0
        and terminal.get("leases_release_authorized") is True
        and type(terminal.get("completed_example_exposures")) is int
        and terminal.get("completed_example_exposures") == 128000
        and type(terminal.get("training_pid")) is int
        and terminal["training_pid"] > 0
        and (terminal.get("process_group_exit_grace") or {}).get("group_present_at_end")
        is False,
        f"{arm} lacks successful training completion",
    )
    references = {"terminal_receipt": terminal_ref}
    for name, field in (("request", "request_sha256"), ("launch", "launch_sha256")):
        data, reference = read_record(
            root, spec["directory"] + f"/{name}_manifest.json", terminal[field]
        )
        record = json.loads(data)
        require(record["plan"] == terminal["plan"], f"{arm} {name} plan differs")
        require(record["source"] == terminal["source"], f"{arm} {name} source differs")
        if name == "launch":
            require(
                record["input_checkpoint"]["sha256"] == MDLM_SHA, "wrong initialization"
            )
        references[name + "_receipt"] = reference
    data, references["protocol"] = read_record(
        root, spec["protocol"], spec["protocol_sha256"]
    )
    plan = terminal["plan"]
    require(
        terminal["source"] == {"head": spec["source"], "upstream": spec["source"]}
        and plan["protocol_sha256"] == spec["protocol_sha256"]
        and plan["protocol"] == json.loads(data)
        and plan["config_sha256"] == digest(plan["config"])
        and plan["checkpoint_sha256"] == MDLM_SHA,
        f"{arm} source, protocol, config or initialization differs",
    )
    checkpoint = terminal["checkpoint"]
    require(isinstance(checkpoint, dict), f"{arm} has no checkpoint receipt")
    require(
        checkpoint["relative_path"] == spec["directory"] + "/checkpoints/1000.ckpt"
        and type(checkpoint["global_step"]) is int
        and checkpoint["global_step"] == 1000
        and type(checkpoint["size_bytes"]) is int
        and checkpoint["size_bytes"] > 0,
        f"{arm} checkpoint receipt differs",
    )
    if arm == "E_CE":
        require(
            checkpoint["sha256"] == CONTROL_CP_SHA,
            "historical control checkpoint changed",
        )
    else:
        require(
            checkpoint.get("udlm_prior_metadata_sha256")
            == spec["prior_metadata_sha256"],
            "V11b terminal prior identity differs",
        )
    return terminal, references


def inspect_checkpoint(root, terminal, arm):
    """Reuse both independent prior/CE inspection and CPU completion validation."""
    spec, expected = ARMS[arm], terminal["checkpoint"]
    path = root / expected["relative_path"]
    metadata = benchmark.checkpoint_metadata(path, expected_sha256=expected["sha256"])
    # The existing synchronous validator scopes FileClaims through its module
    # ROOT. Restore that scope even on failure; no concurrent work runs here.
    previous_root = engine.ROOT
    try:
        engine.ROOT = root
        accepted = engine.validate_checkpoint_output(
            path, terminal["plan"]["config"], expected_steps=1000
        )
    finally:
        engine.ROOT = previous_root
    for key in ("sha256", "size_bytes", "global_step", "finite_tensor_count"):
        require(
            accepted[key] == expected[key], f"{arm} actual checkpoint {key} differs"
        )
    require(
        metadata["sha256"] == expected["sha256"]
        and metadata["size_bytes"] == expected["size_bytes"]
        and metadata["global_step"] == 1000
        and metadata["diffusion_type"] == "udlm"
        and metadata["udlm_inference_eps"] == 1e-5
        and metadata["udlm_exclude_special_tokens"] is False
        and metadata["udlm_prior_variant"] == spec["prior_variant"]
        and metadata["udlm_prior_metadata_sha256"] == spec["prior_metadata_sha256"]
        and metadata.get("udlm_denoiser_metadata") == benchmark.UDLM_DENOISER_METADATA,
        f"{arm} checkpoint CE/prior identity differs",
    )
    return metadata


def build_protocol(*, input_root, config_root=ROOT, v11b_terminal_sha256):
    """Read-only acceptance; publication happens only after every check passes."""
    input_root, config_root = Path(input_root).resolve(), Path(config_root).resolve()
    _, design_ref = read_record(config_root, DESIGN, DESIGN_SHA)
    payload, _ = read_record(config_root, TEMPLATE, TEMPLATE_SHA)
    protocol = json.loads(payload)
    # Check V11b completion before loading either large checkpoint.
    mask, mask_refs = completion_record(input_root, "MASK_CE", v11b_terminal_sha256)
    control, control_refs = completion_record(input_root, "E_CE", CONTROL_TERMINAL_SHA)
    payload, failure_ref = read_record(input_root, FAILURE, FAILURE_SHA)
    failure = json.loads(payload)
    require(
        failure["status"] == "failed" and failure["checkpoint"] is None,
        "original V11 failure changed",
    )
    expected = copy.deepcopy(control["plan"]["config"])
    expected["callback"]["dirpath"] = mask["plan"]["config"]["callback"]["dirpath"]
    expected["training"]["udlm"].update(
        prior_variant="mask_rich_empirical", mask_mixture_weight=0.9
    )
    require(
        mask["plan"]["config"] == expected,
        "MASK training differs beyond prior and namespace",
    )
    common = copy.deepcopy(control["plan"]["config"])
    common["callback"]["dirpath"] = "<arm-output>/checkpoints"
    common["training"]["udlm"].pop("parameterization")
    require(
        digest(common) == protocol["training"]["matched_common_config_sha256"],
        "historical training settings changed",
    )
    for reference in (
        protocol["training"]["original_failed_campaign"],
        protocol["training"]["original_v8_protocol"],
        protocol["training"]["arms"]["CT"]["post_exit_audit"],
        protocol["training"]["arms"]["CT"]["terminal_receipt"],
    ):
        read_record(input_root, reference["relative_path"], reference["sha256"])
    records = {"E_CE": (control, control_refs), "MASK_CE": (mask, mask_refs)}
    metadata = {
        arm: inspect_checkpoint(input_root, record[0], arm)
        for arm, record in records.items()
    }
    entries = []
    for config_id, config_sha in CONFIG_SHAS.items():
        arm = "MASK_CE" if config_id.startswith("mask_") else "E_CE"
        spec, checkpoint = ARMS[arm], records[arm][0]["checkpoint"]
        relative = (
            f"experiments/udlm/protocols/engineering_v12_configs/{config_id}.yaml"
        )
        raw, _ = read_record(config_root, relative, config_sha)
        import yaml

        config = yaml.safe_load(raw)
        temperature = 1.0 if config_id.endswith("100") else 0.5
        expected_config = {
            "model_path": checkpoint["relative_path"],
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
            "prior_variant": spec["prior_variant"],
            "prior_metadata_sha256": spec["prior_metadata_sha256"],
        }
        require(
            config == expected_config, f"frozen sampling settings differ: {config_id}"
        )
        sampling = benchmark.validate_sampling_config(config)
        benchmark.validate_denoiser_sampling_identity(metadata[arm], sampling)
        entries.append(
            {
                "arm_id": arm,
                "config_id": config_id,
                "attempt_id": "v12-" + config_id.replace("_", "-"),
                "candidate_id": ("v11b-mask-ce-" if arm == "MASK_CE" else "v8b-e-ce-")
                + checkpoint["sha256"][:12],
                "checkpoint": checkpoint["relative_path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "config": relative,
                "config_sha256": config_sha,
                "parameterization": "x0_denoiser",
                "prior_variant": spec["prior_variant"],
                "prior_metadata_sha256": spec["prior_metadata_sha256"],
            }
        )
    protocol.update(
        study_id="engineering-v12-ce-mask-prior",
        seeds=[2000, 2001],
        nfe=128,
        num_samples=100,
        entries=entries,
        prospective_design=design_ref,
        purpose="Fixed paired empirical versus MASK-rich CE molecular comparison after successful V11b.",
        output_root="output/udlm/engineering_v12",
        log_root="output/logs/engineering_v12",
        report_root="output/udlm/engineering_v12_reports/complete",
    )
    design = protocol["design"]
    del design["objective_comparison"]
    design["prior_comparison"] = {
        name: {
            "control_config": "e_ce_" + suffix,
            "treatment_config": "mask_ce_" + suffix,
            "temperature": temperature,
        }
        for name, suffix, temperature in (
            ("primary", "t100", 1.0),
            ("secondary", "t050", 0.5),
        )
    }
    design["parameterization_controls"] = (
        "Both CE arms convert clean-denoiser logits to LOO at current state/time before temperature and top-p; no extra model call."
    )
    training = protocol["training"]
    historical_ct = training["arms"]["CT"]
    for key in (
        "prior_variant",
        "prior_metadata_sha256",
        "training_implementation_hashes_sha256",
    ):
        training.pop(key)
    training["original_v8_ct_context"] = historical_ct
    training["original_v11_failure"] = failure_ref
    training["matched_common_config_definition"] = (
        "Empirical-arm configuration after removing parameterization and normalizing "
        "callback.dirpath. The MASK arm matches after restoring empirical_frequency "
        "and removing mask_mixture_weight."
    )
    training["arms"] = {
        arm: {
            **refs,
            "controller_status": "completed",
            "checkpoint_sha256": terminal["checkpoint"]["sha256"],
            "checkpoint_size_bytes": terminal["checkpoint"]["size_bytes"],
            "source_revision": ARMS[arm]["source"],
            "resolved_config_sha256": terminal["plan"]["config_sha256"],
            "prior_variant": ARMS[arm]["prior_variant"],
            "prior_metadata_sha256": ARMS[arm]["prior_metadata_sha256"],
            "udlm_prior_metadata": metadata[arm]["udlm_prior_metadata"],
            "udlm_denoiser_metadata": metadata[arm]["udlm_denoiser_metadata"],
        }
        for arm, (terminal, refs) in records.items()
    }
    protocol["limitations"] = [
        "V9/V10 and the frozen MDLM transfer diagnostic informed this exploratory prior comparison; no superiority or independent confirmation claim.",
        "Both arms use clean CE; changed corruption and reverse laws change reconstruction difficulty. Equal updates/exposures do not imply equal runtime.",
        "Original V11 failed before training and remains failed. V11b is a separate successful fresh attempt; original V8 CT remains failed with a separate checkpoint audit and is context only.",
        "Two seeds of 100 requests are small. Pair by seed, not molecular trajectory. Report every configuration and failure, both decoding branches, all four metrics and signed MASK-minus-empirical contrasts.",
        "Quality counts unique valid QED>=0.6 and SA<=4 molecules divided by all requests. Report recovery and largest-component selection frequencies with denominators.",
        "Disclose final editable MASK counts and their editable-position denominator from summary sampled_token_control_audit; decoded strings skip specials and cannot supply this count.",
        "128000 configured training exposures are not necessarily distinct molecules. Finite final checkpoints do not establish finiteness of every update.",
        "Local MDLM .858 and paper GenMol V1 .846 use different training/sample budgets and are context only; no automatic promotion or extra training. Final seeds0/1/2 remain reserved.",
    ]
    # Check supported prior declarations without inventing completed generation.
    reporter._paired_contrasts(protocol, [])
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--v11b-terminal-sha256", required=True)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Keep the result line machine-readable if imported ML libraries print notices.
    with redirect_stdout(sys.stderr):
        protocol = build_protocol(
            input_root=args.input_root, v11b_terminal_sha256=args.v11b_terminal_sha256
        )
    claim = artifact_io.publish_bytes_exclusive(ROOT, OUTPUT, engine.encode(protocol))
    print(
        json.dumps({"path": str(ROOT / OUTPUT), "sha256": claim.sha256}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
