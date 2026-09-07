"""Wait for both learning arms, audit checkpoints, then evaluate their curve.

This isolated checkout owns all generated configs and samples. Its prospective
evaluation uses CT at temperature .5 / 128 calls and the native MDLM comparator,
100 requests for each fresh seed 17500/17501 at 1k, 2k and 4k extra updates.
An additional MDLM 50k EMA sample provides a same-size zero-extra-update control.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
LEARNING = ROOT.parent / "udlm_followthrough_learning"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.udlm import launch_engineering_training as engine
from scripts.udlm import launch_exploration as exploration
from scripts.udlm import report_exploration as reports

PROTOCOL = Path("experiments/udlm/protocols/followthrough_curve.json")
CONFIG_DIR = Path("experiments/udlm/protocols/followthrough_curve_configs")
OUTPUT = Path("output/udlm/followthrough_curve")
REPORT_DIR = Path("output/udlm/followthrough_curve_reports/complete")
BASELINE_SHA = "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
PRIOR_SHA = "f738b8b17de5c4704058018bbddacd7fed779c85248e33d199b68a648151e612"


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def training_receipts(learning=LEARNING):
    result = {}
    for arm in ("ct", "mdlm"):
        path = learning / f"output/udlm/followthrough_learning_r2/{arm}_4000_b128_w2/terminal_manifest.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if value["status"] != "completed" or value["training_return_code"] != 0:
            raise RuntimeError(f"Training failed; evaluation will not retry or substitute: {path}")
        if value["completed_example_exposures"] != 512000 or not value["leases_release_authorized"]:
            raise RuntimeError("Training exposure/cleanup evidence is incomplete")
        result[arm] = {"receipt": value, "path": str(path), "sha256": exploration.digest(path)}
    return result if len(result) == 2 else None


def config_for(arm, checkpoint):
    config = {"model_path": checkpoint, "num_samples": 100,
              "diffusion_type": "udlm" if arm == "ct" else "mdlm",
              "softmax_temp": 0.5, "randomness": 0.0 if arm == "ct" else 0.5,
              "min_add_len": 40}
    if arm == "ct":
        config.update(num_steps=128, inference_eps=1e-5, exclude_special_tokens=False,
                      prior_variant="empirical_frequency", prior_metadata_sha256=PRIOR_SHA,
                      raw_loo_top_p=1.0)
    return config


def materialize(receipts):
    """Copy immutable local inputs and audit every intermediate checkpoint on CPU."""
    import yaml
    inputs = ROOT / "output/udlm/followthrough_curve_inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    configurations = ROOT / CONFIG_DIR
    configurations.mkdir(exist_ok=False)
    entries, checkpoints = [], []
    for arm, steps in (("mdlm", 0), *((arm, step) for arm in ("ct", "mdlm")
                                    for step in (1000, 2000, 4000))):
        source = (PROJECT / "outputs/paper_v1/checkpoints/50000.ckpt" if steps == 0 else
                  LEARNING / f"output/udlm/followthrough_learning_r2/{arm}_4000_b128_w2/checkpoints/{steps}.ckpt")
        name = f"{arm}_{steps}"
        target = inputs / f"{name}.ckpt"
        before = exploration.digest(source)
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
        if exploration.digest(source) != before or exploration.digest(target) != before:
            raise RuntimeError("Checkpoint changed during copy")
        audit = {"global_step": 50000, "sha256": before}
        if steps == 0:
            if before != BASELINE_SHA:
                raise RuntimeError("Initial MDLM checkpoint differs from the frozen comparator")
        else:
            audit = engine.validate_checkpoint_output(
                target, receipts[arm]["receipt"]["plan"]["config"], expected_steps=steps)
            if steps == 4000 and before != receipts[arm]["receipt"]["checkpoint"]["sha256"]:
                raise RuntimeError("Final checkpoint differs from the terminal receipt")
        relative = target.relative_to(ROOT).as_posix()
        config = config_for(arm, relative)
        config_path = configurations / f"{name}.yaml"
        with config_path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=True)
        entries.append({"arm_id": arm.upper(), "config_id": name,
                        "attempt_id": f"followthrough-curve-{arm}-{steps}",
                        "candidate_id": f"followthrough-{arm}-{steps}-{before[:12]}",
                        "additional_updates": steps, "checkpoint": relative,
                        "checkpoint_sha256": before, "config": config_path.relative_to(ROOT).as_posix(),
                        "config_sha256": exploration.digest(config_path),
                        "parameterization": "raw_loo" if arm == "ct" else None})
        checkpoints.append({"arm": arm, "additional_updates": steps,
                            "source": str(source), "audit": audit})
    protocol = {"schema_version": 1, "study_id": "followthrough-learning-curve",
                "claim": "engineering_screen_only_no_superiority_claim",
                "seeds": [17500, 17501], "num_samples": 100, "entries": entries,
                "gpu_policy": {"max_gpus": 2, "max_utilization_percent": 10, "min_free_memory_mib": 30000},
                "training": {"seed": 17400, "global_batch_size": 128, "optimizer_updates": 4000,
                             "fresh_scheduler_horizon": 4000, "receipts": receipts,
                             "intermediate_checkpoint_audits": checkpoints},
                "limitations": ["One training seed per method; generation seeds are not independent training runs.",
                                "Same extra example exposure does not imply equal training or sampling compute.",
                                "MDLM uses native confidence sampling, temperature0.5 and randomness0.5; CT uses128 predictor calls.",
                                "Final seeds0/1/2 remain untouched. These pilots cannot establish superiority."]}
    write_json(ROOT / PROTOCOL, protocol)
    return protocol


def render_curve(report, protocol):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output/.matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pypdf import PdfWriter
    report["caveats"] = [
        "New CT and MDLM arms start from the same MDLM50k EMA and reset optimizer/EMA state. "
        "Each completes a fresh4k schedule at batch128; checkpoints at1k/2k/4k provide the curve.",
        "Normal Lightning startup seeding is retained. CT includes additional FiLM parameters. "
        "Matching example exposure does not match FLOPs or runtime.",
        *protocol["limitations"], *reports.CAVEATS[3:]]
    paths = reports.write_report(report, ROOT / REPORT_DIR)
    entries = {entry["config_id"]: entry for entry in protocol["entries"]}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (branch, metric) in zip(axes.flat, (("strict", "quality"), ("released_comparable", "quality"),
                                               ("strict", "validity"), ("released_comparable", "diversity"))):
        for arm in ("CT", "MDLM"):
            values = sorted((entries[a["config_id"]]["additional_updates"], a["metrics"][branch][metric])
                            for a in report["aggregates"] if a["arm_id"] == arm)
            ax.errorbar([x for x, _ in values],
                        [100*v["mean"] if v["mean"] is not None else float("nan") for _, v in values],
                        yerr=[100*v["sample_sd"] if v["sample_sd"] is not None else float("nan") for _, v in values],
                        marker="o", capsize=3, label=arm)
        ax.set_title(f"{branch.replace('released_comparable', 'Repaired').title()} {metric}")
        ax.set_xlabel("Additional optimizer updates (batch128)")
        ax.set_ylabel("Percent" if metric != "diversity" else "Diversity ×100")
        ax.set_ylim(0, 102)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Fresh adaptation curve — two generation seeds,100 requests each\nError bars: sample SD across generation seeds; not a superiority claim")
    chart = ROOT / REPORT_DIR / "learning_curve.pdf"
    fig.savefig(chart)
    plt.close(fig)
    combined = PdfWriter()
    appendices = [chart, Path(paths["pdf"]),
                  ROOT.parent / "udlm_diagnostics_followthrough/output/udlm/followthrough_resolution_r1_reports/complete/paired_contrasts.pdf",
                  ROOT.parent / "udlm_diagnostics_followthrough/output/udlm/followthrough_resolution_r1_reports/complete/report.pdf",
                  ROOT.parent / "udlm_genmol_worktree/output/udlm/engineering_v9_reports/complete/report.pdf",
                  ROOT.parent / "udlm_genmol_worktree/output/udlm/engineering_v10_reports/complete/report.pdf"]
    for path in appendices:
        combined.append(str(path))
    with (ROOT / REPORT_DIR / "followthrough_study.pdf").open("xb") as stream:
        combined.write(stream)
    combined.close()
    return paths


def main():
    engine.benchmark._require_project_virtual_environment()
    engine.benchmark._require_tmux_for_execution()
    source = engine.benchmark._require_clean_pushed_source()
    (ROOT / "output").mkdir(exist_ok=True)
    with (ROOT / "output/.followthrough_curve_controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if engine.benchmark._require_clean_pushed_source() != source:
                raise RuntimeError("Evaluation source changed while waiting")
            receipts = training_receipts()
            if receipts is not None:
                break
            print(json.dumps({"event": "waiting_for_both_learning_arms", "time": engine.stamp()}), flush=True)
            time.sleep(30)
        protocol = materialize(receipts)
        staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
        if staged.strip():
            raise RuntimeError("Existing staged changes retained; refusing an automatic commit")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != source["head"]:
            raise RuntimeError("Source changed during checkpoint materialization")
        status = subprocess.check_output(["git", "status", "--porcelain=v1", "-z"], cwd=ROOT).decode()
        for record in filter(None, status.split("\0")):
            path = record[3:]
            if not (path == str(PROTOCOL) or path.startswith(str(CONFIG_DIR) + "/") or path.startswith("output/")):
                raise RuntimeError(f"Unrelated change retained: {path}")
        subprocess.run(["git", "add", str(PROTOCOL), str(CONFIG_DIR)], cwd=ROOT, check=True)
        subprocess.run(["git", "diff", "--cached", "--check"], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", "Freeze audited learning-curve checkpoints and evaluation configs"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "codex/udlm-followthrough-evaluation"], cwd=ROOT, check=True)
        for entry in protocol["entries"]:
            command = exploration.command_for(entry, protocol, ROOT / OUTPUT, ROOT / "output/logs/followthrough_curve")
            subprocess.run([*command, "--dry-run"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "-u", "scripts/udlm/launch_exploration.py", "--protocol", str(PROTOCOL),
                        "--output-root", str(OUTPUT), "--log-root", "output/logs/followthrough_curve"], cwd=ROOT, check=True)
        report = reports.build_report(PROTOCOL, OUTPUT)
        if report["status"] != "complete" or report["accounting"]["independently_rescored_requests"] != 1400:
            raise RuntimeError("Learning-curve evaluation is incomplete")
        paths = render_curve(report, protocol)
        print(json.dumps({"event": "learning_curve_complete", "artifacts": paths,
                          "superiority_established": False}), flush=True)


if __name__ == "__main__":
    main()
