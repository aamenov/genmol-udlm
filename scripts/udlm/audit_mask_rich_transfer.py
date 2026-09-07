"""CPU-only frozen-MDLM transfer diagnostic, not training or molecular scoring.

Fix lambda=0/.9, times=.1/.5/.9 and the first16 previously frozen validation
rows before observing results. All comparisons use the same frozen MDLM EMA.
Report token groups separately because corruption difficulty changes with prior.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

# SAFE/Transformers may print signature notices while defining imported classes.
# Keep stdout machine-readable and retain those diagnostics on stderr.
with redirect_stdout(sys.stderr):
    import torch
    from genmol.diffusion import ContinuousCategoricalDiffusion
    from genmol.sampler import load_model_from_path

CHECKPOINT = PROJECT / "outputs/paper_v1/checkpoints/50000.ckpt"
CHECKPOINT_SHA = "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
PANEL = ROOT / "experiments/udlm/validation_panel/first_256.json"
PANEL_SHA = "e2493da4f3cb3217b48c78dc2901dc7524d7d90dc3959cffa87a1b6f8a9a7658"
FREQUENCY = ROOT / "experiments/udlm/token_frequency/train_first_10000.json"
FREQUENCY_SHA = "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
TIMES = (0.1, 0.5, 0.9)
MIXTURES = (0.0, 0.9)
SEED = 1900
ROWS = 16


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def read_pinned(path, digest):
    payload = path.read_bytes()
    assert sha(payload) == digest, str(path)
    return json.loads(payload)


def group_summary(selected, nll, correct, gradient_norm):
    count = int(selected.sum())
    return {
        "tokens": count,
        "ce_sum": float(nll[selected].sum()),
        "ce_mean": float(nll[selected].mean()) if count else None,
        "correct_top1": int(correct[selected].sum()),
        "top1_accuracy": float(correct[selected].double().mean()) if count else None,
        "mean_unreduced_ce_logit_gradient_l2": (
            float(gradient_norm[selected].mean()) if count else None
        ),
    }


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    panel = read_pinned(PANEL, PANEL_SHA)
    frequencies = read_pinned(FREQUENCY, FREQUENCY_SHA)
    rows = panel["rows"][:ROWS]
    special = panel["tokenizer"]["special_token_ids"]
    pad = panel["tokenizer"]["pad_token_id"]
    lengths = [len(row["input_ids"]) for row in rows]
    x0 = torch.full((ROWS, max(lengths)), pad, dtype=torch.long, device="cpu")
    for i, row in enumerate(rows):
        x0[i, : lengths[i]] = torch.tensor(row["input_ids"], dtype=torch.long)
    content = torch.ones_like(x0, dtype=torch.bool)
    for token in special:
        content &= x0 != token
    attention_mask = x0 != pad
    counts = torch.tensor(frequencies["counts_by_token_id"], dtype=torch.float64)
    base = (1 - 0.0002) * counts / counts.sum() + 0.0002 / len(counts)
    base /= base.sum()
    assert len(counts) == 1880 and (base > 0).all()
    model = load_model_from_path(CHECKPOINT, CHECKPOINT_SHA, require_ema=True)
    model.eval()
    assert model.device.type == "cpu" and model.diffusion_type == "mdlm"
    mask_id = model.tokenizer.mask_token_id
    assert mask_id in special and counts[mask_id] == 0
    outputs = []
    with torch.no_grad():
        for mixture in MIXTURES:
            prior = (1 - mixture) * base
            prior[mask_id] += mixture
            diffusion = ContinuousCategoricalDiffusion(1880, prior, noise_eps=1e-3)
            for time_index, noise_time in enumerate(TIMES):
                # Reset the draw stream within each time across priors. Different
                # categorical laws still produce different corrupted token IDs.
                generator = torch.Generator(device="cpu").manual_seed(SEED + time_index)
                times = torch.full((ROWS,), noise_time, dtype=torch.float32)
                xt = diffusion.forward_process(
                    x0, times, mutable_mask=content, generator=generator
                )
                assert torch.equal(xt[~content], x0[~content])
                logits = model(xt, attention_mask).float()
                assert logits.shape == (*x0.shape, 1880)
                log_probs = logits.log_softmax(-1)
                nll = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
                probabilities = log_probs.exp()
                target_probability = probabilities.gather(-1, x0.unsqueeze(-1)).squeeze(
                    -1
                )
                # Unreduced single-token CE gradient wrt logits is p-one_hot(x0).
                # This is not a gradient norm wrt backbone parameters.
                norm_squared = (
                    probabilities.square().sum(-1) + 1 - 2 * target_probability
                )
                gradient_norm = norm_squared.clamp_min(0).sqrt()
                correct = logits.argmax(-1) == x0
                assert torch.isfinite(nll[content]).all()
                groups = {
                    "all_content": content,
                    "currently_mask": content & (xt == mask_id),
                    "changed_nonmask": content & (xt != x0) & (xt != mask_id),
                    "unchanged": content & (xt == x0),
                }
                summaries = {
                    name: group_summary(selected, nll, correct, gradient_norm)
                    for name, selected in groups.items()
                }
                assert sum(
                    summaries[k]["tokens"] for k in groups if k != "all_content"
                ) == int(content.sum())
                outputs.append(
                    {
                        "mask_mixture_weight": mixture,
                        "time": noise_time,
                        "seed": SEED + time_index,
                        "prior_mask_probability": float(prior[mask_id]),
                        "stationary_probability_float64_bytes_sha256": sha(
                            prior.numpy().tobytes()
                        ),
                        "corrupted_token_ids_int64_bytes_sha256": sha(
                            xt.numpy().tobytes()
                        ),
                        "groups": summaries,
                    }
                )
    source_files = [
        Path(__file__),
        *[
            ROOT / name
            for name in (
                "src/genmol/model.py",
                "src/genmol/backbone.py",
                "src/genmol/diffusion.py",
                "src/genmol/sampler.py",
                "src/genmol/utils/ema.py",
                "src/genmol/utils/utils_data.py",
                "src/genmol/utils/checkpoint_io.py",
            )
        ],
    ]
    result = {
        "schema_version": 1,
        "purpose": "Frozen MDLM EMA token-transfer diagnostic; no optimizer updates or generated molecules",
        "device": "cpu",
        "torch_version": torch.__version__,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {
            str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in source_files
        },
        "checkpoint_sha256": CHECKPOINT_SHA,
        "inference_weights": {"source": "MDLM EMA", "ema_applied": True},
        "panel_sha256": PANEL_SHA,
        "frequency_sha256": FREQUENCY_SHA,
        "selection": "First16 rows of previously used frozen validation panel; no new held-out confirmation",
        "source_row_indices": [row["source_index"] for row in rows],
        "clean_token_ids_int64_bytes_sha256": sha(x0.numpy().tobytes()),
        "shape": list(x0.shape),
        "content_tokens": int(content.sum()),
        "uniform_floor": 0.0002,
        "noise_eps": 0.001,
        "results": outputs,
        "limitations": [
            "All conditions use identical frozen MDLM EMA weights; this does not compare trained models.",
            "Cross-prior CE also changes corruption difficulty and cannot select molecular quality.",
            "Changed-nonmask includes any non-MASK replacement, potentially other special IDs.",
            "Reported gradient norms are unreduced token CE gradients wrt logits, not backbone parameters.",
            "One coupled corruption per time and16 validation rows is a small descriptive diagnostic.",
            "No CE-to-LOO conversion, reverse-chain generation, chemistry metric or optimizer update occurs.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
