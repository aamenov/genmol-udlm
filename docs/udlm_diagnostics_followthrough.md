# Authorized diagnostic sequence

The user requested execution of the objective, resolution, adaptation-budget,
and confirmation sequence in the side conversation on 2026-09-07. This isolated
checkout preserves the running V9 source and all existing experiment outputs.

## Objective comparison

The existing V9 pipeline remains the owner of its generation and CPU report.
Read its terminal report under `../udlm_genmol_worktree/output/udlm/engineering_v9_reports/complete/`.
Do not relaunch its jobs or change its source. Training loss magnitudes cannot
rank the CT and CE objectives. Report both temperatures and both decoding paths.

## Frozen-checkpoint resolution diagnostic

The new protocol is `experiments/udlm/protocols/followthrough_resolution.json`.
Both completed CT and CE checkpoints are crossed with 128 and 512 predictor
evaluations. Temperature is fixed at 1.0, no corrector or nucleus truncation is
used, and the empirical prior, checkpoint bytes, and all other sampling settings
are unchanged. Each configuration requests 100 molecules for each of fresh
engineering seeds 17300/17301: eight runs and 800 requests.

Testing both objectives avoids selecting a checkpoint from the small V9 pilot.
The primary endpoint is strict quality (unique valid QED>=0.6, SA<=4 molecules
divided by requests). Report paired 512-minus-128 differences separately for CT
and CE, with strict/repaired validity, quality, uniqueness, diversity, lexical
ring/parenthesis errors, and runtime. More steps cost four times the model calls;
an improvement is not an equal-compute advantage. Both seeds come from the same
training checkpoint and do not measure independent training-run variability.

Before generation, preserve the committed source and satisfy the existing
upstream-source check. The audited launcher dynamically selects at most two GPU
UUIDs below 10% utilization with at least 30,000 MiB free, rechecks immediately
before launch, and records active processes. It retains existing leases and
never retries failed attempts automatically. Run in a named tmux session:

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python -u scripts/udlm/launch_exploration.py --protocol experiments/udlm/protocols/followthrough_resolution.json --output-root output/udlm/followthrough_resolution --log-root output/logs/followthrough_resolution
CUDA_VISIBLE_DEVICES='' /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python -u scripts/udlm/report_followthrough_resolution.py
```

The report script independently rescores all raw samples before paired analysis.
It refuses incomplete/duplicated runs and preserves undefined metrics. Its output
directory contains the full report, paired CSV/PDF, and lexical counts in JSON.

## Conditional learning curve

Review the complete resolution comparison before training. If the selected
configuration remains below the MDLM comparator, specify a fresh adaptation
from the same MDLM EMA: 4,000 optimizer updates, global batch 128, checkpoints
at 1k/2k/4k, and a learning-rate horizon fixed at 4k from the first update.
Do not extend a decayed 1k schedule and call it the same learning curve.
Use a new training seed and new engineering evaluation seeds, preserving 0/1/2.

Include an MDLM control with equal additional example exposure and a fresh
optimizer/EMA schedule. The current `initialize_from_mdlm_checkpoint` helper
only permits a UDLM destination; the MDLM control needs a separately reviewed
initialization path before launch. A normal Lightning resume would carry old
optimizer/scheduler state and would not implement the proposed matched control.
Keep exposure matching distinct from FLOP/runtime matching and report both.

## Confirmation and next hypothesis

Freeze the selected model and sampler before held-out generation. Match the
baseline's three 1,000-request runs and report the same strict and repaired
definitions, uncertainty across seeds, diversity, runtime, and total training
exposure. A broad GenMol claim additionally needs the corresponding optimization
benchmarks with matched oracle budgets; de novo quality alone cannot establish it.

Mixed masking/replacement adaptation is a new hypothesis only if the preceding
diagnostics fail to close the gap. It is not part of this resolution protocol.
