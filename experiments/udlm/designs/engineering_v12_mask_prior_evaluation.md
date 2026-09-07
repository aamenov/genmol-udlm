# Prospective V12 MASK-prior molecular comparison

This design is fixed after observing V9, V10 and the frozen-MDLM transfer
diagnostic, before observing any molecular output from a trained MASK-rich
model. It is an exploratory prior comparison. The four accompanying inference
YAMLs declare settings now; the executable protocol must wait for a successful
V11b terminal receipt and an independently validated final checkpoint. The
prospective V11b checkpoint path in those YAMLs is not evidence of completion.
The infrastructure amendment below preserves the original V11 failure.

## Question and training comparison

Does adding stationary MASK mass improve molecular generation after otherwise
matched clean-token cross-entropy (CE) adaptation from the same MDLM EMA?

The control is the completed V8b empirical-prior CE model. The treatment is
the prospective V11b CE model with
`pi_mask = 0.9 * delta_MASK + 0.1 * pi_empirical`, where `pi_empirical` includes
the existing 0.0002 uniform smoothing. Both priors have full support on all
1,880 tokenizer IDs. Clean special-token positions remain immutable context
and are excluded from training targets. This changes the forward corruption,
training difficulty and reverse generation law; it does not reproduce the
absorbing MDLM process or isolate an inference-only change.

Both arms start fresh from the original 50,000-update MDLM EMA, use seed 1500,
1,000 optimizer updates, global batch 128 = two GPUs x microbatch 16 x
accumulation four, FiLM conditioning and the same learning-rate schedule.
V11b must match V8b's complete resolved configuration except prior variant,
MASK weight and output namespace. The training sources differ to introduce
the optional prior and its validation; retain both exact revisions. Equal
configured exposures (128,000 per arm) are not necessarily distinct molecules
and do not imply equal stochastic corruptions, difficulty or runtime.

| Identity | Frozen value |
| --- | --- |
| Initial MDLM EMA checkpoint SHA-256 | `8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6` |
| Completed V8b CE checkpoint SHA-256 | `b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1` |
| Completed V8b terminal SHA-256 | `2dd0423257e8908548b69a28b8aa24b3cc956507944c343e24ef615253af2205` |
| V8b empirical prior metadata SHA-256 | `f738b8b17de5c4704058018bbddacd7fed779c85248e33d199b68a648151e612` |
| Expected V11b MASK-rich prior metadata SHA-256 | `4e1febe2684beebdbf3d4c86aa01bc24ed1d3941af67b63473979e2ae810ee45` |
| Pinned frequency artifact SHA-256 | `088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed` |

The original V8 CT controller and campaign remain failed; its checkpoint was
accepted only by a separate post-exit CPU audit. V12 compares two CE arms and
does not use that CT checkpoint. Preserve this history when describing the
study rather than labeling V8 a successful campaign.

## Fixed generation settings

| Setting | Value |
| --- | --- |
| Control / treatment | Completed V8b empirical CE / successfully validated V11b MASK-rich CE |
| Configurations | `e_ce_t100`, `mask_ce_t100`, `e_ce_t050`, `mask_ce_t050` |
| Primary contrast | MASK-rich CE minus empirical CE at temperature 1.0 |
| Secondary contrast | MASK-rich CE minus empirical CE at temperature 0.5 |
| Inference weights | EMA for both 1,000-update checkpoints |
| Parameterization | `x0_denoiser` for both, converted to LOO at current state and time |
| Model-evaluation budget | 128 predictor calls per molecule, zero Gibbs correctors |
| Nucleus threshold | 1.0 |
| Inference endpoint | 1e-5 |
| Other sampler settings | `randomness: 0.0`, `min_add_len: 40` |
| Generation seeds | 2000 and 2001 for every configuration |
| Requests | 100 per seed per configuration; 8 runs and 800 requests total |

Temperature and top-p act after CE-to-LOO conversion. Temperature 1.0 leaves
that conversion untempered and is the primary comparison. Temperature 0.5
is an informed secondary engineering setting. Pair runs by seed, not by
individual generated molecule: changed priors need not follow the same random
trajectory even with equal seeds. Use the same existing length distribution,
metric inputs and decoding implementations for all four configurations.

## Selection history and limits

V9 already showed repaired quality 46.5% CT versus 41.0% CE at temperature
1.0, and 54.5% CT versus 52.5% CE at 0.5. V10 then compared 128/512 predictor
calls at temperature 0.5: repaired quality was 45.5%/45.0% for CT and
46.5%/48.0% for CE. Each setting used two seeds of 100 requests. Four times
the model calls did not close the local MDLM gap; this does not prove
discretization error is irrelevant.

The frozen transfer diagnostic used one fixed MDLM EMA, the first 16 rows
of the previously used validation panel (814 content tokens), MASK weights
0 and 0.9, and times 0.1/0.5/0.9. The MASK-rich corruption yielded lower
clean-token CE in those observations. It changed the corruption difficulty,
used no optimizer or molecular generation, and did not compare trained
models. That observation motivates a transfer hypothesis, not a selected
optimal MASK weight or a molecular-quality prediction. All these observations
precede V12; neither temperature nor the prior was chosen independently of
earlier engineering evidence.

## Checkpoint acceptance, execution and reporting

Only materialize `experiments/udlm/protocols/engineering_v12_mask_prior.json`
after V11b completes successfully with the intended 1,000-update checkpoint.
Bind the actual V11b terminal, checkpoint SHA-256/size, training source,
training protocol/configuration, CE metadata/state marker, EMA evidence and
prior metadata/state marker. Validate the actual prior against the expected
fingerprint above. Recheck the historical control checkpoint bytes, its
successful terminal and its empirical prior identity. Bind this design and
all four config hashes; never insert a placeholder checkpoint hash, evaluate
an incomplete arm or automatically substitute another checkpoint.

Use the existing `launch_exploration.py` and `report_exploration.py` with
fresh `output/udlm/engineering_v12` and
`output/udlm/engineering_v12_reports/complete` namespaces. Preserve source,
controller receipts, raw strings/token IDs, seeds, device mapping, configuration
and runtime. Run in named detached tmux, with at most two dynamically selected
GPUs strictly below 10% utilization and at least 30,000 MiB free each immediately
before launch. Keep existing processes in place. Do not automatically retry
failed runs or perform generation from this design alone.

Independently re-decode and rescore all successful rows on CPU. Report every
configuration and failure, with validity, uniqueness, quality and diversity
under both strict decoding and the released repair/largest-component path.
Quality is the number of unique valid molecules with QED >= 0.6 and SA <= 4,
divided by all requested samples. Retain per-seed values and equal-seed means
and sample standard deviations. Report both MASK-minus-empirical contrasts
for every metric, including negative differences; withhold a contrast summary
unless both declared seed pairs define that metric. Record generation runtime
and disclose recovery/component-selection frequencies with denominators.

The existing `objective_comparison` report feature specifically means CT
versus CE and must not label this prior experiment. Use distinct
`design.prior_comparison` metadata and explicit entry `prior_variant` values
in the eventual protocol. Report the paired prior contrasts with separately
reviewed support for this comparison, or a small fixed V12 supplement derived
from the complete independently rescored report. Preserve existing reports
and use MASK-minus-empirical labels with both arms identified as clean CE.

Local MDLM quality 85.8% (three seeds of 1,000 requests) and paper GenMol V1
quality 84.6% are context, with different adaptation budgets and sample sizes.
This small, adaptively motivated study cannot establish superiority over
either comparator. Final benchmark seeds 0, 1 and 2 remain reserved. There is
no automatic promotion, longer training or further generation after V12.

## Infrastructure amendment before MASK-trained molecular generation

The original design and four configurations were published in commit
`de21ba62f52e7e5e7cff22a55da3d402fa995ba8`. Its checkpoint acceptance
condition could not be fulfilled: V11's final capacity check found only one
GPU satisfying the two-GPU policy at `2026-09-07T11:59:29.914397+00:00`.
This occurred before
any training subprocess was created. There was no training PID, launch
manifest, checkpoint, optimizer update or example exposure. The terminal's
checkpoint, return-code and completed-exposure fields remain null, and both
exact leases were released. Preserve that failed attempt and its empty
training log; do not relabel it as successful training.

The original failed terminal is
`output/udlm/engineering_v11/mask_ce_1000_b128_w2/terminal_manifest.json`,
SHA-256 `6a4aa47e1f7b76ad5efc3ce03cd3e4a55c0db4d95778b0cf60cc55cab7207408`.
Its source revision is `46ec3b0bc0c54208fd546af4b71108a3f1e31015`, and its
training protocol SHA-256 is
`29f46ec0f925eca992c3b9b2ed27386dcfacd6789076efe1840b14319d9d9198`.

V12 will instead use a separately authorized **V11b** attempt in
`output/udlm/engineering_v11b/mask_ce_1000_b128_w2`. It starts fresh from
the same original MDLM EMA with exactly the V11 training settings and a new
output namespace. Its controller may wait for eligible capacity before its
first training subprocess; it must not retry or resume after that subprocess
is created. Bind the new training protocol, actual successful completion
receipt and validated checkpoint before materializing V12, alongside the
unchanged V11 failed receipt above. A failed V11b attempt does not authorize
automatic checkpoint substitution.

Only the treatment checkpoint namespace and required completion evidence
change. Both temperatures and their primary/secondary roles, both priors,
128 predictor calls, all sampler controls, seeds 2000/2001, 100 requests per
seed, all metrics/contrasts and reserved final seeds remain unchanged. No
molecular output from any trained MASK-rich model informed this amendment.
