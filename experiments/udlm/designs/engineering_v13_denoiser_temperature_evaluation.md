# V13: fixed temperature-space comparison on both CE checkpoints

Design fixed on 2026-09-07 while V12 generation was in progress, before reading
any MASK-rich molecular outcome. The algebra in
`docs/udlm_denoiser_temperature_hypothesis.md` motivates this new inference
hypothesis. Historical V5/V6/V9/V10 results informed temperature 0.5; an
unfinished V12 empirical T1 log was incidentally viewed, but no outcome is used
to select a prior, change temperature, or exclude an arm here.

## Fixed panel

| Configuration | Frozen checkpoint | Temperature space |
|---|---|---|
| `e_ce_raw_t050` | V8b empirical CE | `raw_loo` (existing convention) |
| `e_ce_clean_t050` | V8b empirical CE | `x0_denoiser` (new hypothesis) |
| `mask_ce_raw_t050` | V11b MASK-rich CE | `raw_loo` (existing convention) |
| `mask_ce_clean_t050` | V11b MASK-rich CE | `x0_denoiser` (new hypothesis) |

The V8b checkpoint SHA-256 is
`b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1`.
The V11b checkpoint SHA-256 is
`62299de99d8c003e3776215efce9643cca196091a6284f351de2b404a22c2e8d`.
Both use their accepted EMA weights and original trained prior, with 1,000
adaptation updates and 128,000 configured exposures from the same MDLM 50k EMA.
This is an inference ablation, not additional training or a prior swap.

Every configuration uses temperature **0.5**, **128 predictor evaluations**,
top-p **1**, **no Gibbs corrector**, inference epsilon **1e-5**, full active
vocabulary, `min_add_len=40`, and unused UDLM `randomness=0`. Fresh engineering
seeds **2100 and 2101** each request **100** molecules: four configurations,
eight runs, 800 requests and 102,400 molecule-level backbone evaluations.
Same numerical seeds pair arms but do not make stochastic trajectories equal.
Temperature-one equivalence is a CPU correctness check, not another selected
molecular arm. Neither model is selected from V12 for this panel.

## Required contrasts and reporting

Compute `clean-temperature minus raw-LOO-temperature` separately for the
empirical checkpoint (primary contrast) and MASK-rich checkpoint (secondary
contrast), retaining every seed pair. The primary metric is released-comparable
quality: distinct repaired valid molecules meeting QED >= 0.6 and SA <= 4,
divided by requests. Strict quality and all strict/repaired validity,
uniqueness and diversity measurements are mandatory companion endpoints.
Show individual signed seed differences, their mean and sample standard
deviation; two seeds do not support a robust uncertainty or superiority claim.
Missing, invalid or undefined pairs must remain visible and withhold their
aggregate rather than becoming zero or being dropped.

Record exact input/output hashes, source, EMA/checkpoint/prior identities,
all controls, sample counts, seeds, dynamically selected physical GPU UUIDs,
launch probes, runtime and independent rescoring. Count final MASK and other
control-token occurrences from the saved token audit over editable positions;
decoding to text removes special tokens and cannot measure this quantity.
Retain exact request denominators and both strict and repaired outputs.

## Execution boundary

Finish and preserve V12 before merging or launching this feature in its frozen
source checkout. First validate the new temperature ordering, historical
default identity, normalized configuration and independent-rescore bindings.
Then materialize all four concrete configs and the checkpoint-bound protocol,
review, commit, push and run the CPU dry-run before starting the panel once.
Use a new V13 output namespace, a named tmux session and logs under
`output/logs/`. Dynamically use at most two GPUs, each strictly below 10%
utilization and with sufficient free memory immediately before launch.
Preserve all failures; no automatic retry, replacement seed, cherry-picked
subset, final-seed evaluation or automatic final promotion belongs to this
pilot. Final seeds 0/1/2 remain reserved for a later frozen confirmation design.

No benchmark benefit is claimed by this design. Temperature changes neither
the learned checkpoint nor its calibration, and may increase retention of
incorrect tokens or reduce diversity. The local MDLM quality reference is
85.8% over three 1,000-request seeds; this smaller adaptive pilot cannot by
itself establish superiority, and broader GenMol claims need optimization
benchmarks with matched oracle budgets.
