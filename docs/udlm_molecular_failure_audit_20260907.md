# Molecular failure audit: frozen V5 and V6 artifacts

This is a CPU-only diagnosis of existing samples. It does not train, generate,
load checkpoints, change decoding, or promote a model. Counts pooled across
18 configurations and 36 runs are a **heterogeneous diagnostic census**, not
a single-model benchmark estimate. Each configuration has only two 64-sample
seeds. Molecular observations and settings are not assumed independent.

The reproducible script is
[`audit_molecular_failures.py`](../scripts/udlm/audit_molecular_failures.py).
Its [JSON evidence](../experiments/udlm/diagnostics/v5_v6_failure_modes_20260907/audit.json)
records definitions, input SHA-256 values, row counts, per-seed counts,
RDKit version, and training budgets. The
[configuration CSV](../experiments/udlm/diagnostics/v5_v6_failure_modes_20260907/configurations.csv)
contains all 18 settings. The upstream reports and every raw CSV/summary are
checked against their frozen hashes before aggregation.

## Structural consistency dominates strict failures

The audit removes bracket-atom substrings so isotope, charge and atom-map
digits cannot be mistaken for ring labels. It recognizes single-digit, `%NN`
and `%(integer)` ring labels, allowing even-count label reuse. It flags odd
label counts and negative-prefix or nonzero-final parenthesis balance. These
necessary lexical checks are not a complete chemical validator.

| Diagnostic | V5 (1,536 rows) | V6 (768 rows) | Combined (2,304 rows) |
| --- | ---: | ---: | ---: |
| Strictly valid | 631 | 453 | 1,084 |
| Repaired valid | 1,480 | 755 | 2,235 |
| Odd ring-label count | 617 | 198 | 815 |
| Unbalanced parentheses | 266 | 92 | 358 |
| Both flags | 135 | 29 | 164 |
| Either flag | 748 | 261 | 1,009 |
| Recovered strict failure | 849 | 302 | 1,151 |
| Largest component selected | 657 | 247 | 904 |

All 1,009 flagged rows fail strict decoding. They account for **82.70% of the
1,220 strict failures**. Odd ring counts alone occur in 815 failures; the two
checks overlap, so their raw counts must not be added. Direct RDKit parsing
of the remaining 211 unflagged raw SAFE strings fails with 137 kekulization,
47 valence, 14 duplicate-bond, three non-ring aromatic, and ten other parse
errors. This auxiliary error classification does not redefine the upstream
strict SAFE metric.

The historical local MDLM comparator has only 28 lexical flags among 3,000
requests: 27 ring and one parenthesis flag. It has 2,964 strict-valid rows.
This large descriptive difference does not control training or sampling compute.

All upstream final editable-position control counts are zero for BOS, EOS,
PAD, MASK and UNK across **114,066 editable tokens**. Mean editable length is
49.5078125 tokens. This checks final states only; intermediate corruption can
contain control IDs. Excluding those IDs would not directly explain the saved
final syntax failures.

## Repair does not recover the desired quality distribution

After released repair and within-seed deduplication, 2,215 molecules remain;
1,029 pass QED >= 0.6 and SA <= 4. Quality divides this passing count by
requests, not valid molecules. QED summarizes drug-likeness; smaller SA means
easier synthetic accessibility. Among the 1,186 unique quality failures:

- 1,043 fail QED alone;
- 62 fail SA alone;
- 81 fail both.

Thus 1,124/1,186 = **94.77%** of quality failures involve low QED. Repairing
syntax alone cannot be assumed to close the quality gap. The selected repaired
molecules recovered from strict failures have median molecular weight 188.27
and 13 heavy atoms, versus 344.4135 and 24 heavy atoms for the selected outputs
that were already strict-valid. Component removal is consistent with these
smaller fragments, but **this size association is not a causal estimate**:
the groups have different generated structures and are not paired counterfactuals.

The highest repaired-quality configuration, V6 S-Gibbs, has 73/128 repaired
quality and 47/128 strict quality. Among its 123 repaired unique molecules,
48 fail QED and two fail SA. Its repaired quality remains 57.03125% versus
85.8% for the local MDLM three-seed comparator. Strict and repaired quality
must not be compared as though they used the same decoding rule.

## What V5/V6 say about sampling

V5's pooled lexical flags rise from 132/384 at temperature 0.50 to 249/384 at
temperature 1.00 across the three arms. That is a descriptive temperature
pattern; settings remain distinct experimental configurations.

| V6 arm | Predictor lexical flags /128 | Gibbs lexical flags /128 | Predictor strict-valid /128 | Gibbs strict-valid /128 |
| --- | ---: | ---: | ---: | ---: |
| E | 51 | 50 | 69 | 71 |
| S | 43 | 43 | 79 | 71 |
| R | 34 | 40 | 84 | 79 |

The 64-predictor/64-Gibbs allocation does not demonstrate structural
improvement over 128 predictors. This is a finite two-seed comparison, not a
claim that Gibbs correction cannot help. At roughly 50 editable positions,
64 random-scan selections give about 1.28 selections per position on average,
spread over changing noise levels; they are not repeated full clean-sequence
sweeps. Predictions are learned and tempered, so the exact-conditional Gibbs
stationarity theorem does not guarantee this implementation's sample quality.

## Two bounded hypotheses for the next decision

These are prospective hypotheses; no experiment is launched or selected here.
Review the fixed V9 CT/CE comparison first. The smaller frozen-checkpoint
128-to-512-NFE diagnostic below is the next decision to consider before a
substantial training extension.

1. **Adaptation exposure may be insufficient.** Historical R/S/E each received
   only 1,000 updates at batch 16, or 16,000 additional examples. V8's planned
   batch-128 comparison increases that to 128,000. The pinned official
   [QM9 UDLM recipe](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/scripts/train_qm9_no-guidance.sh)
   uses 25,000 updates at global batch 2,048: 51.2 million examples. Its data,
   32-token DiT model, and from-scratch training differ from this MDLM warm
   start, so that number is context, not a required adaptation budget. If V8
   remains below MDLM and learning is still progressing, a newly specified
   4,000-update/batch-128 curve with a frozen 4,000-update LR horizon and
   checkpoints at 1k/2k/4k is more justified than a broad architecture change.
   Include a matched extra-update MDLM control before a superiority claim.
   A fresh 4k schedule is a new experiment; extending the old 1k schedule at
   its final LR floor is a different intervention.
2. **Finite-step simultaneous edits may contribute.** The official
   [`_ddpm_denoise` kernel](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py)
   samples coordinate posteriors in parallel. Exact coordinate marginals need
   not yield the correlated finite-step sequence posterior, as explained in
   [the local derivation](udlm_denoiser_ce_hypothesis.md) and the factorized
   reverse-model setup in [D3PM](https://arxiv.org/abs/2107.03006). A frozen
   checkpoint/temperature comparison of 128 versus 512 plain predictor NFEs
   directly tests this resolution hypothesis through the existing sampler
   step-count option. With inference epsilon 1e-5, the last 512-step predictor
   time is approximately 0.001963, still above training epsilon 0.001. Record
   ring/parenthesis rates, strict validity, quality, diversity and wall time
   on fresh declared engineering seeds. This costs four times the NFE budget;
   an improvement would not be an equal-compute win or prove the whole gap
   was discretization error.

Both hypotheses preserve the existing representation and backbone. Final
UDLM seeds 0, 1 and 2 remain reserved; V8/V9 outcomes must be reviewed before
specifying a new comparison. The historical MDLM baseline already used those
seed numbers and is not a new tuning run.

## Reproduce without GPU access

Run from this checkout using the workspace virtual environment. Omitting
`--output-directory` prints the complete JSON to stdout without artifact writes.
An output directory must be fresh and inside the declared workspace.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python \
  scripts/udlm/audit_molecular_failures.py \
  --artifact-root /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree \
  --workspace-root /home/aidar.alimbayev/Documents/genmolv2
```

The saved syntax counts are stable integer results. Error-message categories
and molecular descriptors are tied to the recorded RDKit version. Tests cover
ring-label reuse, multi-digit labels, bracket isotope/atom-map digits,
parenthesis prefixes, and overlapping flags.
