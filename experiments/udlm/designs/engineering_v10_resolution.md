# Prospective V10 predictor-resolution comparison

This engineering design is fixed after reading the complete, independently
rescored V9 report and before any V10 generation. The V9 report contains
eight runs and 800 requests; its SHA-256 is
`b880c8bbad2e5e1d5f3639ba8f6848da934c87c2ec44d7e9c466f46a941c7fb6`.
V5/V6 and V9 outcomes informed the fixed temperature 0.5. This is an adaptive
engineering follow-up, not an independent confirmation or a new training run.

## Question and scientific rationale

Does increasing plain predictor resolution from 128 to 512 evaluations improve
generation for the frozen CT and CE checkpoints at temperature 0.5?

V9 repaired quality was 0.465 for CT and 0.410 for CE at temperature 1.0;
at temperature 0.5 it was 0.545 and 0.525. Both repaired CE-minus-CT contrasts
were negative. Strict quality instead changed from CT/CE 0.210/0.260 at
temperature 1.0 to 0.440/0.395 at 0.5, illustrating why the decoding branches
must remain separate. At temperature 0.5, strict validity was 0.770 for CT
and 0.670 for CE. Neither objective closed the contextual MDLM quality gap.
Both checkpoints remain in V10; neither is substituted or retrained.

The [molecular audit](../../../docs/udlm_molecular_failure_audit_20260907.md)
found ring/branch imbalance in 1,009 of 1,220 strict failures in the heterogeneous
V5/V6 diagnostic pool. That motivates checking sequence consistency without
changing the representation or backbone. The independently reviewed
[finite-state oracle](../../../docs/udlm_predictor_resolution_audit.md), added
at revision `c0810c1`, isolates finite-step factorization: sampling exact
coordinate reverse marginals independently can lose joint sequence dependence.
In that nine-state, untempered oracle, 128-to-512 steps reduced endpoint total
variation error by about fourfold. This does not establish that molecular
errors have the same cause or that a learned, temperature-0.5 model improves
with more steps. The experiment tests that hypothesis directly.

## Fixed panel

| Setting | Value |
| --- | --- |
| Checkpoints | Separately audited V8 CT and completed V8b CE, both step 1000 |
| Inference weights | EMA |
| Configurations | CT/CE crossed with 128/512 predictor evaluations |
| Temperature | 0.5 for every configuration; informed by V5/V6/V9 |
| Corrector and nucleus | Corrector disabled; top-p 1.0 |
| Inference endpoint | 1e-5 |
| Other sampling settings | randomness 0; min_add_len 40 |
| Prior and alphabet | Frozen checkpoint empirical prior; full vocabulary |
| Seeds | 1700 and 1701 for every configuration |
| Requests | 100 per seed: four configurations, eight runs, 800 requests |

Each checkpoint keeps exactly its V9 inference settings except `num_steps`.
Raw-LOO CT keeps its canonical implicit parameterization; CE explicitly uses
`x0_denoiser` and converts to LOO before temperature. The 512-step treatment
costs four times the predictor evaluations per molecule. This is not an
equal-compute comparison, and wall-clock scaling is measured rather than
assumed. With inference epsilon 1e-5, the last predictor times are approximately
0.00782242 at 128 steps and 0.00196311 at 512; both exceed training epsilon
0.001. Predictions still come from learned models at every step.

No single global NFE describes this panel. Each pinned YAML declares its own
`num_steps`; the protocol's `design.nfe_by_configuration` records the mapping.
The existing launcher binds those configuration bytes and validates observed
UDLM NFE against the loaded `num_steps`. No sampler or launcher behavior is
changed by this materialization.

## Predeclared outcomes and contrasts

For each method separately, pair seed 1700 at 512 steps with seed 1700 at 128,
and similarly for 1701. At each decoding branch and metric, define
`delta_s = metric_512,s - metric_128,s`. Report both per-seed differences,
their equal-seed mean and sample standard deviation (denominator `S - 1`,
where `S = 2` is the declared seed count), including zero and negative results.
SD describes seed spread, not a confidence interval. Seed labels do not imply
coupled molecular trajectories or independence of generated observations.

Report validity, uniqueness, quality and diversity for every run under both
strict SAFE decoding and released repair/largest-component selection. Quality
counts first-occurrence unique valid molecules with QED >= 0.6 and SA <= 4,
divided by all requests. Validity also divides by requests; uniqueness divides
by valid molecules. Diversity uses the existing released fingerprint metric
within each seed and is never pooled. Preserve undefined metrics; do not turn
them into zero. A contrast mean/SD is withheld if any declared seed pair lacks
the metric. Keep every failed, pending or invalid slot visible.

Also disclose observed NFE, generation time, raw lexical ring/parenthesis
flags, and repair frequency where available from the preserved raw artifacts.
The main independently rescored report supplies all run outcomes. The two
within-method resolution contrasts are a separately labeled analysis; they
must not be relabeled as the V9 CE-minus-CT objective contrasts.

## Provenance, execution and stopping boundary

The protocol copies the accepted V9 checkpoint and training provenance.
CT SHA-256 is `48986899c401c09cdc1e9e865e773cadc40899b62129420689f89a8e622a9c99`;
CE SHA-256 is `b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1`.
The original V8 CT controller and campaign remain failed; a separate CPU
post-exit audit accepted its checkpoint. V8b CE completed under its own receipts.
Training implementation and resolved common settings match by recorded hashes;
1,000 updates at batch 128 give 128,000 configured exposures per arm, not a
distinct-molecule census. Finite final checkpoint tensors do not establish
that every intermediate update was finite.

The launcher may use at most two dynamically selected GPUs, each strictly
below 10% utilization with at least 30,000 MiB free immediately before launch.
Execution must use a named detached tmux session and preserve logs, raw samples,
token IDs, configuration hashes, seeds, device mapping, runtime and receipts.
This file creates no job. Failed runs are not automatically retried, and
checkpoints are not substituted. Independently rescore every completed raw run.

The local MDLM mean quality 0.858 over three 1,000-request seeds and paper
GenMol V1 0.846 remain contextual comparisons with different training/sample
budgets. Selection across prior settings, two seeds per setting and additional
sampling compute prevent a superiority claim. Final UDLM seeds 0/1/2 remain
reserved. No automatic promotion, further resolution sweep or longer training
follows V10; review all results before choosing another experiment.
