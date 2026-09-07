# Prospective V9 objective evaluation design

This design was fixed while V8 CT training was starting, before either V8
checkpoint or any V8 molecular output was available. V5/V6 results and the
V7 throughput pilot were already known. The executable protocol will bind
the actual validated checkpoint hashes after V8 completes; this document
does not authorize substitution of an incomplete or failed checkpoint.

## Question and comparison

Does clean-token cross-entropy (CE), converted to the leave-one-out
parameterization used by the UDLM reverse bridge, improve molecular generation
over the continuous-time (CT) objective under the matched V8 training setup?

Both checkpoints must pass the complete V8 campaign at source
`c8434dda105c5fb2a15bb784d24e0387062c733e`, with protocol SHA-256
`a27875752d25a4a7928401434f9bd90ee6e9d01539586bab70ce3ba75499026e`.
Each arm starts from MDLM50k EMA, receives 1,000 batch-128 updates, and uses
seed1500, the empirical prior with uniform mixture0.0002, FiLM conditioning,
the L1 optimizer schedule, the full active vocabulary, and the same
all-special-token clean-target mask. The common resolved training configuration
has SHA-256 `f373121441fca1aaa2f2657a76d7f7ce66be199cd35637597d0df240e257e89a`.

## Fixed generation settings

| Setting | Value |
| --- | --- |
| Checkpoints | V8 CT and CE, each at optimizer step1000 |
| Inference weights | EMA |
| Configurations | CT/CE crossed with temperatures1.0 and0.5 |
| Primary comparison | Paired CE minus CT at temperature1.0 |
| Secondary comparison | Paired CE minus CT at temperature0.5 |
| Sampling budget | 128 predictor model evaluations per molecule |
| Corrector | Disabled |
| Nucleus threshold | 1.0, no truncation |
| Inference endpoint | 1e-5 |
| Prior and alphabet | Validated checkpoint empirical prior, full vocabulary |
| Other sampler settings | randomness0, min_add_len40 |
| Seeds | 1600 and1601 for every configuration |
| Requests | 100 per seed: 8 runs and800 requests total |

Temperature1.0 preserves the untempered CE-to-LOO conversion and is the
primary objective comparison. Temperature0.5 is an engineering setting
informed by the earlier syntax and Gibbs studies; it is not an independently
selected confirmation setting. Temperature and top-p operate after CE-to-LOO
conversion, so these values do not mean temperature scaling of the CE clean
posterior itself. Equal seeds provide paired comparisons but do not imply
identical model trajectories.

## Evidence and analysis

The executable V9 protocol must include explicit entry parameterizations
(`raw_loo` or `x0_denoiser`), immutable checkpoint/configuration hashes,
the shared prior fingerprint, V8 source/protocol/configuration identities,
and the accepted campaign/arm completion-receipt hashes. Raw-LOO inference
retains the historical canonical omission of its default parameterization;
CE inference explicitly sets `parameterization: x0_denoiser`.

Run the existing exploration launcher in a named detached tmux session, with
at most two dynamically selected GPUs, each strictly below10% utilization
and with at least30,000MiB free immediately before launch. Preserve raw samples,
token IDs, full configuration, seeds, device mapping, runtime and failure
receipts. Do not automatically retry failed runs or substitute checkpoints.

Independently re-decode and rescore every successful raw sample on CPU.
Report validity, uniqueness, quality and diversity for both strict decoding
and the released repair/largest-component path, with per-seed values and
equal-seed means/sample standard deviations. Quality is the number of unique
valid molecules satisfying QED>=0.6 and SA<=4 divided by all requested samples.
Report the two per-seed CE-minus-CT differences at each temperature, even
when negative. Report all four configurations and every failure; do not choose
a checkpoint by comparing CT and CE training-loss magnitudes.

The existing MDLM mean quality85.8% (three seeds of1,000) and paper GenMolV1
quality84.6% are contextual comparators. Sample counts, full training budgets
and study purposes differ. This small continuation study can diagnose an
objective effect, but cannot establish superiority over either comparator.
Final UDLM seeds0/1/2 remain reserved. No automatic candidate promotion or
further training follows from this design.
