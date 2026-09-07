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

The original design required both checkpoints to pass the complete V8
campaign. The disclosed infrastructure amendment below replaces that
controller-success condition; its original training source remains
`c8434dda105c5fb2a15bb784d24e0387062c733e`, with protocol SHA-256
`a27875752d25a4a7928401434f9bd90ee6e9d01539586bab70ce3ba75499026e`.
Each arm starts from MDLM 50k EMA, receives 1,000 batch-128 updates, and uses
seed 1500, the empirical prior with uniform mixture 0.0002, FiLM conditioning,
the L1 optimizer schedule, the full active vocabulary, and the same
all-special-token clean-target mask. The common resolved training configuration
has SHA-256 `f373121441fca1aaa2f2657a76d7f7ce66be199cd35637597d0df240e257e89a`.

## Fixed generation settings

| Setting | Value |
| --- | --- |
| Checkpoints | V8 CT and CE, each at optimizer step 1000 |
| Inference weights | EMA |
| Configurations | CT/CE crossed with temperatures 1.0 and 0.5 |
| Primary comparison | Paired CE minus CT at temperature 1.0 |
| Secondary comparison | Paired CE minus CT at temperature 0.5 |
| Sampling budget | 128 predictor model evaluations per molecule |
| Corrector | Disabled |
| Nucleus threshold | 1.0, no truncation |
| Inference endpoint | 1e-5 |
| Prior and alphabet | Validated checkpoint empirical prior, full vocabulary |
| Other sampler settings | randomness 0, min_add_len 40 |
| Seeds | 1600 and 1601 for every configuration |
| Requests | 100 per seed: 8 runs and 800 requests total |

Temperature 1.0 preserves the untempered CE-to-LOO conversion and is the
primary objective comparison. Temperature 0.5 is an engineering setting
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
at most two dynamically selected GPUs, each strictly below 10% utilization
and with at least 30,000 MiB free immediately before launch. Preserve raw samples,
token IDs, full configuration, seeds, device mapping, runtime and failure
receipts. Do not automatically retry failed runs or substitute checkpoints.

Independently re-decode and rescore every successful raw sample on CPU.
Report validity, uniqueness, quality and diversity for both strict decoding
and the released repair/largest-component path, with per-seed values and
equal-seed means/sample standard deviations. Quality is the number of unique
valid molecules satisfying QED >= 0.6 and SA <= 4 divided by all requested samples.
Report the two per-seed CE-minus-CT differences at each temperature, even
when negative. Report all four configurations and every failure; do not choose
a checkpoint by comparing CT and CE training-loss magnitudes.

The existing MDLM mean quality 85.8% (three seeds of 1,000) and paper GenMol V1
quality 84.6% are contextual comparators. Sample counts, full training budgets
and study purposes differ. This small continuation study can diagnose an
objective effect, but cannot establish superiority over either comparator.
Final UDLM seeds 0/1/2 remain reserved. No automatic candidate promotion or
further training follows from this design.


## Infrastructure amendment before CE training or molecular evaluation

V8 CT reached step 1,000 and returned zero, but the controller failed its
immediate process-group cleanup check. CE did not start. A separate CPU audit
subsequently accepted the saved CT checkpoint and released its exact retained
leases after verifying that its controller and training group were absent.
The original failed campaign and terminal receipts remain unchanged. See
[the incident record](../results/engineering_v8_ct_exit_incident.md).

The comparison will therefore use **separately audited V8 CT plus a new V8b
CE run**. CT checkpoint SHA-256 is
`48986899c401c09cdc1e9e865e773cadc40899b62129420689f89a8e622a9c99`;
the separate post-exit audit SHA-256 is
`959151c37072431be23b1f5696d7215da5291e0bace40d6f307864a7262f93c2`.
V8b must preserve the original CE initialization, seed, W2 grouping, resolved
training configuration (apart from output path) and training implementation
hashes. It starts fresh from MDLM EMA and has its own launch/completion
receipts. A bounded process-exit grace repairs the controller independently
of the molecular model.

The primary/secondary temperature comparisons, 128 predictor evaluations,
seeds 1600/1601 and all sample counts are unchanged. No molecular output from
either new checkpoint informed this amendment. A failed V8b arm would remain
disclosed and would not authorize automatic substitution or evaluation.
