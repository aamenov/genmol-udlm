# V13: changing temperature space gives mixed results

All eight generation runs and all 800 independently rescored requests were
accepted. The pipeline ran from 2026-09-07 13:06:59.343798 to 13:15:46.912253
UTC (527.568 seconds); generation ended at 13:14:13.310451 UTC. Both subprocess
return codes were zero. Clean-denoiser temperature increased empirical-prior
repaired quality by 5.5 percentage points but left strict quality unchanged.
For the MASK-rich checkpoint it decreased repaired quality by 2.0 points and
strict quality by 9.5 points. No uniform benefit, superiority or final promotion
is established. Best V13 quality is 52.0%, below local MDLM 85.8%.

## Fixed comparison

Protocol `experiments/udlm/protocols/engineering_v13_temperature_space.json`
has SHA-256
`daf8c8108772e0ebcbdd39f15dab07478fe8945ba6b33f4b68858a8536ad7319`.
Its design was fixed and pushed as `b6f0358` before reading V12 MASK-rich
outcomes. It includes both existing CE checkpoints, so no model was selected
from V12 for this panel. Historical studies informed temperature 0.5; the
prospective design also discloses an incidentally viewed unfinished V12
empirical T1 log. Generation and independent reporting stayed at clean pushed
source `0fdc54d91baade2f84b48de9223ceb19696fadb7` throughout.

Empirical CE checkpoint:
`b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1`.
MASK-rich CE checkpoint:
`62299de99d8c003e3776215efce9643cca196091a6284f351de2b404a22c2e8d`.
Their accepted EMA weights, trained priors and V8b/V11b evidence are unchanged.
Each received 1,000 batch-128 adaptation updates from the same MDLM 50k EMA;
V13 adds no training. Both use their original full active vocabulary.

Every setting uses temperature 0.5, 128 predictor evaluations, top-p one, no
Gibbs, endpoint 1e-5, min_add_len 40 and unused UDLM randomness zero. Fresh
seeds 2100/2101 each request 100 molecules per configuration: four settings,
eight runs, 800 requests and 102,400 molecule-level backbone evaluations.
Only temperature space differs within a pair. The control converts clean
probabilities D to raw LOO weights D/L before temperature; the treatment applies
temperature to D before conversion and uses bridge temperature one. This is
an inference hypothesis, not a repair to the existing raw-LOO convention.
See `docs/udlm_denoiser_temperature_hypothesis.md` and notebook Stage 27.

## Every configuration

Values are equal-seed means. All per-seed values and sample SDs are retained
in the complete report. Strict uniqueness is 100% for every setting.

| Prior | Temperature space | Repaired validity | Repaired uniqueness | Repaired quality | Repaired diversity | Strict validity | Strict quality | Strict diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Empirical CE | Raw LOO | 99.0% | 97.9796% | 46.5% | 0.893390 | 66.5% | 34.5% | 0.877696 |
| Empirical CE | Clean denoiser | 97.5% | 100.0% | 52.0% | 0.891434 | 64.0% | 34.5% | 0.876116 |
| MASK-rich CE | Raw LOO | 99.0% | 99.4949% | 49.5% | 0.881680 | 80.5% | 42.0% | 0.872805 |
| MASK-rich CE | Clean denoiser | 98.5% | 99.4898% | 47.5% | 0.891643 | 66.5% | 32.5% | 0.878335 |

Quality is within-seed unique valid molecules satisfying QED >= 0.6 and SA <= 4
divided by all requests. Validity uses requests; uniqueness uses valid samples;
diversity uses released PyTDC on unique valid molecules. Repaired SAFE decoding
repairs and selects the largest component; strict decoding disables both.
Both begin after tokenizer special-token removal.

## Both signed clean-temperature-minus-raw-LOO-temperature contrasts

Quality differences and sample SDs are in percentage points. SD describes the
spread of two seed differences, not a confidence interval or training variability.

| Prior | Decoding | Seed 2100 | Seed 2101 | Mean | Sample SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Empirical (primary) | Repaired | 0.0 | +11.0 | +5.5 | 7.7782 |
| Empirical (primary) | Strict | -3.0 | +3.0 | 0.0 | 4.2426 |
| MASK-rich (secondary) | Repaired | -6.0 | +2.0 | -2.0 | 5.6569 |
| MASK-rich (secondary) | Strict | -12.0 | -7.0 | -9.5 | 3.5355 |

Empirical strict validity decreases by 2.5 points. MASK-rich strict validity
decreases by 14.0 points even though repaired diversity increases by 0.009963.
These outcomes show why a local algebraic retention example cannot establish
learned molecular benefit. All sixteen metric contrasts remain reported,
including negative and inconsistent seed effects. Fresh-seed control means
differ from V12 and cannot be interpreted as a temperature-space effect.

## Exact token audit, devices and timing

Saved compressed token arrays and editable bitmasks independently reproduce
zero final MASK, UNK, BOS, EOS or PAD occurrences over 39,280 editable positions:
4,876 for seed 2100 and 4,944 for seed 2101 per configuration. No row among
800 contains an editable MASK. This is a final-state count, not a trajectory
survival measurement, and cannot be inferred from CSV text after special-token
removal.

Dynamic physical GPUs 2 and 3 were mapped through UUIDs
`GPU-997e881a-bd08-aa54-45f3-9b3c6fdf6ece` and
`GPU-2cd1aa5b-616e-e6d3-b54d-49241cc8f959`. Every final utilization probe was
7%, 8% or 9%, with sufficient free memory and at most two generation children
concurrent. Existing processes were preserved. Per-100-request generation
time ranged from 14.105 to 20.841 seconds; loading ranged from 12.635 to 14.229
seconds. Generation time includes model sampling/tokenizer plus released
postprocessing, excluding loading, chemistry scoring and token audit. Shared
GPU load prevents interpreting these wall times as a controlled speed comparison.

## Evidence and direction

The original rows, summaries and controller receipts are under
`output/udlm/engineering_v13/`; logs are under `output/logs/engineering_v13/`
and `output/logs/engineering-v13-*`. The report directory is
`output/udlm/engineering_v13_reports/complete/`:

- JSON SHA-256: `7942486344669088179536dd12dc69d0aab91bede8920a68763a00522e23bb52`.
- PDF SHA-256: `b97233ad1b22f88d976d6f5a8bd83613212f17ddc5b291451044e8e2b8774463`.
- Paired CSV SHA-256: `6384ce2a25e499ada66df46c27632f4646adac84057f2335b8af8c22f7c7555e`.

The cumulative 93-page report is
`output/udlm/study_overview_v13_20260907/study_overview.pdf`, SHA-256
`350d365017b5a87291d772389b8767d44094e40dd6cb1e99d30b37951b206fdf`.
Its 291-input manifest has SHA-256
`db76e519891a1891b0587653d9f0c49602f814c739611c430ba800dc72108b20`.
All eight bundle outputs independently reproduce byte-for-byte; all 91 appended
pages preserve their original decoded content streams and extracted text.
The independent V13 audit SHA-256 is
`06b0d8fd1c7af0a6e37cf23c063af53e590fe8f71d895a29a8dbe09001cb5f45`.

The source feature passed 175 production/helper tests, 230 integration tests,
45 paired-report tests and 34 notebook tests (overlapping suites, not additive
independent measurements). The merged focused check passed 72 tests. Default
behavior was checked against historical source, and old V9/V12 paired CSVs
remain byte-identical under the new reporter. The separate V13 evidence audit
recomputed all sixteen contrasts and verified exact saved token arrays.

The new temperature mode remains an explicitly labeled option, with no claim
that it replaces the old convention. The next implementation prepares a
matched-oracle-budget PMO comparison using the released fragment population
policy and gamma zero. A separate prospective task/checkpoint/seed/budget
panel is required before any PMO experiment. Final de novo seeds 0/1/2 remain
reserved; the best earlier selected UDLM pilot is still 57.03%, and broader
GenMol superiority has not been established.
