# V12: MASK-rich CE improves the paired pilot but remains below GenMol

All eight runs completed and independent CPU rescoring accepted all 800
requests. The pipeline ran from 2026-09-07 12:45:39.651741 to 12:56:25.917740
UTC (646.266 seconds); generation ended at 12:54:49.468385 UTC. Both subprocess
return codes were zero. MASK-rich CE increased mean repaired quality by
7.5 percentage points at temperature 1 and 5.5 points at temperature 0.5.
Best V12 quality is 54.0%, below the local MDLM reference of 85.8%.
No superiority or final promotion is established.

## Fixed design and checkpoint identities

The prospective protocol is
`experiments/udlm/protocols/engineering_v12_mask_prior.json`, SHA-256
`f4ea756fabb9e5133c72f302e8398bcbb8d0eabaa81f183b017af8c19b02cadd`.
Generation and independent reporting used clean pushed source
`c61a1ef4b42fd2f8aba817e5e3e3554afc64eaa1` throughout.

Empirical CE uses the accepted V8b checkpoint
`b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1`.
MASK-rich CE uses the accepted V11b checkpoint
`62299de99d8c003e3776215efce9643cca196091a6284f351de2b404a22c2e8d`.
Both start from the same MDLM 50k EMA and receive 1,000 updates at global
batch 128 (128,000 configured exposures), with the same training seed 1500,
A1/L1 settings and clean-target control mask. V11b changes the corruption prior
to a 0.9 MASK point-mass mixture with the smoothed empirical base. It is a
trained prior change, not an inference-time swap. The failed original V11
prelaunch attempt and failed original V8 controller remain separate records.

All four settings use EMA, 128 predictor evaluations, top-p one, no Gibbs
corrector, endpoint 1e-5, min_add_len 40, unused UDLM randomness zero and the
full active alphabet. Temperatures are 1 and 0.5. Fresh seeds 2000/2001 each
request 100 molecules per setting. CE logits are converted to raw LOO logits
before temperature, exactly as declared. The panel costs 102,400 molecule-level
backbone evaluations. It was designed after earlier pilots and a frozen-MDLM
reconstruction diagnostic; those adaptive choices are disclosed.

## Every configuration

Values are equal-seed means. The complete JSON/PDF retains every seed value and
sample standard deviation for both decoding rules.

| Prior | Temperature | Repaired validity | Repaired uniqueness | Repaired quality | Repaired diversity | Strict validity | Strict quality | Strict diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Empirical CE | 1.0 | 99.0% | 99.5% | 42.5% | 0.900820 | 58.5% | 27.5% | 0.885439 |
| MASK-rich CE | 1.0 | 99.5% | 99.5% | 50.0% | 0.904567 | 58.5% | 34.0% | 0.882340 |
| Empirical CE | 0.5 | 98.0% | 100.0% | 48.5% | 0.894485 | 71.5% | 37.5% | 0.883021 |
| MASK-rich CE | 0.5 | 99.5% | 99.4949% | 54.0% | 0.886350 | 76.5% | 42.0% | 0.876468 |

Strict uniqueness is 100% for every setting. Quality counts within-seed unique
valid molecules with QED >= 0.6 and SA <= 4, divided by all requests. Validity
uses all requests; uniqueness uses valid samples; diversity uses the released
PyTDC metric on unique valid molecules. Repaired decoding uses SAFE repair and
largest-component selection; strict decoding disables both. Both follow
tokenizer removal of special tokens.

## Both declared MASK-minus-empirical contrasts

Quality differences and sample SDs below are in percentage points. Sample SD
describes two paired seed differences; it is not a confidence interval.

| Temperature | Decoding | Seed 2000 | Seed 2001 | Mean | Sample SD |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.0 (primary) | Repaired | +16.0 | -1.0 | +7.5 | 12.0208 |
| 1.0 (primary) | Strict | +12.0 | +1.0 | +6.5 | 7.7782 |
| 0.5 (secondary) | Repaired | +6.0 | +5.0 | +5.5 | 0.7071 |
| 0.5 (secondary) | Strict | +5.0 | +4.0 | +4.5 | 0.7071 |

The primary repaired gain varies substantially across seeds, including one
negative difference. At temperature 0.5, repaired diversity decreases by
0.008135 and strict diversity by 0.006553 despite improved quality. All sixteen
metric contrasts are retained in the paired CSV and PDF; none are selected away.

## Token audit, devices and runtime

Independent decoding of the saved token-ID arrays and editable bitmasks found
zero final editable MASK, UNK, BOS, EOS or PAD occurrences. Each configuration
has 4,789 editable positions for seed 2000 and 4,837 for seed 2001: zero control
occurrences over 38,504 positions and zero MASK-containing rows out of 800.
CSV text cannot establish this because tokenizer decoding removes special
tokens. Final occurrence counts do not describe intermediate trajectories.

The launcher dynamically used physical indices 3 and 5, mapped respectively
to `GPU-2cd1aa5b-616e-e6d3-b54d-49241cc8f959` and
`GPU-134e9a1b-cf80-5c0e-b0c4-048fff8e6d3c`. Every final launch probe was
7%, 8% or 9% utilization, with sufficient memory; at most two V12 generation
jobs were concurrent. Full inventories and active-process records are saved.
Existing processes were preserved. Per-100-request generation time ranged
from 14.850 to 15.628 seconds, excluding model loading; loading ranged from
12.617 to 13.759 seconds. Shared-host availability and load affect wall time.
Generation time includes sampling/tokenizer decoding and released postprocessing.

## Published evidence and next experiment

Raw CSVs, summaries, controller receipts and logs remain under
`output/udlm/engineering_v12/` and `output/logs/engineering_v12/`. The pipeline
wrapper, preview, log and terminal status are under `output/logs/engineering-v12-*`.
The complete report JSON SHA-256 is
`133311e81ca7154ab6d4569619ece75d36a8dc5a5ffe927d405d5fd69c3259b8`;
its 13-page PDF SHA-256 is
`164b105d2a9cd03731b358cab23a9698456022472bc44fe12f4c8eed3f9a440c`.

The combined 78-page report is
`output/udlm/study_overview_v12_20260907/study_overview.pdf`, SHA-256
`b6be1949a3f163e90d4437e5e1f66c2a406839fe791fda6d97edc9f9af29e250`.
It covers all 30 settings, 60 completed runs and 4,704 independently rescored
requests, with 253 input hashes. The failed V4 diagnostic's 32 requests remain
separate. All 68 appended original pages were checked against their source
PDF text; new overview pages and the configuration plot were visually reviewed.

V12 does not exceed the best earlier selected pilot (57.03%) or local MDLM
(85.8% repaired quality, 84.7667% strict quality over three 1,000-request seeds).
The next prospective V13 panel compares temperature before versus after CE
conversion on both frozen checkpoints. Its design was fixed and pushed before
reading MASK-rich molecular outcomes; it does not select a model from V12.
This is a new inference hypothesis, not a correction to the existing raw-LOO
temperature convention. Final seeds 0/1/2 remain reserved, and broad GenMol
superiority also requires optimization benchmarks with matched oracle budgets.
