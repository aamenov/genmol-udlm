# V9: matched CT and clean-denoiser CE molecular evaluation

All eight runs completed and independent CPU rescoring accepted all 800
requested rows. The initial pipeline ended at 2026-09-07 11:22:42 UTC. The
best V9 repaired-quality mean is **54.5%**, below local MDLM **85.8%** by
31.3 percentage points. CE does not improve repaired quality at either
predeclared temperature in this small study. No superiority is established.

## Configuration and provenance

- Prospective protocol: `experiments/udlm/protocols/engineering_v9_objectives.json`,
  SHA-256 `718893198b0e08a2405e33177c64989499914207ff39b6880edbc5eff2b35c7d`.
- Generation source: `da8f5403ec9de02792261d1ffc3aa20ac08474e9`.
- CT checkpoint: `48986899c401c09cdc1e9e865e773cadc40899b62129420689f89a8e622a9c99`.
  The original V8 controller remains failed; this checkpoint passed the separate
  post-exit CPU audit. Original failure receipts remain unchanged.
- CE checkpoint: `b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1`.
  This separately completed V8b arm has its own successful receipt.
- Both: fresh MDLM 50k EMA initialization, 1,000 optimizer updates, batch 128,
  training seed 1500, 128,000 configured exposures, matched all-special-token
  clean-target mask and full empirical-prior corruption alphabet, A1/L1.
- Generation: EMA, CT/CE × temperatures 1.0/0.5, top-p 1.0, no corrector,
  128 predictor evaluations per molecule, endpoint 1e-5, randomness 0,
  min_add_len 40. Seeds 1600/1601 each request 100 molecules per setting.
- At most two concurrent GPUs, selected dynamically: physical 7/2 for the
  first pair, then 7/6. All final probes were below 10% utilization with ample
  memory; UUID mappings and active processes are retained in each summary.
- Initial generation-plus-rescoring pipeline: 457.009 seconds. Child generation
  time was 11.929–14.364 seconds per 100 requests, with model-loading and other
  timings reported separately. These sequential jobs are not controlled speed
  comparisons between objectives.

## All four configurations

Numbers below are equal-seed means. Complete per-seed values and sample standard
deviations for validity, uniqueness, quality and diversity under both decoding
rules are in the JSON/CSV/PDF report.

| Objective | Temperature | Repaired validity | Repaired uniqueness | Repaired quality | Repaired diversity | Strict validity | Strict quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CT | 1.0 | 97.5% | 99.4792% | 46.5% | 0.909003 | 39.0% | 21.0% |
| CE | 1.0 | 98.0% | 99.4845% | 41.0% | 0.902542 | 64.5% | 26.0% |
| CT | 0.5 | 98.5% | 99.4949% | 54.5% | 0.883836 | 77.0% | 44.0% |
| CE | 0.5 | 99.0% | 99.4949% | 52.5% | 0.887961 | 67.0% | 39.5% |

Quality counts unique valid molecules satisfying QED >= 0.6 and SA <= 4,
divided by all requests. Uniqueness is within each seed among valid molecules.
Released-comparable decoding uses SAFE repair and largest-component selection;
strict decoding uses no repair. Comparing these decoding branches as if they
were the same metric would be misleading.

## Predeclared CE-minus-CT quality contrasts

Every value below is in **percentage points**. Sample SD describes the two
paired seed differences; it is not a confidence interval or standard error.

| Contrast | Decoding | Seed 1600 | Seed 1601 | Mean | Sample SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Primary, T=1.0 | Repaired | -7.0 | -4.0 | -5.5 | 2.1213 |
| Primary, T=1.0 | Strict | +9.0 | +1.0 | +5.0 | 5.6569 |
| Secondary, T=0.5 | Repaired | -4.0 | 0.0 | -2.0 | 2.8284 |
| Secondary, T=0.5 | Strict | -8.0 | -1.0 | -4.5 | 4.9497 |

At temperature 1.0, CE improves strict validity by 25.5 points while repaired
quality declines. At 0.5, CE also lowers strict validity by 10.0 points. Thus
the observed objective effect differs with temperature; a single
pooled winner or a comparison of training-loss magnitudes would conceal it.

## Structural diagnosis and next hypothesis

The tested lexical audit was applied to the hash-verified, independently
rescored V9 raw rows. Counts per 200 requests are:

| Setting | Odd ring counts | Unbalanced parentheses | Either flag |
| --- | ---: | ---: | ---: |
| CT T=1.0 | 73 | 37 | 96 |
| CE T=1.0 | 43 | 18 | 58 |
| CT T=0.5 | 19 | 23 | 38 |
| CE T=0.5 | 34 | 23 | 52 |

Flags overlap and are necessary lexical checks, not a complete chemical
validator. None of the flagged rows is strict-valid. For CT T=0.5, 86 of 87
repaired-unique quality failures involve low QED, including three also failing
SA. Syntax alone cannot be assumed to explain the quality gap. The reproducible
wrapper is `output/logs/engineering-v9-syntax-audit.py`; exact counts and input
hashes are in `experiments/udlm/diagnostics/engineering_v9_syntax.json`.

The next proposed V10 comparison holds both checkpoints and temperature 0.5
fixed and compares 128 with 512 predictor evaluations on fresh seeds 1700/1701.
This setting is informed by V5/V6/V9, and 512 costs four times the model calls.
The independently reviewed two-position oracle audit motivates the hypothesis;
it does not guarantee an improvement for learned molecular models. V10 requires
its own committed protocol before generation. No further training or final
evaluation is authorized by the V9 protocol itself.

## Published reports and caveats

The original completion report remains at
`output/udlm/engineering_v9_reports/complete/`. The separate paired report is
`output/udlm/engineering_v9_reports/paired_complete/report.pdf`, SHA-256
`4419c7b1ed7466d47d0f118ae4738f615f774de292255fd2c28ef0eb295f1811`.
It adds all predeclared signed contrasts in JSON, PDF and `paired_contrasts.csv`;
it independently rescored the same 800 rows again from source
`269ef30` and preserves the original report. Raw rows, token audits, controller
receipts and logs remain under the V9 namespaces.

Two seeds of 100 requests per setting are exploratory. The MDLM comparator uses
three seeds of 1,000 and a different full training budget; paper GenMol V1 quality
84.6% is also contextual. Historical V5/V6 checkpoints differ in exposure,
masking and training grouping, so cross-study changes are not a causal batch-size
effect. Final UDLM seeds 0/1/2 remain reserved, and no candidate was promoted.
