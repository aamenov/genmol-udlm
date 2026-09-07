# V10: increasing predictor resolution did not close the quality gap

All eight runs completed and independent CPU rescoring accepted all 800
requests. The pipeline ended at 2026-09-07 11:40:49 UTC, after 571.326 seconds.
Increasing 128 to 512 predictor evaluations changed repaired quality by
**-0.5 percentage points for CT** and **+1.5 points for CE**. The best V10 mean
is 48.0%, versus the contextual local MDLM mean 85.8%. No superiority.

The prospective protocol is
`experiments/udlm/protocols/engineering_v10_resolution.json`, SHA-256
`cc6e3ced657f6ec66f611398e65b8b0e79598b3e76acbef77c72cfc4a7d0c4fd`.
Generation used clean pushed source
`c17e848eb46896aff191ecd535d6a3cfa757c37e`. The frozen CT checkpoint is
`48986899c401c09cdc1e9e865e773cadc40899b62129420689f89a8e622a9c99`;
CE is `b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1`.
Their training provenance is recorded in V9: separately audited V8 CT after
controller failure, and completed V8b CE. Neither was retrained or substituted.

Both methods use EMA, temperature 0.5, top-p 1.0, no corrector, endpoint 1e-5,
randomness 0, min_add_len 40, and the same full empirical-prior alphabet.
Seeds 1700/1701 each request 100 molecules for each of four settings.
Temperature 0.5 was informed by V5/V6/V9. The 512-step treatment costs four
times the predictor calls; the entire panel requests 256,000 molecule-NFE.
Each job rechecked its dynamically selected UUID before launch, below 10%
utilization, with at most two GPUs concurrent. Full device/process telemetry,
configurations, seeds and timings are retained in the raw summaries and logs.

## All configurations

Equal-seed means are shown below. The complete report includes per-seed
metrics and sample standard deviations for both decoding rules.

| Method | Predictor NFE | Repaired validity | Repaired uniqueness | Repaired quality | Repaired diversity | Strict validity | Strict quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CT | 128 | 100.0% | 100.0% | 45.5% | 0.891760 | 72.0% | 35.5% |
| CT | 512 | 99.5% | 100.0% | 45.0% | 0.890721 | 73.5% | 32.5% |
| CE | 128 | 99.0% | 99.4898% | 46.5% | 0.892772 | 72.5% | 36.0% |
| CE | 512 | 98.5% | 100.0% | 48.0% | 0.892653 | 71.5% | 38.5% |

Quality counts within-seed unique valid molecules satisfying QED >= 0.6 and
SA <= 4, divided by all requests. Repaired decoding uses released SAFE repair
and largest-component selection; strict decoding does neither. The fresh
128-step controls also differ from V9 means. Those cross-study differences
cannot be attributed to step count because the seeds changed.

## Both predeclared 512-minus-128 contrasts

Quality differences and sample SDs below are in percentage points. SD describes
the spread of two paired differences; it is not a confidence interval.

| Method | Decoding | Seed 1700 | Seed 1701 | Mean | Sample SD |
| --- | --- | ---: | ---: | ---: | ---: |
| CT | Repaired | +3.0 | -4.0 | -0.5 | 4.9497 |
| CT | Strict | +3.0 | -9.0 | -3.0 | 8.4853 |
| CE | Repaired | +4.0 | -1.0 | +1.5 | 3.5355 |
| CE | Strict | +5.0 | 0.0 | +2.5 | 3.5355 |

The equal-seed mean generation-time ratio (512/128) is 3.79366 for CT and
4.02467 for CE. Generation time includes model sampling/tokenizer plus released
postprocessing, with model loading excluded. The supplement retains each ratio,
its sample SD and all original timing fields. These are observed wall times on
shared GPUs, not controlled architecture-speed comparisons.

## Evidence and next step

The original complete report remains under
`output/udlm/engineering_v10_reports/complete/`, JSON SHA-256
`8d9b7df5797338b6a768cfd3489af6abdadabe0cf48a8cec212086cfc7714d16`.
The independently reviewed three-page supplement is
`output/udlm/engineering_v10_reports/resolution_analysis/analysis.pdf`, SHA-256
`6204c58405e5b3a4e710a0407cdb20574dede06f9f77fdd74f758b70cec0230c`.
It includes all 16 paired metric comparisons, ring/parenthesis diagnostics,
repair frequencies, runtime ratios and 38 input hashes. All five supplement
artifacts reproduced byte-for-byte; 23 focused/lexical CPU tests and independent
review passed. Original generation/report artifacts remain unchanged.

This small learned-model result does not reproduce the guaranteed oracle
conditions in the toy resolution audit. More steps did not close the observed
gap, and no configuration was promoted. The next hypothesis is an opt-in
mask-rich empirical prior intended to make transfer from MDLM easier. It needs
a new trained checkpoint and its own prospective evaluation; changing an old
checkpoint's prior during sampling is forbidden. A frozen-MDLM CPU diagnostic
precedes the proposed new training arm. Final UDLM seeds 0/1/2 remain reserved.
