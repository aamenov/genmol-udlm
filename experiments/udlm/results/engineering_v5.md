# UDLM temperature screen: complete, no superiority

All 24 runs (12 configurations, seeds 1200/1201, 64 requests per seed) completed and were independently redecoded and rescored. Every one of the 1,536 raw rows agreed with the producer. The generation source was `12e13ddf9b1323bbfb60fe289b5dd363ca0c3c05`. All runs used 128 backbone evaluations per molecule.

The largest pilot quality mean was **50.78125%**, from schedule-consistent uniform S at temperature 0.85. The audited 50k MDLM comparator has quality **85.8%** at 1,000 requests per seed. This selected small-sample pilot is far below the comparator; it does not establish superiority. The strict-validity leader was release-uniform R at temperature 0.5 (63.28125%). No final evaluation seeds were used.

| Configuration | Repaired validity | Repaired quality | Strict validity | Diversity |
|---|---:|---:|---:|---:|
| s_t085_p100 | 0.960938 | 0.507812 | 0.398438 | 0.904468 |
| r_t050_p100 | 0.992188 | 0.484375 | 0.632812 | 0.880851 |
| e_t070_p100 | 0.984375 | 0.453125 | 0.406250 | 0.899351 |
| e_t050_p100 | 0.976562 | 0.453125 | 0.476562 | 0.892146 |
| r_t070_p100 | 0.976562 | 0.445312 | 0.460938 | 0.894184 |
| s_t070_p100 | 0.976562 | 0.421875 | 0.429688 | 0.898250 |
| s_t050_p100 | 0.976562 | 0.414062 | 0.601562 | 0.896686 |
| e_t085_p100 | 0.945312 | 0.398438 | 0.320312 | 0.903752 |
| e_t100_p100 | 0.937500 | 0.359375 | 0.179688 | 0.920087 |
| r_t085_p100 | 0.968750 | 0.359375 | 0.398438 | 0.911394 |
| s_t100_p100 | 0.929688 | 0.320312 | 0.250000 | 0.924219 |
| r_t100_p100 | 0.937500 | 0.304688 | 0.375000 | 0.910999 |

Each checkpoint was initialized from the 50k MDLM EMA and received only 1,000 additional updates at global batch 16. This is an adaptation study, not equal-compute training or a population-level method claim. Model selection noise and differing sample counts prevent treating these pilot differences as confirmatory effects. Full seed values and sample standard deviations appear in the report.

All 24 final launch probes satisfied utilization strictly below 10% and at least 30,000 MiB free. UUID discovery dynamically selected physical cards 3 and 5; these are observations, not fixed IDs in the launcher. At most two seed jobs ran concurrently. Existing processes were preserved.

The prior v4 diagnostic remains a failed attempt: its generation completed but its launcher rejected the effective-config top-p default. Its raw evidence and failure receipt are preserved. V5 uses a new protocol and fresh seeds.

Reproduce the independent report from this artifact-bearing checkout with the project `.venv`:

```bash
python scripts/udlm/report_exploration.py --protocol experiments/udlm/protocols/engineering_v5.json --report-dir output/udlm/engineering_v5_reports/new_snapshot
```

[Full PDF](../../../output/udlm/engineering_v5_reports/complete/report.pdf) · [JSON](../../../output/udlm/engineering_v5_reports/complete/report.json) · [CSV](../../../output/udlm/engineering_v5_reports/complete/report.csv)

The next specified experiment compares 128 predictors against 64 predictors plus 64 single-coordinate Gibbs corrections. Its six configurations and fresh seeds were committed before the V5 screen completed; it makes no confirmatory superiority claim.
