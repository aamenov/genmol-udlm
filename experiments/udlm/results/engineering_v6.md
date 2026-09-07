# Fixed-budget Gibbs engineering results — 2026-09-07

All 12 runs completed; independent CPU rescoring accepted all 768 raw rows. No superiority over GenMol has been established.

Each row averages seeds 1300/1301 with 64 requested molecules per seed. All settings use temperature 0.5, top-p 1 and 128 model evaluations. Predictor controls use 128 transitions; Gibbs uses 64 transitions and 64 fresh-state, single-coordinate corrections. R/S/E retain their existing 1,000-update, batch-16 MDLM-EMA adaptation checkpoints.

| Arm and sampler | Quality | Repaired validity | Uniqueness | Diversity | Strict validity |
| --- | ---: | ---: | ---: | ---: | ---: |
| e_t050_predictor | 46.875000% | 96.875000% | 99.193548% | 0.895889 | 53.906250% |
| e_t050_gibbs | 51.562500% | 98.437500% | 98.412298% | 0.888567 | 55.468750% |
| s_t050_predictor | 54.687500% | 98.437500% | 99.193548% | 0.882188 | 61.718750% |
| s_t050_gibbs | 57.031250% | 98.437500% | 97.605847% | 0.892101 | 55.468750% |
| r_t050_predictor | 53.906250% | 100.000000% | 98.437500% | 0.884955 | 65.625000% |
| r_t050_gibbs | 47.656250% | 97.656250% | 100.000000% | 0.893241 | 61.718750% |

Within-arm Gibbs quality differences were +4.6875 percentage points for E, +2.34375 for S, and −6.25 for R. These two-seed engineering observations are too small to establish a reliable improvement. The selected pilot leader S+Gibbs has quality 57.03125% (seed sample SD 1.104854 percentage points), repaired validity 98.4375%, uniqueness 97.60585%, and strict validity 55.46875%. The local MDLM comparator has quality 85.8%, repaired validity 100%, uniqueness 99.86667%, and strict validity 98.8% over three 1,000-request seeds. Different sample counts, selection, and unmatched training compute prevent a confirmatory comparison.

Quality counts unique repaired molecules meeting QED ≥ 0.6 and SA ≤ 4, divided by requested count. Repair and largest-component selection can hide malformed structures, so strict results remain essential. Learned and tempered LOO conditionals have no exact Gibbs stationarity guarantee; the fresh final correction is also below the usual training minimum noise time.

The prospective protocol was committed before V5 finished and discloses the partial V5 observations that informed its temperature choice. Final evaluation seeds 0/1/2 remain reserved. Both the earlier v4 failure and all V5/V6 outcomes are preserved; no failed attempt was reclassified.

- Generation source: `84e909d983c0c413a9bce71357deba24df717b85`.
- Protocol: `experiments/udlm/protocols/engineering_v6.json` (SHA-256 `b50d97dcdd10f35c85aba61c9b9b948081ae9726809abb25b6c9e0a8834a92e9`).
- Full independently rescored JSON/CSV/PDF: `output/udlm/engineering_v6_reports/complete/`.
- PDF SHA-256: `c9105da785e30ab3fb50c39f5b101c6e4f5c3c76d4482f86fec3d61c8a2819ca`.
- Durable pipeline: 09:58:43–10:09:33 UTC, generation and report exits both zero; see `output/logs/engineering-v6-pipeline-status.json`.
- At most two GPU jobs overlapped; every launch selected UUIDs dynamically after a final <10% utilization and ≥30,000 MiB free-memory check. Active external processes were recorded and left intact.

The next investigation changes training exposure/objective. CE clean-denoiser code and its exact CE-to-LOO conversion are implemented and CPU-tested but have not trained a molecular model. V7 first measures 20 CT-E updates at batch128; subsequent CT/CE comparison must start fresh and use identical vocabulary and target-mask policies.
