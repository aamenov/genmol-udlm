# V11b: MASK-rich CE training completed

V11b completed at 2026-09-07T12:36:31.214684+00:00, with 1,000 optimizer updates and
128,000 configured example exposures. The final checkpoint passed the exact
resolved-configuration, CE identity, EMA-update and 1,157-tensor finiteness
checks, plus independent prior reconstruction against the prospective MASK-rich
fingerprint. This is training evidence; no molecular improvement is established.

The original V11 capacity rejection remains a failed prelaunch attempt with
no child or updates. V11b is its separate fresh attempt, not a resumed checkpoint
or a relabeling of that failure.

## Configuration and identity

| Setting | Value |
| --- | --- |
| Runtime source | `be18a3244a717978e027d6e43ba0ac20b8137e4a` |
| Protocol SHA-256 | `5ecf638fa8fc7a3707a0497c8a358697610f52c8af7f2d6468458890e26368eb` |
| Initial weights | Original MDLM 50k EMA, fresh optimizer and EMA |
| Initialization SHA-256 | `8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6` |
| Objective | Clean-token CE, `x0_denoiser`; current-state/time LOO conversion at inference |
| Prior | `mask_rich_empirical`, MASK mixture 0.9, empirical base smoothing 0.0002 |
| Prior metadata SHA-256 | `4e1febe2684beebdbf3d4c86aa01bc24ed1d3941af67b63473979e2ae810ee45` |
| Seed / optimizer updates | 1500 / 1,000 |
| Batch | Global 128 = 2 GPUs x microbatch 16 x accumulation 4 |
| Learning rate | A1 FiLM; L1 peak 3e-4, 50-update warmup, 1,000-update horizon, floor 3e-6 |
| Clean-target mask | All tokenizer specials immutable/excluded; full 1,880-token corruption alphabet |
| Final checkpoint SHA-256 | `62299de99d8c003e3776215efce9643cca196091a6284f351de2b404a22c2e8d` |
| Checkpoint size | 1,636,493,702 bytes; weights retained on host |
| Terminal SHA-256 | `891f5d6d609146a02ebb18caa7176bf555e3f67fe2493795c8a3e1fccb264779` |

The resolved training settings equal V8b empirical CE except prior variant,
MASK weight and output namespace. They equal the failed V11 request except
output namespace; all 16 training source files remain unchanged from V11.
Shared settings and exposures do not imply equal corruptions, task difficulty
or wall time. Exposures are not a count of distinct molecules.

## Runtime and resources

The controller started at 2026-09-07T12:14:43.637509+00:00. The dynamic capacity check
passed on its first round in 0.819007474
seconds. Physical GPUs 6 and 7 were selected by UUID; both were at 0% utilization
in the final individual probes immediately before launch. Their mapping is
retained in the launch receipt and full telemetry. Existing processes remained
in place. No third GPU was used.

Training subprocess runtime was 1273.872136587 seconds;
controller runtime was 1307.577202052 seconds including initial
checks, waiting and final validation. Subprocess throughput was
100.481042268 configured exposures/second.
Maximum observed aggregate device use was 19,987 and 23,443 MiB, including
other processes; these are sampled device totals, not allocator peaks. Shared
hardware and startup costs prevent interpreting these times as a method speed
comparison. No telemetry errors were recorded.

The child returned zero. Its process group disappeared during the 15-second
cleanup grace after 0.751276228 seconds
and four probes. Both exact leases were released and their absence was verified.
The final checkpoint's finite tensors do not establish that every intermediate
state was finite.

## Validation and next comparison

129 controller/compatibility tests, 19 follow-up tests and independent reviews
passed before launch; the clean pushed canonical CPU dry-run also passed.
The original request/launch/terminal records, GPU telemetry, Hydra settings,
training metrics and logs are preserved byte for byte. The final checkpoint is
`output/udlm/engineering_v11b/mask_ce_1000_b128_w2/checkpoints/1000.ckpt`.

The already declared V12 comparison will use this checkpoint and completed
V8b empirical CE at temperatures 1.0/0.5, 128 predictor evaluations, seeds
2000/2001 and 100 requests per seed: four configurations and 800 requests.
Its executable protocol is gated on actual checkpoint and receipt verification.
Report both signed prior contrasts and exact final editable MASK counts from
saved token IDs, since decoded CSV text omits special tokens. The local MDLM
quality mean 85.8% remains a contextual baseline; final seeds 0/1/2 are reserved.
