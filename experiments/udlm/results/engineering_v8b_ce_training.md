# V8b CE adaptation completed

The separately recorded CE run completed 1,000 optimizer updates at
2026-09-07 11:08:38 UTC and passed CPU checkpoint validation. Its exact saved
configuration, 1,156 checked finite tensors, optimizer state, 1,000 EMA updates,
CE denoiser metadata and scalar int64 version marker were accepted. This is
training evidence; molecular evaluation remains pending.

- Source: `b84f21ba1f30ceb62d2626daf2caad8adb6534cf`.
- Protocol SHA-256: `acb286ac4b938aa049327740affd2dea5beb0ded745cd51dca627f5e8fd9de8b`.
- Fresh initialization: MDLM 50k EMA, SHA-256
  `8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`.
- Training: seed 1500, global batch 128 = 2 GPUs × microbatch 16 × accumulation 4,
  1,000 updates and 128,000 configured example exposures. These are not
  necessarily distinct molecules.
- Matched settings: empirical prior with mixture 0.0002, full active alphabet,
  all-special-token clean-target mask, A1 conditioner and L1 scheduler. The
  training implementation hashes match the separately audited CT run.
- Checkpoint: `output/udlm/engineering_v8b/ce_e_1000_b128_w2/checkpoints/1000.ckpt`,
  1,636,492,999 bytes, SHA-256
  `b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1`.
  The large weight file remains on this host.
- Training subprocess: 706.595 seconds including startup and saving;
  controller total: 743.413 seconds. End-to-end throughput is 181.150 configured
  examples per second. Sequential runs with external processes do not support
  a controlled speedup claim against CT.
- Dynamic GPU UUIDs: `GPU-997e881a-bd08-aa54-45f3-9b3c6fdf6ece` (physical 2)
  and `GPU-7ae42da2-3c8d-efa0-99ec-845b7f85399a` (physical 7), at 7% and 6%
  utilization respectively in the final launch probes. Both met the 30,000 MiB
  free-memory requirement. Existing processes were retained.
- Maximum observed aggregate memory: 9,273 and 11,139 MiB, including external
  processes. Two-second snapshots are not allocator peak measurements.
- Cleanup: a remaining training process group exited within 0.751 seconds,
  after four probes in the new 15-second grace period. Both leases were
  released and the training processes/tmux session exited.
- Terminal receipt SHA-256:
  `2dd0423257e8908548b69a28b8aa24b3cc956507944c343e24ef615253af2205`.
  All request, launch, terminal and telemetry evidence is under
  `output/udlm/engineering_v8b/ce_e_1000_b128_w2/`; logs are under
  `output/logs/engineering_v8b/`.

The original V8 campaign remains failed after the CT process-exit incident;
V8b does not alter that status. The next comparison uses the separately audited
V8 CT checkpoint (`48986899…a9c99`) and this V8b CE checkpoint, following the
disclosed V9 design amendment. It retains four settings (CT/CE × temperatures
1.0/0.5), seeds 1600/1601, 100 requests per seed and 128 predictor evaluations
per molecule. No new generation output has been observed or used for selection.

Final checkpoint finiteness does not prove every intermediate update was
finite. Equal training settings and example exposures do not imply equivalent
loss scales or equally optimized methods. Both methods still share the large
MDLM pretraining budget, and superiority over the local MDLM benchmark remains
unproven.
