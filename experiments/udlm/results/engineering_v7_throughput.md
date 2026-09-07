# Larger-batch throughput pilot — 2026-09-07

The fresh CT-E pilot completed 20 optimizer updates at global batch128,
microbatch16 and accumulation8 on one dynamically selected RTX A6000. The
step20 checkpoint, saved configuration, all 1,155 checked tensors, optimizer
state and twenty EMA updates passed CPU validation. No molecular benchmark
was run from this resource checkpoint; it will not initialize the next study.

- Initialization: local MDLM50k EMA, SHA-256 `8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`.
- Seed:1400; source:`e78267b098103baa822273e82e5a654a97b9daf4`.
- Configuration: historical CT/raw-LOO E; empirical floor0.0002; FiLM; L1 peak3e-4/warmup50/horizon1000/floor3e-6; alphabet includes controls, while the historical target mask excludes BOS/EOS/PAD.
- Completed example exposures:2,560. These are not necessarily distinct molecules.
- Training subprocess:146.333seconds, including initialization and checkpoint saving; end-to-end throughput17.494examples/second. Controller total169.116seconds includes validation.
- Progress reached160 microbatches in17seconds before saving and27seconds including the final checkpoint. This is an approximate console observation, not separately instrumented steady-state timing. Short-run startup overhead makes linear extrapolation of either measure uncertain.
- Maximum observed aggregate GPU usage:9,637MiB, including external processes; external sampling every2seconds is not an allocator-peak measurement.
- Dynamically selected UUID:`GPU-997e881a-bd08-aa54-45f3-9b3c6fdf6ece`, physical2 exposed only as logical cuda:0. Final launch probe verified utilization<10% and free memory≥30,000MiB; active external processes were recorded and left intact.
- Output checkpoint: `output/udlm/engineering_v7/ct_e_throughput20_b128_w1/checkpoints/20.ckpt`, 1,636,492,040bytes, SHA-256 `07ad66498ccaf629fd61e8485c6ddb310c84c4a86770a0be5e3d4911fe341cc0`. Weights remain on this host; Git contains the digest and complete launch/terminal evidence.
- Request, launch, terminal and telemetry artifacts: `output/udlm/engineering_v7/ct_e_throughput20_b128_w1/`.
- Training log: `output/logs/engineering_v7/ct_e_throughput20_b128_w1.training.log`.

The pilot supports proceeding to a separate, fresh-start CT/CE1000-update
comparison. It does not establish convergence, two-GPU throughput or molecular
superiority. Both new arms will explicitly share an all-tokenizer-control
target mask; this is a new CT control setting, not a reinterpretation of the
historical CT mask used here. V8 will use the same selected GPU count for both
arms and independently recheck resources at each launch.
