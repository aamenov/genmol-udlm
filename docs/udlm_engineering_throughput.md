# Fixed v7 CT-E throughput pilot

This prospective manual engineering pilot measures whether the existing E
training stack can run microbatch 16 efficiently. It performs exactly twenty
optimizer updates at effective batch 128 (2,560 example exposures), with seed
1400. It does not automatically start longer training, resume the pilot, select
a model, decode molecules or claim benchmark improvement.

The frozen configuration is `configs/udlm_e_throughput20.yaml`; the protocol is
`experiments/udlm/protocols/engineering_v7_throughput.json`. It retains E's
categorical CT loss, empirical prior with uniform mixture 0.0002, FiLM/AdaLN
conditioning, and fresh initialization from the SHA-pinned MDLM50k EMA. The
optimizer/EMA start fresh. The L1 schedule has peak LR 3e−4, fifty-update
warmup, a 1,000-update half-cosine horizon including warmup, and floor 3e−6.
Thus this short run ends during warmup. It cannot determine late-training
stability or the best learning rate.

One GPU uses accumulation eight; two use accumulation four. Both use local
microbatch 16 and one data-loader worker per rank. The repaired hosted loader
uses the actual trainer rank/world size. Direct training's restricted
post-initialization reseeding remains disabled. Anomaly detection and the old
registered pilot's gradient audits are disabled to measure the ordinary
training path. Completion independently checks the step, configuration,
model/optimizer/EMA tensor finiteness and twenty EMA updates on CPU.
The active-alphabet setting is explicitly `exclude_special_tokens=false`.
A subsequent CE arm must use the same value rather than its CE-config default;
otherwise the comparison also changes the corruption alphabet and token prior.

## Preview and launch

The project virtual environment and clean pushed source are required. A CPU
preview is valid from either the development checkout or the main artifact
checkout and creates no output or leases and performs no GPU query:

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python \
  scripts/udlm/launch_engineering_training.py --gpu-count 1 --dry-run
```

Live execution is allowed only after merging into the canonical artifact
checkout, and requires tmux. A concrete one-GPU invocation is:

```bash
tmux new-session -d -s genmol_v7_ct_e_throughput_w1 \
  -c /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree \
  'env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python -u scripts/udlm/launch_engineering_training.py --gpu-count 1'
```

For two GPUs use `--gpu-count 2` and a distinct tmux session ending in `w2`.
One GPU is the default. The launcher acquires both existing generation and
training lock paths before probing, so it cannot overlap either workflow in
the canonical checkout. Never run another workflow from a separate checkout
to evade those locks. Existing or stale locks fail closed; there is no stale
lock removal, automatic retry or resume. Finish the v6 generation controller
and report before launching this pilot.

Selection examines all GPUs, including their compute processes. Each selected
GPU must have strictly less than 10% utilization, at least 30,000 MiB free and
non-prohibited compute mode. Existing processes are allowed and preserved.
The launcher selects at most two GPUs, re-probes their exact UUIDs immediately
before creating the training subprocess, and exposes only those UUIDs through
`CUDA_VISIBLE_DEVICES`. It never accepts physical GPU IDs.

## Evidence and limits

Outputs are exclusive under
`output/udlm/engineering_v7/ct_e_throughput20_b128_w{1,2}/`. The training log is
`output/logs/engineering_v7/ct_e_throughput20_b128_w{1,2}.training.log`.

- `request_manifest.json` pins the source, prospective protocol, resolved
  configuration, exact argv, initialization checkpoint and requested budget.
- `launch_manifest.json` records the selected UUIDs, complete selection and
  final probes, process inventory, controlled environment and both lease
  identities. A failure before process launch may have only a request and
  terminal manifest; this is explicitly a failed attempt.
- `gpu_telemetry.jsonl` retains periodic external NVIDIA snapshots, including
  existing compute processes and query errors. The nominal polling interval is
  two seconds plus query overhead.
- `terminal_manifest.json` retains process return code, runtime, source,
  checkpoint digest and validation, telemetry/log/checkpoint hashes and errors.
  It is published before exact lease release. Failure artifacts are retained.
  If the child process group remains or a lease changes identity, the launcher
  keeps its resource block and records the problem.

The reported memory maximum is the largest **observed aggregate GPU use**,
including other users' processes. It is neither PyTorch allocator peak nor an
exact continuous-time peak; brief spikes can be missed. Runtime measures the
whole training subprocess, including imports, checkpoint loading, dataset
startup and checkpoint saving. `2,560 / subprocess_seconds` is therefore
end-to-end throughput, not isolated steady-state optimizer throughput. The
training progress log provides additional context. A successful pilot shows
this specific data prefix fitted; later batches can require different memory.

Before any longer experiment, review the terminal validation, log losses,
observed memory headroom and throughput. Restart both CT and proposed CE arms
fresh from the common MDLM EMA for a matched 1,000-update batch-128 comparison;
do not silently reuse this twenty-update checkpoint. Keep the same GPU count,
batch grouping, schedule, data start and initialization/RNG policy. Final
benchmark seeds 0/1/2 remain reserved. Longer adaptation and a matched
extra-update MDLM control require a new declared experiment.
