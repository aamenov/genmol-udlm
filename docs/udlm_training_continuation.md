# Longer UDLM training: engineering plan

This is a prospective engineering plan, not a trained checkpoint or benchmark
result. Existing R/S/E 1,000-update training and the v4 failure remain immutable
in the artifact-bearing `udlm_genmol_worktree`. This branch repairs direct
multi-GPU hosted-data sharding without changing those runs or their receipts.

## Data correctness repair

`scripts/train.py` previously partitioned the pinned hosted SAFE stream only
for registered pilots. Direct multi-GPU calls constructed the unpartitioned
iterable loader on every rank; Lightning cannot insert a DistributedSampler
for that dataset, so ranks received duplicate examples. Direct hosted training
now constructs the loader after Trainer construction and passes the Trainer's
actual global rank and world size to the existing Hugging Face split function.
Each rank is split exactly once; single-rank ordering, local-file loading and
the registered pilot's stricter rank/environment checks remain unchanged.

## Proposed next run

After the temperature screen, choose the arm using disclosed engineering
results, then freeze its new budget before training. A useful next budget is
10,000 new updates at global batch 128: 1,280,000 example exposures. This is
80 times the exposure of the existing 1,000-update batch-16 pilot, not a minor
extension. Try 20 updates first to measure throughput and peak memory at the
intended microbatch; use a separate output namespace and label that check.

The following exact training arguments compose on CPU. They propose fresh
initialization from the 50k MDLM EMA, empirical-prior E, the A1 FiLM conditioner,
and a new half-cosine schedule: peak 3e-4, 500-update warmup, 10,000-update
horizon including warmup, floor 3e-6. The longer LR schedule is a new hypothesis.
The inference uniform-mixture weight remains 0.0002. Use microbatch 16 and
accumulation 8 on one GPU, or microbatch 16 and accumulation 4 on two GPUs.

Run from this checkout. Before starting the command in a detached tmux session,
the scheduling wrapper must inspect utilization, free memory and active
processes, select one or two GPUs at strictly below 10% utilization with at
least 30,000 MiB free, then re-probe those exact UUIDs immediately before
exposing them through `CUDA_VISIBLE_DEVICES`. Never supply physical GPU IDs.
`GENMOL_SELECTED_GPU_UUIDS` below means that dynamically selected comma-separated
UUID list; it is not a default selection. Preserve the complete inventory and
exact command in the launch log. No scheduling or GPU launch has been performed
for this plan.

```bash
env CUDA_VISIBLE_DEVICES="$GENMOL_SELECTED_GPU_UUIDS" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONPATH="$PWD/src:$PWD" PYTHONHASHSEED=23 \
  /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python -u scripts/train.py \
  --config-name udlm_categorical seed=23 \
  trainer.devices=1 trainer.num_nodes=1 trainer.max_steps=10000 \
  loader.global_batch_size=128 loader.batch_size=16 loader.num_workers=1 \
  trainer.accumulate_grad_batches=8 trainer.detect_anomaly=false \
  optim.lr=0.0003 \
  optim.scheduler.name=half_cosine_with_linear_warmup_and_floor \
  optim.scheduler.warmup_updates=500 optim.scheduler.horizon_updates=10000 \
  optim.scheduler.decay_floor_lr=0.000003 \
  training.udlm.conditioning_variant=film_adaln \
  training.udlm.zero_init_conditioning=false \
  training.udlm.empirical_uniform_mix=0.0002 \
  training.reseed_after_model_initialization=false \
  training.init_from_mdlm_checkpoint=/home/aidar.alimbayev/Documents/genmolv2/outputs/paper_v1/checkpoints/50000.ckpt \
  training.init_from_mdlm_checkpoint_sha256=8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6 \
  callback.every_n_train_steps=1000 \
  callback.dirpath="$PWD/output/udlm/engineering-v6-e-fresh-10k-b128/checkpoints" \
  hydra.run.dir="$PWD/output/udlm/engineering-v6-e-fresh-10k-b128/hydra" \
  > output/logs/engineering-v6-e-fresh-10k-b128.log 2>&1
```

For two GPUs change only `trainer.devices=2` and
`trainer.accumulate_grad_batches=4`, and use a fresh namespace. The direct
entrypoint has no GPU inventory policy or global training lease; a scheduling
wrapper must supply both before executing this command. Do not run it alongside
existing generation jobs if that would exceed the user's total two-GPU limit.
The manual path does not emit registered schema-5 pilot receipts. It retains
Hydra config, Lightning logs and checkpoint files; a future launcher must record
checkpoint digests, runtime, exact source revision and device inventory.

## Initialization and comparison choices

Fresh MDLM-EMA initialization resets optimizer, scheduler, global step and EMA.
The E prior and timestep conditioner are newly constructed. Direct execution
currently requires `reseed_after_model_initialization=false`: the explicit
post-initialization reseeding option is restricted to registered pilots. Thus
different architectures can consume different random draws before training,
even at the same seed. A later matched-control launcher should repair this
restriction with explicit new provenance rather than mislabel the initialization.

True E1000 resume is a different experiment. Copy the checkpoint byte-for-byte
into a fresh callback checkpoint directory as `1000.ckpt`; the entrypoint
automatically resumes the largest numbered checkpoint in that directory. It
retains model, optimizer, scheduler, EMA and global step. `max_steps=11000`
then requests 10,000 additional updates. Do not place new checkpoints into the
historical directory. The old scheduler is already at its 3e-6 floor; leaving
its horizon at 1,000 keeps that floor. Replacing the horizon on resume retains
the old scheduler counter and first-step LR, then recalculates with the new
curve, potentially creating an abrupt LR increase. A warm restart from E1000
weights with fresh optimizer/EMA would require a new explicit initializer;
`init_from_mdlm_checkpoint` does not implement it.

Hosted data is pinned, streamed in order and unshuffled. Fresh runs begin at
the start. Resume does not restore the dataset cursor: `fast_forward_*` fields
are read from checkpoint metadata but unused by the loader. Do not claim exact
datastream continuation or disjoint post-pilot examples. Microbatch changes also
change grouping in the released mean-of-local-token-ratios objective.

A useful control is MDLM initialized from the same 50k EMA and given the same
new example budget, batch grouping, seed, optimizer restart and LR schedule.
The existing initializer rejects an MDLM target, so that control needs a small
generic weights-only initializer before execution. Merely resuming the old
MDLM checkpoint to step 60k preserves its old optimizer and constant-LR history;
it is an additional descriptive control, not a matched restart. E1000 warm
restart also carries 16,000 extra exposures that should be disclosed or matched.

## Resource estimate

The observed E1000 pilot trained 8,000 microbatches of two examples in 54m01s
on one shared RTX A6000, approximately 4.94 examples/second, with anomaly and
gradient audits enabled. At that observed throughput 1.28 million examples
would require about 72 GPU-hours; ideal two-GPU scaling would be 36 wall-hours.
Larger microbatches and omitting expensive engineering audits can improve
throughput, but neither improvement has been measured. These estimates are
extrapolations, not promised runtimes.

The A1 model has 102,254,168 trainable parameters and each full checkpoint is
about 1.64 GB. Ten new checkpoints require roughly 16.4 GB plus logs and any
input copy. Optimizer/weight/EMA state is only part of GPU memory; sequence
length and activations determine the microbatch limit. A 30 GB free-memory
launch check does not establish that microbatch 16 fits. The short throughput
check must measure actual peak memory before the long run.
