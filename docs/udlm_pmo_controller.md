# Finite PMO campaign controller

`scripts/udlm/launch_pmo_campaign.py` runs an explicitly declared engineering
panel through the released PMO population policy and opt-in sampling adapter.
It does not select a task, checkpoint, seed, or scientific comparison. A panel
must be committed and pushed before live execution. No study panel is supplied
by this implementation change.

Invoke the project `.venv/bin/python` from the executing checkout, first with
`--panel <checkout-relative JSON> --panel-sha256 <actual SHA-256>`
`--gpu-count 1|2 --dry-run`. A dry run checks all requests and actual checkpoint
metadata on CPU; it does not reserve output, query GPUs, construct an oracle,
or sample. It does not claim full EMA application: the child adapter verifies
the loaded EMA tensors before constructing its evaluator.

Live execution uses the same arguments without `--dry-run`, from a named tmux
session whose controller stdout/stderr are logged under `output/logs/`.
The checkout must be clean and equal its pushed upstream. Unset a pre-existing
`CUDA_VISIBLE_DEVICES`; specify a count, never a physical index.

## Panel schema 1

The top-level keys are exactly:

- `schema_version: 1`, `campaign_id`, `oracle`, and a nonempty `scientific_status`.
  The task must be in the runner's `ORACLES` list. All children use explicit
  `--variant released --gamma 0`; unsupported task/library combinations must
  be rejected by the separate scientific preflight.
- `output_root` under `output/udlm/` and `log_root` under `output/logs/`.
  Both namespaces must be absent. Existing state is never resumed or replaced.
- Finite positive `child_timeout_seconds`; integer `capacity_wait_seconds`
  from zero through 21,600. Zero requests immediate capacity admission.
- `runner`, containing exactly `max_iterations`, `population_size`, `warmup`,
  Boolean `legacy_warmup_off_by_one`, `reporting_frequency`, `checkpoint_every`,
  `guidance_scale`, `min_mol_size`, and `max_mol_size`. Counts are explicit;
  iterations must exceed warmup. The released per-task size rules still apply.
- Nonempty `entries`, each containing exactly `id`, `seed`, `checkpoint`,
  `checkpoint_sha256`, `sampling_config`, `sampling_config_sha256`, `vocabulary`,
  `vocabulary_sha256`, and `max_oracle_calls`. Entry IDs are unique. Each budget
  must exceed 1,000; each seed is a nonnegative integer. Sampling YAML controls
  temperature and randomness and must use the adapter's effective length 18.
- Nonempty `input_files`, each exactly `{ "path": ..., "sha256": ... }`.
  Pin task implementation/package metadata/native libraries and any oracle
  model. GSK3B additionally requires `oracle/gsk3b_current.pkl` to be pinned;
  this condition alone does not establish model/library compatibility.

File paths may be checkout-relative or absolute within the containing project.
Use absolute paths for shared `.venv` packages outside the checkout. Paths with
parent traversal or symlink components are rejected. Output/log paths remain
checkout-relative. Hashes are lowercase SHA-256. Duplicate JSON keys and
non-finite values are rejected. Checkpoint bytes, CE/prior metadata, vocabulary,
all YAMLs, and auxiliary inputs are checked before any process starts. Each
child repeats full adapter checkpoint/EMA validation before oracle scoring.

## Execution and acceptance

Jobs run in declared waves of up to the requested count, never more than two.
Each wave inventories every physical GPU. Each child's chosen UUID is probed
again immediately before its `Popen`; utilization must be strictly below 10%
and free memory at least 30,000 MiB. Existing compute processes are allowed
under those same guards and their inventory is retained. The UUID maps to
logical `cuda:0` only inside that child.

All children set `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `TOKENIZERS_PARALLELISM=false`. The runner's seed
argument still controls its explicit random streams. Sampling YAML values
override the runner's legacy temperature/randomness defaults.

The optional capacity wait polls every 30 seconds only before the first child
starts. A final-probe rejection fails the attempt; no child is retried. After
any `Popen`, insufficient capacity, a child failure, changed source/input, or
timeout stops the campaign and prevents later waves. The controller only
signals process groups it created with `start_new_session=True`: TERM, up to
10 seconds, then KILL and up to 5 seconds. Ordinary parent exit uses the
existing 15-second process-group grace before acceptance. A remaining group or
changed lease prevents release of the exact repository generation lease.

Success requires exit zero, group exit, exact run/config/checkpoint/source/
sampling bindings, verified EMA receipt, and the full declared oracle budget.
`max_iterations_reached` is a failed campaign, not full-budget success. The
controller binds saved PMO score artifacts; it does not independently reevaluate
oracle scores or establish a molecular improvement.

## Artifact locations

Each run writes beneath
`<output_root>/runs/<entry.id>/<oracle>/released/seed_<seed>/`:
`manifest.json`, `summary.json`, `events.jsonl`, and `state/latest.pkl`.
Each stdout/stderr log is `<log_root>/<entry.id>.log`.

The controller writes exclusive `request_manifest.json`, per-entry
`<entry.id>.launch.json`, numbered `telemetry/*.json`, and
`terminal_manifest.json` directly under `output_root`. The request contains
the panel hash, all input SHA-256/size records, normalized child configurations,
exact commands, source commit, and resource policy. The terminal records PID,
UUID/physical mapping, return code, elapsed time, acceptance artifact hashes,
failures/timeouts/cleanup, unlaunched entries, final input validation, and log
hashes. A launch receipt records intent before `Popen`; a non-null PID in the
terminal records that a child actually started. Failure evidence remains intact.
Lease release is authorized only after terminal evidence is durable and all
owned process groups have exited. No stale lease is automatically removed.
