# Standalone zero-update transfer probe

`scripts/udlm/run_transfer_denovo_probe.py` is an engineering diagnostic driver.
Its artifact kind is `zero_update_transfer_denovo_probe`, distinct from accepted
trained-checkpoint benchmark artifacts. It performs one seeded generation batch
and preserves every requested row, including decoding failures. No actual molecular
run is part of implementing this driver. The prospective V15 panel is declared
separately; this script does not select candidates, schedule a campaign, retry
jobs, or claim an improvement.

The two arm kinds are `frozen_mdlm_ema_transfer` and `mdlm_ema_reference`. A transfer
must have an independently completed export proof, exact zero UDLM updates and
exposures, actual applied EMA with zero updates, and the selected `raw_loo` or
`x0_denoiser` interpretation. These selectors do **not** imply CT/CE training or
calibrated predictions. The reference must be ordinary MDLM at step 50000 with
actual positive-update EMA. Existing trained benchmark validators and their
positive-EMA requirement remain unchanged.

## Inputs and default preview

The spec has exactly these fields:

| Field | Meaning |
|---|---|
| `schema_version`, `artifact_kind` | `1`, `zero_update_transfer_denovo_probe` |
| `entry_id`, `arm_kind` | Unique run identifier and one of the two kinds above |
| `seed`, `num_samples` | Explicit integer seed and requested batch size |
| `checkpoint`, `sampling_config`, `design` | Pinned file references |
| `export_manifest` | Pinned completed export manifest; `null` for MDLM |
| `input_sha256` | Exact role-to-SHA mapping from `direct_input_records(sampling)` |

A file reference contains `path`, `sha256`, and optionally `size_bytes`. Relative
paths resolve under `--input-root`; every artifact must remain inside the project.
The direct-input map binds live generation/helper source bytes, the empirical
frequency artifact for transfers, generation-length bytes, the pinned SA fragment
table, and TDC metric implementation files. The tokenizer's existing constructor
checks its fixed revision/file digest, and the loaded tokenizer vocabulary and
backend identities are recorded. The spec and source revision are supplied
separately to avoid a circular commit/spec hash dependency.

Without `--execute`, the CLI only validates source, configuration, hashes, and the
export's saved proof closure. It does not unpickle a checkpoint or length/SA table,
construct a model, query CUDA, evaluate metrics, or create output directories:

```bash
.venv/bin/python scripts/udlm/run_transfer_denovo_probe.py \
  --input-root /absolute/project/input/root \
  --spec relative/spec.json --expected-spec-sha256 SHA256 \
  --expected-source-revision PUSHED_COMMIT
```

The source must be clean and match its pushed upstream revision. Actual execution
additionally requires the project `.venv`, offline model/tokenizer assets,
`--execute`, a fresh `--output-dir`, and `--device`. CPU is supported; CUDA requires
exactly one UUID in `CUDA_VISIBLE_DEVICES` and logical `cuda:0`. Immediately before
device placement the worker checks that UUID, utilization below 10%, and at least
30000 MiB free. A separate campaign controller owns global leases, the two-GPU
limit, durable execution, timeouts, and prerequisite gates.

Sampling YAML uses the existing normalizer, with an explicit allowed-field list.
Any supplied model path or sample count must equal the spec. Transfers require
the full-alphabet MASK-rich empirical prior, top-p 1, and predictor-only sampling.
Context guidance and Gibbs correction are rejected. The existing temperature-space
validation is retained: optional clean-denoiser temperature requires the denoiser
interpretation. Device selection comes from the CLI and is recorded separately.

## Proof, generation, and saved evidence

Before device placement, the driver loads the pinned checkpoint on CPU, applies
its real EMA, and validates its actual state. For transfers it checks the embedded
export record, zero counters/compatibility loops, absence of training state,
serialized tensor hashes before loader normalization, complete base-state and
buffer identity, neutral FiLM projections, prior, interpretation, and fresh EMA.
JSON object-key normalization follows the exporter's serialized comparison; tensor
hashes remain exact. The original source MDLM checkpoint is not loaded again for
each transfer: its frozen EMA reference is taken from the pinned completed export,
and actual target tensors must match that reference. The archived exporter and
its runtime source bytes must match the current helpers, even if Git HEAD differs.
Unused MDLM configuration fields are not eagerly resolved by this driver.

Generation reuses `benchmark.generate_raw_model_text` and the existing sampler.
Backbone hooks observe every batch size: NFE must equal the returned protocol count
and every call must process the entire requested batch. Candidate-equivalent work
is the sum of observed batch sizes; reference MDLM uses its actual confidence-based
step count. Equal row-weighted NFE does not equalize token lengths, padding, attention
cost, or wall time. Initial/final token IDs and the immutable editable-position
mask are retained using the existing exact binary token audit. BOS/EOS/PAD framing
must remain unchanged. Counts include all five final editable control IDs; these
are final occurrences, not counts of trajectory visits.

The exclusive output directory contains:

- `request.json`: source, full spec/input bindings, execution device and start time.
- `raw_generation.json`: all raw text slots, generation protocol and exact token audit,
  saved before chemical decoding.
- `raw_samples.csv`: one ordered row per request, using the existing raw-sample schema
  and LF line endings, including failed strict/released decodes.
- `summary.json`: normalized sampling, checkpoint/EMA proof, source/software/device,
  tokenizer/metric inputs, both metric branches, controls, observed NFE and timings.
- `terminal_manifest.json`: completed/failed execution status, request/output hashes,
  final source/input revalidation, elapsed time, and an error when applicable.

A late source/input mismatch makes the terminal failed, even if diagnostic outputs
were already written. Existing directories are never reused. An exception retains
available evidence; a hard process kill may leave no terminal, which the external
controller must classify as incomplete. Completed execution remains
`pending_independent_rescore`, with `final_promotion_eligible=false`.

## Metrics and limits

The existing decoder and evaluator are reused unchanged. Released-comparable
decoding applies SAFE repair and retains the largest disconnected component by
the released string-length rule. Strict decoding performs no repair or component
selection. In each branch, validity is valid rows divided by requested rows;
uniqueness is unique valid strings divided by valid rows; quality is unique valid
strings satisfying QED >= 0.6 and SA <= 4 divided by requested rows. Diversity uses
the existing TDC evaluator on unique valid strings. Empty valid sets leave
uniqueness/diversity undefined, not zero. QED estimates drug-likeness; SA is the
synthetic-accessibility heuristic, with lower values indicating easier synthesis.
Neither proves efficacy or practical synthesizability.

Scoring uses the pinned QED/SA/diversity implementations and pinned resident SA
table. Generation runtime includes sampling/tokenizer work plus released repair;
strict decoding and metric evaluation are separately timed. Independent CPU
rescoring and complete paired reporting are separate required follow-ups, not
implemented or claimed by this generation driver. A finite adaptive pilot,
zero-update transfer, or successful serialization is not evidence of GenMol
superiority. Tests use synthetic small checkpoints and deterministic fake sampling
and metric functions; no actual model checkpoint, oracle task, or GPU is exercised.

Run focused checks with pytest's temporary directory inside the project, for
example `--basetemp=output/tmp/probe_tests`, because artifact validation deliberately
rejects paths outside the project. Use a fresh test directory for each run.
