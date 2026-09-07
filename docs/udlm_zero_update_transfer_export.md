# Zero-update MDLM EMA transfer exporter

This is an inference-only artifact builder, not training or a molecular experiment.
`scripts/udlm/export_zero_update_transfer.py` gives frozen MDLM50k EMA logits an
explicit categorical interpretation and a positive MASK-rich prior. It performs
no model forward, optimizer step, data loading, property evaluation, or GPU work.
No actual full-size checkpoint has been exported as part of implementing it.
An actual export panel and its inputs must be fixed and pushed separately before
execution.

The two supported selectors, `raw_loo` and `x0_denoiser`, specify how the sampler
would interpret the copied logits. Neither selector establishes CT or CE training,
calibration, molecular improvement, or equality to an absorbing MDLM reverse law.
In particular, the copied logits need not give visible observations point-mass
clean predictions, and they need not suppress the clean MASK logit. Existing CE
metadata includes the configured objective name for loader compatibility; the
additional `zero_update_transfer_metadata` explicitly records
`trained_as_ct_or_ce=false`, `training_objective_applied=null`, and zero UDLM
updates/exposures. Source MDLM step 50000 and its actual positive EMA count remain
separate historical fields.

## API and input contract

The CLI requires `--config`, `--source-checkpoint`, `--expected-source-sha256`,
`--seed`, and `--output-dir`. It accepts a JSON or YAML **fully resolved** config
containing only `model` (the source-compatible BERT architecture) and `training`.
Hydra defaults/interpolations, optimizer settings, trainer settings, loader
settings, callback settings, and implicit parameterization selection are rejected.

The exact `training` keys are `diffusion: udlm`, `ema` strictly between zero and
one, boolean `antithetic_sampling`, `global_mean_loss`, `use_bracket_safe`,
`sampling_eps` strictly between zero and one, and `udlm`. The latter requires:

| Field | Required value |
|---|---|
| `prior_variant` | `mask_rich_empirical` |
| `parameterization` | explicit `raw_loo` or `x0_denoiser` |
| `mask_mixture_weight` | finite scalar strictly between 0 and 1 |
| `empirical_uniform_mix`, `noise_eps` | finite scalars strictly between 0 and 1 |
| `exclude_special_tokens` | `false` (full corruption alphabet) |
| `mask_all_special_tokens` | `true` (configured clean-position mask; no training occurs) |
| `conditioning_variant` | `film_adaln` |
| `zero_init_conditioning` | `false` (normally initialized time MLP; zero FiLM output projections) |
| `time_embedding_size` | positive integer |

The stationary distribution is
`pi_lambda = lambda * delta_MASK + (1 - lambda) * pi_E`, where `pi_E` is rebuilt
by the unchanged model code from its pinned token frequencies and explicit
uniform floor. For `0 < lambda < 1` the existing implementation retains positive
full support. Its exact metadata, prior probabilities/mapping, lambda state marker,
and optional denoiser marker are included in the checkpoint and validated by the
unchanged strict model hooks. Metadata hashes use canonical compact sorted JSON;
file hashes refer to the exact saved bytes.

Use the project `.venv`. The CLI hides CUDA before importing Torch and requires
offline model/tokenizer assets; every explicit input/output artifact must resolve
inside the project. Execution requires a clean source checkout at its pushed
upstream revision. The output directory must be fresh. No default source
checkpoint, lambda, seed, or output namespace is silently chosen.

## What is checked

The source checkpoint's expected SHA-256 is verified on a stable descriptor before
and after deserialization. It must identify MDLM at global step 50000 with a finite,
positive-update EMA of the correct parameter topology. A separate MDLM model is
constructed on CPU, its actual checkpoint hooks and strict state loader run, and
its EMA is applied. Its complete **named** backbone state supplies an independent
reference for all parameters, persistent buffers, and tied state aliases. All
named buffers, including nonpersistent BERT position/type buffers, are also
recorded. Parameters come from source EMA; persistent buffers come from the source
raw checkpoint, while nonpersistent buffers come from its strict architecture.
The source config fingerprint covers its complete **unresolved** configuration,
with `config_sha256_scope` stating that meaning. Unused callback/output/schedule
interpolations are retained literally rather than eagerly resolved or assigned
invented values. GenMol receives that OmegaConf structure and resolves the fields
its constructor actually accesses normally; an unresolved required architecture
field still fails. The explicit target export config remains fully resolved.

The categorical target is initialized using the existing
`initialize_from_mdlm_checkpoint(use_ema=True, expected_sha256=...)`. Its full base
state and normalized BERT configuration must match the independent reference
exactly; a same-shaped activation or normalization-setting change is rejected.
Every new FiLM output
weight and bias must be exactly zero, so finite time embeddings have no modulation
effect. The time MLP itself is newly initialized from the explicit seed. Every
fresh target EMA shadow must equal its corresponding initialized parameter
exactly, with `num_updates=0`; the saved EMA is not presented as an average over
UDLM training. All target state tensors must be finite and on CPU.

The artifact is serialized, then `GenMol.load_from_checkpoint(...,
map_location='cpu', strict=True)` runs without a Trainer. The independent base,
buffer, FiLM, EMA, prior, and parameterization checks run again, and the complete
target state tensor digest table must match the pre-save table. Source checkpoint,
exporter/runtime source, and config bytes are rechecked before completion.
The actual serialized target tensor table is also checked before loading, because
strict PyTorch loading can still cast dtypes or overwrite inconsistent tied aliases.
The independently reconstructed source raw state must match its serialized tensor
table before EMA is applied for the same reason.

## Honest inference-only serialization

The checkpoint contains the installed Lightning version, `state_dict`,
`hyper_parameters.config`, zero `global_step`/`epoch`, truthful EMA state,
prior/conditioning/optional denoiser metadata, and transfer provenance. The only
`loops` fields are two zero completed counters read by the existing
`fast_forward_info` loader. They are **serialization compatibility fields**, not
evidence that a Trainer ran. No optimizer, scheduler, callback, or dataloader state
is created. The exporter does not call the Trainer-dependent `on_save_checkpoint`
hook or attach a fake Trainer.

The exclusive bundle contains `request.json`, `transfer.ckpt`,
`resolved_config.json`, `exporter_source.py`, and `manifest.json`. Only a successful
strict roundtrip and final input checks produce `status=completed`; a failure after
reservation leaves an explicit failed manifest and any existing diagnostic bytes.
Existing directories are never reused or overwritten. Manifests bind source,
config, checkpoint, reference-state digests, software versions, and runtime.

The existing sampler can inspect an EMA with zero updates. The existing **trained
benchmark validator deliberately rejects it** when EMA is required, because that
gate demands a positive update count. No loader, benchmark validator, final-study
gate, or production sampling code is modified here. `validate_checkpoint` is the
exporter's additional provenance check; the ordinary GenMol loader does not by
itself validate the custom transfer record. Any future zero-update diagnostic
must explicitly verify the completed export bundle, rather than relabel this
artifact as a trained checkpoint or alter its EMA count.

## Verification scope

Tests use tiny real BERT/GenMol architectures with synthetic tokenizers, frequencies,
and source checkpoints. They exercise actual serialization, strict Lightning load
hooks, named buffers/tied weights, deliberately distinct raw/EMA weights, zero FiLM,
and metadata/state corruption. Model forwards, Trainer/optimizer construction,
CUDA discovery, real source checkpoints, and molecular/oracle evaluations are not
part of the exporter tests. These checks establish artifact consistency only.
