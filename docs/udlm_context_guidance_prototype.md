# Experimental posterior-space context guidance API

This prototype implements the [reviewed categorical hypothesis](udlm_context_guidance_hypothesis.md)
as a separate sampler method. It has been tested with synthetic CPU predictors
and categorical processes only. There is no molecular efficacy result, GPU
run, oracle evaluation, new training, or new benchmark protocol in this change.

The public entrypoint is `Sampler.generate_context_guided(token_ids, *, config)`.
It returns `{"token_ids": Tensor[B,L], "receipt": dict}`. It does not decode,
repair or filter molecules. An already constructed sampler must have recorded
EMA inference weights, a validated categorical S/E/MASK process, and an
explicitly evaluated model (`sampler.model.eval()`). Legacy released-uniform
R and MDLM processes are rejected. The method does not change evaluation mode
or load a checkpoint itself.

```python
# Given an already initialized sampler and int64 token_ids[B,L]:
sampler.model.eval()
result = sampler.generate_context_guided(
    token_ids,
    config={
        "schema_version": 1,
        "method": "posterior_context",
        "gamma": 0.2,
        "scale": 2.0,
        "context_seed": 419,
        "predictor_steps": 128,
        "execution": "serial",  # or "packed"
    },
)
```

The configuration must contain exactly these fields. `gamma` is in [0,1],
`scale` is finite and at least 1, `context_seed` is an integer in
[0,2^63−1], and `predictor_steps` is a positive integer. Temperature is fixed
at 1, top-p at 1, and Gibbs correction is disabled. `predictor_steps` counts
transitions; it does **not** redefine the existing `generate(num_steps=...)`
NFE argument. Unknown keys and unsupported identities fail before RNG draws.

Original editable positions are captured before the prior draw. Context
eligibility is computed independently for each row from original immutable
positions, excluding every tokenizer-declared control ID. Each active step
selects `floor(gamma * eligible_count)` positions per row using a dedicated,
seeded CPU Torch generator. Masking affects the poor predictor's view only.
Every editable noisy token, time and prior supplied to both analytic bridges
is the same original state. CE predictions receive the audited D/L conversion;
raw-LOO predictions retain their existing meaning.

For active guidance, the helper obtains float64 reverse probabilities from the
existing categorical process, combines their logarithms with coefficients
`scale` and `1-scale`, and normalizes. It rejects nonfinite logits at editable
active-vocabulary entries, malformed laws, arithmetic overflow, and any loss
of positive support after exponentiation. It applies no floor or clipping to
editable probabilities. Unused distributions at immutable coordinates are
replaced by uniform active-alphabet dummies, which prevents irrelevant
excluded-token sentinels from causing underflow failures. Their sampled
values are discarded, and every immutable output is clamped to its original
token. The receipt records this dummy policy. This token invariant does not
prove substructure retention after a later decoder or repair operation.

`gamma=0`, `scale=1`, no selected context, or no editable tokens delegate to
the exact existing predictor sampler at temperature 1. They do not construct
or draw from the context generator, make a second prediction, or use the new
float64 guidance arithmetic. Synthetic tests require identical raw IDs and
Python, NumPy and Torch RNG states against the historical call. The active
float64 path is a declared numerical difference from lower-precision
historical inference; packed and serial calls are required to agree within
tolerance on deterministic fixtures, not bit-for-bit on arbitrary kernels.

Serial active execution makes two predictions of batch B per transition.
Packed execution makes one prediction of batch 2B. Both count 2BN
candidate-equivalent evaluations for B candidates and N transitions. A
backbone hook records every actual input batch size and rejects an unexpected
work count. Mixed batches still evaluate both full B-row branches, including
rows without eligible context; those rows use their conditional reverse law.
The receipt does not claim per-row RNG or trajectory equality with separately
sampled rows. Identity fastpaths count BN evaluations.

The returned receipt includes the normalized configuration and hash, exact
source hashes, runtime prior metadata and tensor identities, CE identity where
applicable, the load-time EMA receipt, precision/order/control policies,
original-input/edit-mask hashes, selected-subset digest and draw count,
observed forward batch sizes, candidate-equivalent counts, runtime, and
before/after sampling RNG-state hashes. All returned metadata is detached from
caller/model dictionaries. Source and runtime identities are rechecked before
return. A thrown exception returns no completed receipt; callers must preserve
that failure rather than silently retrying it.

This is **not a benchmark acceptance receipt**: it explicitly records
`checkpoint_bytes_revalidated=false`. The sampler method does not reopen or
rehash neural checkpoints, and the load-time EMA fact does not certify that
external code has never modified backbone weights. A later experiment must
bind the actual checkpoint, complete runtime setup, seeds and source, and
independently preserve generated-token and molecular evidence.

The old `generate` defaults retain their exact existing execution and RNG
paths, and this helper is imported only when the new method is called.
Two narrow guards reject the reserved `context_guidance` key or exact
`method="posterior_context"` value in `generate` and benchmark normalization;
otherwise their existing code is unchanged. This prevents explicit new
settings from silently running an unguided sampler. Other historical unknown
keys retain their previous behavior. The PMO adapter remains unchanged and
keeps its UDLM gamma-zero restriction. No general
`generate(context_guidance=...)` keyword API is introduced. Enabling
fragment/PMO guidance requires separate prospective configuration, artifact,
NFE and evidence integration and review.
