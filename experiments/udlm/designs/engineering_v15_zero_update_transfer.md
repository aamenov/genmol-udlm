# V15 prospective zero-update transfer probe

Declared 2026-09-07 before exporting these checkpoints or generating any V15
molecules. This is an adaptive engineering probe of a new interpretation of
frozen MDLM weights. It is not a trained CT/CE comparison, an exact reproduction
of released uniform UDLM, or a final GenMol-superiority study. V14 remains fixed.

## Motivation and known observations

Earlier trained UDLM pilots remain below the local MDLM50k de novo reference.
The separately published frozen CPU diagnostic evaluated sixteen reused rows
under increasingly MASK-rich corruption. Token reconstruction improved while
the task became easier, and unmasked raw logits were not point-mass predictions.
The mathematical audit at `codex/udlm-near-absorbing-limits` distinguishes
raw-LOO and clean-denoiser interpretations of these logits: a near-MASK prior
alone does not make them equivalent. Raw-LOO copying requires positive observed
clean-token weight; a fixed misspecified clean posterior can still flip a
visible token. Temperature after D/L conversion changes this limit further.

Those findings, prior de novo results and the ongoing V14 runtime observations
informed this design. No V15 molecular outcome is available. The CPU diagnostic
is reused exploratory evidence, not held-out confirmation. The two finite
mixtures below keep full support and avoid claiming the singular absorbing
endpoint. A molecule-level result is needed to assess this transfer hypothesis.

## Frozen source and derived checkpoint identities

All derivatives start from the **same MDLM50k EMA**, SHA-256
`8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`,
at `/home/aidar.alimbayev/Documents/genmolv2/outputs/paper_v1/checkpoints/50000.ckpt`.
Export exactly four inference-only checkpoints, crossing:

- MASK prior mixture **0.99 / 0.999**, each with the existing empirical prior
  and uniform floor **0.0002**, all1,880 corruption categories active;
- **raw_loo / x0_denoiser** parameterization, interpreted at inference only.

Every export uses initialization seed **2509**, neutral A1 FiLM conditioning,
the unchanged BERT architecture, and zero optimizer updates/data exposures.
The copied base parameters must exactly equal the original source EMA; copied
buffers and any tied aliases must also be checked. Neutral conditioning and
fresh EMA shadows must be independently verified before and after a strict CPU
checkpoint-load roundtrip. New EMA `num_updates=0`, `global_step=0` and
inference-only compatibility counters remain honest. No fake trainer,
optimizer state, scheduler or training history is created.

The two parameterizations must have identical named backbone tensors at each
mixture and across mixtures. Their process/parameterization buffers and metadata
may differ as explicitly declared. Existing source or trained checkpoints are
never relabeled. Fully resolved export configs, exporter source, exact output
checkpoint hashes and transfer manifests must be pushed/bound before generation.
The old trained-EMA benchmark and final-study guards stay unchanged; these
zero-update artifacts require their own explicit probe and verification path.

## Fixed molecular panel

Seven configurations, each with **seeds2500/2501 and100 requested samples**:

| Configuration | Mixture | Interpretation | Temperature space | Temperature |
|---|---:|---|---|---:|
| `mdlm` | absorbing reference | original MDLM | original confidence sampler | 0.5 |
| `r099` | 0.99 | raw LOO | raw LOO | 0.5 |
| `ce_raw099` | 0.99 | clean D, then D/L | raw LOO after conversion | 0.5 |
| `ce_clean099` | 0.99 | clean D, then D/L | clean D before conversion | 0.5 |
| `r0999` | 0.999 | raw LOO | raw LOO | 0.5 |
| `ce_raw0999` | 0.999 | clean D, then D/L | raw LOO after conversion | 0.5 |
| `ce_clean0999` | 0.999 | clean D, then D/L | clean D before conversion | 0.5 |

This is **14 runs / 1,400 requests**, retaining every requested slot, including
invalid outputs. All use `min_add_len=40`, the same frozen length distribution
and tokenizer, no molecular context guidance, and a single batch of100 requests.
UDLM uses **128 predictor steps**, top-p1, no Gibbs corrector, inference
epsilon1e-5, and ignored confidence randomness0. MDLM uses its original adaptive
confidence decoder, temperature0.5 and randomness0.5. Common checkpoint noise
epsilon is0.001. MDLM and UDLM have different laws/NFE; this tests configured
pipelines, without a matched-compute claim.

Execute in the displayed configuration order for seed2500, then that same order
for2501, in waves of at most two freshly qualifying GPUs. Inspect utilization,
free memory and active processes immediately before each launch; utilization
must be strictly below10% and free memory at least30,000MiB. UUID-map logical
devices dynamically, never choose a fixed physical GPU. Only after V14 and its
independent acceptance terminate may this study acquire the generation lease.
Use a named tmux session, complete logs and a fresh exclusive namespace.
Allow up to six hours total capacity waiting across waves; cap each child at
one hour. Any launch/child/identity failure stops remaining jobs and preserves
completed, failed and unlaunched entries. No automatic retries or namespace reuse.

## Metrics, comparisons and scope

Reuse the existing released-comparable and strict de novo definitions. For
each branch: validity is valid/requested; uniqueness is unique/valid; quality is
unique valid molecules with **QED>=0.6 and SA<=4.0**, divided by requested;
diversity is the pinned evaluator on unique valid molecules. The released branch
includes SAFE repair and largest-component selection; the strict branch uses
direct decode and explicit RDKit sanitization. Retain raw token IDs, original
editable mask, all decoded rows, repair/failure categories and final controls.
Record actual forwards, candidate-equivalent work, loading/generation/metric
runtime, source/inputs/device/seed and precise score counts. Metric evaluation
is distinct from V14's Fexofenadine optimization budget.

Primary mechanistic contrast: **r0999 minus ce_raw0999**, by same seed, for
released and strict quality. Secondary contrasts retain r099 minus ce_raw099,
ce_clean minus ce_raw at each mixture, and0.999 minus0.99 within each of the
three interpretations. Also retain **all six candidate-minus-MDLM** contrasts
for validity, uniqueness, quality and diversity in both metric branches, all
per-seed values, mean signed differences and sample SDs. No cherry-picked
winner-only table, pooled seed denominator or comparison of this small adaptive
probe to the paper mean as though it were an independent reproduction.

Every run needs independent CPU decoding/rescoring and exact transfer/source
identity acceptance before any engineering comparison is qualified. A candidate
is eligible only for a separately declared replication when both paired
released-quality differences exceed zero, mean strict-quality difference is
nonnegative, and mean released validity/uniqueness/diversity differences are
each at least-0.01. Undefined metrics or incomplete runs withhold eligibility.
All14 runs must complete and pass acceptance; retain every qualifying arm.
The0.01 tolerance is an engineering screen, not statistical noninferiority.

Two seeds and adaptive choice support no significance or general superiority
claim. Final de novo seeds0/1/2 remain reserved for a separate prospective study.
The final PDF must retain all configurations, signed effects, failures,
transfer/compute caveats, and earlier study appendices. The overarching goal
remains to improve GenMol benchmarks; this probe is a bounded next experiment.
