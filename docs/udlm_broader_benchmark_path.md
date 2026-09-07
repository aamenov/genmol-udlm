# Broader GenMol benchmarks and controlled PMO sampling

CPU/code audit, 2026-09-07; reference source `0fdc54d`. This adds an inference
adapter, not an executed benchmark or evidence of molecular improvement. No
oracle budgets, GPU jobs, docking jobs, or new model loads were used for this
implementation; its checkpoint tests use a tiny synthetic CPU fixture.

## Verified benchmark paths

| Task | Existing local entrypoints | Comparison and remaining work |
| --- | --- | --- |
| Property optimization (PMO) | `scripts/exps/pmo/run.py`, `main/genmol/hparams.yaml`, `eval.py`; controlled runner `scripts/exps/pmo/run_ablation.py` | The controlled `released` policy already has canonical-molecule caching, a strict call budget, saved population/events, and AUC trajectories. The opt-in adapter below supplies the missing UDLM law and checkpoint binding. |
| Fragment constraints | `scripts/exps/frag/run.py`, `hparams.yaml`, `data/fragments.csv`; shared `Sampler.fragment_linking`, `fragment_linking_onestep`, `fragment_completion` | UDLM supports fixed context with gamma zero. The runner also lists one-step linking. Add actual seed initialization and saved per-input results/identities before formal evaluation; retain empty-input failures and undefined diversity/distance rather than averaging only defined inputs without disclosure. |
| Lead optimization | `scripts/exps/lead/run.py`, `eval.py`, `docking/docking.py`, existing receptor/active-ligand files | Gamma-zero mutation is available. The current seed flag names outputs but does not initialize RNGs; evaluation hardcodes a 1,000 denominator and does not require improvement over the starting docking score. Bind seeds, actual attempts, docking versions/settings, and the starting-score comparison before a paper comparison. |

The paper specifies PMO's 23 tasks, 10,000-call budget, top-10 AUC, and three-run
means. Its no-MCG attaching-plus-fragment-remasking ablation totals **18.208**;
full GenMol totals **18.362**. Lead optimization has 30 tasks: five targets,
three starting ligands, and two similarity thresholds. It requires QED ≥0.6,
SA ≤4, similarity ≥0.4/0.6, and improved binding; docking results are three-run
means. Fragment settings use 100 requests per drug and three runs. These are
paper context, not values measured by this adapter.
[Primary GenMol paper, §§5.2–5.5, Tables 3–5, Appendix D](https://arxiv.org/html/2501.06158v3).

A full PMO panel therefore allows at most 690,000 online calls per model, or
1,380,000 for a matched pair. Existing task-specific `scripts/exps/pmo/vocab/*.csv`
contain offline fragment scores; reuse and hash the same files in both arms.
Their offline scoring is separate from the online budget. Do not silently run
`get_vocab.py` and exclude those new calls. The released optimizer changes to
remasking at `iteration > warmup`; the controlled runner exposes that exact
off-by-one convention with `--legacy-warmup-off-by-one`. With warmup 1,000, a
small-budget run may finish using attachment alone, so it cannot establish a
diffusion comparison. Neither a budget nor a task/seed panel is selected here.

The paper's gamma-zero PMO tasks are encoded in `run_ablation.PAPER_GAMMA`;
eleven tasks already use that setting. Gamma zero across all 23 tasks instead
compares to the no-MCG ablation. The fragment YAML also has nonzero-gamma tasks,
so replacing all guidance with zero must be labeled an ablation. In lead code,
stored `DS` is clipped negative docking energy (higher is better), while stored
`SA` is `(10 - raw_SA)/9`; its threshold `6/9` means raw SA ≤4. Generated
candidates are not equivalent to PMO's unique charged calls.

## Why official UDLM guidance is not a drop-in GenMol replacement

Current `src/genmol/sampler.py::generate` rejects nonzero `gamma*w` for UDLM.
GenMol gamma is the fraction of context masked for its self-guidance. Official
UDLM's `guidance.gamma` is a conditional-guidance strength: its conditional and
unconditional predictions depend on the trained class-conditioning/dropout
setup. For uniform diffusion, `_cfg_denoise` constructs reverse probabilities
before combining their logs. This does not supply GenMol's context self-guidance
for our existing unconditional checkpoint.
[Official UDLM repository](https://github.com/kuleshov-group/discrete-diffusion-guidance),
[pinned `_cfg_denoise` implementation](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1255).

A future context-guidance adaptation could compare two context-conditioned
categorical reverse laws at the same state/time/prior, combine them in log
probability space, and clamp immutable context. That is a hypothesis requiring
normalization/limiting-case tests and explicit extra NFE accounting, not released
behavior. It is outside this adapter.

## Opt-in PMO contract

`scripts/exps/pmo/run_ablation.py` now accepts `--sampling-config PATH` for a
fresh `--variant released` run. The YAML requires `checkpoint_sha256` and an
explicit `diffusion_type`, plus the existing de novo sampling schema:
temperature, randomness, effective `min_add_len: 18`; UDLM additionally declares
`num_steps`, `inference_eps`, special-token support, prior variant/hash, and
optional CE parameterization/temperature space. The common benchmark validator
enforces their compatibility. Gibbs is initially excluded. YAML temperature and
randomness override the old CLI values and are copied into the resolved run
config, so recorded values equal forwarded values. UDLM ignores the MDLM-only
randomness parameter; this is explicit in its receipt.

The helper `scripts/exps/pmo/udlm_sampling.py` verifies checkpoint bytes before
loading, checks actual diffusion/prior/CE identity, requires installed verified
EMA weights, and records tokenizer and source hashes after reusable validators
run. Opt-in preflight occurs before constructing the oracle. UDLM gamma must be
exactly zero, even if a caller attempts to disable guidance through its scale.
Implicit UDLM without a YAML fails before any score. Without the new option,
MDLM keeps its existing configuration/artifact fields and constructor order.

The adapter calls the existing `mask_modification` with the declared NFE and
temperature law. Backbone hooks record actual calls, including failed attempts;
accepted candidate events retain that candidate's counts, while summary totals
also include rejected proposals. Attaching-only warmup has zero NFE. Ordinary
pre-generation SAFE failures retain the released parent fallback and are counted
explicitly at zero NFE. Failures inside `generate`, including interruptions
otherwise swallowed by `addmask`, propagate. Multiple generation calls or a
configured/observed UDLM NFE mismatch fail. Source/YAML hashes are rechecked
before terminal acceptance; failures retain the runner's failed summary.

The released completion path calls `_insert_mask` without forwarding its length
kwargs: effective minimum added length is **18**, and `mask_len` from `addmask`
is ignored there. Longer inputs retain released remasking with a random 5–15
token chunk subject to capacity. This adapter preserves that behavior; it does
not introduce a length repair, running-mean population rule, or delta credit.
It rejects opt-in resume initially. Actual PMO execution still needs a separate
prospective budget/seed/checkpoint panel, reviewed device allocation, named tmux,
and a frozen source. No command here starts that campaign.
