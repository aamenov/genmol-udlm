# V14: prospective matched-budget PMO pilot

Declared before any PMO optimization, oracle evaluation or molecular outcome in
this study. This is a task-specific engineering screen, not a full GenMol
benchmark or a claim of superiority. All six scheduled runs remain reportable.

## Question and candidate selection

Can an existing UDLM sampler improve fragment-guided property optimization
under the released GenMol population policy and the same online oracle budget?
Poor de novo quality need not imply poor conditional local search; this is a new
hypothesis, not an observed result. No new model training belongs to this panel.

The task is **fexofenadine_mpo**, which already uses gamma zero in the released
GenMol settings. It combines atom-pair similarity to a fixed reference molecule,
TPSA and logP. The exact installed PyTDC 0.4.1 implementation and reference
string will be hash-pinned by the executable protocol. It needs no learned
oracle checkpoint. GSK3B was initially considered, but a read-only inspection
found a scikit-learn 0.23.0 pickle and installed 1.7.2 tree-schema incompatibility.
No GSK3B model was unpickled or scored and no task outcome informed this choice.
The deferred GSK3B model's SHA-256 is
`d3a20701b80e5179c88c3ad4dc3483dd7ab35c50dc055c6773a7f5b63e89b6d5`.

Three fixed arms use explicit PMO sampling YAMLs:

| Arm | Checkpoint and training | Sampling |
| --- | --- | --- |
| MDLM control | Local GenMol 50k checkpoint, EMA | Released PMO temperature 1.2, confidence randomness 2.0; adaptive observed NFE |
| MASK CE (primary treatment) | V11b MASK-rich CE, 1,000 batch-128 updates from the same MDLM EMA, prior MASK weight 0.9 | Raw-LOO temperature 0.5, 128 predictor NFE, top-p one, no Gibbs |
| S CT (secondary treatment) | Schedule-consistent uniform S, 1,000 batch-16 updates from the same MDLM EMA | Raw-LOO temperature 0.5, 128 predictor NFE, top-p one, no Gibbs |

Checkpoint SHA-256 values, respectively:

- MDLM: `8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`.
- MASK CE: `62299de99d8c003e3776215efce9643cca196091a6284f351de2b404a22c2e8d`.
- S CT: `100f467b94766f2c87cc734398c8590bd14a1c8e0722ce31dbca7b446d986e72`.

Both UDLM arms use full active support, endpoint 1e-5 and the checkpoint's
verified prior. CE uses its required D/L conversion. Temperature remains in
raw-LOO space; V13 did not establish a general benefit for the new clean-space
option. The unused UDLM randomness field is zero. All arms require actual EMA
weights and preserve the released completion minimum added length 18 and
long-input remasking rule. Their training amounts and NFE are unequal and must
be reported; the comparison matches oracle budget and population policy.

Candidate and temperature choices are adaptive to earlier de novo studies:
MASK CE is the newer transfer hypothesis with higher strict validity in V13's
raw mode; S CT provides the historical schedule-consistent alternative. This
panel does not tune either on PMO outcomes. The historical best Gibbs result is
not silently reproduced with a predictor-only setting: Gibbs is outside this
initial PMO adapter and each configured sampler is labeled explicitly.

## Fixed panel and optimizer policy

Fresh seeds **2300 and 2301**, each with **2,000 unique canonical molecule oracle
calls per arm**: six runs and at most 12,000 online oracle calls. Final de novo
seeds 0/1/2 remain unused. The existing task vocabulary is identical and hashed
for every arm; its offline fragment scores are separate from the online budget.
Do not regenerate it. No outcome-dependent retries, substitutions, temperature
changes, early winner selection or expansion belong to V14.

All runs use `run_ablation.py --variant released`, gamma 0, guidance scale 2,
population size 100, warmup 1,000 with `--legacy-warmup-off-by-one` (remasking
starts at iteration 1,001), molecule size 20..40 atoms, reporting/checkpoint
interval 100, maximum 5,000 iterations and a 3,600-second child wall limit.
The existing 1,000-proposal inner limit and released duplicate population
updates remain unchanged. Resume is disabled. Invalid candidates and canonical
cache hits do not consume new oracle calls. A failed oracle evaluation never
becomes a fabricated zero: opt-in scoring uses TDC's singleton-list path,
retaining its evaluator and normalization while allowing exceptions to escape.
Legitimate finite zero scores are still counted normally.

The deterministic wave order is MDLM2300/S2300, MASK2300/MDLM2301,
S2301/MASK2301. This is an execution order, not a selection procedure. Every
child binds exact configuration/checkpoint/vocabulary/source/package hashes.
Use at most two dynamically chosen GPUs strictly below 10% utilization and
with at least 30,000 MiB free immediately before launch. Map GPU UUIDs to each
isolated process. The controller records full launch inventory/processes,
source, commands, runtime and terminal status. Run from named tmux and keep
canonical source/HEAD frozen until all children and evidence acceptance end.
Capacity waiting may occur before the first launch only; a started child is
never automatically retried. Failure leaves its observed budget and evidence
intact, stops further scheduled launches, and cannot be scored as a full run.

## Metrics, evidence and interpretation

The primary metric is top-10 AUC versus unique online oracle calls. With points
at call counts 0,100,...,2,000, let m(c) be the mean of the best min(10,c)
scores observed by c; m(0)=0. Integrate successive points with the trapezoidal
rule and divide by 2,000. Report terminal top-1/top-10/top-100 means, observed
calls, proposals/iterations, duplicates, remasking events, fallback counts,
actual backbone evaluations and runtime. Runtime is on shared GPUs, so it does
not establish a controlled speed ratio. Partial-run curves may be displayed
with their observed endpoint; a runner's padded AUC must not be presented as
completed full-budget evidence.

Compute both signed treatment-minus-MDLM contrasts per seed and their mean and
sample SD. The primary comparison is MASK CE minus MDLM; S CT minus MDLM is
secondary. Do not merge seeds or tasks into an undeclared aggregate. Preserve
all raw event/score/population records and independently replay call indices,
canonical cache accounting, curves, terminal metrics and contrast calculations.
For each seed, the attaching-only warmup's selected fragments, molecules,
scores and population evolution must match across arms (ignore timestamps,
checkpoint metadata and explicit sampler fields). Require positive observed
remasking and backbone evaluations before interpreting a run as a diffusion
comparison; warmup-only completion or failed verification withholds that claim.

An arm is a candidate for a later, separately declared replication only if
both paired AUC differences are positive and its mean terminal top-10 score is
at least the MDLM mean, with all six runs complete and identity/audit checks
passing. This is an engineering gate, not a statistical superiority test. If
both qualify, report both; do not silently pick the larger noisy estimate.
If neither qualifies, retain the negative result and reconsider the hypothesis.

The paper's PMO benchmark uses 23 tasks, 10,000 calls and three runs. Its full
GenMol and unguided fragment-remasking totals are context only and cannot be
numerically compared to this two-seed, one-task, 2,000-call pilot. Even passing
this pilot cannot satisfy the broader goal of beating GenMol. A PDF must show
all settings, both contrasts, failures/caveats and the preserved prior studies.

Primary paper: https://arxiv.org/html/2501.06158v3 . See
`docs/udlm_broader_benchmark_path.md` for the released-code audit and the
unsupported-guidance distinction. The executable JSON and input manifest will
be added after source review and before any oracle evaluation or GPU launch.
