# A proposed posterior-space molecular context guidance rule

**Status: unimplemented inference hypothesis, 2026-09-07.** The accompanying
standard-library categorical audit establishes algebraic properties only.
No neural model, checkpoint, molecular data, oracle score, or GPU was used.
Production guidance guards and the running V14 gamma-zero panel are unchanged.
This note does not establish a GenMol improvement or reproduce official UDLM
conditional training.

## Verified source distinction

The pinned official UDLM uniform-noise `_cfg_denoise` obtains conditional and
unconditional predictions from the **same noisy sequence and time**, computes
their reverse probabilities separately, and combines their log probabilities.
Its `guidance.gamma` is the extrapolation coefficient. The conditioning input
is a class label; training randomly replaces that label with a dedicated
unconditional value. See [official code, lines 1255–1343](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1255),
[conditioning dropout, lines 654–662](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L654),
and the [D-CFG paper, Section 3.1](https://arxiv.org/html/2412.10193v3#S3.SS1).

GenMol's molecular context guidance instead reuses one model on an input with
additional MASK tokens. The released implementation combines its **clean
logits** as `w * logits + (1-w) * logits_poor`, before confidence decoding.
Its `gamma` is a context-masking fraction, separate from scale `w`.
Eligibility is recomputed from current row-zero tokens, excluding BOS, EOS,
MASK and PAD; selected positions are then masked in every row. Previously
filled editable positions can consequently become eligible. See the
[pinned GenMol sampler, lines 64–78](https://github.com/NVIDIA-BioNeMo/genmol/blob/add09fc83b7255bd09c797e527c0f4b51f5fb7c1/src/genmol/sampler.py#L64)
and [GenMol Section 4.3](https://arxiv.org/html/2501.06158v3#S4.SS3).

The proposed rule borrows UDLM's **location of the combination** and GenMol's
**degraded-context prediction**. The latter is neither a learned null-class
prediction nor a true unconditional law by construction. This is a new
adaptation, with different context eligibility, not equivalence to either
released algorithm.

## State, context and the two predictions

Let `B` be batch size, `L` sequence length and `K` the active categorical
alphabet size. Token arrays have shape `[B,L]`, and prediction arrays
`[B,L,K]`. Let `x_init` be the original input and `e` its fixed Boolean
editable mask, captured before stationary-prior initialization. `x_t` is
the current sequence at noise time `t`; originally immutable coordinates
remain equal to `x_init`. Let `C_b` contain row `b`'s originally immutable
coordinates whose original IDs are outside a declared control set `S`.
For a prospective implementation, `S` would include all tokenizer controls,
including BOS/EOS/PAD/MASK/UNK. This excludes controls from degradation
eligibility, not from the checkpoint's corruption alphabet.

Choose a subset `H_b` of size `floor(gamma * |C_b|)`, with `gamma` in [0,1].
A future experiment must specify the RNG stream and whether a new subset is
drawn at every predictor step. The simple candidate would draw once per row
per step from a dedicated recorded generator. The toy supplies `H_b`
explicitly and uses no randomness. Construct a degraded view `x_t_poor` by
replacing only `H_b` with MASK, retaining the original attention mask.
Every editable coordinate of both views remains exactly the original `x_t`.

This explicitly changes released eligibility and row-zero sharing. Applying
the release's current-token rule to UDLM would usually classify ordinary
noisy editable tokens as context immediately. They must not be masked by a
rule described as preserving the editable noisy observation.

At the same time `t`, evaluate the same frozen EMA model in deterministic
evaluation mode (dropout disabled) on `x_t` and `x_t_poor`. Denote the resulting
clean categorical predictions at one editable
coordinate by `D_c(j)` and `D_u(j)`, where `j` indexes a possible clean token.
Subscript `u` means **context-degraded**, not demonstrated unconditional.
All prior, schedule, checkpoint, support and parameterization identities stay
fixed. A first prospective implementation should isolate the guidance order
with temperature 1, top-p 1 and no Gibbs corrector; temperature interactions
would require a separate declared ordering.

## Reverse laws and guidance

For one editable coordinate, let `k=x_t[l]` be its original noisy token and
`i` a possible next token at time `s<t`. Let `pi_i>0` be the stationary
probability of token `i`, with sum 1. The rank-one process is

\[
q_t(k\mid j)=\alpha_t\mathbf1[k=j]+(1-\alpha_t)\pi_k.
\]

`alpha_t` and `alpha_s` are retained-signal probabilities. The strict-support
proof uses `0<alpha_t<alpha_s<1`, as on distinct positive-time grid points.
Define `r=alpha_t/alpha_s`, likelihood
`L_j=alpha_t*1[j=k]+(1-alpha_t)*pi_k`, and
`A_i=r*1[i=k]+(1-r)*pi_k`. For branch `b` in `{c,u}`, convert a CE denoiser to
leave-one-out weights

\[
R_b(j)=\frac{D_b(j)/L_j}{\sum_h D_b(h)/L_h},\qquad
P_b(i)=\frac{A_i[\alpha_sR_b(i)+(1-\alpha_s)\pi_i]}
{\sum_h A_h[\alpha_sR_b(h)+(1-\alpha_s)\pi_h]}.
\]

`h` is a vocabulary summation index. Both branches use the **same original
`k`, `alpha_t`, `alpha_s` and `pi`**. A historical raw-LOO checkpoint supplies
`R_b` directly; it must not be interpreted as a clean CE denoiser. At a
mutable coordinate, the CE conversion is algebraically identical to the
posterior-weighted mixture

\[
P_b(i)=\sum_j D_b(j)
\frac{[r\mathbf1[i=k]+(1-r)\pi_k]
[\alpha_s\mathbf1[i=j]+(1-\alpha_s)\pi_i]}{L_j}.
\]

The bracketed quantities are multiplied. This identity is coordinate-wise;
independent coordinate sampling is not an exact correlated sequence posterior.

For guidance scale `w`, define the proposal

\[
P_g(i)=\frac{P_c(i)^wP_u(i)^{1-w}}
{\sum_hP_c(h)^wP_u(h)^{1-w}}
=\operatorname{softmax}_i\{w\log P_c(i)+(1-w)\log P_u(i)\}.
\]

A prospective API would initially support `w>=1`. Algebraically `w=0`
returns the poor branch, but the existing GenMol CLI's `if gamma and w`
bypasses guidance at zero; this note does not redefine that existing API.
When `w>1`, token odds increase when the conditional-to-poor odds ratio is
larger. The following properties follow directly:

- `w=1` gives exactly `P_c`. If `gamma=0`, the selected subset is empty,
  or the two predictions are identical, then `P_c=P_u` and `P_g=P_c`.
- On the stated interior grid, `A_i>0` and
  `(1-alpha_s)*pi_i>0`, hence both reverse laws and the guided law are
  positive and normalized even if some clean denoiser weights are zero.
- Sample only original editable coordinates. Equivalently, multiply the
  editable product kernel by `1[x_s[l]=x_init[l]]` at each immutable
  coordinate. This clamps all original context, including coordinates hidden
  from the poor predictor, and preserves normalization. It is not a guarantee
  of chemical substructure retention after SAFE decoding and repair.

Identity fastpaths in a future implementation must bypass the second model
call **and context-mask RNG draws** for `w=1`, `gamma=0`, and empty eligible
subsets. Equality of two evaluated laws is an algebraic check, not permission
to retroactively discount their actual computation.

## Exact counterexample and numerical limitations

The three-category toy uses `pi=(1/5,1/2,3/10)`, `alpha_s=2/3`,
`alpha_t=1/3`, `k=0`, `D_c=(3/5,3/10,1/10)` and
`D_u=(1/5,1/5,3/5)`. The reverse laws are

\[
P_c=(24/35,31/140,13/140),\quad P_u=(3/7,29/140,51/140).
\]

At `w=2`, posterior guidance gives
`(141984/175679,245055/1405432,24505/1405432)`. Combining clean logits first
would produce clean weights proportional to `D_c^2/D_u`; their subsequent
bridge instead gives `(1929/2380,73/476,43/1190)`. Simply tempering `P_c`
also differs. Thus the operations cannot generally be interchanged.
The predictor tables are synthetic normalized weights, not molecular
measurements or a trained model's behavior across an entire trajectory.

In floating-point code, use full reverse **log** probabilities and
log-sum-exp normalization, avoiding direct powers. Subtracting separate
branch constants is harmless. Reject nonfinite arithmetic; do not silently
clip or floor tiny probabilities, which changes the heuristic. Strict
mathematical support does not imply that every `exp(log_p)` is representable
in float32/float64. The toy explicitly demonstrates finite log probabilities
whose exponentials underflow to zero. A production implementation would need
a declared precision/sampling treatment and stress tests. Zero-support priors,
truncation, identical-time steps and the exact clean endpoint fall outside
this strict-support proof; negative exponents make unmatched zeros unsafe.

Large `w` can amplify calibration errors in a weak poor branch. Masked context
can be outside the checkpoint's useful training distribution, especially for
an empirical prior with tiny MASK mass. Full-support corruption does not
make an input degradation a learned null-condition model. Learned denoisers,
finite time steps, factorized sequence transitions and the initial stationary
approximation further limit any ideal reverse interpretation. Even exact
conditional and unconditional coordinate laws do not establish that the
resulting finite-grid trajectory samples a desired globally tilted molecular
distribution. No property oracle enters this guidance rule.

## Compute accounting and the next bounded step

An active guided predictor step requires two predictions per candidate.
For `B` candidates and `N` predictor steps, serial branches make `2N`
backbone calls; concatenating them into batch `2B` can reduce Python/module
invocations to `N`, but still costs `2BN` candidate-equivalent evaluations,
or `2N` per candidate. For example `B=4,N=128` means 1,024 candidate-equivalent
evaluations, whether the two branches are batched or serial. Existing hook
counts that only count forward invocations would understate a batched rule's
work; a future receipt must preserve batch-weighted evaluations, actual
forward calls and runtime. Identity fastpaths cost `BN` evaluations.

Before any molecular experiment, the minimal implementation work would be
an opt-in guidance identity and posterior combination with fixed edit masks,
per-row context selection, exact no-op/RNG fastpaths, precision checks,
clamping tests, and updated batch-aware NFE/source receipts. Existing defaults
and the current UDLM gamma guard must remain until that work is independently
reviewed. A later prospective fragment/PMO panel would need both no-guidance
and released-MCG controls, explicit context eligibility and guidance scales,
matching checkpoint/prior/EMA and seeds, and disclosed compute differences.
Neither that implementation nor an experiment is authorized by this memo.

Run the deterministic audit from this checkout:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python \
  scripts/udlm/audit_context_guidance.py
```

Observed CPU result: 243 categorical settings, 486 exact bridge equalities,
972 positive normalized guided laws, and 8,748 enumerated two-position states
with immutable-context checks; maximum log-space float probability error
`3.3306690738754696e-16` against rational arithmetic. The 24 focused tests
also cover controls/per-row eligibility, boundary exclusion, underflow and
overflow, and identical standalone replay. These counts are algebra checks,
not model evaluations or oracle calls.

Source audit base: project `48473d4febbd06d9bc07986ca96ceb93926c91ec`;
its `src/genmol/sampler.py` SHA-256 is
`35db5af0aaf10d3f4a871008f26839d79ceb43a6a4edf0ac808627dd39964c36`.
The official UDLM `diffusion.py` at `edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661`
has SHA-256 `4757675020ea98a33e53bfe941ae7aa2f2c236e4efb9f5dc1e62df67c733395c`.
Local integration points are `src/genmol/sampler.py` (UDLM guard, original
editable mask and sampling loop), `src/genmol/denoiser.py` (CE conversion)
and `src/genmol/diffusion.py` (categorical bridge). All remain unchanged.
