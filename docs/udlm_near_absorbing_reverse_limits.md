# Near-absorbing reverse limits depend on logit interpretation

This is a one-coordinate mathematical audit and transfer hypothesis, not a
molecular efficacy result. A prior approaching MASK does not by itself make
arbitrary frozen MDLM backbone logits a calibrated clean denoiser. In the
example below, interpreting the same weights as raw LOO preserves a visible
token in the limit; interpreting them as a clean denoiser can still flip or
remask it. Temperature after D/L conversion can amplify that difference.
No production code, checkpoints, trained-EMA gates, or experiment protocols
are changed. No neural model or oracle is evaluated by this audit.

## Definitions and finite-prior identity

There are three categories, clean tokens A and B and control token M=MASK.
Let `b=(1/5,1/2,3/10)` be a full-support base distribution,
`0 <= lambda < 1` the MASK mixture weight, and `epsilon=1-lambda`.
The stationary distribution is

\[
\pi_\lambda(i)=\lambda\mathbf1\{i=M\}+\epsilon b_i.
\]

Every category has positive prior mass at every finite lambda. Clean model
weights have `R_M=D_M=0`, an idealized suppression assumption; this does not
set the stationary MASK mass to zero. Finite neural logits need not realize
exact suppression, and floating underflow is outside this rational toy.

Let `x_0` be the clean token, `z_t=k` the observed noisy token and `z_s=i` an
earlier state. Fixed signal levels satisfy `0 < alpha_t < alpha_s < 1`, and
`r=alpha_t/alpha_s`. The forward marginal and transition are

\[
q(z_t=k\mid x_0=j)=L_j=\alpha_t\mathbf1\{j=k\}+(1-\alpha_t)\pi_k,
\quad q(z_t=k\mid z_s=i)=r\mathbf1\{i=k\}+(1-r)\pi_k.
\]

Here `R` is a normalized learned LOO weight vector; `D` is a normalized
putative clean posterior `P(x_0=j | noisy context, z_t=k)`. These are different
objects even when their input numerical weights happen to be identical.
The raw-LOO bridge has the exact normalizer

\[
P_R(i\mid k)=
\frac{[r\mathbf1\{i=k\}+(1-r)\pi_k]
      [\alpha_sR_i+(1-\alpha_s)\pi_i]}
     {\alpha_tR_k+(1-\alpha_t)\pi_k}.
\tag{1}
\]

For finite positive pi, setting `R=normalize(D/L)` makes (1) exactly equal to

\[
P_D(i\mid k)=\sum_j D_j
\frac{[\alpha_s\mathbf1\{i=j\}+(1-\alpha_s)\pi_i]
      [r\mathbf1\{i=k\}+(1-r)\pi_k]}{L_j}.
\tag{2}
\]

Normalization of `D/L` matters: if `C=sum_j D_j/L_j`, the denominator in (1)
is `1/C`; multiplication cancels C and gives (2). The refresh likelihood
uses the **observed** mass pi_k, not candidate mass pi_i. The script computes
(1) and (2) independently and checks exact equality.

## A visible token: different limiting conditionals

Fix a clean observation `k != M`. Then `pi_k=epsilon b_k -> 0`. If R is fixed
and `R_k>0`, the numerator for i=k and denominator of (1) both approach
`alpha_t R_k`; every other numerator vanishes. Thus `P_R -> delta_k`.
More generally, `R_k/pi_k -> infinity` suffices. This is not uniform in
arbitrarily small R_k, time endpoints, or changing model predictions.

For fixed D with `D_M=0`, define

\[
c=\frac{\alpha_t(1-\alpha_s)}{\alpha_s(1-\alpha_t)},\quad
u=\frac{\alpha_s-\alpha_t}{1-\alpha_t},\quad
v=\frac{(\alpha_s-\alpha_t)(1-\alpha_s)}{\alpha_s(1-\alpha_t)}.
\]

All are positive and `c+u+v=1`. In (2), the j=k conditional tends to delta_k.
For any clean j!=k, conditioning on the rare refresh that produced k gives
the limiting law `c delta_k + u delta_j + v delta_M`. Consequently,

\[
P_D(k\mid k)\to D_k+(1-D_k)c,\quad
P_D(j\mid k)\to uD_j\;(j\ne k,M),\quad
P_D(M\mid k)\to v(1-D_k).
\tag{3}
\]

These conditional ratios remain finite even though their conditioning event
has probability `L_j=(1-alpha_t) epsilon b_k -> 0`. A fixed, uncalibrated D
assigns those counterfactual clean states nonvanishing posterior weight.
Equivalently D/L forces the converted R_k to shrink at order epsilon,
violating the fixed-positive-R_k premise above. This is not a Bayes algebra
bug. A Bayes-consistent `D_lambda=normalize(L R_true)` would instead satisfy
`D_lambda,k -> 1` whenever fixed `R_true,k>0`; its converted bridge agrees
with the raw bridge for every finite lambda.

With `alpha_t=2/5`, `alpha_s=7/10`, we have `(c,u,v)=(2/7,1/2,3/14)`.
Take the same fixed numerical weights `R=D=(3/5,2/5,0)`, observed k=A:

| Interpretation and temperature order | Limit P(A), P(B), P(MASK) |
|---|---|
| Raw LOO, T=1 | (1, 0, 0) |
| Clean D, convert D/L, T=1 | (5/7, 1/5, 3/35) |
| Raw LOO, T=1/2 | (1, 0, 0) |
| Clean D, T=1/2 **before** conversion | (71/91, 2/13, 6/91) |
| Clean D, convert, raw-LOO T=1/2 **after** conversion | (2/7, 1/2, 3/14) |

Thus the fixed CE interpretation at T=1 retains 5/7 rather than 1; cooling
the converted LOO weights retains only 2/7 in this example. This is a
counterexample, not a prediction of any checkpoint's molecular results.
Setting R_k exactly zero also gives a non-copying raw limit: positive R_k is
an essential assumption, not a cosmetic restriction.

## MASK observations and temperature order

For k=M and zero clean MASK weight, every nonzero component of D has the same
likelihood `(1-alpha_t) pi_M`. Therefore `normalize(D/L)=D` exactly at every
finite lambda. In the limit, raw R or converted clean D yields

\[
P(i\ne M\mid M)\to
\frac{\alpha_s-\alpha_t}{1-\alpha_t}D_i,\qquad
P(M\mid M)\to\frac{1-\alpha_s}{1-\alpha_t}.
\tag{4}
\]

The example gives `(3/10,1/5,1/2)` at T=1. At T=1/2, normalized squared clean
weights are `(9/13,4/13,0)`; all three temperature interpretations for this
MASK observation agree, giving `(9/26,2/13,1/2)` in the limit.

For general positive finite temperature T, let `h=1/T`. Tempering raw LOO
means `R_T=normalize(R^h)`. Fixed positive R_k stays positive, so visible
copying still holds. Directly tempering clean D means replacing D in (3)
with `normalize(D^h)`, which need not be a point mass.

Tempering **after** D/L conversion instead induces

\[
D_{\mathrm{eff},j}\propto D_j^h L_j^{1-h}.
\tag{5}
\]

For clean k, fixed `0<D_k<1` and D_M=0, equation (5) implies:

- T<1: effective mass at k tends to zero; other clean weights approach
  `normalize_{j!=k}(D_j^h)`. Equation (3) then gives carry c and remask v.
- T=1: equation (3) uses D unchanged.
- T>1: effective mass at k tends to one, so the bridge tends to delta_k.

These are fixed-T limits; convergence need not be rapid near T=1. The main
audit uses integer h=1,2,3 to keep every calculation rational; an additional
exact-square construction tests T=2 (h=1/2) with rational square roots. The
general positive-T limits follow from the powers of epsilon in (5).

## Limits and transfer interpretation

At T=1, sending fixed denoiser error `eta=1-D_k` to zero after (3) also
recovers copying; the bounded mixture does not create a noncommuting
two-limit counterexample there. At raw-space T=1/2 it does. For
`D=(1-eta,eta,0)`, the effective wrong/current odds scale as
`eta^2/epsilon` up to a positive constant. Sending epsilon to zero first at
any eta>0 yields the non-copying `(c,u,v)` law. Sending eta to zero first
at finite epsilon yields the exact positive-prior bridge with `R=delta_k`;
it still assigns nonzero mass to other earlier states. Only the subsequent
epsilon-to-zero limit yields delta_k. Joint paths with `eta=o(sqrt(epsilon))`
restore copying in the limit. The raw bridge also lacks uniform convergence if R_k
vanishes with pi_k. We cannot substitute exactly lambda=1 into D/L at a
clean observation: some L_j become zero and those conditionals are undefined.

The pinned MDLM implementation's substitution parameterization overwrites
visible-coordinate log probabilities with a point mass at the input token
before the loss is evaluated. Its loss thus provides no direct local
reconstruction gradient for raw backbone logits at visible targets; shared
parameters can still learn from other masked targets. It does not require
those **unprocessed** logits to be delta_current. See
[MDLM c112c526, substitution lines 261–277](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L261)
and [continuous loss lines 883–894](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L883).
The inspected file SHA-256 is
`f8b4f1fdea2ffc960c9852583bee58865aace44baf958f83d5f9b8e352752121`.

[Released GenMol add09fc83, training_step](https://github.com/NVIDIA-BioNeMo/genmol/blob/add09fc83b7255bd09c797e527c0f4b51f5fb7c1/src/genmol/model.py#L116)
passes raw BERT logits to BioNeMo MDLM.loss. The locally installed
`bionemo-moco==0.0.2.1` file
`.venv/lib/python3.10/site-packages/bionemo/moco/interpolants/continuous_time/discrete/mdlm.py`
has SHA-256 `28b45c90d98a9d142421494d39e0c3c80cf5369b68be901a42133c5fefee9a54`;
its lines 166–171 and 190–213 apply that same substitution before loss.
This is an inspected local dependency, not a claim that every historical
BioNeMo version is identical. Our bridge and D/L equations correspond to
`src/genmol/diffusion.py::ContinuousCategoricalDiffusion.posterior_probs`
and `src/genmol/denoiser.py::denoiser_to_loo_logits` at frozen local revision
`48473d4febbd06d9bc07986ca96ceb93926c91ec`; this audit does not import them.

These coordinate calculations do not prove joint sequence consistency,
calibration on generated contexts, equivalence to GenMol's confidence
schedule, or benefit at finite lambda/NFE. Frozen weights can produce
different logits as the noisy context changes. A future zero-update
transfer must be labeled as an interpretation of MDLM EMA weights, with
zero UDLM updates/exposures and independently bound original weights; it
must not be labeled CE-trained or bypass existing trained-EMA acceptance.

Run from this checkout with the project Python:

```bash
../../.venv/bin/python -I scripts/udlm/audit_near_absorbing_limits.py
PYTHONPATH=. ../../.venv/bin/python -m pytest tests/test_udlm_near_absorbing_limits.py -q
```

The standalone script prints deterministic JSON: 648 exact finite-prior
bridge/mixture identities, the symbolic rational limits, and 60 finite-prior
method/observation rows. At epsilon=10^-12, every displayed example is within
10^-10 total variation of its stated limit. Tests also cover multiple base
prior ratios and times, missing observed raw mass, Bayes-consistent D,
temperature order, singular-prior rejection, and byte-identical repeated
CLI output. This proves those finite calculations and supports the analytic
derivation; finite numerical checks alone are not a proof of a limit.
