# A mask-rich stationary prior as a future transfer hypothesis

This is a prospective hypothesis, recorded against source `269ef30` on
2026-09-07. It proposes no model change or launch. Review the complete V9 and
predictor-resolution follow-up results before deciding whether to test it.
There is no empirical improvement or superiority evidence for this prior.

The motivation is specific to our initialization: MDLM was pretrained to
recover masked tokens, whereas the current empirical UDLM prior almost never
produces MASK. Giving MASK more stationary mass may make the initial
denoising task closer to that pretraining while retaining reversible token
replacement. It may also reduce useful correction freedom and diversity.

**1. The process is mathematically coherent and has not been tested in our
registered R/S/E panels.** Let the active vocabulary have size \(V\), let
\(m\) denote its MASK token, and let \(c_j\) be the recorded frequency of
token \(j\). Define a strictly positive base distribution and its mixture by

\[
b_j=(1-w)\frac{c_j}{\sum_i c_i}+\frac{w}{V},\qquad
\pi_{\lambda,j}=\lambda\mathbf1[j=m]+(1-\lambda)b_j,
\quad 0\leq\lambda<1.
\]

Here \(w\) is the existing uniform smoothing weight; \(\lambda\) is a new,
separate MASK mixture weight. For times \(0\leq s<t\leq1\), let
\(\alpha(t)=1-(1-\epsilon)t\), where the current noise endpoint is
\(\epsilon=0.001\), and set \(a=\alpha(t)/\alpha(s)\). The forward
transition matrix is

\[
Q_{s,t}(i,j)=a\mathbf1[i=j]+(1-a)\pi_{\lambda,j}.
\]

It is normalized, has stationary distribution \(\pi_\lambda\), composes
correctly across time, and obeys detailed balance: for \(i\ne j\),
\(\pi_{\lambda,i}Q_{s,t}(i,j)=(1-a)\pi_{\lambda,i}\pi_{\lambda,j}\).
The existing `ContinuousCategoricalDiffusion` already implements this
arbitrary-prior family and its coordinate reverse bridge. Three-state CPU
checks with mixture weights 0, 0.9 and 0.99 confirmed composition, detailed
balance and finite normalized production bridge outputs; these are algebra
checks, not molecular experiments.

The pinned frequency artifact is
`experiments/udlm/token_frequency/train_first_10000.json`, SHA-256
`088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed`.
For the current full vocabulary, \(V=1880\), \(w=0.0002\), and
\(c_m=0\), so \(b_m=1.0638297872340426\times10^{-7}\).
With \(\lambda=0.9\), MASK mass becomes 0.9000000106382979 and the
smallest remaining probability becomes about \(1.064\times10^{-8}\).
At a mutable clean non-MASK position, the forward MASK probability is
\((1-\alpha(t))\pi_{\lambda,m}\): about 44.955% at \(t=0.5\) and
89.910% at \(t=1\). Ordinary-token corruption remains substantial across
a whole sequence; 90% mixture mass does not imply equivalence to MDLM.

Existing R/S/E configurations use released uniform, schedule-corrected
uniform, or smoothed empirical priors. Their smoothing studies vary
\(w\), not \(\lambda\). Mixed MASK/uniform transition kernels are already
discussed in [D3PM, Sections 3.1 and 4](https://arxiv.org/html/2107.03006).
The proposed empirical-mixture transfer experiment is our hypothesis; mixed
discrete corruption itself is not a new technique.

**2. Implementation is feasible, but requires a new training and checkpoint
identity.** Preserve the generic process equations. Extend the explicit prior
builder, variant validation and CE compatibility whitelist in
`src/genmol/model.py`; record \(\lambda\), actual MASK token ID, base
artifact/smoothing, token ordering and resulting probability hash in immutable
prior metadata. Existing checkpoint metadata and runtime prior checks must
keep their current behavior. Do not mutate or override an E checkpoint's
prior at inference: its model learned a different corruption law.

A prospective comparison should initialize both arms freshly from the same
MDLM 50k EMA weights, with fresh optimizer and EMA states. Keep
`exclude_special_tokens=false` so MASK remains in the corruption alphabet.
The common `mask_all_special_tokens=true` clean-position policy remains
compatible: it makes clean control positions immutable without excluding
MASK from noise at ordinary positions. The sampler fixes editable positions
before drawing its prior, so an incidental MASK draw does not freeze that
position. New identities must be accepted explicitly by generation and
independent rescoring, with the old prior paths unchanged.

**3. The absorbing limit is singular and does not automatically reproduce
MDLM.** As \(\lambda\to1\), the forward kernel tends to absorbing MASK
diffusion. Exactly \(\lambda=1\) loses full support and irreducibility;
the current constructor, density ratios and logarithms require positive
probabilities and cannot simply receive that value.

For a mutable position \(\ell\), assume its clean token is never MASK.
Let \(Z_{t,\ell}\) be its noisy token, \(X_{0,\ell}\) its clean token,
and \(Z_{t,-\ell}\) all other noisy positions. Write
\(R_j=P(X_{0,\ell}=j\mid Z_{t,-\ell})\) for the exact clean
leave-one-out posterior. In the absorbing process, an observed non-MASK
token copies backward exactly. At a MASK observation, for \(j\ne m\),

\[
P(Z_{s,\ell}=m\mid Z_t)=\frac{1-\alpha(s)}{1-\alpha(t)},\qquad
P(Z_{s,\ell}=j\mid Z_t)=
\frac{\alpha(s)-\alpha(t)}{1-\alpha(t)}R_j.
\]

These are exact coordinate marginals, not an assertion that multiplying them
recovers the correlated sequence joint. At visible observations the CE-to-LOO
division has zero-likelihood states in this limit; use a support-aware
derivation, not division by zero. Near the limit, tiny ordinary-token masses
can amplify calibration and density-ratio errors despite stable logarithms.

There is also an endpoint mismatch: fixed \(\epsilon>0\) leaves clean
mass in \(q(Z_1\mid X_0)\), whereas the stationary prior tends to all MASK.
For a clean non-MASK token, the terminal KL divergence to \(\pi_\lambda\)
diverges as \(\lambda\to1\) at fixed \(\epsilon\). Exact all-MASK
terminal consistency additionally needs \(\alpha(1)\to0\) or explicit
terminal handling. If both limits are taken together, vanishing terminal KL
also needs an appropriate rate, such as
\(\epsilon\log(1/(1-\lambda))\to0\). The absorbing forward-kernel limit
therefore does not establish equivalence of training objectives, architecture
or decoding.

The next small test, only if the V9/resolution review supports it, is a CPU
fixed-minibatch diagnostic using frozen MDLM EMA with \(\lambda=0\) and
0.9 across the same time grid. Separate MASK, changed ordinary-token and
unchanged-token CE/gradient statistics: total loss across different priors
confounds model fit with corruption difficulty. If justified, predeclare a
matched 100-update, batch-128 CE control/treatment pilot with fresh common
initialization, data ordering, seed and grouping, and common cross-prior
validation. That would be a new experiment, not a promotion or continuation
of an existing checkpoint. Keep final seeds 0/1/2 reserved.
