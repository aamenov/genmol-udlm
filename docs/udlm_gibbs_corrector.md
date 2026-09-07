# Opt-in random-scan Gibbs corrector

`src/genmol/corrector.py` provides a standalone kernel for a future sampling
experiment. It is not connected to `Sampler`, the benchmark runner, or any
registered protocol. Existing checkpoint, training, and inference behavior is
unchanged. No molecular performance improvement has been established.

The implementation follows the conditional identity discussed in Appendix E
of [Uniform Diffusion Models Revisited](https://arxiv.org/abs/2605.22765) and its
[reference implementation](https://github.com/samsongourevitch/rev_udm). At time
`s`, let `r_l(j)` denote the clean leave-one-out probability of token `j` at
position `l`, given the other noisy positions; let `alpha_s` be the actual
process's clean fraction and `pi_j` the stationary noise probability. The
single-coordinate noisy conditional is

```text
c_l(j | x_-l, s) = alpha_s * r_l(j) + (1 - alpha_s) * pi_j.
```

Select one coordinate uniformly from each row's fixed editable set, sample its
new token from `c_l`, and retain every other token. The coordinate may retain its
old value. Rows with no editable coordinates are identity transitions. This is
a Gibbs correction at fixed time; it does not use the observed-token likelihood
from the reverse bridge. The release residual-clean endpoint is retained.

`gibbs_conditional_probs(diffusion, logits, s)` maps full model logits `[B,L,K]`
to float64 probabilities `[B,L,A]` in active-alphabet order. Here `B` is batch
size, `L` is sequence length, `K` is full vocabulary size, and `A` is the number
of allowed categories. It handles both the uniform prior and the categorical
process's stored float64 prior. Excluded controls may occur at immutable
positions. `random_scan_gibbs_step` accepts int32/int64 IDs `[B,L]`, a Boolean
`mutable_mask`, and an optional caller-owned `torch.Generator` controlling both
coordinate and token draws. Neither function changes its input tensors.

Exact random-scan stationarity assumes compatible, exact LOO conditionals,
temperature 1, top-p 1, and an editable set independent of current token values.
Learned own-token dependence, prediction error, temperature/top-p transforms,
confidence-dependent coordinate selection, and simultaneous updates with stale
logits remove that guarantee. In particular, sharpening local conditionals does
not automatically define a consistently tempered joint distribution. Evaluate
the network at the current state and time, and refresh logits after each update.

The CPU test enumerates a two-coordinate binary clean target
`[[0.4, 0.1], [0.2, 0.3]]`, corrupts it exactly with clean fraction 0.6, and
constructs oracle LOO predictions independently. It reconstructs all actual
kernel branches and checks normalized rows, stationarity, and detailed balance
for uniform and nonuniform priors. A simultaneous-update negative control
fails stationarity, confirming that the fixture detects the difference.
Separate tests cover framing, active-ID mapping, caller RNG reproducibility,
uniform coordinate selection, empty editable sets, and small prior masses.

A future engineering comparison can keep 128 network evaluations fixed and
compare 128 predictor steps against 64 predictor plus 64 fresh-logit corrector
steps. That experiment needs its own prospective specification and disclosed
engineering seeds. This module alone does not authorize GPU generation or
alter the active temperature screen.
