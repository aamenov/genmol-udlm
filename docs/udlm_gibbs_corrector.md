# Opt-in random-scan Gibbs corrector

`src/genmol/corrector.py` provides a Gibbs kernel for a future sampling
experiment. `Sampler.generate(..., gibbs_corrector=True)` opts into its use;
the argument defaults to `False` and requires a strict Boolean. Existing
checkpoint, training, and default sampling behavior is unchanged. No frozen
protocol enables the corrector and no molecular performance improvement has
been established.

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
engineering seeds. The implementation alone does not authorize GPU generation
or alter the active temperature screen.

## Reusable teaching stage: correction under a fixed computation budget

**Paper correspondence.** The original UDLM release uses ancestral reverse
bridges. The later paper's Appendix E derives informed corrections at a fixed
noise level. NVIDIA GenMol instead uses absorbing MDLM and confidence-based
revelation. Our option retains the UDLM predictor and inserts random-scan Gibbs
updates; it does not import GenMol's confidence ordering into the Gibbs kernel.

**Intuition and motivation.** A predictor moves toward less noise. A corrector
revisits one uncertain molecular sequence through a conditional update at the
same noise level. To measure whether that computation helps, reserve half the
existing network budget for corrections. This gives each predictor a larger
time interval, so correction must compensate for fewer predictor transitions.
Even an exact invariant corrector does not guarantee a better finite-budget
generator.

**Mathematics and defined symbols.** Let `N` be the requested total number of
network-function evaluations (NFE), `M=N/2` the predictor count, and `delta` the
inference endpoint (`inference_eps`, normally 0.00001). For integer
`i=0,...,M`, define `t_i=1-(1-delta)*i/M`. Predictor i maps the current noisy
sequence `x_i` at `t_i` to a provisional sequence `y_i` at `s_i=t_(i+1)` using
the existing reverse bridge. A new network evaluation at `(y_i,s_i)` produces
the clean LOO probabilities `r_l(j)` used in the conditional defined above.
One uniformly selected editable coordinate is resampled, giving the next
predictor's input `x_(i+1)`. Thus `N = M predictor calls + M corrector calls`.
The final correction is evaluated at `delta`; there is no extra terminal model
call outside this accounting.

**Concrete example.** For `N=4` and `delta=0.00001`, the predictor grid is
`(1,0.500005,0.00001)`. Network evaluations occur in order at
`(1,0.500005,0.500005,0.00001)`. The two calls at the middle time use different
states: one sees the provisional predictor sample; the next sees its corrected
sample. Time equality does not make their predictions interchangeable.

**Code below, shapes, and invariants.** This reusable CPU cell previews that
accounting without a model or GPU. Production token IDs and fixed editable masks
have `[B,L]` shape, where `B` is batch size and `L` sequence length; logits have
`[B,L,K]` shape for full vocabulary size `K`. Corrector probabilities have
`[B,L,A]` shape for active vocabulary size `A`, and each row sums to one. The
editable mask is captured from the original masked template and reused after
both kinds of update; incidental MASK tokens sampled from a full prior do not
change it. Supplied fragments, BOS/EOS, and padding remain immutable. A corrector
can leave its chosen token unchanged. An all-immutable row still receives the
budgeted model evaluations but its IDs never change.

```python
total_nfe = 4
inference_eps = 1e-5
assert type(total_nfe) is int and total_nfe >= 2 and total_nfe % 2 == 0
predictor_count = total_nfe // 2
grid = [
    1 - (1 - inference_eps) * i / predictor_count
    for i in range(predictor_count + 1)
]
evaluation_ledger = []
for index in range(predictor_count):
    evaluation_ledger.append(("predictor", grid[index], "current state"))
    evaluation_ledger.append(("corrector", grid[index + 1], "fresh predictor sample"))
assert len(evaluation_ledger) == total_nfe
print(evaluation_ledger)
```

**Differences from the release and approximation boundary.** This is an opt-in
inference hypothesis. Defaults preserve the original sampled token IDs and RNG
state. MDLM rejects the option; UDLM requires an even integer budget at least
two, including when the budget is read from checkpoint configuration. Both
predictor and corrector use the requested raw-LOO temperature and top-p. Their
tempered or learned conditionals generally have no exact invariant-distribution
guarantee. Exact oracle stationarity tests establish the kernel's algebra, not
the accuracy or compatibility of a trained network's conditionals. The fresh
final call at `inference_eps` also reaches below the usual training lower bound
`sampling_eps=0.001`, which must be disclosed in empirical interpretation.

**Comprehension checkpoint.** Why reject odd NFE budgets? Expected reasoning:
each predictor/corrector pair uses exactly two evaluations. Why keep the original
editable mask? Expected reasoning: generated token values cannot revoke or add
permission to change framing or supplied context. Why refresh logits at the
middle time? Expected reasoning: correction changed the state even though time
is unchanged. Does stationarity on a four-state toy target prove better molecular
quality? Expected reasoning: trained predictions are approximate, temperatures
may alter conditionals, and the sampler now has fewer predictor transitions.
