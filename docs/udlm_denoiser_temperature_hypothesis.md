# Temperature on the clean denoiser: a new inference hypothesis

Prepared while the fixed V12 comparison was running, before inspecting any
MASK-rich molecular output. This is a proposed sampling control, not a repair
to the released raw-LOO UDLM temperature convention. V12 retains its original
settings and source. No checkpoint or training objective changes are proposed.

## Why the location of temperature matters

For one editable position, let `k` be its observed noisy token, `j` a candidate
clean token, `t` the current diffusion time, and `pi[k]` the positive stationary
probability of `k`. The retained clean mass is
`alpha(t) = 1 - (1-noise_eps)*t`. Define the local likelihood

```text
L[j] = alpha(t) * 1[j=k] + (1-alpha(t)) * pi[k].
```

The CE model predicts clean-token posterior probabilities `D[j]`. Conversion
to leave-one-out probabilities is `R[j] proportional to D[j]/L[j]`, with each
distribution normalized over the active vocabulary. Existing sampling applies
positive temperature `T` to these converted logits, producing

```text
R_old[j] proportional to (D[j]/L[j])^(1/T).
D_implied_old[j] proportional to R_old[j]*L[j]
                 proportional to D[j]^(1/T) * L[j]^(1-1/T).
```

Thus `T=0.5` gives an additional inverse-likelihood factor in the implied clean
posterior. At a rare visible token, the likelihood of retaining that token may
be much larger than the likelihood of another clean token. The inverse factor
can then offset even strong denoiser confidence in the current token. At a
MASK position, the likelihoods are equal among all non-MASK clean candidates,
so their relative probabilities receive the usual denoiser temperature; the
distinction still applies to the MASK candidate itself.

The proposed control first tempers the clean logits and then converts them:

```text
D_new[j] proportional to D[j]^(1/T).
R_new[j] proportional to D_new[j]/L[j].
```

The existing categorical reverse bridge then consumes `R_new` at temperature
one. Applying temperature a second time would implement another rule and is
incorrect for this declared control. At `T=1`, both rules have the same law.
All symbols and the conversion derivation also appear in
`udlm_denoiser_ce_hypothesis.md`; the residual-MASK note describes the current
temperature ordering without assuming that it is a defect.

## Concrete example and executable check

Use three tokens, with token zero called MASK, stationary probabilities
`pi=(0.9, 0.0999, 0.0001)`, observed token `k=2`, time `t=0.5`, and
`noise_eps=0.001`. Suppose the learned clean posterior is
`D=(0.001, 0.009, 0.99)`. Compare the implied clean posterior and the actual
reverse transition under both rules at `T=0.5`. Run the companion CPU audit:

```bash
CUDA_VISIBLE_DEVICES='' /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python \
  scripts/udlm/audit_denoiser_temperature.py
```

The audit enumerates exact rational identities, independently mixes normalized
forward bridges using the tempered denoiser, and checks the existing production
conversion/bridge against that mixture. It uses no checkpoint, molecular data,
GPU, generation seed, or training output. Its example deliberately isolates
the mathematical effect; it does not estimate its frequency in real samples.

For that example, the implied clean probability of the observed token is
54.3949% under raw-LOO temperature and 99.9916% under denoiser temperature.
The actual reverse-step retention probability from `t=0.5` to `s=0.4` is
84.8083% versus 99.9959%. These are distinct quantities: the first describes
the clean prediction implied by the logits, and the second describes the
partially noisy state after one reverse transition. The CPU audit checks 243
rational cases and 1,458 production scalar probabilities, including 81
temperature-one cases; maximum absolute discrepancy was `1.45e-15` or less.

## Narrow implementation and experimental boundary

An explicit inference field `temperature_space: x0_denoiser` should enable this
control only for CE-parameterized UDLM. The absent/default `raw_loo` convention
must preserve historical behavior and normalized configuration identities.
Initially require top-p one and predictor-only sampling, so a second heuristic
does not obscure the comparison. The chosen mode and order must be bound in
the saved sampling configuration and independent rescore. Existing checkpoints,
immutable context, NFE accounting and prior identity remain unchanged.

A later prospective pilot must compare both temperature spaces on the same
frozen checkpoint, same temperature, NFE, prior and fresh engineering seeds.
Any model choice informed by V12 must be disclosed. Do not change V12, pool its
seeds into the new pilot, or consume final seeds 0/1/2 for tuning. Report strict
and repaired chemistry, exact request denominators, diversity, runtime and
editable special-token occurrences for every declared arm.

## Limits and comprehension checkpoint

Temperature is a heuristic for both parameterizations when `T != 1`.
Independently tempered coordinate posteriors need not describe one consistent
joint molecular distribution. Learned calibration errors, terminal-prior
mismatch, finite-step factorization and positive final diffusion time remain.
The proposal neither imposes a chemistry constraint nor removes MASK output.
Greater local retention could preserve an incorrect token as well as a correct
one, and reduced diversity could outweigh any improvement in validity.

Why do the two rules agree at temperature one? The likelihood exponent in the
old implied posterior vanishes, leaving the same `D`. Why can a temperature
below one alter retention? For `j=k`, the likelihood includes the retained
clean mass; for other candidates it does not, and the negative likelihood
exponent changes their relative weights. Why is a positive toy result not a
benchmark win? It checks an algebraic control on chosen probabilities rather
than learned molecular behavior or held-out benchmark performance.
