# CE denoiser adaptation: a prospective hypothesis

Original audit status: CPU algebra and gradient evidence only. The optional CE
implementation is now documented in [the implementation note](udlm_ce_implementation.md).
CE molecular benchmark results are pending, and **no empirical improvement
over UDLM or GenMol has been established**. The V7 resource pilot passed and
[the matched V8 study](../experiments/udlm/protocols/engineering_v8_objectives.json)
is now implemented; check live status in `PROJECT_CONTEXT.md`. Existing R/S/E training, v4's failed
diagnostic attempt, and the prospective v5 temperature screen retain their
original identities and conclusions. This document does not reinterpret them.

The hypothesis is to adapt the existing MDLM backbone by predicting clean
tokens with cross entropy (CE), then convert those predictions to the
leave-one-out (LOO) distribution expected by our current reverse bridge.
Predicting the clean token is compatible with the general denoising task of
the MDLM initialization, but uniform/categorical replacement creates different
observations from masking. Compatibility does not establish faster adaptation.

## Defined symbols and forward process

Let the active alphabet contain K tokens with indices i,j,k in {0,...,K−1}.
The clean sequence is x₀; zₜ and zₛ are noisy sequences at times 0 ≤ s < t ≤ 1.
The position under discussion is ℓ, and k = zₜ,ℓ is its observed current token.
The notation zₜ,−ℓ means all noisy positions except ℓ. Forward corruptions are
independent across positions conditional on x₀; the data distribution itself
may correlate positions. Immutable control/padding positions are outside the
active objective and must be masked explicitly.

The stationary prior π has πᵢ > 0 and Σᵢπᵢ = 1. Uniform diffusion takes πᵢ = 1/K;
the categorical extension also permits a nonuniform prior. The clean-retention
coefficient αᵤ at time u decreases from α₀ = 1. Assume
0 < αₜ < αₛ ≤ 1. The current release schedule is
αᵤ = 1 − (1−ε)u, where ε = 0.001 is residual clean mass at u = 1.
Define r = αₜ/αₛ and let δᵢⱼ equal 1 when i=j and 0 otherwise. The kernels are

```text
q(zₜ,ℓ = k | x₀,ℓ = j) = αₜ δⱼₖ + (1−αₜ) πₖ = Lⱼ
q(zₛ,ℓ = i | x₀,ℓ = j) = αₛ δᵢⱼ + (1−αₛ) πᵢ = Bᵢⱼ
q(zₜ,ℓ = k | zₛ,ℓ = i) = r δᵢₖ + (1−r) πₖ = Aᵢ
Σᵢ Aᵢ Bᵢⱼ = Lⱼ.
```

The refresh term in Aᵢ uses the **observed** token's prior πₖ, not πᵢ.
Full support and t>0 ensure every likelihood Lⱼ is positive.

## Converting a clean denoiser to the current bridge

Let Dⱼ be the network's clean-token probability at this position. CE targets
Dⱼ = P(x₀,ℓ=j | zₜ), where the conditioning includes the current position.
The following algebra also holds for any approximate normalized D. Define

```text
C = Σⱼ Dⱼ/Lⱼ
Rⱼ = (Dⱼ/Lⱼ)/C.
```

If D is the true conditional distribution, Bayes' rule gives
Rⱼ = P(x₀,ℓ=j | zₜ,−ℓ). Indeed, conditional independence of the local forward
noise gives Dⱼ ∝ P(x₀,ℓ=j | zₜ,−ℓ)Lⱼ. Thus the division removes the likelihood
contribution of the observed local token. For an approximate network, R is a
derived probability vector, not a guarantee of globally compatible LOO
conditionals across all positions or inputs.

The current raw-LOO bridge constructs the unnormalized earlier-token weights

```text
uᵢ = Aᵢ [αₛ Rᵢ + (1−αₛ)πᵢ] = Aᵢ Σⱼ Bᵢⱼ Rⱼ.
```

Its normalization follows from the forward kernel composition:

```text
Σᵢ uᵢ = Σⱼ Rⱼ Σᵢ AᵢBᵢⱼ = Σⱼ RⱼLⱼ = 1/C.
```

Consequently its normalized probabilities are

```text
uᵢ / Σₕuₕ = C Aᵢ Σⱼ Bᵢⱼ (Dⱼ/Lⱼ)/C
           = Σⱼ Dⱼ AᵢBᵢⱼ/Lⱼ
           = Σⱼ Dⱼ q(zₛ,ℓ=i | zₜ,ℓ=k, x₀,ℓ=j).
```

This is exactly a mixture of **normalized forward posteriors**. When D is the
true conditional, it equals P(zₛ,ℓ=i | zₜ), even if clean sequence positions
are correlated. Sampling every position independently from these marginals
still approximates the full joint reverse transition: the product generally
does not equal P(zₛ | zₜ). Finite-step molecular generation therefore remains
approximate. The identity also does not remove residual terminal clean mass.

The final s=0 step is valid because conversion uses t>0. At t=0, some Lⱼ vanish;
do not evaluate this formula there. At αₜ=0 the conversion becomes R=D, but the
current finite-noise schedule never reaches that endpoint exactly.

## Relationship to the primary literature

Austin, Johnson, Ho, Tarlow and van den Berg's
[D3PM paper, Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
provides the forward posterior in Eq. 3 and introduces clean-token prediction
and an auxiliary denoising CE term in §§3.3–3.4. Its Eq. 5 combines that CE term
with a variational objective. It does not establish this proposed pure-CE
molecular adaptation experiment.

There is a material distinction in
[D3PM Eq. 4](https://arxiv.org/html/2107.03006v3#S3.SS3): it sums **joint** kernels,
`pθ(zₛ | zₜ) ∝ Σⱼ q(zₛ,zₜ | x₀=j) p̃θ(j | zₜ)`.
Putting D directly into that expression weights the normalized posteriors
proportionally to DⱼLⱼ. Our Dⱼ/Lⱼ conversion recovers weights Dⱼ instead.
This document derives a CE-to-LOO reparameterization; it does not claim to
reproduce Eq. 4 literally or to introduce a published molecular improvement.

## Executable exact evidence

Run from this checkout with the project virtual environment:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python \
  scripts/udlm/audit_denoiser_conversion.py
```

The script writes JSON only to stdout and does not train, inspect GPU inventory,
decode molecules, or load checkpoints. Do not use Python's `-O`, since the
proof checks are assertions. It uses exact `fractions.Fraction` arithmetic to
enumerate a three-token, two-position example with correlated clean weights

```text
P(x₀) = (1/35) [[11, 1, 2],
               [ 1, 9, 1],
               [ 2, 1, 7]].
π = (1/3,1/3,1/3) or (72/100,23/100,5/100).
(αₛ,αₜ) = (4/5,1/3), (1,9/10), or (1/5,1/100).
```

For each of the six prior/time settings it enumerates all nine x₀, nine zₛ
and nine zₜ sequences: 4,374 trajectories in total. At all 54 observed cases,
two positions and three token values, it asserts 324 exact scalar equalities
between the converted bridge, the normalized posterior mixture, and the
independently enumerated true reverse marginal. Another 324 equalities compare
the conversion with independently enumerated LOO probabilities.

The same examples exercise the actual `posterior_probs` methods in
`src/genmol/diffusion.py`: 54 categorical calls plus 27 legacy uniform calls,
using float64 CPU tensors and log-space conversion. They must agree within
1e−12; the initial saved-script execution agreed at machine precision. The
JSON records the actual maximum error, Torch version, and script/diffusion
source SHA-256 values on every invocation.

Two negative examples guard the interpretation:

| Deliberately different calculation | Maximum observed discrepancy |
| --- | ---: |
| Pass D directly to the raw-LOO bridge | 0.324170435 absolute probability error |
| Product of exact reverse marginals vs true joint reverse | 0.217759240 total variation |

Total variation here is one half the sum of absolute differences over the
nine earlier sequences. These are exhaustive results for this finite toy,
not molecular benchmark results or a numerical proof for every distribution.
The symbolic derivation above covers arbitrary K, positive π and admissible
times independently of the finite enumeration.

## Why CE and the current CT objective target different logits

For completeness, fix zₜ and one position. In this section h denotes a possible
clean token, j a possible noisy token, and k the observed current token. Define
the noised clean point mass mₕ(j) = αₜδₕⱼ+(1−αₜ)πⱼ and the noised model
distribution mᵥ(j) = αₜvⱼ+(1−αₜ)πⱼ for a raw probability vector v.
Define the stationary-density ratios

```text
ρⱼʰ = [mₕ(j)/πⱼ] / [mₕ(k)/πₖ]
σⱼ(v) = [mᵥ(j)/πⱼ] / [mᵥ(k)/πₖ]
βₜ = −α′ₜ/αₜ = (1−ε)/αₜ, where α′ₜ is the time derivative of αₜ
φ(a) = exp(a) − 1 − a.
```

The current schedule-consistent categorical per-token loss is
`βₜ Σⱼ≠ₖ πⱼ ρⱼʰ φ(log σⱼ(v) − log ρⱼʰ)`.
Taking its conditional expectation over h with probabilities Dₕ, the
derivative with respect to σⱼ is
`βₜ πⱼ [1 − E(ρⱼʰ | zₜ)/σⱼ]`.
Hence each ratio is minimized at `σⱼ = E(ρⱼʰ | zₜ)`. Substituting
`Dₕ = RₕLₕ / ΣₐRₐLₐ` cancels the local likelihood and yields
`E(ρⱼʰ | zₜ) = σⱼ(R)`. The raw-LOO target R realizes every optimal ratio.

By contrast, conditional expected CE is `−ΣₕDₕ log vₕ`; its minimizing
probabilities are D. For a single clean label h, CE's gradient with respect to
softmax logits is `v − one_hot(h)`, with Euclidean norm at most √2. Thus adding
CE directly to logits interpreted as raw R generally introduces conflicting
population targets. If a hybrid is tested later, CT must consume R(D), not
the unconverted denoiser logits. Pure CE changes the objective and time
weighting; it is not the current CT NELBO. A positive weighting depending only
on time preserves the unlimited-capacity per-time CE optimum, but changes
finite-model optimization and shared-parameter tradeoffs.

The executable audit averages exactly over all nine clean sequences conditional
on zₜ=(1,2), using the skew prior above and αₜ=1−0.999t. It takes the mean over
the two token losses and measures the Euclidean gradient norm across the six
logits. At the corresponding true D and R, it obtains:

| t | CT gradient at D | CT gradient at R | CE gradient at D | CE gradient at R |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 1.172389077 | <1e−10 | <1e−12 | 0.830412200 |
| 0.50 | 0.484095078 | <1e−10 | <1e−12 | 0.453888904 |
| 0.99 | 0.022410950 | <1e−10 | <1e−12 | 0.026734496 |

A separate single-token example fixes clean h=0, noisy k=1, and logits
`[0,8,0]`, favoring an erroneous copy. The CT loss interprets those logits as
raw-LOO scores; CE interprets them as denoiser scores. CE loss is 8.000670700
and its gradient norm is 1.413502475 at every time. Current CT values are:

| t | CT loss | CT gradient norm |
| ---: | ---: | ---: |
| 0.001 | 14242.215225 | 450.120417 |
| 0.010 | 1002.886063 | 6.365664 |
| 0.500 | 5.679427 | 0.004687330 |
| 0.990 | 0.131946 | 0.000204676 |
| 0.9999 | 0.013487 | 0.000021463 |

This demonstrates variable local gradient scales, not gradient variance over
the training corpus, effective optimizer updates, or convergence speed. The
same numerical logits describe different statistical targets in the two
columns. CT's weights encode its variational objective, so larger or smaller
gradients alone do not establish a defect. The example motivates a controlled
optimization experiment without predicting its outcome.

## Minimal implementation needed before an experiment

1. Add explicit, checkpoint-bound parameterization/objective fields, retaining
   `raw_loo/ct` as the default for all historical checkpoints. The proposed
   alternative is `x0_denoiser/ce`; incompatible combinations must fail clearly.
2. Train CE on the denoiser logits with the existing corruption process,
   admissible alphabet, content mask, time sampling and reduction. Include
   eligible unchanged tokens; selecting only changed tokens would change the
   target. Preserve the current mean-of-local-token-ratios reduction unless
   it is a separately declared ablation.
3. At inference compute `loo_logits = denoiser_logits − log(L)`, then use the
   current bridge. Subtracting the log-softmax normalization is unnecessary,
   since the subsequent softmax removes it. Compute `log(L)` with log-add-exp;
   never divide already underflowed probabilities.
4. Apply temperature/top-p after conversion if retaining existing **raw-LOO**
   control semantics. Applying them to D first is a different intervention.
   Start at temperature 1 and top-p 1 to preserve the proved identity. Record
   objective, conversion and controls in checkpoint/config/report provenance.
5. Test the conversion at s=0 and low t, uniform and skew priors, excluded
   immutable tokens, default compatibility and matching CT/CE initialization.
   The toy proof is evidence for the equations; it is not full integration
   coverage for a future training implementation.

For E's mixture floor 0.0002 with 1,880 active tokens, a previously unseen
token has prior approximately 1.06e−7. At t=0.001, an off-diagonal likelihood
can be approximately 1.06e−10, and inference nearer zero makes it smaller.
Log-space conversion is essential. Learning accurate D values and then
dividing by tiny likelihoods may amplify calibration errors; bounded CE
gradients do not remove that risk. Low-noise CE examples also often reward
copying an unchanged input, so CE does not automatically cure copying behavior.
Denoiser calibration and converted R concentration should therefore be
measured alongside molecular metrics.

## Suggested matched first experiment

Start with one selected S or E prior and two fresh warm starts from the exact
same 50k MDLM EMA checkpoint. S uses the uniform prior through the
schedule-consistent categorical implementation; E uses its frozen empirical
prior and mixture. Do not begin with legacy R: its loss uses α=1−t while its
corruption/sampler use α=1−(1−ε)t. CE on the actual corruption would also remove
that mismatch, confounding the objective comparison.

Freeze a CT-vs-CE panel of **1,000 new updates, effective batch 128** per arm:
128,000 example exposures each. Use the same MDLM EMA bytes, conditioner
initialization, seed, pinned ordered data start, microbatch grouping,
accumulation, optimizer, LR schedule and sampling budget. A candidate starting
schedule is the existing 1k L1 schedule: peak 3e−4, warmup 50 updates,
half-cosine decay over 1,000 updates including warmup, and floor 3e−6. It must
be identical in both arms, resolved and recorded before launch, not selected
from the outcome. Optimizer and EMA are fresh for both arms. E1000
initialization would carry another
16,000 exposures and a CT-adapted state; save that as a separately labeled
continuation hypothesis.

First run a separate short, approximately 20-update throughput/memory pilot
at the intended batch grouping. One GPU with microbatch 16 and accumulation 8,
or two GPUs with microbatch 16 and accumulation 4, gives batch 128. Keep the
same GPU count and grouping in both objective arms; equal global batch alone
does not imply equal time-sampling and dropout RNG streams. The subsequent
[V7 pilot](../experiments/udlm/results/engineering_v7_throughput.md) verified
the one-GPU grouping; the V8 launch also reached finite training on two GPUs.
Recorded aggregate GPU usage is a sampled resource observation, not an
allocator peak. Inspect the actual manifests before interpreting throughput. Each launch must dynamically select at most
two GPUs below 10% utilization with sufficient free memory, preserve other
processes, re-probe immediately before exposure, and use tmux plus persistent
logs. Do not overlap with another job beyond the global two-GPU cap.

The original proposal suggested explicit post-initialization reseeding to
ensure matching. The implemented V8 pair instead keeps
`reseed_after_model_initialization=false` in both arms. The CE metadata/state
marker adds no random draws: focused tests verify identical constructor RNG
states, A1 backbone weights, masks, times and corruptions for the matched CT/CE
configurations. Thus reseeding is not a prerequisite for this implementation.
Both arms retain the same seed and grouping; this does not assert bitwise
identity of different objectives or GPU executions. Hosted data resumes do
not restore their cursor, so V8 uses fresh, disclosed data starts.
See [the direct-training plan](udlm_training_continuation.md) for initialization,
scheduler, sharding, runtime and checkpoint-storage limitations.

After the throughput pilot, use prospectively reserved engineering evaluation
seeds ≥1000 that do not collide with existing attempts; disclose all outcomes.
Keep **final benchmark seeds 0, 1 and 2 reserved**, and do not use them to choose
the loss, prior, checkpoint or sampling controls. Independently rescore raw
outputs under both repaired and strict decoding, recording validity,
uniqueness, diversity, quality, NFE, sample counts, runtime and exact hashes.
A 128,000-exposure adaptation comparison answers a learning-efficiency
question. It does not establish superiority over the original MDLM50k model;
a matched extra-update MDLM control and a frozen larger final evaluation are
still needed before that claim.
