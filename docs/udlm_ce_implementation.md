# Optional CE denoiser implementation

This implements a prospective clean-token cross-entropy (CE) adaptation. It has
CPU algebra, integration and checkpoint tests, but no trained CE checkpoint or
molecular benchmark evidence. Stage 23 in `genmol_from_scratch.ipynb` teaches
the conversion with an exact two-token example. The full derivation, literature
distinction, and proposed matched experiment are in
[the CE hypothesis](udlm_denoiser_ce_hypothesis.md).

## Configuration and training

`configs/udlm_ce.yaml` opts into `training.udlm.parameterization: x0_denoiser`,
the schedule-consistent uniform categorical prior, and special-token exclusion.
Override `training.udlm.prior_variant=empirical_frequency` for a separately
matched empirical-prior arm. Existing empirical-mixture configuration still
applies and must be recorded. Legacy `release_uniform` and MDLM are unsupported
for this option. There is one selector: `x0_denoiser` implies CE; absent or
`raw_loo` means the existing CT objective. Existing configuration objects are
not rewritten to insert the default.

`GenMol.forward` always returns raw backbone logits `[B,L,K]`. CE normalizes
over the active alphabet and targets the clean token at every eligible content
position, including positions left unchanged by corruption. Framing, padding,
MASK, UNK and other tokenizer special IDs are masked from CE. The existing
global-token-mean or mean-of-per-row-token-means reduction is preserved. CE
does not use the CT loss's time-dependent weight. Use the existing explicit
MDLM backbone warm start for fresh adaptation; loading a raw-LOO UDLM checkpoint
as CE is rejected.

## Inference and checkpoint interpretation

For candidate token `j`, observed token `k`, retention `alpha(t)` and stationary
prior `pi`, `denoiser_to_loo_logits` subtracts
`log(alpha(t)*1[j=k] + (1-alpha(t))*pi[k])` from each active denoiser logit.
Log-add-exp and the prior's float64 logarithm avoid division by tiny
probabilities. This produces logits whose softmax is proportional to `D[j]/L[j]`.
Immutable positions and excluded vocabulary columns retain their input logits;
lower precision is promoted to float32 consistently with existing UDLM math.
Conversion requires `0<t<=1`; a predictor may transition to `s=0`, but conversion
of a new model evaluation at zero is undefined and rejected.

`GenMol.sampling_logits` returns the identical logits object for raw-LOO models.
For CE, `Sampler.generate` applies conversion after **every** model evaluation:
at the predictor's current state/time, and again at the fresh post-predictor
state/time for Gibbs correction. Temperature and top-p then act on the converted
LOO logits. Their default values of one preserve the coordinate posterior
identity; modifying denoiser logits before conversion is a different operation.
Gibbs still uses the fixed editable mask and existing total-NFE allocation.

CE checkpoints contain `udlm_denoiser_metadata` identifying the actual CE
objective and conversion, plus the scalar int64 state marker
`_udlm_denoiser_ce_version=1`. Loading hooks validate both; even non-strict direct
state loads reject cross-parameterization weights. Historical checkpoints add
neither record. The existing `udlm_prior_metadata` remains the identity of the
underlying process; its CT `objective_scope` names the integrand that process
implements, while the separate CE metadata identifies the optimized objective.

The helper import occurs only for CE so default runtime provenance remains
unchanged. Benchmark provenance must additionally capture `genmol.denoiser`
and checkpoint denoiser metadata before collecting registered CE benchmark
evidence. This change does not register a CE campaign or launch training.

## Limits and verification

Exact clean conditionals give exact coordinate reverse marginals after
conversion. Their independent product generally approximates the correlated
sequence reverse joint. A learned CE distribution may copy low-noise input and
have calibration errors amplified by small forward likelihoods. CE is neither
an established improvement nor the released UDLM variational objective.

`tests/test_udlm_denoiser.py` checks posterior mixtures under uniform and skew
priors, active/full-vocabulary mapping, tiny priors, masked gradients, unchanged
token supervision, CPU backbone differentiation, checkpoint round trips and
cross-parameterization rejection, default state/RNG compatibility, and fresh
conversion before both predictor and Gibbs sampling controls. Run these together
with the existing model, sampler and corrector tests using the project virtual
environment and `CUDA_VISIBLE_DEVICES=`.
