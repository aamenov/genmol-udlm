# Opt-in empirical prior with additional MASK mass

`mask_rich_empirical` uses the existing rank-one categorical diffusion with
stationary probabilities

\[
\pi_{\lambda,j}=(1-\lambda)b_j+\lambda\mathbf1[j=m].
\]

Here `b` is the canonical, smoothed empirical prior already used by E; `m`
is the tokenizer's actual MASK ID; and `lambda` is an explicit new mixture
weight. The base still uses the pinned frequency artifact and its independent
uniform smoothing weight `w`. Giving MASK more probability is a hypothesis
about transfer from MDLM pretraining. It has no molecular improvement evidence.
Mixed absorbing/uniform kernels already appear in
[D3PM Section 4 and Appendix A.2.6](https://arxiv.org/html/2107.03006).

The configuration keys are:

```yaml
training:
  diffusion: udlm
  udlm:
    prior_variant: mask_rich_empirical
    empirical_uniform_mix: 0.0002
    mask_mixture_weight: 0.9
    exclude_special_tokens: false
    mask_all_special_tokens: true
    parameterization: x0_denoiser
```

This fragment illustrates the future 0.9 hypothesis; it is not a complete
training configuration or an approved experimental protocol. The new prior
also supports the existing raw-LOO CT parameterization. CE uses its existing
clean-token loss and its current-state/current-time CE-to-LOO conversion
before sampling controls. The backbone, loss formula, reverse bridge and
corrector are reused; existing inference defaults remain unchanged.

The mixture weight must be an explicitly supplied finite real number in
`[0, 1)`; Booleans, strings, missing values and exactly one are rejected.
The base smoothing weight remains in `(0, 1)`. MASK must belong to the active
corruption alphabet. The clean-position mask and corruption alphabet remain
separate: immutable clean special tokens do not prevent ordinary editable
positions from receiving MASK noise. At zero mixture weight the stationary
probability bytes match E exactly, while the new variant identity remains
distinct. At positive weight the builder mixes the normalized E probabilities
in float64, then uses the existing categorical normalization and full-support
checks.

The frozen new metadata record preserves every base-artifact, tokenizer,
active-alphabet, schedule and probability identity field and adds only:

- `mask_mixture_weight`: the separate MASK weight;
- `mask_token_id`: its full tokenizer ID, with compact ordering bound by the
  existing active-token hash;
- `base_stationary_probs_sha256`: the canonical E prior before the MASK mixture.

`stationary_probs_sha256` identifies the final mixed distribution. Existing
R/S/E metadata exports keep their exact old fields and values. The new variant
alone stores `_udlm_mask_rich_mixture_weight_bits`, a scalar int64 encoding
the float64 weight. This keeps the identity intact under half/bfloat16 casts
and rejects E-to-mask-mixture state relabeling even at weight zero. Checkpoint
loading validates metadata, weight marker, active mapping and prior buffers
before accepting weights, including non-strict state loads. Changing the
weight requires a new checkpoint identity and fresh training; it is not an
inference override for an E checkpoint.

Benchmark checkpoint inspection independently rebuilds both distributions
from the pinned counts and validates the added fields and state marker.
Inference configurations bind the resulting prior metadata hash; launcher
validation and independent rescoring reuse that strict identity check.

Exactly one is a singular absorbing limit: ordinary-token stationary mass
vanishes, while the present density ratios require full support. At fixed
noise endpoint epsilon 0.001, the terminal forward distribution of a clean
non-MASK token still retains clean signal, so its KL to the stationary prior
diverges as the weight tends to one. Finite weights near one can also amplify
calibration errors. These
limits prevent an automatic MDLM-equivalence claim.

CPU tests check explicit mixture probabilities and compact mapping, finite
full support, historical metadata digests, initialization/RNG preservation,
checkpoint round trips and relabeling rejection, mixed-precision identity,
clean-control gradient masking and analytical CE-to-LOO/posterior recovery.
No training or generation run is created by this implementation. Any trial
must be specified after reviewing the current resolution experiment, with
fresh common initialization and its own configuration and receipts.
