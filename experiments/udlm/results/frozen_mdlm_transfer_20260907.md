# Frozen MDLM transfer diagnostic, 2026-09-07

The same frozen MDLM EMA model had lower token cross-entropy (CE) and higher
top-1 accuracy under the MASK-rich corruption in this small panel. Changing
the prior also changes the prediction task: this is evidence about initial
denoising difficulty, not improved trained models or molecular quality.

The CPU diagnostic used the first 16 rows of the previously used frozen
validation panel, containing 814 content tokens in a padded `[16,106]` tensor.
It excluded all clean special-token positions from scoring and corruption.
The full 1,880-token corruption alphabet remained active. It used the original
MDLM 50,000-update checkpoint, SHA-256
`8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`,
with EMA applied and no weight updates. The empirical prior had uniform
smoothing 0.0002, MASK mixture weight λ in {0, 0.9}, and noise endpoint 0.001.
Base seed 1900 became seeds 1900/1901/1902 for times 0.1/0.5/0.9, resetting the
draw stream across priors at each time. These are coupled draws, not repeated
independent trials or molecule-level matches.

| λ | Time | Draw seed | Content CE, nats/token | Correct / 814 | Top-1 accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.1 | 1900 | 0.676942 | 685 | 84.1523% |
| 0 | 0.5 | 1901 | 2.341534 | 386 | 47.4201% |
| 0 | 0.9 | 1902 | 4.237881 | 137 | 16.8305% |
| 0.9 | 0.1 | 1900 | 0.422874 | 738 | 90.6634% |
| 0.9 | 0.5 | 1901 | 1.319160 | 574 | 70.5160% |
| 0.9 | 0.9 | 1902 | 3.840302 | 213 | 26.1671% |

The [second run's original JSON](../diagnostics/frozen_mdlm_transfer_20260907/mask-rich-frozen-transfer-r2.json)
retains full precision, all six corruption hashes and separate currently-MASK,
changed-non-MASK and unchanged-token statistics. Those three groups partition
the 814 content positions. Currently-MASK counts were 0/0/0 for λ=0 and
77/364/652 for λ=0.9. Changed-non-MASK can include other special IDs. Gradient
norms describe unreduced per-token CE with respect to logits, computed as
the norm of `softmax(logits) - one_hot(target)`; they are not backbone-parameter
gradient norms. Independent CPU review checked the corruption hashes, group
partitions and this formula against autograd.

Both original runs are preserved byte for byte in the
[diagnostic archive](../diagnostics/frozen_mdlm_transfer_20260907/provenance.json).
The first `.json` file is the raw stdout stream: it contains one library
notice before its JSON payload. Revision `79ab550` changed only import-time
stdout handling, directing that notice to the `.log` stderr stream. Its
second `.json` parses directly. The six numerical result records and all
common provenance fields match exactly between revisions
`aab96a1a02c46e25e5dbeb9d9b3a44d07ad46a1e` and
`79ab55018e0e553a47f43bed3e5bc2be55458bdb`; source records, completion times
and runtimes differ. Reported runtimes were 19.994 and 19.551 seconds; this
repeat is a stream-output repair and reproducibility check, not extra evidence
from an independent sample.

All eight recorded source hashes were verified against each run's recorded
Git revision. The publication base `d32ff53` has subsequently integrated the
MASK-rich model implementation, so its `model.py` differs from the diagnostic
source. The sidecar records both hashes; it does not relabel the original run.

One field needs precise interpretation: the original
`stationary_probability_float64_bytes_sha256` hashes the **constructor input**
prior, not the stored normalized process buffer. At λ=0 the vectors coincide.
At λ=0.9 the input sums to `1.0000000000000002`; the constructor normalizes it
to 1, changing a probability by at most `2.220446049250313e-16`. The sidecar
records both exact byte hashes and MASK probabilities. These are hashes of
native float64 bytes, not canonical-JSON checkpoint metadata hashes. This
clarification was reconstructed on CPU using the unchanged diffusion code;
publication did not reload a model or rerun denoising.

No CE-to-LOO conversion, reverse-chain generation, chemistry scoring or
optimizer update occurred. A reused 16-row panel and one corruption draw per
time cannot establish a benchmark gain. Molecular evaluation of a separately
trained, checkpoint-identified prior remains necessary.
