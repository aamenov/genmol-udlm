# V11: one fresh MASK-rich CE adaptation

The prospective training arm changes the empirical stationary law to
`pi = 0.9 * delta_MASK + 0.1 * pi_empirical`. This is an engineering hypothesis
about transfer from MDLM pretraining. It preserves full support; it does not
reproduce the absorbing MDLM process or establish molecular improvement.
The implementation is the independently reviewed, opt-in
`mask_rich_empirical` variant in commit
`f7cc78838275ff8137b6a749f0f56035324ebd86`.

The completed V10 resolution check informed this next step. All eight runs
and 800 requests were independently rescored. Increasing predictor calls
from 128 to 512 changed repaired quality from 45.5% to 45.0% for CT and
46.5% to 48.0% for CE, at temperature 0.5. These are two seeds with 100
requests each per configuration, compared with 3,000 requests in the local
MDLM quality result of 85.8%. The fourfold call budget did not close this gap;
the small comparison does not prove that discretization error is irrelevant.
The input report SHA-256 is
`8d9b7df5797338b6a768cfd3489af6abdadabe0cf48a8cec212086cfc7714d16`.

The new arm starts **fresh from the original MDLM 50,000-update EMA**, with
a fresh optimizer and EMA. It does not extend either 1,000-update UDLM model.
Its settings match the completed V8b empirical CE arm: seed 1500, 1,000
updates, global batch 128 = two GPUs × microbatch 16 × accumulation four,
A1 conditioner, L1 scheduler, learning rate 0.0003, 50-update warmup,
1,000-update horizon and floor 0.000003. The empirical smoothing weight is
0.0002. All 1,880 tokenizer IDs are active corruption states; all special
tokens in the clean sequence remain immutable context and excluded targets.
CE training and current-state/current-time CE-to-LOO inference conversion
remain unchanged. The controller compares the full resolved configuration
against the historical CE record, allowing only the new prior, MASK weight
and output directory.

The prior law and its identity are separate from the empirical control:

| Quantity | V11 value |
|---|---|
| MASK weight | 0.9 |
| Full tokenizer MASK ID | 4 |
| Stationary MASK probability | 0.9000000106382977 |
| Final prior SHA-256 | `07d011c07bdac8fdbe096f514ded6e2d1b6850268c48bec8dd0e3f3853d013eb` |
| Complete prior metadata SHA-256 | `4e1febe2684beebdbf3d4c86aa01bc24ed1d3941af67b63473979e2ae810ee45` |
| Empirical control metadata SHA-256 | `f738b8b17de5c4704058018bbddacd7fed779c85248e33d199b68a648151e612` |

The pinned frequency artifact, normalized empirical base and added MASK mass
are independently reconstructed during identity validation. Existing R/S/E
metadata and process buffers were checked byte-for-byte against prechange
code. The new source revision differs from V8b; the launcher records the
actual clean, pushed implementation at execution. Matching configuration
and initialization does not couple stochastic corruption or imply equal
optimization difficulty, loss scales or runtime. The 128,000 configured
example exposures are not necessarily distinct molecules.

The protocol binds V8b's completed terminal receipt, original initialization,
checkpoint and empirical-prior identity. It also retains the original V8
campaign's failed status and the separate CT post-exit audit. V11 uses its
own namespace, `output/udlm/engineering_v11/mask_ce_1000_b128_w2`, with one
attempt and no automatic retry or resume. The controller reuses exclusive
leases, dynamic selection of two GPUs strictly below 10% utilization with
at least 30,000 MiB free memory each, and the reviewed 15-second process-group
cleanup grace. Existing processes remain in place. The final checkpoint
must pass CPU configuration, finiteness, EMA and new prior identity checks.
Final checkpoint finiteness does not establish every intermediate update
was finite.

This is a training protocol only. It launches no molecular benchmark and
makes no promotion or superiority claim. After the training outcome is
accepted, any molecular comparison needs a separate prospective protocol
with the frozen new checkpoint, both repaired and strict metrics, all
outcomes and fresh exploratory seeds. Final benchmark seeds 0, 1 and 2
remain reserved.
