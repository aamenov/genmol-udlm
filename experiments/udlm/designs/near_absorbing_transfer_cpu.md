# Prospective frozen-MDLM near-absorbing diagnostic

Declared 2026-09-07 before executing this diagnostic. This is a CPU token
reconstruction study, with zero training updates, generated molecules, property
oracle calls or GPU use. V14 source, settings and running jobs stay fixed.

Earlier adaptation pilots leave a large de novo quality gap to the local MDLM
50k reference. A new hypothesis is that stronger MASK concentration can reduce
the corruption mismatch faced by the same pretrained clean-token predictor.
This extends the earlier lambda 0/0.9 frozen diagnostic; it is not an independent
held-out confirmation and cannot choose a molecular winner.

Fix the local MDLM 50k EMA checkpoint (SHA-256
`8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`),
the first 16 rows of the existing validation panel (814 content tokens), the
existing empirical frequency table and its 0.0002 uniform smoothing floor.
Their exact hashes are bound by the reused diagnostic source.

Evaluate all four mixture weights **0.9, 0.99, 0.999, 0.9999**, each at noise
times **0.1, 0.5, 0.9**. Use CPU corruption seeds **2400, 2401, 2402** respectively,
reset within each time for every mixture. These give common RNG streams, not
an assertion of maximal or nested coupling. All 12 conditions are retained.
Each condition makes one frozen-model forward on all 16 rows. Four CPU intra-op
threads, one inter-op thread, evaluation mode and no gradients are fixed.

For empirical prior pi_E and MASK token m, use

    pi_lambda = lambda * delta_m + (1-lambda) * pi_E,
    q_t(.|j) = alpha_t * delta_j + (1-alpha_t) * pi_lambda,
    alpha_t = 1 - 0.999*t.

Here j is the clean token and delta_j is its point mass. Every tested lambda is
strictly below one, so all 1,880 categories remain in positive support. Original
special-token positions are immutable; corrupted content may contain controls.
Report all-content, currently-MASK, changed-non-MASK and unchanged groups:
counts, mean/sum cross entropy, top-1 accuracy and the norm of the unreduced
single-token cross-entropy gradient with respect to logits (not parameters).

As an analytic check, compare each forward law to the absorbing marginal at
the same alpha_t. Its one-token total variation distance is exactly

    TV = (1-alpha_t) * (1-lambda) * (1-pi_E[m]).

This identity holds independently of j. Also report the mean exact terminal
KL(q_1(.|j) || pi_lambda) over these observed clean content tokens, with
alpha_1=0.001. The latter does not vanish just because lambda approaches one;
at fixed positive alpha_1 it diverges in the absorbing limit for non-MASK clean
tokens. The exact lambda=1 endpoint is not evaluated by the full-support code.

Lower cross entropy across priors partly reflects changing task difficulty.
Sparse changed-non-MASK groups may have zero observations and must retain null
means. There is no reverse sampling, CE-to-LOO conversion, checkpoint relabeling,
adaptation, time-conditioning change, molecule-quality estimate or promotion.
A later zero-shot conversion or trained near-MASK benchmark requires its own
explicit source/checkpoint identity and prospective experiment design.

This is an application of known structured discrete corruption, not a claim to
invent mixed absorbing transitions; see [D3PM Appendix A.2.6](https://arxiv.org/html/2107.03006#A1.SS2.SSS6).
The local terminal-KL and residual-MASK notes supply the endpoint caveats.
