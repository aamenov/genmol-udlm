# UDLM scientific audit — 2026-09-07

The reviewed checkpoint is an MDLM-warmed UDLM adaptation pilot, not evidence
of superiority. The first registered v4 structural diagnostic generated 32
molecules and then failed its launcher completion check. That campaign and its
failure remain immutable; engineering v5 is a separately specified screen.

## Implementation and scientific references

Local official UDLM reference: `tmp/official_udlm_edb0f8c` under the parent
workspace, particularly `diffusion.py::_compute_posterior`,
`_forward_pass_diffusion`, `_ddpm_denoise`, and
`scripts/train_qm9_no-guidance.sh`. The corresponding upstream sources are
[official UDLM](https://github.com/kuleshov-group/discrete-diffusion-guidance),
[official GenMol](https://github.com/NVIDIA-BioNeMo/GenMol), and
[official MDLM](https://github.com/kuleshov-group/mdlm).

The uniform bridge in `src/genmol/diffusion.py` agrees algebraically with the
released UDLM bridge. The categorical extension correctly uses the observed
category's stationary mass in the refresh likelihood. R deliberately retains
the official residual-clean forward schedule versus idealized loss mismatch;
S uses a consistent schedule, and E changes S's stationary prior to empirical
SAFE frequencies. No confirmed diffusion-equation defect emerged in this audit.
This is a bounded source review, not a proof of the complete implementation.

The original [UDLM paper](https://arxiv.org/abs/2412.10193) motivates uniform
noise for iterative token editing and discrete guidance. GenMol's SAFE
representation, fixed sequence framing, BERT backbone, and MDLM warm start are
local adaptation choices; the official UDLM QM9 recipe instead trains DiT from
scratch. The current R/S/E runs each used 1,000 updates with effective batch 16,
or 16,000 additional example exposures. Checkpoints share MDLM pretraining but
do not establish a converged UDLM training budget.

EMA is not frozen near initialization: `src/genmol/utils/ema.py` uses update
warmup `min(decay, (1+n)/(10+n))`. At update 1,000, the effective coefficient is
1,001/1,010, not the nominal 0.9999.

## Archived D observations

All paths here are relative to the artifact-bearing UDLM worktree.

- Attempt directory:
  `output/udlm/de_novo_candidate_campaign_v1/attempts/stage-d-e_t100_p100/seed_1100/`.
- Checkpoint: `output/udlm/scaleup-w1-e-dcb271453411/checkpoints/1000.ckpt`;
  SHA-256 `dce870e8d63453f73c428b9115f33556f67a21d45788d3112c2777ea82623005`.
- Source: `34c990ebb3e5d51656e860c54ba306f5ede35d06`; seed 1100; 32 requests;
  raw-LOO temperature 1; top-p 1; 128 network evaluations per molecule.
- Raw CSV SHA-256:
  `7315c8b0dfd896ffe9b7ec113e5f1d354a623038a81546d3ee243ade506d830a`.
- Summary SHA-256:
  `0e4df461425b6513f8f3a7764207b6b5f9c9820ed7dc19b69dff876979bbb4f2`.

The child completed generation. `failure_receipt.json` records that launcher
completion validation expected an effective configuration without the implicit
`raw_loo_top_p=1.0`, whereas the producer included that field. Literal mappings
and their hashes differed despite equal intended sampling semantics. This is
a validation defect; changing the archived receipt would not be a valid repair.

The unpromoted raw artifacts reported repaired validity 29/32 (90.625%), strict
validity 6/32 (18.75%), repaired quality 12/32 (37.5%), and repaired diversity
0.916025. These 32 samples are descriptive engineering observations, not a
registered benchmark estimate comparable to GenMol's three-seed means.
Quality uses QED >= 0.6 and SA <= 4 after released-compatible repair and
deduplication; its denominator is the requested count. Strict decoding disables
repair and largest-component selection.

An independent CPU syntax recount removed bracket atom substrings, counted
ring labels with `%\d\d|\d`, and flagged any label with an odd occurrence
count. It flagged 20/32 rows. A running parenthesis balance flagged 4/32 rows
for negative intermediate depth or nonzero final depth. The two diagnostics
can overlap; neither is a complete molecular validator. SAFE fragment repair
recovered 23 strict failures, and largest-component selection affected 15 rows.
The complete token audit recorded no editable BOS/EOS/PAD/MASK/UNK in the final
tokens. Banning control symbols therefore does not directly explain these
observed failures.

## Next experiments

The immediate experiment is
`experiments/udlm/protocols/engineering_v5.json`: the three existing R/S/E
checkpoints crossed with temperatures 0.50, 0.70, 0.85, and 1.00; top-p remains
1.00; each setting uses fresh engineering seeds 1200/1201 and 64 requests at
128 NFE. This tests a quality/diversity frontier with no retraining. Failure
artifacts remain disclosed. Final evaluation seeds 0/1/2 are not tuning inputs.
The notebook's Stage 21 only previews this protocol and checks configuration
hashes on CPU.

For a subsequent prospectively specified study, prioritize an informed
predictor–corrector ablation. [Uniform Diffusion Models Revisited](https://arxiv.org/pdf/2605.22765)
identifies the plug-in predictor as leave-one-out (LOO) and derives a Gibbs
conditional from it. For position l and category j, the conditional is
`alpha_s * r_theta,l(j) + (1-alpha_s) * pi_j`, where s is current time, alpha_s
is its clean fraction, r is the predicted clean LOO distribution, and pi is the
stationary prior. This permits correction without a new trained model.

Compare 128 predictor evaluations against 64 predictor plus 64 corrector
evaluations using identical checkpoints, temperatures, engineering seeds, and
requested counts. Refresh predictions after each predictor and resample one
uniformly chosen editable position per corrector. Random-scan one-coordinate
Gibbs preserves the target marginal with exact conditionals. Learned LOO error,
tempering, state-dependent low-margin selection, and parallel multi-coordinate
updates require explicit approximation labels. The authors' [reference code](https://github.com/samsongourevitch/rev_udm)
includes a LOO sensitivity probe; measure how changing only a position's own
noisy observation changes its predicted LOO probabilities before interpreting
the learned conditional as exact. Current BERT does not enforce this invariance.

The current user authorization is at most **two** GPUs, each strictly below
10% utilization immediately before launch, with sufficient free memory and
dynamic UUID mapping. V5 uses a conservative 30,000 MiB free-memory threshold.
Recorded active processes can coexist and must never be interrupted. Historical
three-GPU protocol fields are not current authorization. Long jobs run in named
tmux sessions and log under `output/logs/`.
