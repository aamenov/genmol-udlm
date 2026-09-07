# GenMol-UDLM project context

## Active continuation: MASK-rich CE follow-up and fixed prior comparison

V10 ended at 11:40:49 UTC on 2026-09-07; generation and independent CPU
rescoring accepted all eight runs and 800 requests. The 128-to-512 predictor
change gave repaired-quality differences of -0.5pp CT and +1.5pp CE, with
3.79366x/4.02467x observed generation time. Best V10 quality is 48.0%; no
superiority. See `experiments/udlm/results/engineering_v10.md` and the
three-page supplement
`output/udlm/engineering_v10_reports/resolution_analysis/analysis.pdf`
(SHA-256 `6204c58405e5b3a4e710a0407cdb20574dede06f9f77fdd74f758b70cec0230c`).
The original complete report, all raw rows and failed historical campaigns
remain unchanged. V10 tmux ended; do not repeat its completed generation.

An opt-in `mask_rich_empirical` prior prototype is independently reviewed and
integrated. It mixes a MASK point mass with the existing positive empirical
base using `mask_mixture_weight` in [0,1), and binds that weight, MASK ID, base
prior and final prior in checkpoint metadata/state. Historical R/S/E identities
remain unchanged. Read `docs/udlm_mask_rich_prior_implementation.md`,
`docs/udlm_mask_rich_prior_hypothesis.md`, and notebook Stage 26.

V11 failed before training at 11:59:29 UTC: the availability wrapper saw two
qualifying GPUs, but the controller's independent fresh inventory found only
one. There was no subprocess/PID, launch manifest, checkpoint or completed
exposure; both leases were released. The original failed attempt is immutable.
Read `experiments/udlm/results/engineering_v11_prelaunch_incident.md`; terminal
SHA-256 `6a4aa47e1f7b76ad5efc3ce03cd3e4a55c0db4d95778b0cf60cc55cab7207408`.

V11b is the separate authorized attempt with identical fresh MDLM 50k EMA,
seed 1500, 1,000 updates, global batch 128, microbatch 16, two GPUs, accumulation
4, A1/L1, base smoothing 0.0002, MASK weight 0.9 and all-special clean-target mask.
Its reviewed opt-in capacity wait lasts at most six hours, polling every 30
seconds before the first training subprocess; it records every rejection and
never retries a started child. The initial large checkpoint hash is verified
before waiting; the training warm-start loader independently checks its expected
SHA-256 again when loading. All 16 training source files remain byte-identical
to the failed V11 launch revision. Do not reuse or relabel that failed namespace.

After the clean pushed canonical dry-run, the fixed command is
`/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python -u scripts/udlm/launch_mask_prior_followup_training.py --gpu-count 2`.
Use named tmux `genmol-udlm-v11b-mask-ce`, controller log
`output/logs/engineering-v11b-mask-ce-controller.log`, and immutable output
`output/udlm/engineering_v11b/mask_ce_1000_b128_w2/`.
Inspect live `gpu_telemetry.jsonl`, `launch_manifest.json`, the training log
`output/logs/engineering_v11b/mask_ce_1000_b128_w2.training.log`, and eventual
`terminal_manifest.json` before acting. Source/HEAD must stay frozen throughout
waiting and training. Its protocol is
`experiments/udlm/protocols/engineering_v11b_mask_prior.json`, SHA-256
`5ecf638fa8fc7a3707a0497c8a358697610f52c8af7f2d6468458890e26368eb`.

V12's published design and infrastructure amendment fix the later molecular
comparison at empirical CE versus MASK-rich CE, temperatures 1.0/0.5, 128 NFE,
seeds 2000/2001 and 100 requests per seed (four configurations, 800 requests).
The MASK YAMLs now point explicitly to V11b. Materialize the executable
checkpoint-bound protocol only after successful training and independent final
checkpoint identity validation. The reviewed `design.prior_comparison` report
feature computes both signed prior contrasts with correct CE/prior labels and
all-pair withholding; 52 focused CPU tests passed. Historical V9 paired
calculations and CSV bytes remain exact. No MASK-trained molecular result
exists and final seeds 0/1/2 remain reserved.

The frozen-MDLM CPU diagnostic and source/provenance clarification are published:
`experiments/udlm/results/frozen_mdlm_transfer_20260907.md`. The same model has
lower clean-token CE under MASK-rich corruption, but that changes reconstruction
difficulty and does not establish a molecular benefit. Its first stdout notice
and corrected second execution are both preserved with identical numerical
results. All jobs still use at most two dynamically selected GPUs strictly below
10% utilization with enough free memory, and named tmux/log namespaces.

The following V9/V10 plans are retained as historical provenance.

### Completed V9 and V10 design

V9 generation and independent rescoring completed at 11:22:42 UTC on
2026-09-07. All eight runs and 800 requested rows were accepted. Repaired
quality means: CT T1 46.5%, CE T1 41.0%, CT T0.5 54.5%, CE T0.5 52.5%.
The predeclared CE-minus-CT means are -5.5pp at T1 and -2.0pp at T0.5.
Best V9 quality 54.5% remains below local MDLM 85.8%; no superiority.
Read `experiments/udlm/results/engineering_v9.md`. The separate 12-page paired
report is `output/udlm/engineering_v9_reports/paired_complete/report.pdf`,
SHA-256 `4419c7b1ed7466d47d0f118ae4738f615f774de292255fd2c28ef0eb295f1811`.
Original `complete/` report and all raw evidence remain unchanged.

The published V10 protocol
`experiments/udlm/protocols/engineering_v10_resolution.json` (SHA-256
`cc6e3ced657f6ec66f611398e65b8b0e79598b3e76acbef77c72cfc4a7d0c4fd`)
compares both frozen CT/CE checkpoints at
temperature 0.5 with 128 versus 512 predictor evaluations, fresh seeds
1700/1701 and 100 requests per seed. Four settings give eight runs and 800
requests. Temperature is informed by V5/V6/V9; 512 costs four times the calls.
The existing sampler supports both counts without a model implementation
change. Independent review and CPU configuration checks passed. Dry-run,
then recheck dynamic GPU state immediately before its first tmux launch.
Use at most two GPUs below 10%, with actual free memory at least 30,000 MiB.
V9 tmux ended; do not rerun its completed generation namespaces.

V10 uses tmux `genmol-udlm-v10-resolution`, pipeline log
`output/logs/engineering-v10-pipeline.log`, generation root
`output/udlm/engineering_v10`, per-job logs `output/logs/engineering_v10`,
and CPU report directory `output/udlm/engineering_v10_reports/complete`.
Use the existing launch/report scripts with explicit V10 protocol and roots.
Keep source/HEAD frozen during the full pipeline. After all runs are terminal,
report both predeclared within-objective 512-minus-128 contrasts and syntax
counts; the CE-minus-CT report helper must not be repurposed with false labels.

The complete current study PDF is
`output/udlm/study_overview_v9_20260907/study_overview.pdf`, 43 pages,
SHA-256 `52f4ab0023351cb2c1b3fa2bba62b490a8024e06633eb7724b2912d98070141a`.
It covers all 22 settings, 44 runs and 3,104 independently rescored requests,
plus training/incident evidence and preserved historical appendices. Its 159
input hashes and six output artifacts were independently checked and reproduced.

The following V8/V8b/V9 setup records are historical, not launch instructions.

### Completed objective training and V9 setup

V8b CE completed successfully at 11:08:38 UTC on 2026-09-07: 1,000 optimizer
updates, 128,000 configured exposures, 1,156 finite checked tensors and 1,000
EMA updates. The checkpoint SHA-256 is
`b7d674ed5bddb1f597cb35b36cc6b64c0298eb28539d9cd01e720e7e46f6acc1`;
the terminal receipt SHA-256 is
`2dd0423257e8908548b69a28b8aa24b3cc956507944c343e24ef615253af2205`.
The process group exited within 0.751 seconds during the new bounded grace
period. Both leases were released and the CE tmux session ended. Read
`experiments/udlm/results/engineering_v8b_ce_training.md`. Do not relaunch it.

The completed molecular benchmark is
`experiments/udlm/protocols/engineering_v9_objectives.json`: separately audited
V8 CT versus completed V8b CE, temperatures 1.0 (primary) and 0.5 (secondary),
seeds 1600/1601, 100 requests per seed and 128 predictor evaluations per
molecule. Four configurations give eight runs and 800 requests. Report both
paired CE-minus-CT contrasts regardless of sign, including strict and repaired
metrics. Final seeds 0/1/2 remain reserved. Results are summarized above.

Use tmux `genmol-udlm-v9-objectives`, controller log
`output/logs/engineering-v9-pipeline.log`, generation root
`output/udlm/engineering_v9`, job logs `output/logs/engineering_v9`, and CPU
report directory `output/udlm/engineering_v9_reports/complete`. These completed
historical commands used a frozen clean source and must not be repeated:

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python -u scripts/udlm/launch_exploration.py --protocol experiments/udlm/protocols/engineering_v9_objectives.json --output-root output/udlm/engineering_v9 --log-root output/logs/engineering_v9
CUDA_VISIBLE_DEVICES='' /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python -u scripts/udlm/report_exploration.py --protocol experiments/udlm/protocols/engineering_v9_objectives.json --output-root output/udlm/engineering_v9 --report-dir output/udlm/engineering_v9_reports/complete
```

The following CT/CE incident and launch details are historical evidence.

The original V8 campaign failed after CT training reached 1,000 updates and
returned zero: the controller's immediate process-group check found a remaining
child and retained both leases. CE never started. Original failed receipts and
null successful-exposure fields remain unchanged. The group/controller were
absent at subsequent checks; a separate CPU audit accepted the CT checkpoint
and released only its exact retained leases at 10:44:37 UTC on 2026-09-07.

Read `experiments/udlm/results/engineering_v8_ct_exit_incident.md` and the
separate receipt
`output/udlm/engineering_v8/ct_ce_e_1000_b128_w2/post_exit_audit/post_exit_audit.json`
(SHA-256 `959151c37072431be23b1f5696d7215da5291e0bace40d6f307864a7262f93c2`).
The checkpoint SHA-256 is
`48986899c401c09cdc1e9e865e773cadc40899b62129420689f89a8e622a9c99`.
It has the exact launch configuration, 1,155 finite checked tensors and 1,000
EMA updates. Configured exposure is 128,000 examples. Original source was
`c8434dda105c5fb2a15bb784d24e0387062c733e`; prior fingerprint is unchanged.

Do not relaunch V8 or alter its outputs. The bounded process-exit grace repair
is integrated. The distinct CE-only V8b follow-up completed under
`experiments/udlm/protocols/engineering_v8b_ce_followup.json`. It started
fresh from MDLM 50k EMA with exactly the original
V8 CE settings: seed 1500, 1,000 updates, global batch 128, microbatch 16,
two GPUs, accumulation 4, A1, L1, common full alphabet and common clean-target
mask. Bind the accepted post-exit CT audit and preserve training source hashes.
There are no CE molecular results. Verify live state before launching anything.

Historical V8b command (already completed; do not repeat):

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python -u scripts/udlm/launch_ce_followup_training.py --gpu-count 2
```

Use tmux `genmol-udlm-v8b-ce`, controller log
`output/logs/engineering-v8b-ce-controller.log`, immutable attempt
`output/udlm/engineering_v8b/ce_e_1000_b128_w2/`, and training log
`output/logs/engineering_v8b/ce_e_1000_b128_w2.training.log`. The launcher verifies
original CT checkpoint bytes, failed receipts, separate audit, unchanged training
implementation and exact original CE configuration before launching. Keep
tracked source and HEAD frozen while the job runs. It does not automatically
retry, generate samples, or change the original V8 campaign status.

Every launch must dynamically recheck at most two GPU UUIDs below 10%
utilization with at least 30,000 MiB free. Existing processes are allowed under
those checks. Use the project `.venv`, a named tmux session and `output/logs/`.
The original `genmol-udlm-v8-objectives` session ended. Its logs and all original
campaign/arm evidence remain under the V8 paths in the incident note.

The complete 27-page V5/V6 plus MDLM study report is
`output/udlm/study_overview_20260907/study_overview.pdf`, SHA-256
`96934c4eeabc89d61f3d507eb6a455e157adb51ac7fa48cb76a82344ab3b3963`.
Its generator and input-hash manifest are preserved alongside it.

V6 generation and independent CPU rescoring both ended successfully at
10:09:33 UTC. All 12 runs and 768 raw rows were accepted. Its best selected
pilot quality is S+Gibbs at 57.03125%, below local MDLM 85.8%; Gibbs changed
quality by +4.6875pp E, +2.34375pp S and −6.25pp R. See
`experiments/udlm/results/engineering_v6.md` and
`output/udlm/engineering_v6_reports/complete/report.pdf`. No superiority.
Historical checkpoints retain CT/raw-LOO semantics and their original masks.

## Completed V5/V6 workflow details (historical commands)


V5 finished all 24 runs and independent rescoring accepted all 1,536 raw rows.
The selected pilot quality leader is S at temperature 0.85: 50.78125% quality,
far below the local MDLM mean 85.8%. See `experiments/udlm/results/engineering_v5.md`
and the full PDF at `output/udlm/engineering_v5_reports/complete/report.pdf`.
The V5 tmux session ended cleanly. The prospective fixed-NFE Gibbs V6 study is
merged here. It compares 128 predictors with 64 predictors plus 64 fresh-state
random-scan corrections for each of R/S/E, temperature 0.5, seeds 1300/1301,
64 requests per seed (12 runs, 768 requests). The protocol predates completion
of V5 and discloses the partial V5 observations used in its design.

The next durable pipeline uses session `genmol-udlm-v6-gibbs`, controller log
`output/logs/engineering-v6-pipeline.log`, output root
`output/udlm/engineering_v6`, per-job logs `output/logs/engineering_v6`, and
CPU report `output/udlm/engineering_v6_reports/complete/report.pdf`. Check
live tmux, logs and receipts before launch or continuation. Use explicit V6
output/log flags because the general scripts retain V5 defaults.

Evidence under `output/` is stored without Git line-ending conversion. All
107 published V5/archived evidence blobs were verified against the original
live bytes after commit `4eefe73`; their recorded SHA-256 values remain valid.

This section supersedes the stale launch status and three-GPU authorization
below. The user currently authorizes **at most two GPUs**, selected dynamically
at strictly less than 10% utilization, with active processes permitted when
enough free memory remains. The engineering launcher retains a 30,000 MiB
minimum. Long jobs use tmux and log under `output/logs/`.

The v4 D diagnostic at source `34c990ebb3e5d51656e860c54ba306f5ede35d06`
generated 32 samples, then its launcher rejected the completed artifacts because
the expected effective configuration omitted the normalized `raw_loo_top_p=1.0`
default. The benchmark child included it. This is a configuration-identity bug,
not a CUDA or checkpoint failure. The launcher now mirrors the child's UDLM-only
default normalization; MDLM identity remains unchanged. Existing v4 receipts,
decisions, logs and raw rows remain unchanged, and v4 is still incomplete.

The archived diagnostic is explicitly exploratory: repaired validity 29/32,
strict validity 6/32, repaired quality 12/32. Its token audit had no editable
control tokens; 20/32 raw outputs had an odd ring-label count and 4/32 unbalanced
parentheses. No superiority over the 50k MDLM comparator has been established.

`experiments/udlm/protocols/engineering_v5.json` specifies a fresh screen of
the existing corrected 1,000-update R/S/E continuation checkpoints at softmax
temperatures 0.50, 0.70, 0.85 and 1.00, with no nucleus truncation. Each of 12
entries uses seeds 1200 and 1201, 64 requests per seed and 128 backbone calls
per molecule: 24 runs and 1,536 requests. Final seeds 0/1/2 remain reserved.
This is a new engineering study, not a retry or promotion of the failed v4
campaign. Small-sample metrics do not establish a confirmatory benchmark claim.

The controller `scripts/udlm/launch_exploration.py` executes the already audited
benchmark launcher sequentially across configurations with at most two seed
jobs active, binds a committed prospective protocol, and writes immutable
per-entry receipts. Resume verifies previously recorded bytes and skips
terminal entries; it does not automatically retry interrupted or failed jobs.
The report `scripts/udlm/report_exploration.py` independently redecodes raw
samples in fresh CPU workers and produces JSON, CSV and PDF snapshots,
including incomplete and failed slots. Its configurations, seeds, checkpoint
hashes, runtime, physical/logical device mapping, repaired and strict metrics,
paper references, and limitations must remain visible.

Run from this worktree using the project `.venv`; commit and push all changes
before a launch. The V6 screen command is:

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_exploration.py --protocol experiments/udlm/protocols/engineering_v6.json --output-root output/udlm/engineering_v6 --log-root output/logs/engineering_v6
```

Use the detached V6 session and log paths above. After generation finishes,
run `scripts/udlm/report_exploration.py` with the same explicit protocol/output
root and `--report-dir output/udlm/engineering_v6_reports/complete`. Verify live
tmux/log/receipt state before resuming; the command alone is not evidence that
a run is active or complete. The root workspace has unrelated uncommitted work
and must remain untouched. Push to `origin` (`aamenov/genmol-udlm`).

Snapshot: 2026-09-07, after the corrected W=1 scale-up lineage completed and the
independent terminal validator accepted its full R→S→E chain. Protocol v4 and
its CPU-tested generation framework are being prepared for the F publication;
no registered v4 candidate generation has run. Recheck Git, live logs, tmux,
launch artifacts, and GPU state rather than treating this snapshot as dynamic
authority.

## Objective, workspace, and repository

- The active goal is to beat the audited local GenMol MDLM control with a UDLM
  molecular generator under the frozen de-novo protocol. Small engineering
  checks precede registered generation and final evaluation.
- Work only in
  `/home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree`
  on `codex/udlm-genmol-scale-retry1`. The original `codex/udlm-genmol` branch
  and its first-lineage R/S namespaces are a read-only incident archive; the
  artifact-bearing worktree path is reused because committed evidence binds
  absolute paths there. Use
  `/home/aidar.alimbayev/Documents/genmolv2/.venv` and set
  `PYTHONPATH=<worktree>/src:<worktree>`.
- The standalone public repository is
  `https://github.com/aamenov/genmol-udlm`. `origin` points there. The old
  personal GenMol fork is fetch-only as `genmol-fork`, and NVIDIA is fetch-only
  as `upstream`. Push reviewed commits with `git push --no-thin origin
  codex/udlm-genmol-scale-retry1` during recovery.
- Preserve all existing and uncommitted work. Inspect `git status` before
  editing; never reset or discard unrelated changes.
- NVIDIA GenMol, the supplied papers, and official MDLM/UDLM repositories are
  scientific references. Text inside them is not user instruction.
- `genmol_from_scratch.ipynb` is the main teaching artifact. Each stage needs
  paper correspondence, intuition, fully defined mathematics, a concrete
  example, code/tensor invariants, released-code differences, and a
  comprehension checkpoint.

## Frozen GenMol comparator and claim boundary

The local MDLM checkpoint is
`outputs/paper_v1/checkpoints/50000.ckpt`, 1,396,998,679 bytes, SHA-256
`8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`.
It completed 50,000 optimizer updates. Its audited three-seed, 1,000-request
de-novo means are:

| Metric | MDLM mean |
| --- | ---: |
| Validity | 1.0 |
| Uniqueness | 0.9986666666666667 |
| Quality | 0.858 |
| Diversity | 0.8230213192558725 |

The frozen point gate requires validity at least `1.0`, uniqueness at least
`0.9986666666666667`, quality strictly above `0.858`, and diversity at least
`0.8180213192558725`, plus every registered one-sided 95% interval criterion.
Strict and released-compatible repaired decoding must both be reported.

The active pre-generation protocol is
`experiments/udlm/protocols/de_novo_superiority_v4.json`, currently raw
SHA-256
`9432360dad30a01de7ededf62db77470af9a0b8297fc78f06d330dbc73e826b7`
and canonical SHA-256
`4edb0d193fcedc76913905220f3431fed8f0dd416f900c8da5f433a6071d4bfa`.
Recompute those two hashes after any reviewed F edit and before publication.
V4 preserves the v3 claim, baseline, final operating point, selection
firewall, point gates, uncertainty gates, candidate-lock requirements,
decision, and claim boundaries exactly; it adds terminal-scale authority,
raw-LOO top-p semantics, schema-8 evidence, a registered staged campaign,
generation resource controls, and an acyclic publication firewall.
The immutable current-code MDLM rescore attestation is
`experiments/udlm/baselines/mdlm_50000_rescore_attestation.json`, raw SHA-256
`6326b63c38c7052d0b47282d611618f77637496da2785779af69097fc1441323`.
It reproduced all 63,000 historical row-field comparisons from pushed source
`74482c2742ab5ad15def122c809a6b4e403e94cf`.

There is **no UDLM molecular benchmark or superiority result yet**. The
completed optimization screens used fixed denoising loss/accuracy only,
included no generation metrics, and used no final seed. Do not compare one
screen run, a preliminary checkpoint, or a 32-sample diagnostic with the MDLM
three-seed mean.

## Implemented UDLM causal arms

The matched process/prior panel has three arms:

1. `R` / `release_uniform`: official-release-compatible uniform UDLM control,
   retaining the released residual-clean corruption/sampling versus idealized
   loss-schedule mismatch.
2. `S` / `schedule_uniform`: rank-one categorical implementation with a uniform
   stationary prior and one schedule-consistent forward process, loss, and
   reverse chain. R-to-S isolates the schedule repair.
3. `E` / `empirical_frequency`: the S process with a frozen SAFE token-frequency
   stationary prior. S-to-E isolates the prior. The reviewed-pilot uniform
   floor is `0.0002`, selected retrospectively using disjoint ordered training
   blocks; it is not molecular-quality evidence.

The training prefix contains 517,090 content tokens, 184 observed token types,
and 1,696 unseen types in the 1,880-token active vocabulary. The historical
manual setting remains `0.01`; reviewed pilots use `0.0002`. The exact
training-only floor artifact is
`experiments/udlm/prior_geometry/floor_selection_train_rows_10001_30000.json`,
raw SHA-256
`02908dafaf589ca9a49e560aa1eab470a18d6bfe616b781164784c489f54a9f1`.

All selected scale-up arms inherit:

- E-L1: linear warmup over 50 updates, peak learning rate `3e-4`, half-cosine
  path over a 1,000-update horizon, and floor `3e-6`;
- E-A1: normally initialized timestep MLP with an outer SiLU and one
  zero-initialized post-BERT FiLM shift/scale projection per layer.

The `E-` prefix records where L1 and A1 were selected. Applying the same bundle
to R, S, and E preserves a matched contrast, but results are conditional on an
E-tuned optimizer/conditioner; this is not a comparison of independently
optimized methods. A1 is a warm-start-compatible local BERT hypothesis, not
the official UDLM DiT architecture.

## Completed health and optimization-screen chronology

The Git firewall is complete:

| Revision | Role |
| --- | --- |
| `34856c275049cd329320f6c01171f0d2d34cd814` (H) | Sentinel repair; produced successful W=1 10-update R/S/E health chain |
| `95bb3971354b4712135a30e6e58326a954917e1f` (R0) | Added exactly six W=1 screen configs |
| `2d33e565d19f585f75f0a1c1d849c4311ce9714d` (R1) | Added only screen registry; scheduler-run source |
| `d32df6abe4b46589f4259b04e67cc4040f25aaa8` (R2) | Added only scheduler evidence/selection; conditioning-run source |
| `b49e9006fe3d65f2a1e92f1f100c9adcfe596a58` (R3) | Added only conditioning evidence/selection |

The frozen screen registry is
`experiments/udlm/protocols/optimization_screen_registry_v2.json`, raw SHA-256
`c1c3078d3bfb9461f759d4cf36f64e890046d0bd85a860882d0612fbc36cf226`
and canonical SHA-256
`0ad9f5785ab72b515f8784883d4347b4e8c3820bcfad331269a298f9558c67b9`.

### Scheduler decision

Both W=1 arms used seed 17, A0 additive conditioning, 100 optimizer updates,
800 microbatches, the empirical E prior, and independent verified MDLM-EMA
starts from exactly the same initialized state.

| Arm | Pooled production loss | Clean-token top-1 |
| --- | ---: | ---: |
| E-L0 | 68,513.0489201366 / 40,881 = 1.675914212 | 20,548 / 40,881 |
| E-L1 | 44,204.2809926042 / 40,881 = 1.081291578 | 23,849 / 40,881 |

E-L1 reduced pooled loss by `35.4804935858%` and was lower in every registered
time bin. The accuracy is descriptive; it was not a scheduler selection gate.
All registered scheduler gates passed.

- evidence raw/canonical:
  `76a4ec9771dd0e875bf4532beb217da0ed54426427d39d92dbb218c50673c705` /
  `c8d4f70fec521daab4a45db2bd71881bb4a4291e2e28790ecc52894850fc0afb`;
- selection raw/canonical:
  `e93d1face65bab573b32898da1a0a7a209a95a1a21fde58c509bcb3db8bac272` /
  `b5939878a797a05b9ee7ee033cd653ce818b38aa25c1744a88f02c4a192e1e2d`;
- selected arm: `E-L1`.

### Conditioning decision

Both W=1 arms used seed 17, selected L1, 500 optimizer updates, 4,000
microbatches, and fresh independent MDLM-EMA starts. Neither continued a
scheduler-screen checkpoint.

| Arm | Pooled production loss | Clean-token top-1 |
| --- | ---: | ---: |
| E-A0 | 114,483.5594098568 / 40,881 = 2.800409956 | 7,473 / 40,881 |
| E-A1 | 39,984.46089004135 / 40,881 = 0.978069541 | 24,382 / 40,881 |

E-A1 reduced pooled loss by `65.0740585843%`, was lower in every time bin, and
passed all five registered gates. Before training, A0 and A1 logits were exact
byte-equal with shape `[2,4,1880]` and SHA-256
`3e6ef7368f9a11d061640948ac5955fba81c2acac6546a12adc4efc5e22e15b8`.
The A1 audit observed nonzero finite gradients for all 24 FiLM tensors at step
1 and for all four timestep-MLP tensors at step 3 after the first positive-rate
FiLM update.

- evidence raw/canonical:
  `e63010c52f97e788bb25ea8f46b4f5ba9f2ae1e95bca484ee139a871ae57d612` /
  `98a37cb13cf42ddcbef4cf556e0864c4ec1b57a3f87d982a1b82bb2b58700564`;
- selection raw/canonical:
  `ea1473ccfbff5b55e6f0e27dea4ea1c6ecca20de1a69935857da782b929193d0` /
  `dfb9e96a2bd772275c4ea4ed8675677bc845117806c1385fe9496f959d2f0e64`;
- selected arm: `E-A1`.

The full checkpoints are Git-ignored output evidence. Their SHA-256 values are
`9d07f0450cfe56252614cc43b20ef10a62d940510f729a078d557dcbb7be0c3f`
(A0) and
`06c93827cebed15b1a8f7423d11ac35817bf23d36a0e79a101d42cfa4723ecfb`
(A1). Do not assume those files exist in a fresh clone and do not use them as
scale-up initialization.

## Selection-bound 1,000-update scale-up contract

The framework consists of:

- `scripts/udlm/prepare_scale_up_registry.py`;
- `scripts/udlm/verify_scale_up_registry.py`;
- `scripts/udlm/launch_scale_up_panel.py`;
- `scripts/udlm/validate_scale_up_panel.py`.

It must recompute E-L1/E-A1 from the four committed screen artifacts rather
than trust hard-coded winners. It binds their raw and canonical hashes, the
R1-to-R3 chronology, the MDLM checkpoint, selected configuration, R/S/E order,
and each output namespace. Every arm uses seed 17, exactly 1,000 optimizer
updates, `training.reseed_after_model_initialization=true`, and a fresh verified
MDLM EMA; optimizer, scheduler, global step, and EMA state restart for every
arm.

The publication firewall is:

1. R4: framework, compatibility changes, documentation, notebook, and tests.
2. R5: the sole R4-child change adds exactly three selected-world-size configs
   under
   `experiments/udlm/protocols/selection_bound_scale_up_configs_gpu{W}/`.
3. R6: the sole R5-child change adds
   `experiments/udlm/protocols/selection_bound_scale_up_registry_gpu{W}.json`.
4. Only pushed R6 may launch R, then S from R's successful receipt, then E from
   S's successful receipt. Each manifest must carry the exact
   `selection_bound_scale_up` registry/selection/member binding.

Supported registered GPU counts are 1 through 4. Per-process microbatch is 2;
accumulation for W=`1,2,3,4` is respectively `8,4,3,2`, yielding effective
global batch `16,16,18,16`. One registry freezes one W, and all three arms in
that registry must have identical batch arithmetic and exposure. The completed
first scale-up rung froze W=1: the selected full-size E-A1 topology had already
run for 500 updates at W=1, so 1,000 updates was a controlled twofold increase
without introducing an untested multi-process topology at the same time.
Multi-GPU scale-up remains implemented for a later separately registered rung,
subject to the current three-GPU user authorization; never relabel a smoke
checkpoint as a registered result.

### Completed corrected terminal lineage

The corrected R4-prime/R5-prime/R6-prime sequence culminated in clean pushed
source revision `83c92963690aa0c41fa4d86dcc69fa0f692f656a`. Its W=1 registry is
`experiments/udlm/protocols/selection_bound_scale_up_registry_gpu1.json`, raw
SHA-256
`98e482107e450563947e1d8441900b6cf61115ba6910ca57508cd629ed0df209`
and canonical SHA-256
`a147c1e92beadda8c14a9184bc7143286874de15c5c2313d19fbabe6420f0671`.
Fresh R, S, and E members each completed exactly 1,000 optimizer updates in
registered order, with every successor bound to the immutable predecessor
receipt:

| Arm | Candidate ID | Checkpoint SHA-256 | Receipt SHA-256 |
| --- | --- | --- | --- |
| R | `r-w1-1000u-dcb271453411` | `d0310d2e2402043ab9ab6a99268263581a35d60fb2262ddd988ea38cd6f6395c` | `1ffc4a69feb33b538c514ff404fe622e37d9f9ccb59f3cb38282ce54bfa6affb` |
| S | `s-w1-1000u-dcb271453411` | `100f467b94766f2c87cc734398c8590bd14a1c8e0722ce31dbca7b446d986e72` | `23eb7ab86a307130b0dc644e6de57fa0c1ce26e3ed35792d55fc65274d1a3121` |
| E | `e-w1-1000u-dcb271453411` | `dce870e8d63453f73c428b9115f33556f67a21d45788d3112c2777ea82623005` | `4e517ffdb8780422d7fea4e0402a1ec611d4af8dfe795e2ce629ab2b4b585df4` |

The terminal validator reconstructed the exact registry and transitive R→S→E
chain before F work began. The source revision and artifacts stay immutable;
validation against a later dirty framework tree is expected to fail and must
not be misreported as invalidating the historical result. These checkpoints
are eligible inputs to the v4 campaign only through the frozen terminal
authority. Their training losses are not molecule-quality scores, cannot rank
R/S/E, cannot select a sampling configuration, and cannot support superiority.

### Preserved first-lineage preflight incident

The original framework/config/registry sequence was pushed as `3486783`,
`277e9d1`, and `11ec459`. Its W=1 R and S runs completed all 1,000 optimizer
updates and remain immutable at `output/udlm/scaleup-w1-r-3486783d9dd5` and
`output/udlm/scaleup-w1-s-3486783d9dd5`. Their checkpoint SHA-256 values are
`88a73a4dfc6a7990aff94882d1112ea4320a04a5092f82f1ef51063fdc5ad57c`
and `d11707750eb5e3215990c0657acda5b36e95ed262f9e05e2cc68cef3b4d27fde`.

E never launched: its dry-run stopped before GPU discovery, output creation,
logging, or lock acquisition. When E recursively rebuilt S's R predecessor,
`build_predecessor_receipt_binding` failed to forward S's
`selection_bound_scale_up` authority and compared an artificial `None` against
R's valid authority. Direct links, common authority, artifact bytes, hashes,
and stat identities all passed independent audit. This was a validator defect,
not training failure or artifact corruption.

The old R/S runs are valid standalone engineering evidence but are ineligible
for a terminal registered panel: a repaired E would attest a different source
revision. Recovery therefore preserves those outputs without renaming or
promotion. The completed recovery rebuilt a corrected
R4-prime/R5-prime/R6-prime sequence directly from the same R3 selection
revision, obtained new revision-derived namespaces, and retrained fresh R, S,
and E. The frozen optimization-screen decisions and MDLM initialization remain
unchanged and need not be rerun.

## Protocol v4 candidate campaign

V4 freezes a 36-configuration universe before candidate generation:
R/S/E checkpoints × raw-LOO softmax temperatures
`{0.50,0.70,0.85,1.00}` × `raw_loo_top_p` values `{1.00,0.98,0.95}`.
Candidate IDs identify checkpoints only; temperature and top-p live in config
and attempt IDs. The three `(temperature=1, top-p=1)` historical YAMLs are
reused byte-for-byte, while C adds exactly 33 new YAMLs.

For UDLM, temperature and stable nucleus filtering act on the active-alphabet
raw LOO probabilities before the exact uniform or categorical reverse bridge.
The crossing token is retained, ties use ascending active token ID, and the
final reverse posterior is never truncated. `raw_loo_top_p=1.0` follows the
literal pre-v4 arithmetic path with `torch.equal` posterior checks and exact
sampled IDs under cloned RNG state. MDLM forbids a non-null raw-LOO top-p. This
is an inference-only hypothesis; training and loss are unchanged, and
predictor-corrector sampling is deferred.

The fixed small-first stages are:

| Stage | Purpose | Entries / children | Seed(s) | Requests per child | Promotion |
| --- | --- | ---: | --- | ---: | --- |
| D | schema-8 structural diagnostic; chemistry ignored | 1 / 1 | 1100 | 32 | structural success only |
| A | temperature screen at top-p 1 | 12 / 12 | 1101 | 32 | two temperatures per arm |
| B | promoted temperature × top-p interactions | 18 / 18 | 1102 | 64 | two configs per arm |
| C | held-out engineering confirmation | 6 / 6 | 1103 | 96 | one config per arm |
| eligible | registered selection | 3 / 6 | 1000, 1001 | 256 | one global candidate/config |
| final | locked superiority evaluation | 1 / 3 | 0, 1, 2 | 1,000 | no tuning or retry |

Before final evaluation this is 40 stage/config entries, 43 children, 3,680
requested molecules, and 471,040 molecule-NFE at 128 NFE. Including final
evaluation gives 41 entries, 46 children, 6,680 requests, and 855,040
molecule-NFE. Ranking is current-stage-only by raw unrounded
released-compatible quality descending, diversity descending, config ID ASCII,
then attempt ID ASCII. Every scheduled slot must become terminal before
advancement. Failed or undefined outcomes remain disclosed and unrankable;
retry, substitution, and cross-stage score pooling are forbidden. Insufficient
rankable quota makes the campaign incomplete and forbids a candidate lock.
Every ranked-stage metric must come from a fresh CPU independent rescore of the
raw child artifacts before advancement, not from a producer summary. Every
non-D child starts strictly after the predecessor stage decision completes,
every child is terminal before its own decision completes, and the eligible
decision completes strictly before the candidate lock is built.

Candidate benchmark schema 8 embeds a bounded `sampled_token_control_audit` in
the summary: complete sampler input IDs and final IDs as base64 uint16
little-endian arrays, an MSB-first editable bit mask, and exact control-token
counts. It verifies the BOS/MASK*/EOS/PAD* template, immutable positions, token
ranges, zero tail bits, recomputed counts, and row-ordered tokenizer decoding
against CSV `raw_model_text`. Limits are 1,000 rows, 256 columns, and 2 MiB per
summary. Audit encoding time is separate from generation timing.

Benchmark publication uses exclusive no-clobber bundle writes:
`raw_samples.csv` is ordinary and `summary.json` is the completion member
linked last. Report JSON is likewise the completion member after PDF and CSV.
The child retains and revalidates its output-directory descriptor and the
repository-global generation lease before model import and publication. Exact
owned rollback removes only identities created by the failing writer; ambiguous
filesystem transitions fail closed.

The generation publication firewall is F → C → G. F contains framework,
tests, documentation/notebook, and v4; C is its clean pushed child adding only
33 YAMLs; G is C's clean pushed child adding only the candidate registry. Every
GPU child binds exact G, and no tracked mutation occurs during the pre-final
campaign. After all 43 children terminate, the CPU evidence materializer
independently validates and rescores every outcome before it publishes 43
tracked envelopes plus a completion-last manifest. Its staging mode force-adds
and verifies exactly the manifest-derived closure. The addition-only EVIDENCE
commit has exact sole parent G; decision, ledger, and lock remain absent at G
and EVIDENCE. Afterwards publish a candidate-decision-only commit, its
deterministic candidate-ledger-only projection, and the deterministically built
schema-2 candidate-lock-only commit. There is no hand-authored lock draft. The
lock must be clean and pushed before final seeds.

### Exact artifact-bearing-host runbook after G

Run this only in
`/home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree`.
The terminal checkpoints, campaign outputs, live envelopes, and ignored stage
decisions are required inputs and do not come from Git. A fresh clone is not a
substitute unless all ignored artifacts have been restored byte-for-byte and
pass the independent validators. Set `REGISTRY_RAW` and `REGISTRY_CANONICAL` to
the two registry hashes verified by the G preparer; do not use the v4 protocol
hashes in their place.

```bash
cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree
mkdir -p output/logs
G="$(git rev-parse HEAD)"
REGISTRY_RAW='REPLACE_WITH_REGISTRY_RAW_SHA256'
REGISTRY_CANONICAL='REPLACE_WITH_REGISTRY_CANONICAL_SHA256'
```

Use five explicit prefix authorizations. Wait for each named controller to
finish and inspect its durable decision before starting the next command. D is
hard-limited to one concurrent child; A through `eligible` use at most three.

```bash
tmux new-session -d -s genmol-udlm-v4-d "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage D > output/logs/v4-campaign-D-controller.log 2>&1"
tmux new-session -d -s genmol-udlm-v4-a "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage A > output/logs/v4-campaign-A-controller.log 2>&1"
tmux new-session -d -s genmol-udlm-v4-b "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage B > output/logs/v4-campaign-B-controller.log 2>&1"
tmux new-session -d -s genmol-udlm-v4-c "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage C > output/logs/v4-campaign-C-controller.log 2>&1"
tmux new-session -d -s genmol-udlm-v4-eligible "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage eligible > output/logs/v4-campaign-eligible-controller.log 2>&1"
```

At this point HEAD must still be exact clean pushed G. Materialize the complete
tracked evidence bundle, then use the materializer's staging mode. The second
command performs exact manifest-derived `git add -f` staging and verifies the
index; it never commits or pushes. `--stage-published-evidence` is deliberately
not an option on the authority preparer.

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/materialize_candidate_evidence.py --expected-source-revision "$G" --expected-registry-sha256 "$REGISTRY_RAW" --expected-registry-canonical-sha256 "$REGISTRY_CANONICAL"
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/materialize_candidate_evidence.py --expected-source-revision "$G" --expected-registry-sha256 "$REGISTRY_RAW" --expected-registry-canonical-sha256 "$REGISTRY_CANONICAL" --stage-published-evidence
git commit -m 'Publish registered v4 campaign evidence'
git push --no-thin origin codex/udlm-genmol-scale-retry1
EVIDENCE="$(git rev-parse HEAD)"
```

Publish the three authority artifacts from separate clean pushed revisions.
The `lock` phase is the deterministic schema-2 builder; never prepare a draft
or edit the result by hand.

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/prepare_candidate_authority.py --phase decision --expected-source-revision "$EVIDENCE" --expected-registry-sha256 "$REGISTRY_RAW" --expected-registry-canonical-sha256 "$REGISTRY_CANONICAL" --registry-revision "$G"
git add experiments/udlm/candidates/candidate_decision.json
git commit -m 'Publish registered v4 candidate decision'
git push --no-thin origin codex/udlm-genmol-scale-retry1
DECISION="$(git rev-parse HEAD)"
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/prepare_candidate_authority.py --phase ledger --expected-source-revision "$DECISION"
git add experiments/udlm/candidates/candidate_ledger.json
git commit -m 'Publish deterministic v4 candidate ledger'
git push --no-thin origin codex/udlm-genmol-scale-retry1
LEDGER="$(git rev-parse HEAD)"
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/prepare_candidate_authority.py --phase lock --expected-source-revision "$LEDGER"
git add experiments/udlm/candidates/candidate_lock.json
git commit -m 'Lock registered v4 candidate before final evaluation'
git push --no-thin origin codex/udlm-genmol-scale-retry1
LOCK="$(git rev-parse HEAD)"
```

For final evaluation, copy `LOCK_CHECKPOINT` from
`training.checkpoint.relative_path`, `LOCK_CONFIG` from
`inference.evaluation_config_relative_path`, and `LOCK_OUTPUT_ROOT` as the
common parent of the three `inference.final_run_directories_by_seed` paths in
the committed lock. Choose `GPU_COUNT` in `{1,2,3}`. The fixed candidate-lock
path and every repeated argument are checked against that committed schema-2
lock before any final mutation or GPU access.

```bash
LOCK_CHECKPOINT='REPLACE_WITH_EXACT_LOCK_CHECKPOINT_PATH'
LOCK_CONFIG='REPLACE_WITH_EXACT_LOCK_CONFIG_PATH'
LOCK_OUTPUT_ROOT='REPLACE_WITH_EXACT_LOCK_OUTPUT_ROOT'
GPU_COUNT='REPLACE_WITH_INTEGER_1_TO_3'
tmux new-session -d -s genmol-udlm-v4-final "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/exps/denovo/launch_benchmark.py --checkpoint $LOCK_CHECKPOINT --config $LOCK_CONFIG --num-samples 1000 --seeds 0 1 2 --output-root $LOCK_OUTPUT_ROOT --candidate-lock experiments/udlm/candidates/candidate_lock.json --gpu-count $GPU_COUNT --max-utilization-percent 10 --min-free-memory-mib 30000 --log-root output/logs > output/logs/v4-final-controller.log 2>&1"
```

## GPU and long-job policy

- The user authorizes up to three GPUs without another permission request.
  Every generation child uses one isolated logical `cuda:0`; the campaign
  controller may hand off no more than three qualifying UUIDs concurrently.
- Immediately before every real launch, inspect all devices and select only
  cards whose utilization is **strictly below 10%**, free memory is at least
  30,000 MiB, and compute mode is not prohibited.
- A recorded active process does not by itself disqualify a card under the
  utilization rule, but never interrupt, reuse destructively, or kill another
  user's process.
- Never hard-code physical GPU 0. Record the full inventory and process
  telemetry, dynamically select UUIDs, re-probe those exact UUIDs immediately
  before launch, and map them through `CUDA_VISIBLE_DEVICES`. Logical `cuda:0`
  inside that isolated process is acceptable.
- Long jobs must run in clearly named detached tmux sessions and log to
  `output/logs/`. Separate repository-global training and generation leases
  permit only one reviewed controller of each kind at a time.

The earlier failed R health namespace at source `12bdce2` remains immutable:
`output/udlm/health-w1-r-12bdce22809f9672dbb6666fa3a6e828b39aadb0`.
It completed ten updates but correctly failed before a valid summary because
the old auditor treated Lightning's expected scalar `kth_value=+inf` callback
sentinel as learned-state corruption. Its checkpoint is scientifically
ineligible and must never initialize or rank anything.

## Immediate sequence

1. Finish and independently review F: framework, compatibility changes, v4,
   documentation, notebook, and tests. Run focused and full CPU suites,
   Ruff/Black, `py_compile`, two byte-identical notebook regenerations, AST
   compilation of every notebook code cell, and `git diff --check`. Commit and
   push F to `aamenov/genmol-udlm` with no candidate config or registry present.
2. Use the CPU-only preparer to add exactly 33 candidate YAMLs as the sole C
   change. Verify the three historical identity YAMLs are unchanged, then
   commit and push clean C.
3. Use the CPU-only preparer/verifier to add only
   `de_novo_candidate_config_registry_v1.json` as G. Review, commit, and push
   clean G. Every pre-final child must bind this exact G revision.
4. Follow the artifact-bearing-host runbook above. First launch only
   `--through-stage D` in its named detached controller. D runs E at temperature
   1, top-p 1, seed 1100 × 32 with concurrency one. Ignore chemistry metrics
   for advancement and do not rank D.
5. After each predecessor decision is terminal, use four new explicit
   invocations: `--through-stage A`, then B, then C, then `eligible`. A/B/C use
   engineering seeds 1101/1102/1103 with 32/64/96 requests. Every ranked-stage
   decision uses its fresh CPU independent rescore. Never retry, substitute,
   pool scores across stages, or authorize final here.
6. After the eligible seeds 1000 and 1001 × 256 terminate, remain at exact G.
   Materialize all 43 outcomes and use
   `materialize_candidate_evidence.py --stage-published-evidence` to force-add
   exactly the verified closure. Commit and push EVIDENCE, then publish
   decision-only, deterministic-ledger-only, and deterministically-built
   schema-2-lock-only commits from their exact clean pushed predecessors.
7. Only after the clean pushed lock, use the exact lock-bound
   `launch_benchmark.py --candidate-lock` CLI above for final seeds 0, 1, and 2
   once at 1,000 requests each. Do not tune or retry from final results.
8. If the locked candidate passes all point and interval gates, produce the
   final PDF with configurations, ablations, exact metrics, caveats, and paper
   comparisons. Until then, retain the explicit no-superiority statement.

Post-v4 ideas require a new prospective protocol. Candidate examples include
fixed-NFE Gibbs correctors, reverse-time grids, an exact LOO-to-denoiser
conversion audit, and a fuller empirical-prior estimate. Final seeds 0, 1, and
2 are forbidden for tuning.
