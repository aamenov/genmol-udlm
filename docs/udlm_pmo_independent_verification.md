# Independent verification of saved PMO oracle scores

`scripts/udlm/rescore_pmo_run.py` verifies one completed Fexofenadine PMO run
from the fixed campaign panel. It is a CPU postprocessing tool. Its oracle
evaluations happen **after optimization, outside the online oracle budget,
with no feedback to optimization**. A complete V14 run has 2,000 charged
unique molecules and therefore 2,000 additional verification evaluations.
The campaign's six complete runs would require 12,000 verification
evaluations, reported separately from its 12,000 online calls.

The tool refuses to score until the campaign and selected job have successful
terminal receipts, the exact requested configuration and artifact hashes
match, the online budget is complete, and the original run's process lock
can be acquired for shared reading. It checks every frozen panel input
before and after scoring, including oracle implementation and package pins.
It additionally checks that the imported TDC implementation and selected
RDKit modules actually come from the pinned paths. It does not read or
unpickle any optimization state or neural checkpoint.

It independently parses the original JSONL events, canonicalizes each child
with RDKit, and requires the declared 20–40 atom bounds. First-charge
indices must be consecutive; each cache hit must refer to the exact earlier
canonical molecule, score, and call index. Event and cumulative call indices
must agree with the saved manifest and summary. Parent scoring is rejected
for this released-policy, gamma-zero panel.

A fresh TDC oracle receives each unique charged canonical SMILES exactly once
through the singleton-list interface, which propagates evaluator failures.
The return must be one finite real number in [0,1]. Every score is compared
using absolute tolerance `1e-12`. Finite mismatches are collected across all
unique molecules and produce a failed receipt. An evaluator exception or
malformed result stops verification without a retry. The receipt records
attempted and completed evaluations separately; a legitimate zero remains
a valid score.

Example command, **only after the entire campaign has completed**:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python \
  scripts/udlm/rescore_pmo_run.py \
  --input-root /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree \
  --protocol experiments/udlm/protocols/engineering_v14_pmo.json \
  --entry-id v14-mdlm-2300 \
  --run-directory /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree/output/udlm/engineering_v14_pmo/runs/v14-mdlm-2300/fexofenadine_mpo/released/seed_2300 \
  --output output/udlm/pmo_verification_v14/v14-mdlm-2300.json
```

Run the command from the reviewed verifier checkout. `--input-root` may be a
different, frozen checkout in the project workspace. The output must be a
fresh file outside the original run directory; its name is reserved before
any scoring, so an occupied output cannot trigger duplicate evaluations.
Existing runs, summaries, events, and prior verification receipts remain
unchanged. A missing, partial, failed, or hash-inconsistent campaign never
receives a `verified` receipt.

Schema 1 receipts contain `inputs` (absolute paths, SHA-256, byte sizes),
`run` (entry, seed, online budget, source, checkpoint digest), `verification`
(unique count, duplicate count, attempted/completed calls, mismatch count,
tolerance, maximum error, and per-charge original/rescored rows), `source`,
`packages`, and `runtime`. `status="verified"` requires every check to pass.
Injected evaluators used in tests are explicitly marked
`evaluation_mode="synthetic_test"`; those receipts are not real oracle
evidence. Failure receipts can contain only partial provenance, and must
remain unrankable in downstream reports.

The focused CPU tests include actual `run_ablation`/`CachedOracle` artifact
production with a synthetic sampler and synthetic evaluator, plus malformed
ledger/hash/controller cases, duplicate accounting, zero scores, tolerance,
exceptions, active locks, mutation detection, and exclusive output behavior.
No real PMO oracle calls or neural checkpoint loads were used to develop or
test this tool. Verification establishes saved chemistry and score
consistency; it does not by itself establish molecular utility, full PMO
performance, or superiority over GenMol.
