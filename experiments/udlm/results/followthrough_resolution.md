# Frozen-checkpoint sampling resolution diagnostic

The temperature1.0 comparison completed all800 requests: CT/CE ×128/512 model
calls × seeds17300/17301 ×100 molecules. Independent CPU rescoring accepted all
eight runs. This is an engineering pilot, with two generation seeds and one
checkpoint per objective; it does not establish superiority over GenMol.

| Objective | Model calls | Strict validity | Strict quality | Repaired validity | Repaired quality |
|---|---:|---:|---:|---:|---:|
| CT |128|54.5%|28.5%|97.5%|44.0%|
| CT |512|53.0%|25.5%|95.5%|43.0%|
| CE |128|58.0%|28.0%|98.5%|40.5%|
| CE |512|62.0%|30.0%|97.5%|44.0%|

The within-seed512-minus128 strict-quality differences average−3 percentage
points for CT and+2 for CE. Repaired quality changes−1 and+3.5 points,
respectively. More sampling calls did not consistently improve molecular quality.
This result weakens a sampling-resolution explanation at these checkpoints and
temperature; it does not rule out other samplers or better-trained models.

The raw samples, per-run summaries and controller receipts are under
`output/udlm/followthrough_resolution_r1/`. Full metric definitions, uncertainty,
configuration, checkpoint hashes, runtime and device evidence are in the JSON,
CSV and PDF files under `output/udlm/followthrough_resolution_r1_reports/complete/`.
The paired report preserves every contrast regardless of sign. Strict metrics
use direct decoding; repaired metrics use the released comparable repair path.

The first infrastructure-only attempt is retained under
`output/udlm/followthrough_resolution/`: no seed worker started because the local
SA scoring input was missing. The r1 repair copied and hash-verified that input
and added actual child-launcher CPU preflights; scientific settings were retained.

The next experiment is a fresh4k adaptation curve at batch128 from a common
MDLM50k EMA, with an equally exposed MDLM control. The separate learning checkout
records the prospective objective/temperature selection using completed pilot
evidence. Final confirmation seeds0/1/2 remain reserved.
