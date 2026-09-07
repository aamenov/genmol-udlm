# Reporting empirical versus MASK-rich CE priors

The exploration report now recognizes `design.prior_comparison` separately
from `design.objective_comparison`. This prevents a comparison of two CE models
from being described as CT versus CE. It reuses the existing independently
rescored metrics, signed per-seed differences, CSV publication and PDF tables;
it changes no training or generation setting.

Each `primary`/`secondary` declaration specifies `control_config`,
`treatment_config` and a positive finite `temperature`. The control entry must
explicitly identify `parameterization: x0_denoiser`,
`prior_variant: empirical_frequency` and its `prior_metadata_sha256`. The
treatment uses the same CE parameterization with `mask_rich_empirical` and its
own prior metadata digest. Every completed run must agree with these declared
prior identities and temperature. The existing independent checkpoint/raw-output
validation still establishes the underlying identity; labels cannot substitute
for it. Declaring both comparison kinds is rejected.

Both contrasts use `MASK_minus_empirical` direction. All four metrics and both
decoding branches retain signed differences, including negative outcomes.
Quality subtraction uses exact certified count/request fractions. A missing,
failed, invalid or undefined pair remains visible and withholds that metric's
paired mean and sample SD; it is not replaced with zero or an available-case
summary. Seed pairing does not imply paired molecular trajectories. The PDF
and caveats describe the changed corruption difficulty and two CE objectives.
The prior-comparison CSV adds an explicit direction column; historical
objective-comparison CSV columns and bytes are retained.

Validation: 50 focused report tests passed, including synthetic full-report
prior comparisons, negative/missing pairs, wrong parameterization/prior/hash,
ambiguous comparison kinds and actual PDF/CSV labels. The synthetic PDF was
visually inspected. Actual historical V9 paired calculations and CSV bytes
match the predecessor implementation exactly. These fixtures are not molecular
results. No historical report was overwritten or generation repeated.
