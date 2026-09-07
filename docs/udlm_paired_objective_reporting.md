# Paired objective contrasts in engineering reports

The V9 protocol predeclares CE minus CT at temperature 1.0 (primary) and 0.5
(secondary). Earlier report code exported each run and each configuration's
mean, but did not calculate these paired contrasts. The report now resolves
the named control and treatment configurations from
`design.objective_comparison`; it never chooses a comparison from outcomes.
Historical protocols without explicit control/treatment declarations retain
their existing report structure.

For each decoding branch and metric, let `d_s = CE_s - CT_s` at seed `s`.
The mean gives every declared seed equal weight. Sample standard deviation
uses the denominator `S - 1`, where `S` is the number of seeds. For quality,
the calculation starts from independently rescored integer accepted-molecule
counts and the requested-sample denominator, using exact fractions before
final JSON conversion. The other metrics use the recorded independent rescore
values. Strict and released repair/largest-component results remain separate;
diversity is never pooled across seeds.

Every scheduled pair stays visible, including negative differences, failed
runs and undefined metrics. A missing member or undefined metric leaves that
pair's difference null. The mean and SD for that contrast/branch/metric are
withheld until all declared pairs define it; SD also requires at least two
pairs. Available per-seed differences remain visible in incomplete snapshots.
This conservative summary policy prevents a partial run from looking like
the complete predeclared contrast. SD describes seed spread, not a confidence
interval. Matching seed labels does not establish molecule-level coupling or
independence of generated observations.

Each new exclusive report bundle contains:

- `report.json`: existing outcomes/provenance plus `paired_contrasts`, with
  per-seed values/differences and means, SDs and pair accounting for both
  branches and validity, uniqueness, quality and diversity.
- `report.csv`: the existing per-run outcomes and provenance table.
- `paired_contrasts.csv`: seed rows and summary rows for every declared
  contrast, branch and metric. Its SHA-256 is bound by
  `report_artifacts.paired_contrasts_csv_sha256` in the JSON.
- `report.pdf`: existing outcomes and incident disclosure, followed by readable
  per-seed contrast and summary tables. Values use the metric's 0-to-1 scale;
  a quality difference of 0.01 means one percentage point.

The JSON completion marker is published only after every bundle member.
Existing report files are never overwritten. New calculations do not change
source samples, independent rescoring, inference, checkpoint selection or
the V9 training-recovery disclosure. Recovered V8 CT and separate V8b CE remain
explicit; this small engineering comparison cannot establish superiority
over the contextual MDLM or paper means.

`tests/test_udlm_paired_report.py` uses explicitly synthetic counts. At ten
requests per seed, primary repaired CT counts `[5, 7]` and CE counts `[6, 6]`
give differences `[+0.1, -0.1]`, mean zero and sample SD approximately 0.141421.
Its primary strict counts instead give mean -0.15. These fixtures test signs,
pair identity, missing pairs, undefined diversity and exports; they are not
V9 molecular results. The mathematics corresponds to the standalone
Stage 25 teaching example, without launching or loading a model.
