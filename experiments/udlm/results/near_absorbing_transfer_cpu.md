# Frozen MDLM near-absorbing token diagnostic

Completed 2026-09-07 at 14:18:07 UTC, after the design and executable producer
were pushed at `e108f1184c6e7fd062e952a1f20f342eb62344dc`. All twelve fixed CPU
forwards completed: four MASK mixture weights, three noise times, sixteen
previously used validation rows and 814 content tokens. The frozen local
MDLM50k EMA was unchanged. There were zero optimizer updates, new molecules,
property-oracle calls and GPU jobs.

All-content raw-logit cross entropy, in nats per token:

| MASK mixture weight | t=0.1 | t=0.5 | t=0.9 | Mean terminal KL |
|---|---:|---:|---:|---:|
| 0.9 | 0.356401 | 1.388782 | 3.670211 | 0.000117005 |
| 0.99 | 0.345955 | 1.153420 | 2.857185 | 0.000628705 |
| 0.999 | 0.345955 | 1.046817 | 2.545282 | 0.002018659 |
| 0.9999 | 0.345955 | 1.046817 | 2.536285 | 0.004072913 |

These numbers compare different corruption tasks. At t=0.9, changed non-MASK
observations fell from 72 to 10, 1 and 0 as MASK concentration increased.
Empty-group means remain null; all 48 group records are retained. Seeds
2400/2401/2402 give shared RNG streams at each time, without asserting maximal
or nested coupling. Producer runtime was 28.270s; full child wall time 41.096s.

Forward marginals become closer to absorbing noise, but this alone does not
validate the learned reverse process. The mean terminal KL above uses
alpha_1=0.001 and the observed clean tokens. At fixed positive alpha_1 the KL
diverges as lambda approaches one, even though every tested finite prior has
full support. The exact absorbing endpoint was not evaluated.

The installed MDLM loss substitutes a point mass at already unmasked tokens;
its raw logits at those coordinates receive no direct masked-target objective.
This diagnostic intentionally evaluated unrestricted model logits. For example,
at lambda0.999 and t=0.1 the 737 unchanged content positions have CE0.352536 and
top-1 accuracy90.638%. Interpreting such logits as a clean posterior under a new
categorical process therefore needs separate justification. This observation
motivates a mathematical raw-LOO versus CE transfer comparison; it does not
establish that either interpretation improves molecular generation.

The [prospective design](../designs/near_absorbing_transfer_cpu.md) fixes the
configuration, definitions and caveats. The separate
[two-page PDF](../../../output/udlm/near_absorbing_transfer_cpu_20260907/report_v2/report.pdf)
and [complete group table](../../../output/udlm/near_absorbing_transfer_cpu_20260907/report_v2/groups.csv)
retain all conditions and the analytic interpretation. Their scope is distinct
from de novo molecular quality and the active V14 property-optimization pilot.

Exact evidence identities:

- Raw result SHA-256: `c007b026da8fbcc66986b8cad2bc8e975dcd8b0bb1d42b48d08d344c03c41e8f`.
- CPU terminal SHA-256: `e7bddfcf9fe1cdd34b77ecb3d67cfc567cd284e48e73b0ae9292b6d30d52b782`.
- Final PDF SHA-256: `8251a97254ea3521c7c5b934618d041506884ac0a177053af4a78c4aebe2740d`.
- Final report manifest SHA-256: `6a754c1c40fc25227e3c7c69464657d94a938f0cb0753db8fd59bbfc7e82300f`.

The earlier local PDF layout draft is retained on the host; `report_v2` is the
published bundle. It changes pagination only and performs no new evaluation.
No molecular promotion or GenMol superiority follows from this diagnostic.
