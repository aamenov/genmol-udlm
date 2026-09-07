# V15 CPU export attempt 1: loader failure, no derived checkpoints

The fixed four-artifact export panel stopped at its first entry, `r099`, on
2026-09-07 at 14:47:41.620205 UTC. The other three entries were unlaunched.
Wrapper duration was 25.880396 seconds; the first child ran 21.731579 seconds
including imports, with 9.036530 seconds recorded inside the exporter.

The frozen source checkpoint was hash-verified and deserialized on CPU. Source
configuration conversion then raised `UnsupportedInterpolationType: cwd` at
`callback.dirpath`. The exporter had requested resolution of the entire historical
training configuration, including a callback path unused by inference. This
occurred before MDLM model construction, target initialization, or serialization.
There were zero model forward calls, GPU jobs, optimizer updates, generated
molecules, or property-oracle calls. No `transfer.ckpt` exists for this attempt.

The wrapper used source `1b9e2b6b44eba706883f5c276422cff9da3c119a`, exporter
`92181278d05d98f137d7d4c2df84d403332a3678`, and panel SHA-256
`5635177b8568485ca301dddffdca7ea6a69fb3d5d10ba6484d7018760aa6a1cb`.
The unchanged source MDLM50k checkpoint is identified in the panel and request.
The failed terminal SHA-256 is
`223a9875377e8e82538255664effb469f1cf3381189cff6b704ac65c1d97cae1`.
Its owned process group had exited; no cleanup signal or automatic retry ran.

All six request/log/terminal files under
`output/udlm/engineering_v15_transfer_exports_20260907/`, plus the wrapper log,
are preserved byte-for-byte. Any corrected export must use a reviewed loader
repair, a separately pinned panel, and a fresh namespace. The four model configs,
initialization seed 2509, and prospective 14-run molecular design remain fixed;
this loader incident supplies no molecular evidence or scientific selection.
