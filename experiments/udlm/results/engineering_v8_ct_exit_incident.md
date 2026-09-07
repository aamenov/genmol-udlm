# V8 CT training and process-exit incident

CT completed 1,000 optimizer updates and its training parent returned zero.
The controller nevertheless marked the original V8 campaign failed at
2026-09-07 10:41:42 UTC because its immediate post-exit check still found the
training process group. CE never started. The controller retained both job
leases and did not validate the checkpoint or populate successful-exposure
fields. Those original failed receipts remain unchanged.

The process group and controller were absent when inspected afterwards. This
is consistent with a transient distributed-process teardown race; the exact
members present at the original check were not captured. No model divergence,
CUDA failure or nonzero training return was observed. This diagnosis does not
retroactively make the original controller successful.

## Separate CPU checkpoint audit and manual lease recovery

At 10:44:37 UTC, a separately recorded CPU audit accepted the saved checkpoint:

- Original source: `c8434dda105c5fb2a15bb784d24e0387062c733e`.
- Checkpoint: `output/udlm/engineering_v8/ct_ce_e_1000_b128_w2/ct/checkpoints/1000.ckpt`.
- SHA-256: `48986899c401c09cdc1e9e865e773cadc40899b62129420689f89a8e622a9c99`.
- Size: 1,636,492,424 bytes; optimizer step 1,000; 1,155 finite checked tensors;
  optimizer state present; 1,000 EMA updates.
- Saved configuration exactly matches the original launch. It declares CT
  raw-LOO semantics and has no CE denoiser metadata or state marker.
- Empirical prior fingerprint:
  `f738b8b17de5c4704058018bbddacd7fed779c85248e33d199b68a648151e612`.
- Configured exposure: 128,000 examples (1,000 updates × batch 128), supported
  by checkpoint step/configuration. This is not a distinct-molecule census or
  an independent trace of live dataset rows.
- Training subprocess: 785.822 seconds, including startup and checkpoint
  saving. Console training/save interval was approximately 10:02.
- GPU UUIDs were dynamically selected: physical 7 at 0% utilization and
  physical 2 at 7% in the final launch probes. Peak observed aggregate usage
  was 9,439 and 11,035 MiB respectively, including external processes.

The manual audit verified that the controller and training process group were
absent before and after checkpoint validation, checked the exact saved lease
identities, preserved their bytes, and released only those two leases. It did
not signal any process. The completion receipt is
`output/udlm/engineering_v8/ct_ce_e_1000_b128_w2/post_exit_audit/post_exit_audit.json`,
SHA-256 `959151c37072431be23b1f5696d7215da5291e0bace40d6f307864a7262f93c2`.
Its input claims include the original failed terminal/campaign and checkpoint,
plus training implementation hashes. The raw full-host process listing remains
on the host; its digest is retained, while public evidence omits unrelated
process arguments. The large checkpoint also remains on this host.

## Next experiment and interpretation

The cleanup repair gives child processes a bounded grace period to exit after
the training parent is reaped. A process group surviving that bound must still
fail and retain its leases.

A separate prospective V8b CE run will start fresh from the same MDLM 50k EMA,
with the same seed, data, vocabulary, masks, batch grouping, optimizer and
1,000-update budget as the original V8 CE plan. Its namespace and receipts will
be distinct. Before launch, the separately audited CT checkpoint and original
training implementation must still match their recorded hashes. It does not
resume CT or change the failed V8 campaign status.

The V9 evaluation design must disclose this infrastructure amendment before
any CE model output is observed. All four generation configurations, fresh
seeds, sample counts and sampling budgets remain as originally specified.
No CT molecule benchmark has been run from this checkpoint, and no new
superiority claim follows from training or the post-exit audit.
