# V11: GPU capacity rejection before training

The MASK-rich CE attempt failed before starting a training subprocess at
2026-09-07 11:59:29.914397 UTC. No training PID, launch manifest, checkpoint or
completed-example count exists. Both resource leases were released. The original
failed namespace and its null training fields are preserved unchanged.

The one-shot waiting wrapper observed two eligible GPUs at 11:59:24.460351 UTC
and invoked the fixed controller once. The controller made its own fresh full
inventory check at 11:59:29.897513 UTC and found only one eligible GPU. It rejected
launch under the user-authorized maximum of two GPUs, each strictly below 10%
utilization with at least 30,000 MiB free. No foreign process was interrupted.

This is an infrastructure outcome with no molecular or training result. The
wrapper/availability stream and empty training log are retained alongside the
original request, terminal and GPU telemetry. Runtime source was clean pushed
`46ec3b0bc0c54208fd546af4b71108a3f1e31015`.

- Terminal SHA-256: `6a4aa47e1f7b76ad5efc3ce03cd3e4a55c0db4d95778b0cf60cc55cab7207408`.
- Request SHA-256: `1cf5a1c0b25a63ee6189aa77b94aad3e14ec25a666acd7eab358ef117f6b5110`.
- Protocol SHA-256: `29f46ec0f925eca992c3b9b2ed27386dcfacd6789076efe1840b14319d9d9198`.
- Controller runtime after request creation: 0.347756907 seconds.
- Wrapper invoked the controller once and did not retry it.

A distinct V11b namespace is being prepared for exactly the same fresh-MDLM
training settings. Its opt-in bounded availability wait will run inside the
controller before its first training subprocess, preserving every capacity
rejection. Waiting does not resume or repeat a started training job. V12's
prospective molecular design will explicitly update the checkpoint namespace
while retaining its settings, sample counts and seeds. Original V11 remains
failed; no checkpoint can be accepted from it.
