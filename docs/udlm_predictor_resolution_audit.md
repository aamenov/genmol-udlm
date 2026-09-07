# Finite-step factorization: a CPU oracle diagnostic

This diagnostic isolates an approximation in the existing parallel UDLM
predictor. It uses a fully known probability distribution over two categorical
positions, each with three values. It does not use learned weights, molecular
data, a GPU, or Monte Carlo samples. The script is
[`audit_predictor_resolution.py`](../scripts/udlm/audit_predictor_resolution.py).

## What is being isolated

Let a clean sequence be \(x=(x_1,x_2)\in\{0,1,2\}^2\), with distribution
\(p_0(x)\) given by the matrix

\[
p_0=\frac1{35}\begin{pmatrix}11&1&2\\1&9&1\\2&1&7\end{pmatrix}.
\]

Rows index the first position and columns the second. Matching values have
high probability, so the coordinates are dependent. Let \(\pi\) be the
stationary categorical distribution: either uniform or \((0.72,0.23,0.05)\).
At time \(t\), the retained-clean weight is
\(\alpha_t=1-(1-10^{-3})t\). The coordinate transition is
\(Q_t(j,k)=\alpha_t\mathbf1[j=k]+(1-\alpha_t)\pi_k\), where \(j\) and
\(k\) are the earlier and later values and \(\mathbf1\) is the indicator.
The sequence transition is the product of the two coordinate transitions.

For \(s<t\), write \(p_s\) for the exact noisy sequence distribution and
\(Q_{t\mid s}\) for the sequence transition with retained weight
\(\alpha_t/\alpha_s\). Bayes' rule gives the exact reverse probability

\[
R_{s\mid t}(y\mid z)=
\frac{p_s(y)Q_{t\mid s}(y,z)}{p_t(z)}.
\]

Here \(z\) is the observed later sequence and \(y\) the proposed earlier
sequence. The parallel predictor instead samples the two marginal
distributions of this row independently. Each coordinate posterior is exact
in this oracle example, but their product can lose sequence correlations.
One large jump already illustrates the issue: the matching-value dependence
is weakened when both values are independently redrawn.

## Enumeration and controls

The script propagates complete nine-state probability vectors through 1, 8,
32, 64, 128, 256 and 512 evenly spaced predictor steps from 1 to \(10^{-5}\).
Total variation distance is \(\operatorname{TV}(p,q)=\frac12\sum_x|p(x)-q(x)|\).
It separately reports error against the true endpoint distribution and clean
data. These differ because the endpoint retains a small amount of noise.

Two starts are reported: the exact terminal forward distribution and the iid
stationary prior actually used by the sampler. They differ because the forward
schedule retains 0.001 of the clean signal at time 1. The oracle is queried at
every state and time, so model calibration and out-of-distribution prediction
errors are absent. Temperature is 1.0, with no truncation or corrector.

Controls verify that the exact joint reverse chain recovers the endpoint to
floating-point precision, and that product sampling is exact for an independent
clean-data distribution with the exact start. Six cases, including the final
128/512-step transitions, compare all nine states' coordinate posteriors with
the production categorical bridge. Maximum absolute disagreement is
\(4.44\times10^{-16}\); the independent-data maximum TV is
\(1.83\times10^{-15}\) or less. Arithmetic is float64, not symbolic exactness.

| Prior | 128-step TV | 512-step TV |
| --- | ---: | ---: |
| Uniform | 0.00443343 | 0.00110174 |
| Skewed | 0.00373081 | 0.00092713 |

These entries compare factorized sampling from the exact start with the true
noisy endpoint. The full evidence also includes stationary starts, clean-data
distances, residual initialization error, and every tested step count.

## Scientific boundary

The observed roughly fourfold error reduction supports a bounded molecular
resolution test as a hypothesis. It does not prove that 512 steps improve a
learned model, that the molecular gap is caused by this approximation, or that
temperature-controlled conditionals preserve the oracle argument. Real SAFE
sequences are much longer than two positions. A 512-step comparison costs four
times the predictor calls of 128 steps and cannot establish an equal-compute
advantage. V9 remains the current molecular experiment; no new molecular
protocol, configuration, selection, or job is created by this audit.

Reproduce from this checkout with the project virtual environment:

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/audit_predictor_resolution.py
```

The script checks its invariants and prints JSON without writing files. Saved
evidence is under
[`experiments/udlm/diagnostics/predictor_resolution_oracle_20260907.json`](../experiments/udlm/diagnostics/predictor_resolution_oracle_20260907.json).
