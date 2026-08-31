# Frozen Benchmark v0.2

## Outcome

The preregistered paired-evaluator advantage was **not supported**.

The sealed run used protocol SHA-256
`ac5904a9b60a6b117b3c6151be7e1462a237fb1097bb175a3f3064ce5fcba1dc` at source commit
`909f80f`. The protocol was not changed after outcomes were revealed.

## Ranking recovery

Mean exhaustive-oracle fraction across eight operator cases:

| Split | Evaluator | 20 | 40 | 80 | 120 | 220 |
|---|---|---:|---:|---:|---:|---:|
| Interpolation | naive intrinsic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Interpolation | paired discrepancy | 0.985 | 0.985 | 1.000 | 1.000 | 1.000 |
| Held-out family | naive intrinsic | 0.986 | 1.000 | 1.000 | 1.000 | 1.000 |
| Held-out family | paired discrepancy | 0.761 | 0.788 | 0.836 | 0.865 | 0.907 |

At the primary budget of 220, the held-out-family paired-minus-intrinsic difference was `-0.0929`
with a 95% case-bootstrap interval of `[-0.2298, 0.0000]`. The sign-flip p-value was `0.511`; the
Holm-adjusted value was `1.000`. With only eight heterogeneous cases, this is not strong evidence
that the paired evaluator is uniformly worse. It is decisive evidence that this benchmark provides
no support for the proposed advantage.

Top-5% witness discovery at budget 220 was `0.75` for the paired evaluator and `1.00` for the
intrinsic baseline on held-out families. The paired evaluator also consumed 4,096 early screening
mechanism steps per case before the 440 full validation executions. Its ranking deficit therefore
cannot be excused by a lower information or execution budget.

## What failed

The failure is concentrated in the held-out spectral-cutoff family:

| Held-out family | Paired regression Pearson | Paired R² | Paired RMSE |
|---|---:|---:|---:|
| Advection scale | 0.968 | 0.847 | 0.0154 |
| Forcing bias | 0.550 | -67.659 | 0.0102 |
| Spectral cutoff | -0.979 | -41,511.547 | 43.8458 |
| Viscosity scale | 0.930 | 0.856 | 0.0086 |

When spectral cutoff was absent from calibration, the frozen ridge mapping extrapolated the early
paired features catastrophically and reversed the within-family ordering. The paired evaluator's
oracle fraction for the two spectral variants was only about `0.10` at budget 20 and remained about
`0.62–0.64` at budget 220. The intrinsic baseline recovered both oracles.

This does not show that early residuals are uninformative. Within known families, paired regression
fit was excellent (`R² = 0.989`, Pearson `0.994`). It shows that raw generic early-discrepancy
features plus one global ridge calibration do not transfer reliably across distinct error
mechanisms.

## Baseline difficulty

The intrinsic baseline was stronger than expected. Candidate-independent shape summaries of the
initial condition ranked the maximum-discrepancy probe first in nearly every case. This makes the
interpolation benchmark saturated and means the current constructed probe grammar contains a
simple intrinsic route to the witness.

That is a benchmark diagnosis, not a reason to weaken the baseline after seeing the result.

## Scientific interpretation

v0.2 accomplished its project-wide purpose: it prevented a favorable within-family regression
result from being mistaken for generalization. The correct conclusions are:

1. The paired evaluator is useful for known-family prediction in this controlled system.
2. The claimed held-out-family ranking advantage is rejected for the frozen v0.2 contract.
3. An FNO must not be used to tune this evaluator and then presented as independent confirmation.
4. Any mechanism normalization, nonlinear calibration, revised descriptor, harder intervention
   grammar, or alternative target belongs to a separately sealed protocol version.

The run remains synthetic predictive validation only. It establishes nothing about trained neural
operators, causality, physical truth, weather, climate, plasma, or deployed systems.
