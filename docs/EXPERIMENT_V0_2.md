# Experiment v0.2: Sealed Operator-Family Validation

## Question

Does a paired discrepancy-aware evaluator recover high-discrepancy interventions better than a
naive intrinsic baseline under strict interpolation and held-out operator-family protocols?

This is a validation of an evaluator contract on controlled numerical mechanisms. It is not a
validation of a neural operator or a causal-identification procedure.

## Frozen protocol

The canonical manifest is `configs/burgers_v0_2.sealed.json`. Its payload SHA-256 is:

```text
ac5904a9b60a6b117b3c6151be7e1462a237fb1097bb175a3f3064ce5fcba1dc
```

The seal covers every decision that may affect the headline comparison:

| Component | Frozen choice |
|---|---|
| Reference | periodic pseudo-spectral viscous Burgers, RK4 |
| Candidate families | spectral cutoff, viscosity scale, advection scale, forcing bias |
| Variants | two fixed parameter values per family |
| Calibration pool | 96 probes, seed `20260901` |
| Evaluation pool | 256 probes, seed `20260902` |
| Early response | steps 0–8 |
| Prediction target | steps 9–64 only |
| Estimator | standardized ridge, penalty `0.001` |
| Budgets | 20, 40, 80, 120, 220 |
| Random baselines | seeds 11, 23, 47 |
| Witness threshold | evaluation-case 95th percentile |
| Uncertainty | 2,000 paired bootstrap/sign-flip repeats; Holm correction across budgets |
| Stopping | run every budget for every case |

## Split protocols

### Interpolation

Every operator family is present in calibration and evaluation, but the intervention pools are
generated independently. Calibration labels train the two ridge evaluators. Evaluation labels are
withheld until predictions are committed.

### Leave one operator family out

For each of four folds, all variants and all probes from one misspecification family are excluded
from training. The fitted evaluator is applied to an independently generated evaluation pool from
only that family. This tests transfer across mechanisms, not merely across random samples.

## Evaluators

`naive_intrinsic` receives eight summaries of the initial field: RMS, normalized extrema, gradient
RMS, total variation, skewness, kurtosis, and zero-crossing rate.

`paired_discrepancy` receives the same inputs plus ten generic summaries of the early residual:
final/max/mean L2, L2 slope, final/max Linf, final/max residual-gradient L2, and final/max energy
error. It receives no family label, parameter, or implementation detail.

The late target is the frozen v0.1 vector-discrepancy acquisition score evaluated strictly after
the early window. Full residual fields remain the scientific evidence in v0.1; v0.2 evaluates
ranking recovery and does not redefine a scalar as physical truth.

## Leakage controls

1. Calibration and evaluation use different frozen seeds.
2. Operator-family folds are defined before execution.
3. Both evaluators use the same model class, ridge penalty, calibration labels, and stopping rule.
4. The target window does not overlap the paired feature window.
5. Evaluators receive arrays, never reference/candidate objects or operator parameters.
6. Both evaluation prediction vectors are SHA-256 committed before targets are revealed.
7. The run records hashes for pools, registry, targets, predictions, manifest, and source modules.

These are reproducibility and accidental-leakage controls, not adversarial security boundaries.

## Outcomes

At each frozen budget and for every operator case, report:

- best recovered score and exhaustive oracle score;
- oracle fraction and simple regret;
- probability of discovering a top-5% witness;
- rank cost to the first top-5% witness;
- mechanism executions and measured comparison time;
- early screening mechanism steps for the paired evaluator.

The primary paired estimand is the case-level difference in oracle fraction,
`paired_discrepancy - naive_intrinsic`, at the frozen maximum budget of 220. Every budget receives a
bootstrap interval and sign-flip randomization test; p-values are Holm-adjusted across the five
budgets. Fixed enumeration and three seeded random rankings are descriptive controls.

## Interpretation rule

A weak, inconsistent, or negative held-out-family result narrows or rejects the proposed evaluator;
it must not trigger post-hoc tuning of this protocol. Any changed feature, family, parameter,
budget, seed, target, or statistic is a new protocol version with a new seal.
