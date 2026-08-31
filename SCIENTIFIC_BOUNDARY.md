# Scientific Boundary

## What v0.1 can establish

Within a finite, generated pool of admissible Burgers initial conditions, v0.1 can compare search
policies by:

- best discrepancy discovered at a fixed execution budget;
- executions and wall-clock time required to reach a discrepancy threshold;
- the spatial, temporal, and spectral structure of a discovered witness;
- local retention of the discrepancy under small parameter perturbations.

## What v0.1 cannot establish

It cannot establish that:

- the adaptive policy generalizes beyond this problem;
- the controlled spectral surrogate represents a trained neural operator;
- the reference discretization is physical ground truth;
- a high discrepancy is automatically a safety failure or regression;
- a low discrepancy outside the tested pool implies global equivalence;
- any conclusion transfers to weather, climate, plasma, reactors, or deployed hardware.

## Trust boundaries

The reference operator is a numerical discretization of viscous Burgers dynamics, not nature. Its
status depends on the equation, periodic boundary, viscosity, resolution, timestep, dealiasing, and
floating-point implementation chosen here.

The candidate is intentionally truncated. The search policy is blind to that implementation, but
the benchmark designer is not. Therefore this is instrument validation, not independent discovery.

The scalar acquisition score is an operational search oracle. Scientific interpretation must use
the retained residual field, per-time norms, spectral bands, and invariant errors.

## Stronger evidence required

A stronger claim requires all of the following:

1. a frozen search policy and metric contract;
2. an independently trained or third-party surrogate;
3. independently arising failure conditions;
4. comparison against fixed, random, and appropriate active-learning baselines;
5. replication across equations, geometries, resolutions, and seeds;
6. confirmation that discovered witnesses remain physically admissible;
7. explicit separation between numerical-reference agreement and real-world validity.

## What v0.2 adds

v0.2 can test whether one frozen evaluator ranks high-discrepancy interventions better than another
within four controlled numerical misspecification families. It adds:

- deterministic calibration/evaluation pools generated from distinct seeds;
- interpolation and leave-one-operator-family-out protocols;
- a naive intrinsic baseline with the same regression class as the paired evaluator;
- prediction hashing before sealed evaluation targets are revealed;
- exhaustive-oracle regret, threshold discovery, cost-to-threshold, random/fixed baselines, and
  paired uncertainty estimates;
- manifest, pool, operator-registry, target, prediction, and source-code hashes.

The paired evaluator observes an early candidate/reference response. This is an interventional
measurement, not free side information; its screening mechanism steps are reported separately.

## What v0.2 still cannot establish

v0.2 cannot establish that:

- the early discrepancy causes the late discrepancy;
- a recovered numerical distinction is a physical failure;
- the four constructed misspecification families represent independently arising defects;
- performance transfers to a trained FNO, another equation, another geometry, or a third-party
  operator;
- the pseudo-spectral reference is physical ground truth;
- a statistically positive paired comparison validates the feature contract as a scientific
  measure.

The correct claim level is `synthetic_predictive_validation_only`. Causal recovery, physical
validity, and out-of-domain generalization remain separate hypotheses.
