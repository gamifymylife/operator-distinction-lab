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

