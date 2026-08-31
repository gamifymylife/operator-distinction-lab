# Experiment v0.1: Controlled Burgers Boundary Search

## Research question

Under equal execution budgets, does the frozen adaptive policy discover stronger and more locally
persistent discrepancies than fixed enumeration or seeded random search?

## Mechanisms

Both mechanisms evolve the periodic equation

```math
u_t + u u_x = \nu u_{xx}.
```

The reference uses the configured spatial resolution with 2/3 spectral dealiasing. The candidate
projects dynamics onto a smaller set of Fourier modes after every Runge-Kutta stage. This creates a
known but hidden-to-the-policy validity boundary around initial conditions containing consequential
fine-scale structure.

## Intervention space

Each probe defines amplitudes and phases for modes 1 through 10. The resulting field is centered and
normalized to the configured RMS amplitude. Every policy receives the same deterministic pool.

## Evidence

For each probe, retain:

- the complete reference and candidate trajectories;
- the residual field over space and time;
- L2 and L-infinity discrepancies by time;
- low-, mid-, and high-frequency residual energy by time;
- mean and energy invariant discrepancies.

The selection score is a fixed weighted combination of normalized components. It is frozen before
the comparative run.

## Policies and budgets

- fixed enumeration;
- adaptive v0.1;
- random seeds 11, 23, and 47;
- prefix budgets 20, 40, 80, 120, and 220.

Each policy executes once to the maximum budget. Lower-budget results are prefixes of the same run,
preventing rerun noise and accidental retuning.

## Success and failure

The experiment succeeds as instrumentation if it is deterministic, produces complete witness
artifacts, and reports honest comparisons.

The adaptive hypothesis is supported only if its advantage persists across multiple pool seeds or
independent problems. A win on this single constructed surrogate is insufficient.

