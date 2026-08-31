# Operator Distinction Lab

**Active discovery of validity boundaries for learned and approximate physical operators.**

Scientific machine-learning papers usually evaluate a surrogate on a preselected test set. This
project asks a harder question:

> What physically admissible initial condition most clearly exposes the difference between a
> surrogate and its reference mechanism?

For a reference operator \(S\), candidate operator \(N\), admissible intervention set
\(\mathcal I\), and vector-valued observable space \(Y\), the project studies bounded operational
equivalence:

```math
N \sim_{\mathcal I,Y,\varepsilon} S
\iff
\sup_{u\in\mathcal I} d_Y(N(u),S(u)) \leq \varepsilon.
```

The engine searches for a witness

```math
u^\star = \arg\max_{u\in\mathcal I} d_Y(N(u),S(u)),
```

while preserving the full discrepancy field rather than pretending that one scalar score is the
scientific result.

## v0.1

The first experiment deliberately avoids expensive neural-network training. It validates the
distinction instrument using:

- a deterministic pseudo-spectral solver for one-dimensional viscous Burgers dynamics;
- a deliberately imperfect low-mode spectral surrogate;
- vector-valued spatial, temporal, spectral, and invariant discrepancies;
- fixed enumeration, frozen adaptive search, and three seeded random searches;
- budgets of 20, 40, 80, 120, and 220 operator comparisons;
- wall-clock and execution-cost accounting;
- neighborhood tests showing whether a witness is isolated or marks a local failure region.

The surrogate is a controlled stand-in, **not** a neural operator and not evidence about real-world
weather or fluid models. The next milestone replaces it with a trained FNO adapter without changing
the search or evaluation contract.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
operator-distinction run --output results/v0_1
```

The run writes:

- `summary.json`: prefix results at every frozen budget;
- `evaluations.jsonl`: every tested intervention and scalar selection score;
- `best_witness.npz`: initial field, reference trajectory, candidate trajectory, and full residual;
- `generalization.json`: perturbation-neighborhood retention around the best adaptive witness.

## Experimental contract

Search policies may access only:

1. cheap descriptors of admissible initial conditions;
2. the scalar selection score returned after an expensive comparison;
3. their own previous observations.

They may not inspect the reference or candidate implementation. The scalar score selects
experiments; it does not replace the vector-valued scientific evidence.

See [SCIENTIFIC_BOUNDARY.md](SCIENTIFIC_BOUNDARY.md) and
[docs/EXPERIMENT_V0_1.md](docs/EXPERIMENT_V0_1.md) before interpreting results.

The first frozen run is reported in [docs/BENCHMARK_V0_1.md](docs/BENCHMARK_V0_1.md). Adaptive v0.1
found the strongest observed witness within 40 comparisons; fixed enumeration required 220. This is
encouraging instrument evidence, not a generalization claim.

## Roadmap

1. **Instrument validation:** controlled Burgers reference/surrogate experiment.
2. **Neural operator:** train and freeze a small FNO on the same admissible domain.
3. **Independent boundary test:** evaluate out-of-distribution and rare-regime witnesses.
4. **Formal replay:** translate one discovered invariant violation into a Lean/TorchLean obligation.
5. **External replication:** test a third-party pretrained physical surrogate without modifying the
   frozen search policy.

## Status

Research prototype. No claim of generalization to neural operators, weather, climate, plasma,
control systems, or physical reality is made by v0.1.
