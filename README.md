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
weather or fluid models.

## v0.2: sealed operator-family validation

v0.2 asks a narrower and harder validation question before an FNO is introduced:

> Does a paired discrepancy-aware evaluator recover high-discrepancy interventions better than a
> naive intrinsic baseline under deterministic interpolation and held-out operator-family splits?

The protocol is committed as a SHA-256-sealed manifest. It freezes the intervention grammar,
operator families, feature contracts, ridge parameters, pool seeds, budgets, random baselines,
threshold, stopping rule, statistics, and artifact schema. Any edit invalidates the seal.

The comparison is deliberately symmetric:

- `naive_intrinsic` receives eight candidate-independent summaries of the initial field;
- `paired_discrepancy` receives those summaries plus ten generic summaries of an early paired
  candidate/reference response;
- both predict discrepancy over a strictly later, non-overlapping time window using the same frozen
  ridge estimator;
- evaluation labels remain sealed until both prediction vectors have been hashed;
- leave-one-family-out evaluation withholds every example from one misspecification mechanism.

The four controlled families are spectral cutoff, viscosity scaling, advection scaling, and an
unmodeled forcing term. This is a test of synthetic predictive recovery. It is not causal
identification, neural-operator validation, or evidence of physical truth.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
operator-distinction run --output results/v0_1
operator-distinction verify-manifest configs/burgers_v0_2.sealed.json
operator-distinction run-v0-2 --output results/v0_2
```

The run writes:

- `summary.json`: prefix results at every frozen budget;
- `evaluations.jsonl`: every tested intervention and scalar selection score;
- `best_witness.npz`: initial field, reference trajectory, candidate trajectory, and full residual;
- `generalization.json`: perturbation-neighborhood retention around the best adaptive witness.

The v0.2 run writes the sealed manifest, summary, split audit, provenance with source hashes,
prediction commits, per-case JSONL/CSV metrics, and a standalone HTML report.

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

The v0.2 design is specified in [docs/EXPERIMENT_V0_2.md](docs/EXPERIMENT_V0_2.md). Its paired
evaluator has a real screening cost: early paired mechanism steps are reported separately so any
ranking gain is not presented as free.

## Roadmap

1. **Instrument validation:** controlled Burgers reference/surrogate experiment.
2. **Sealed validation:** interpolation and held-out operator-family evaluation without FNO tuning.
3. **Neural operator:** train and freeze a small FNO separately, then connect it through the frozen
   operator contract.
4. **Independent boundary test:** evaluate out-of-distribution and rare-regime witnesses.
5. **Formal replay:** translate one discovered invariant violation into a Lean/TorchLean obligation.
6. **External replication:** test a third-party pretrained physical surrogate without modifying the
   frozen search policy.

## Status

Research prototype. No claim of generalization to neural operators, weather, climate, plasma,
control systems, or physical reality is made by v0.1 or v0.2.
