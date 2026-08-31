from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .burgers import BurgersConfig, BurgersReference, LowModeBurgersSurrogate
from .metrics import DiscrepancyReport, compare_trajectories
from .probes import Probe, generate_probe_pool, perturb_probe
from .search import AdaptivePolicy, Evaluation, FixedPolicy, RandomPolicy, run_search


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _json_dump(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _evaluate_probe(
    probe: Probe,
    x: np.ndarray,
    target_rms: float,
    reference: BurgersReference,
    candidate: LowModeBurgersSurrogate,
) -> tuple[DiscrepancyReport, float, np.ndarray, np.ndarray, np.ndarray]:
    started = time.perf_counter()
    initial = probe.field(x, target_rms)
    reference_trajectory = reference.rollout(initial)
    candidate_trajectory = candidate.rollout(initial)
    report = compare_trajectories(reference_trajectory, candidate_trajectory)
    elapsed = time.perf_counter() - started
    return report, elapsed, initial, reference_trajectory, candidate_trajectory


def _prefix_summary(evaluations: list[Evaluation], budgets: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for budget in budgets:
        prefix = evaluations[:budget]
        best = max(prefix, key=lambda evaluation: evaluation.report.selection_score)
        rows.append(
            {
                "budget": budget,
                "best_score": best.report.selection_score,
                "best_probe_id": best.probe.probe_id,
                "operator_comparisons": budget,
                "mechanism_executions": budget * 2,
                "cumulative_seconds": float(sum(item.elapsed_seconds for item in prefix)),
            }
        )
    return rows


def run_experiment(config: dict[str, Any], output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    burgers_config = BurgersConfig(
        grid_size=int(config["grid_size"]),
        viscosity=float(config["viscosity"]),
        dt=float(config["dt"]),
        steps=int(config["steps"]),
    )
    reference = BurgersReference(burgers_config)
    candidate = LowModeBurgersSurrogate(burgers_config, int(config["candidate_modes"]))
    pool = generate_probe_pool(
        size=int(config["pool_size"]), modes=int(config["modes"]), seed=int(config["seed"])
    )
    x = np.linspace(0.0, 2.0 * np.pi, burgers_config.grid_size, endpoint=False)
    target_rms = float(config["target_rms"])
    budgets = [int(value) for value in config["budgets"]]
    maximum_budget = max(budgets)

    def evaluator(probe: Probe) -> tuple[DiscrepancyReport, float]:
        report, elapsed, _, _, _ = _evaluate_probe(probe, x, target_rms, reference, candidate)
        return report, elapsed

    policies = [FixedPolicy(), AdaptivePolicy()] + [
        RandomPolicy(int(seed)) for seed in config["random_seeds"]
    ]
    all_evaluations: dict[str, list[Evaluation]] = {}
    summary: dict[str, object] = {
        "experiment": "controlled_burgers_boundary_search_v0_1",
        "config": config,
        "claim_level": "instrument_validation_only",
        "policies": {},
    }

    with (output / "evaluations.jsonl").open("w", encoding="utf-8") as stream:
        for policy in policies:
            evaluations = run_search(pool, policy, maximum_budget, evaluator)
            all_evaluations[policy.name] = evaluations
            summary["policies"][policy.name] = _prefix_summary(evaluations, budgets)
            for rank, evaluation in enumerate(evaluations, start=1):
                stream.write(
                    json.dumps(
                        {
                            "policy": policy.name,
                            "rank": rank,
                            "pool_index": evaluation.sequence,
                            "probe": evaluation.probe.to_dict(),
                            "elapsed_seconds": evaluation.elapsed_seconds,
                            "metrics": evaluation.report.summary(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    adaptive_best = max(
        all_evaluations["adaptive_v0_1"],
        key=lambda evaluation: evaluation.report.selection_score,
    )
    report, _, initial, reference_trajectory, candidate_trajectory = _evaluate_probe(
        adaptive_best.probe, x, target_rms, reference, candidate
    )
    np.savez_compressed(
        output / "best_witness.npz",
        x=x,
        initial_state=initial,
        reference_trajectory=reference_trajectory,
        candidate_trajectory=candidate_trajectory,
        residual=report.residual,
        l2_by_time=report.l2_by_time,
        linf_by_time=report.linf_by_time,
        spectral_energy_by_time=report.spectral_energy_by_time,
        mean_error_by_time=report.mean_error_by_time,
        energy_error_by_time=report.energy_error_by_time,
    )

    neighborhood_scores: list[float] = []
    for neighbor in perturb_probe(
        adaptive_best.probe,
        count=int(config["neighborhood_size"]),
        sigma=float(config["neighborhood_sigma"]),
        seed=int(config["seed"]) + 1,
    ):
        neighbor_report, _ = evaluator(neighbor)
        neighborhood_scores.append(neighbor_report.selection_score)
    original_score = adaptive_best.report.selection_score
    generalization = {
        "source_probe": adaptive_best.probe.to_dict(),
        "source_score": original_score,
        "count": len(neighborhood_scores),
        "median_score": float(np.median(neighborhood_scores)),
        "minimum_score": float(np.min(neighborhood_scores)),
        "maximum_score": float(np.max(neighborhood_scores)),
        "retained_fraction_at_80_percent": float(
            np.mean(np.asarray(neighborhood_scores) >= 0.8 * original_score)
        ),
        "scores": neighborhood_scores,
    }
    summary["best_adaptive_witness"] = {
        "probe": adaptive_best.probe.to_dict(),
        "metrics": adaptive_best.report.summary(),
    }
    summary["burgers_config"] = asdict(burgers_config)
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "generalization.json", generalization)
    return summary
