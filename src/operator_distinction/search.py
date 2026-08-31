from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .metrics import DiscrepancyReport
from .probes import Probe


@dataclass(frozen=True)
class Evaluation:
    sequence: int
    probe: Probe
    report: DiscrepancyReport
    elapsed_seconds: float


Evaluator = Callable[[Probe], tuple[DiscrepancyReport, float]]


class SearchPolicy(Protocol):
    name: str

    def choose(
        self, pool: list[Probe], evaluations: list[Evaluation], available: set[int]
    ) -> int: ...


class FixedPolicy:
    name = "fixed"

    def choose(self, pool: list[Probe], evaluations: list[Evaluation], available: set[int]) -> int:
        del evaluations
        return min(available)


class RandomPolicy:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.name = f"random_{seed}"
        self._rng = np.random.default_rng(seed)

    def choose(self, pool: list[Probe], evaluations: list[Evaluation], available: set[int]) -> int:
        del pool, evaluations
        options = np.asarray(sorted(available), dtype=np.int64)
        return int(self._rng.choice(options))


class AdaptivePolicy:
    """Frozen ridge-plus-novelty acquisition over cheap probe descriptors."""

    name = "adaptive_v0_1"

    def __init__(self, warmup: int = 8, ridge: float = 1e-3, exploration: float = 0.35) -> None:
        self.warmup = warmup
        self.ridge = ridge
        self.exploration = exploration

    @staticmethod
    def _feature_matrix(pool: list[Probe]) -> np.ndarray:
        features = np.stack([probe.features() for probe in pool])
        mean = np.mean(features, axis=0)
        scale = np.std(features, axis=0)
        scale[scale < 1e-12] = 1.0
        normalized = (features - mean) / scale
        return np.column_stack([np.ones(len(pool)), normalized])

    def choose(self, pool: list[Probe], evaluations: list[Evaluation], available: set[int]) -> int:
        x = self._feature_matrix(pool)
        options = np.asarray(sorted(available), dtype=np.int64)
        if not evaluations:
            return int(options[np.argmax(np.linalg.norm(x[options, 1:], axis=1))])

        tested = np.asarray([evaluation.sequence for evaluation in evaluations], dtype=np.int64)
        if len(evaluations) < self.warmup:
            differences = x[options, None, 1:] - x[tested, 1:][None, :, :]
            nearest = np.min(np.linalg.norm(differences, axis=2), axis=1)
            return int(options[np.argmax(nearest)])

        y = np.asarray([evaluation.report.selection_score for evaluation in evaluations])
        xt = x[tested]
        penalty = self.ridge * np.eye(xt.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(xt.T @ xt + penalty, xt.T @ y)
        prediction = x[options] @ coefficients
        differences = x[options, None, 1:] - x[tested, 1:][None, :, :]
        nearest = np.min(np.linalg.norm(differences, axis=2), axis=1)
        novelty = nearest / max(float(np.max(nearest)), 1e-12)
        acquisition = prediction + self.exploration * max(float(np.std(y)), 1e-6) * novelty
        return int(options[np.argmax(acquisition)])


def run_search(
    pool: list[Probe], policy: SearchPolicy, budget: int, evaluator: Evaluator
) -> list[Evaluation]:
    if budget < 1 or budget > len(pool):
        raise ValueError("budget must be between 1 and pool size")
    available = set(range(len(pool)))
    evaluations: list[Evaluation] = []
    for _ in range(budget):
        index = policy.choose(pool, evaluations, available)
        if index not in available:
            raise RuntimeError(f"policy {policy.name} selected unavailable index {index}")
        report, elapsed = evaluator(pool[index])
        evaluations.append(
            Evaluation(
                sequence=index,
                probe=pool[index],
                report=report,
                elapsed_seconds=elapsed,
            )
        )
        available.remove(index)
    return evaluations
