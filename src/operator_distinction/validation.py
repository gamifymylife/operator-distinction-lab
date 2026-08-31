from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .manifest import sha256_json


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        design = np.column_stack([np.ones(len(features)), normalized])
        return design @ self.coefficients


def fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float) -> RidgeModel:
    if features.ndim != 2 or target.shape != (len(features),):
        raise ValueError("ridge inputs must be [samples, features] and [samples]")
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (features - mean) / scale
    design = np.column_stack([np.ones(len(features)), normalized])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return RidgeModel(mean=mean, scale=scale, coefficients=coefficients)


def intrinsic_features(initial_state: np.ndarray) -> np.ndarray:
    """Candidate-independent, deliberately naive summaries of an intervention."""
    centered = initial_state - np.mean(initial_state)
    rms = max(float(np.sqrt(np.mean(centered * centered))), 1e-12)
    normalized = centered / rms
    gradient = np.roll(normalized, -1) - normalized
    signs = np.signbit(normalized)
    return np.asarray(
        [
            rms,
            float(np.max(normalized)),
            float(np.min(normalized)),
            float(np.sqrt(np.mean(gradient * gradient))),
            float(np.mean(np.abs(gradient))),
            float(np.mean(normalized**3)),
            float(np.mean(normalized**4)),
            float(np.mean(signs != np.roll(signs, 1))),
        ],
        dtype=np.float64,
    )


def paired_discrepancy_features(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Generic early-response summaries; no operator parameters or implementation are exposed."""
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("paired responses must be same-shaped [time, space] arrays")
    residual = candidate - reference
    l2 = np.sqrt(np.mean(residual * residual, axis=1))
    linf = np.max(np.abs(residual), axis=1)
    gradient = np.roll(residual, -1, axis=1) - residual
    gradient_l2 = np.sqrt(np.mean(gradient * gradient, axis=1))
    reference_energy = 0.5 * np.mean(reference * reference, axis=1)
    candidate_energy = 0.5 * np.mean(candidate * candidate, axis=1)
    energy_error = np.abs(candidate_energy - reference_energy)
    time = np.arange(len(l2), dtype=np.float64)
    slope = float(np.polyfit(time, l2, 1)[0]) if len(l2) > 1 else 0.0
    return np.asarray(
        [
            float(l2[-1]),
            float(np.max(l2)),
            float(np.mean(l2)),
            slope,
            float(linf[-1]),
            float(np.max(linf)),
            float(gradient_l2[-1]),
            float(np.max(gradient_l2)),
            float(energy_error[-1]),
            float(np.max(energy_error)),
        ],
        dtype=np.float64,
    )


class SealedTargets:
    """Commit predictions before exposing evaluation targets and metrics."""

    def __init__(self, targets: np.ndarray, target_digest: str) -> None:
        self.__targets = np.asarray(targets, dtype=np.float64).copy()
        observed = hashlib.sha256(self.__targets.tobytes()).hexdigest()
        if observed != target_digest:
            raise ValueError("sealed target digest mismatch")
        self.target_digest = target_digest
        self._revealed = False

    @classmethod
    def from_targets(cls, targets: np.ndarray) -> SealedTargets:
        copied = np.asarray(targets, dtype=np.float64).copy()
        return cls(copied, hashlib.sha256(copied.tobytes()).hexdigest())

    def commit_and_reveal(self, predictions: dict[str, np.ndarray]) -> tuple[dict[str, str], np.ndarray]:
        if self._revealed:
            raise RuntimeError("sealed targets may be revealed only once")
        for name, values in predictions.items():
            if np.asarray(values).shape != self.__targets.shape:
                raise ValueError(f"prediction shape mismatch for {name}")
        commits = {
            name: hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()
            for name, values in sorted(predictions.items())
        }
        self._revealed = True
        return commits, self.__targets.copy()


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    rmse = float(np.sqrt(np.mean(residual * residual)))
    baseline = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / max(baseline, 1e-12)
    centered_target = target - np.mean(target)
    centered_prediction = prediction - np.mean(prediction)
    denominator = float(
        np.sqrt(np.sum(centered_target**2) * np.sum(centered_prediction**2))
    )
    correlation = float(np.sum(centered_target * centered_prediction) / max(denominator, 1e-12))
    return {"rmse": rmse, "r2": r2, "pearson": correlation}


def ranking_rows(
    target: np.ndarray,
    order: np.ndarray,
    budgets: list[int],
    threshold_quantile: float,
) -> list[dict[str, float | int | bool]]:
    oracle_best = float(np.max(target))
    threshold = float(np.quantile(target, threshold_quantile))
    threshold_positions = np.flatnonzero(target[order] >= threshold)
    cost_to_threshold = int(threshold_positions[0] + 1) if len(threshold_positions) else len(order) + 1
    rows: list[dict[str, float | int | bool]] = []
    for budget in budgets:
        selected = order[:budget]
        best = float(np.max(target[selected]))
        rows.append(
            {
                "budget": budget,
                "best_score": best,
                "oracle_best": oracle_best,
                "oracle_fraction": best / max(oracle_best, 1e-12),
                "simple_regret": oracle_best - best,
                "threshold": threshold,
                "threshold_discovered": bool(best >= threshold),
                "cost_to_threshold": cost_to_threshold,
            }
        )
    return rows


def paired_effect(
    candidate: np.ndarray, baseline: np.ndarray, bootstrap_repeats: int, seed: int
) -> dict[str, float]:
    differences = np.asarray(candidate) - np.asarray(baseline)
    if differences.ndim != 1 or len(differences) == 0:
        raise ValueError("paired effects require non-empty one-dimensional arrays")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(differences), size=(bootstrap_repeats, len(differences)))
    bootstrap = np.mean(differences[samples], axis=1)
    observed = float(np.mean(differences))
    # Monte Carlo sign-flip randomization test, frozen by the manifest seed.
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(bootstrap_repeats, len(differences)))
    null = np.mean(signs * differences, axis=1)
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (1 + len(null)))
    return {
        "mean_difference": observed,
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "sign_flip_p_value": p_value,
        "paired_cases": len(differences),
    }


def stable_seed(base_seed: int, label: str) -> int:
    digest = sha256_json({"base_seed": base_seed, "label": label})
    return int(digest[:8], 16)
