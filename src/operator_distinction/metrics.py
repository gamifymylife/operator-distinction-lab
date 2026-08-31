from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscrepancyReport:
    residual: np.ndarray
    l2_by_time: np.ndarray
    linf_by_time: np.ndarray
    spectral_energy_by_time: np.ndarray
    mean_error_by_time: np.ndarray
    energy_error_by_time: np.ndarray
    selection_score: float

    def summary(self) -> dict[str, object]:
        return {
            "selection_score": self.selection_score,
            "max_l2": float(np.max(self.l2_by_time)),
            "final_l2": float(self.l2_by_time[-1]),
            "max_linf": float(np.max(self.linf_by_time)),
            "max_spectral_energy": [float(v) for v in np.max(self.spectral_energy_by_time, axis=0)],
            "max_mean_error": float(np.max(self.mean_error_by_time)),
            "max_energy_error": float(np.max(self.energy_error_by_time)),
        }


def _spectral_bands(residual: np.ndarray) -> np.ndarray:
    modes = np.abs(np.fft.fftfreq(residual.shape[1], d=1.0 / residual.shape[1]))
    spectrum = np.abs(np.fft.fft(residual, axis=1)) ** 2 / residual.shape[1] ** 2
    cut1 = max(1, residual.shape[1] // 16)
    cut2 = max(cut1 + 1, residual.shape[1] // 6)
    masks = [modes <= cut1, (modes > cut1) & (modes <= cut2), modes > cut2]
    return np.stack([np.sum(spectrum[:, mask], axis=1) for mask in masks], axis=1)


def compare_trajectories(reference: np.ndarray, candidate: np.ndarray) -> DiscrepancyReport:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate must be same-shaped [time, space] arrays")
    residual = candidate - reference
    l2_by_time = np.sqrt(np.mean(residual * residual, axis=1))
    linf_by_time = np.max(np.abs(residual), axis=1)
    spectral_energy = _spectral_bands(residual)
    mean_error = np.abs(np.mean(candidate, axis=1) - np.mean(reference, axis=1))
    reference_energy = 0.5 * np.mean(reference * reference, axis=1)
    candidate_energy = 0.5 * np.mean(candidate * candidate, axis=1)
    energy_error = np.abs(candidate_energy - reference_energy)

    scale = max(float(np.sqrt(np.mean(reference * reference))), 1e-12)
    normalized_l2 = float(np.max(l2_by_time)) / scale
    normalized_linf = float(np.max(linf_by_time)) / scale
    spectral_total = float(np.max(np.sum(spectral_energy, axis=1))) / (scale * scale)
    normalized_energy = float(np.max(energy_error)) / (scale * scale)
    # Frozen acquisition score. It selects probes but is not the scientific evidence itself.
    score = (
        0.45 * normalized_l2
        + 0.20 * normalized_linf
        + 0.25 * np.sqrt(max(spectral_total, 0.0))
        + 0.10 * normalized_energy
    )
    return DiscrepancyReport(
        residual=residual,
        l2_by_time=l2_by_time,
        linf_by_time=linf_by_time,
        spectral_energy_by_time=spectral_energy,
        mean_error_by_time=mean_error,
        energy_error_by_time=energy_error,
        selection_score=float(score),
    )
