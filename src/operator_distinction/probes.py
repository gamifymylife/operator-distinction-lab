from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Probe:
    """A physically admissible periodic initial condition parameterization."""

    probe_id: int
    amplitudes: tuple[float, ...]
    phases: tuple[float, ...]

    def field(self, x: np.ndarray, target_rms: float) -> np.ndarray:
        if len(self.amplitudes) != len(self.phases):
            raise ValueError("amplitudes and phases must have equal length")
        u = np.zeros_like(x, dtype=np.float64)
        for mode, (amplitude, phase) in enumerate(
            zip(self.amplitudes, self.phases, strict=True), start=1
        ):
            u += amplitude * np.sin(mode * x + phase)
        u -= np.mean(u)
        rms = float(np.sqrt(np.mean(u * u)))
        if rms == 0.0:
            return u
        return u * (target_rms / rms)

    def features(self) -> np.ndarray:
        """Cheap descriptors available to search policies before execution."""
        amplitudes = np.asarray(self.amplitudes, dtype=np.float64)
        phases = np.asarray(self.phases, dtype=np.float64)
        modes = np.arange(1, amplitudes.size + 1, dtype=np.float64)
        energy = amplitudes * amplitudes
        split = max(1, amplitudes.size // 2)
        return np.concatenate(
            [
                amplitudes,
                np.abs(amplitudes),
                np.sin(phases),
                np.cos(phases),
                np.array(
                    [
                        np.sum(energy[:split]),
                        np.sum(energy[split:]),
                        np.sum(modes * energy),
                    ]
                ),
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "amplitudes": list(self.amplitudes),
            "phases": list(self.phases),
        }


def generate_probe_pool(size: int, modes: int, seed: int) -> list[Probe]:
    """Generate one deterministic, shuffled intervention pool."""
    if size < 1 or modes < 1:
        raise ValueError("size and modes must be positive")
    rng = np.random.default_rng(seed)
    mode_numbers = np.arange(1, modes + 1, dtype=np.float64)
    probes: list[Probe] = []
    for probe_id in range(size):
        decay = rng.uniform(0.7, 1.8)
        amplitudes = rng.normal(0.0, 1.0, size=modes) / np.power(mode_numbers, decay)
        # Occasionally enrich the high-frequency tail without revealing it to the evaluator.
        if probe_id % 7 == 0:
            amplitudes[modes // 2 :] *= rng.uniform(1.5, 3.0)
        phases = rng.uniform(-np.pi, np.pi, size=modes)
        probes.append(
            Probe(
                probe_id=probe_id,
                amplitudes=tuple(float(v) for v in amplitudes),
                phases=tuple(float(v) for v in phases),
            )
        )
    order = rng.permutation(size)
    return [probes[int(index)] for index in order]


def perturb_probe(
    probe: Probe,
    count: int,
    sigma: float,
    seed: int,
    starting_id: int = 1_000_000,
) -> Iterable[Probe]:
    rng = np.random.default_rng(seed)
    amplitudes = np.asarray(probe.amplitudes)
    phases = np.asarray(probe.phases)
    for offset in range(count):
        yield Probe(
            probe_id=starting_id + offset,
            amplitudes=tuple(
                float(v) for v in amplitudes + rng.normal(0.0, sigma, amplitudes.size)
            ),
            phases=tuple(float(v) for v in phases + rng.normal(0.0, sigma, phases.size)),
        )
