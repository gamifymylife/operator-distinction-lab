from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BurgersConfig:
    grid_size: int = 64
    viscosity: float = 0.02
    dt: float = 0.0025
    steps: int = 64

    def __post_init__(self) -> None:
        if self.grid_size < 16 or self.grid_size % 2:
            raise ValueError("grid_size must be an even integer of at least 16")
        if self.viscosity <= 0.0 or self.dt <= 0.0 or self.steps < 1:
            raise ValueError("viscosity, dt, and steps must be positive")


class PhysicalOperator(Protocol):
    def rollout(self, initial_state: np.ndarray) -> np.ndarray: ...


class BurgersReference:
    """Periodic pseudo-spectral viscous Burgers solver with RK4 integration."""

    def __init__(self, config: BurgersConfig) -> None:
        self.config = config
        self.wave_numbers = np.fft.fftfreq(config.grid_size, d=1.0 / config.grid_size)
        self.dealias_mask = np.abs(self.wave_numbers) <= config.grid_size / 3

    def _project(self, state: np.ndarray, max_mode: int | None = None) -> np.ndarray:
        spectrum = np.fft.fft(state)
        mask = self.dealias_mask.copy()
        if max_mode is not None:
            mask &= np.abs(self.wave_numbers) <= max_mode
        spectrum[~mask] = 0.0
        return np.fft.ifft(spectrum).real

    def _rhs(self, state: np.ndarray, max_mode: int | None = None) -> np.ndarray:
        state = self._project(state, max_mode)
        spectrum = np.fft.fft(state)
        derivative = np.fft.ifft(1j * self.wave_numbers * spectrum).real
        laplacian = np.fft.ifft(-(self.wave_numbers**2) * spectrum).real
        rhs = -(state * derivative) + self.config.viscosity * laplacian
        return self._project(rhs, max_mode)

    def _step(self, state: np.ndarray, max_mode: int | None = None) -> np.ndarray:
        dt = self.config.dt
        k1 = self._rhs(state, max_mode)
        k2 = self._rhs(state + 0.5 * dt * k1, max_mode)
        k3 = self._rhs(state + 0.5 * dt * k2, max_mode)
        k4 = self._rhs(state + dt * k3, max_mode)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self._project(next_state, max_mode)

    def _rollout(self, initial_state: np.ndarray, max_mode: int | None) -> np.ndarray:
        if initial_state.shape != (self.config.grid_size,):
            raise ValueError(f"initial_state must have shape ({self.config.grid_size},)")
        trajectory = np.empty((self.config.steps + 1, self.config.grid_size), dtype=np.float64)
        state = self._project(np.asarray(initial_state, dtype=np.float64), max_mode)
        trajectory[0] = state
        for step in range(1, self.config.steps + 1):
            state = self._step(state, max_mode)
            if not np.all(np.isfinite(state)):
                raise FloatingPointError(f"non-finite state at step {step}")
            trajectory[step] = state
        return trajectory

    def rollout(self, initial_state: np.ndarray) -> np.ndarray:
        return self._rollout(initial_state, max_mode=None)


class LowModeBurgersSurrogate(BurgersReference):
    """Controlled approximate operator that discards modes above ``modes_kept``."""

    def __init__(self, config: BurgersConfig, modes_kept: int) -> None:
        super().__init__(config)
        if modes_kept < 1 or modes_kept >= config.grid_size // 3:
            raise ValueError("modes_kept must be inside the reference spectral range")
        self.modes_kept = modes_kept

    def rollout(self, initial_state: np.ndarray) -> np.ndarray:
        return self._rollout(initial_state, max_mode=self.modes_kept)
