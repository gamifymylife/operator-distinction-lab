from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .burgers import BurgersConfig, BurgersReference, LowModeBurgersSurrogate, PhysicalOperator


class AdvectionScaleBurgers(BurgersReference):
    """Burgers operator with a misspecified nonlinear transport coefficient."""

    def __init__(self, config: BurgersConfig, scale: float) -> None:
        super().__init__(config)
        self.scale = scale

    def _rhs(self, state: np.ndarray, max_mode: int | None = None) -> np.ndarray:
        state = self._project(state, max_mode)
        spectrum = np.fft.fft(state)
        derivative = np.fft.ifft(1j * self.wave_numbers * spectrum).real
        laplacian = np.fft.ifft(-(self.wave_numbers**2) * spectrum).real
        rhs = -self.scale * state * derivative + self.config.viscosity * laplacian
        return self._project(rhs, max_mode)


class ForcedBurgers(BurgersReference):
    """Burgers operator with a fixed, unmodeled spatial forcing term."""

    def __init__(self, config: BurgersConfig, amplitude: float, mode: int = 2) -> None:
        super().__init__(config)
        self.amplitude = amplitude
        x = np.linspace(0.0, 2.0 * np.pi, config.grid_size, endpoint=False)
        self.forcing = amplitude * np.sin(mode * x)

    def _rhs(self, state: np.ndarray, max_mode: int | None = None) -> np.ndarray:
        return self._project(super()._rhs(state, max_mode) + self.forcing, max_mode)


@dataclass(frozen=True)
class OperatorCase:
    case_id: str
    family: str
    parameter: str
    value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "parameter": self.parameter,
            "value": self.value,
        }


def operator_cases(families: list[dict[str, Any]]) -> list[OperatorCase]:
    cases: list[OperatorCase] = []
    for family in families:
        name = str(family["name"])
        parameter = str(family["parameter"])
        for index, raw_value in enumerate(family["values"]):
            value = float(raw_value)
            cases.append(
                OperatorCase(
                    case_id=f"{name}:{index}",
                    family=name,
                    parameter=parameter,
                    value=value,
                )
            )
    return cases


def build_candidate(case: OperatorCase, config: BurgersConfig) -> PhysicalOperator:
    if case.family == "spectral_cutoff":
        return LowModeBurgersSurrogate(config, modes_kept=int(case.value))
    if case.family == "viscosity_scale":
        candidate_config = BurgersConfig(
            grid_size=config.grid_size,
            viscosity=config.viscosity * case.value,
            dt=config.dt,
            steps=config.steps,
        )
        return BurgersReference(candidate_config)
    if case.family == "advection_scale":
        return AdvectionScaleBurgers(config, scale=case.value)
    if case.family == "forcing_bias":
        return ForcedBurgers(config, amplitude=case.value)
    raise ValueError(f"unknown operator family: {case.family}")
