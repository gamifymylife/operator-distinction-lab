"""Active validity-boundary discovery for approximate physical operators."""

from .burgers import BurgersConfig, BurgersReference, LowModeBurgersSurrogate
from .metrics import DiscrepancyReport, compare_trajectories
from .probes import Probe, generate_probe_pool

__all__ = [
    "BurgersConfig",
    "BurgersReference",
    "DiscrepancyReport",
    "LowModeBurgersSurrogate",
    "Probe",
    "compare_trajectories",
    "generate_probe_pool",
]
