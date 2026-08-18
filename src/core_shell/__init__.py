"""Core-shell acoustic resonance tools used in the article repository."""

from .physics import (
    eta_from_alpha_beta,
    make_materials_from_alpha_beta,
    total_internal_energy,
)
from .regimes import REGIMES, RegimeConfig, DetectorConfig

__all__ = [
    "REGIMES",
    "RegimeConfig",
    "DetectorConfig",
    "eta_from_alpha_beta",
    "make_materials_from_alpha_beta",
    "total_internal_energy",
]
