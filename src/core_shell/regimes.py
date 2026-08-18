from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DetectorConfig:
    prominence_min: float
    min_peak_distance_samples: Optional[int] = None
    max_candidate_fwhm: Optional[float] = None
    min_peak_background_ratio: Optional[float] = None
    d0_background_ratio_max: float = 0.02
    d0_energy_ratio_min: Optional[float] = None
    zoom_windows: Tuple[float, ...] = (5e-4, 2e-5)
    zoom_points: int = 250
    match_factor: float = 5.0
    min_match_tolerance: float = 1e-5
    q_min_base: Optional[float] = None
    q_min_many_peaks: Optional[float] = None
    many_peaks_threshold: Optional[int] = None
    thin_fwhm_threshold: Optional[float] = None


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    contrast_in: str
    mix_factor: float
    detector: DetectorConfig
    selected_x_max: float
    selected_x_ticks: Tuple[float, ...]
    selected_summary: str
    mean_map_fixed_max: Optional[float]
    mean_map_ticks: int
    alpha_beta_legend_side: str


REGIMES = {
    "density": RegimeConfig(
        name="density",
        contrast_in="density",
        mix_factor=0.5,
        detector=DetectorConfig(
            prominence_min=0.3,
            min_peak_distance_samples=40,
            max_candidate_fwhm=0.1,
            min_peak_background_ratio=400.0,
            d0_background_ratio_max=0.02,
        ),
        selected_x_max=1.0,
        selected_x_ticks=(0.25, 0.50, 0.75, 1.00),
        selected_summary="two_peaks",
        mean_map_fixed_max=2.0,
        mean_map_ticks=3,
        alpha_beta_legend_side="below",
    ),
    "velocity": RegimeConfig(
        name="velocity",
        contrast_in="sound_speed",
        mix_factor=0.5,
        detector=DetectorConfig(
            prominence_min=0.75,
            d0_background_ratio_max=0.02,
            d0_energy_ratio_min=100.0,
            q_min_base=400.0,
            q_min_many_peaks=100.0,
            many_peaks_threshold=100,
            thin_fwhm_threshold=0.002,
        ),
        selected_x_max=0.020,
        selected_x_ticks=(0.005, 0.010, 0.015, 0.020),
        selected_summary="statistics",
        mean_map_fixed_max=None,
        mean_map_ticks=5,
        alpha_beta_legend_side="above",
    ),
    "mixed": RegimeConfig(
        name="mixed",
        contrast_in="mixed",
        mix_factor=0.5,
        detector=DetectorConfig(
            prominence_min=0.75,
            d0_background_ratio_max=0.02,
            d0_energy_ratio_min=100.0,
            q_min_base=400.0,
            q_min_many_peaks=100.0,
            many_peaks_threshold=100,
            thin_fwhm_threshold=0.002,
        ),
        selected_x_max=0.020,
        selected_x_ticks=(0.005, 0.010, 0.015, 0.020),
        selected_summary="statistics",
        mean_map_fixed_max=None,
        mean_map_ticks=5,
        alpha_beta_legend_side="above",
    ),

}


def get_regime(name: str) -> RegimeConfig:
    try:
        return REGIMES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown regime: {name!r}. Choose from {tuple(REGIMES)}.") from exc
