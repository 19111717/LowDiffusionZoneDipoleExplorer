"""Diffusion-energy conversions used by the standalone explorer."""

from __future__ import annotations

import numpy as np

from cygbubble import config


def Anisotropic_ratio_to_energy(anisotropic_ratio: float) -> float:
    """Infer energy in TeV from ``D_parallel/D_perpendicular``."""
    return float(
        (anisotropic_ratio / config.anisotropic_ratio_at_10TeV)
        ** (-1.0 / config.Delta_delta)
        * 10.0
    )


def energy_to_low_parallel_ratio(energy_tev):
    """Return ``D_low/D_parallel`` at an energy in TeV."""
    energy = np.asarray(energy_tev, dtype=np.float64)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("energy_tev must contain finite positive values")
    reference_ratio = float(config.A_at_10TeV)
    exponent_difference = float(config.Delta_low_parallel)
    if not np.isfinite(reference_ratio) or reference_ratio <= 0.0:
        raise ValueError("config.A_at_10TeV must be finite and positive")
    if not np.isfinite(exponent_difference):
        raise ValueError("config.Delta_low_parallel must be finite")
    ratio = reference_ratio * (energy / 10.0) ** exponent_difference
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError("Converted D_low/D_parallel is not finite and positive")
    return float(ratio) if ratio.ndim == 0 else ratio


def low_parallel_ratio_to_energy(d_low_over_d_parallel):
    """Infer energy in TeV from ``D_low/D_parallel``."""
    ratio = np.asarray(d_low_over_d_parallel, dtype=np.float64)
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError("d_low_over_d_parallel must contain finite positive values")
    reference_ratio = float(config.A_at_10TeV)
    exponent_difference = float(config.Delta_low_parallel)
    if not np.isfinite(reference_ratio) or reference_ratio <= 0.0:
        raise ValueError("config.A_at_10TeV must be finite and positive")
    if not np.isfinite(exponent_difference) or exponent_difference == 0.0:
        raise ValueError(
            "config.Delta_low_parallel must be finite and nonzero to infer energy"
        )
    energy = 10.0 * (ratio / reference_ratio) ** (
        1.0 / exponent_difference
    )
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("Converted energy is not finite and positive")
    return float(energy) if energy.ndim == 0 else energy


def diffusion_coefficient_parallel(
    energy_tev,
    delta: float = config.delta_Diffuse,
    D0: float = config.D_0_10TeV,
):
    """Return parallel diffusion coefficient in cm^2/s."""
    return D0 * (np.asarray(energy_tev) / 10.0) ** delta

