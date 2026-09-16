"""Coordinate conversions required by the standalone explorer."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cygbubble import config


def galactic_lonlat_from_cartesian_direction(
    direction: Sequence[float],
) -> tuple[float, float]:
    """Convert a project Cartesian direction to Galactic ``(l, b)`` degrees.

    In this basis ``-x``, ``+y``, and ``+z`` point toward Galactic ``l=0``,
    ``l=90 deg``, and the north Galactic pole, respectively.
    """
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("direction must contain three finite Cartesian components")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return float("nan"), float("nan")

    toward_galactic_center = -float(vector[0])
    toward_l_90 = float(vector[1])
    toward_north = float(vector[2])
    horizontal = float(np.hypot(toward_galactic_center, toward_l_90))
    if horizontal <= 32.0 * np.finfo(np.float64).eps * norm:
        longitude = 0.0
    else:
        longitude = (
            np.degrees(np.arctan2(toward_l_90, toward_galactic_center))
            % 360.0
        )
    latitude = np.degrees(np.arctan2(toward_north, horizontal))
    return float(longitude), float(latitude)


def cartesian_direction_from_galactic_lonlat(
    longitude_deg: float,
    latitude_deg: float,
) -> np.ndarray:
    """Convert Galactic longitude and latitude to a Cartesian unit vector."""
    longitude = float(longitude_deg)
    latitude = float(latitude_deg)
    if not np.isfinite(longitude) or not np.isfinite(latitude):
        raise ValueError("Galactic longitude and latitude must be finite")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("Galactic latitude must lie in [-90, 90] degrees")
    longitude_rad = np.deg2rad(longitude)
    latitude_rad = np.deg2rad(latitude)
    horizontal = np.cos(latitude_rad)
    return np.asarray(
        [
            -horizontal * np.cos(longitude_rad),
            horizontal * np.sin(longitude_rad),
            np.sin(latitude_rad),
        ],
        dtype=np.float64,
    )


def simulation_to_equatorial_components(vector) -> np.ndarray:
    """Project vectors onto the J2000 RA=0 h, RA=6 h, and NCP axes."""
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("The final vector dimension must contain three values")
    if not np.all(np.isfinite(values)):
        raise ValueError("vector must contain finite values only")
    matrix = np.asarray(
        config.SIMULATION_TO_EQUATORIAL_J2000,
        dtype=np.float64,
    )
    return np.einsum("ij,...j->...i", matrix, values)

