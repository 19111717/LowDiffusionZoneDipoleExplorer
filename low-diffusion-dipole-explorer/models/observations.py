"""Read the vector-digitized Li et al. (2024) Figure 3 observations."""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any


MARKERS = {
    "circle": "o",
    "square": "s",
    "triangle_up": "^",
    "triangle_down": "v",
}

DISPLAY_LABELS = {
    "MACRO": "MACRO",
    "Super-Kamiokande": "Super-K",
    "Milagro": "Milagro",
    "Tibet ASgamma": r"Tibet AS$\gamma$",
    "HAWC": "HAWC",
    "HAWC-IceCube": "HAWC-IceCube",
    "EAS-TOP": "EAS-TOP",
    "IceTop": "IceTop",
    "IceCube": "IceCube",
    "ARGO-YBJ": "ARGO-YBJ",
    "KASCADE-Grande": "K-Grande",
}

REQUIRED_COLUMNS = {
    "experiment",
    "amplitude_energy_GeV",
    "dipole_amplitude",
    "amplitude_err_minus",
    "amplitude_err_plus",
    "phase_energy_GeV",
    "phase_hour",
    "phase_err_minus_hour",
    "phase_err_plus_hour",
    "amplitude_lower_clipped",
    "phase_missing",
    "marker_shape",
    "rgb",
}


def _as_float(value: str | None) -> float:
    if value is None or not value.strip():
        return math.nan
    return float(value)


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _as_rgb(value: str) -> tuple[float, float, float]:
    channels = tuple(float(channel.strip()) for channel in value.split(","))
    if len(channels) != 3 or any(
        not 0.0 <= channel <= 1.0 for channel in channels
    ):
        raise ValueError(f"Invalid RGB value {value!r}")
    return channels


def load_digitized_data(
    csv_path: Path,
) -> OrderedDict[str, list[dict[str, Any]]]:
    """Load and validate observations, grouped by experiment."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Digitized data CSV does not exist: {csv_path}")

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            try:
                experiment = row["experiment"].strip()
                point = {
                    "amplitude_energy_GeV": _as_float(
                        row["amplitude_energy_GeV"]
                    ),
                    "dipole_amplitude": _as_float(row["dipole_amplitude"]),
                    "amplitude_err_minus": _as_float(
                        row["amplitude_err_minus"]
                    ),
                    "amplitude_err_plus": _as_float(
                        row["amplitude_err_plus"]
                    ),
                    "amplitude_lower_clipped": _as_bool(
                        row["amplitude_lower_clipped"]
                    ),
                    "phase_energy_GeV": _as_float(row["phase_energy_GeV"]),
                    "phase_hour": _as_float(row["phase_hour"]),
                    "phase_err_minus_hour": _as_float(
                        row["phase_err_minus_hour"]
                    ),
                    "phase_err_plus_hour": _as_float(
                        row["phase_err_plus_hour"]
                    ),
                    "phase_missing": _as_bool(row["phase_missing"]),
                    "marker_shape": row["marker_shape"].strip(),
                    "color": _as_rgb(row["rgb"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid value on CSV line {line_number}: {exc}"
                ) from exc

            if not experiment:
                raise ValueError(f"Missing experiment name on CSV line {line_number}")
            if point["marker_shape"] not in MARKERS:
                raise ValueError(
                    f"Unknown marker shape {point['marker_shape']!r} "
                    f"on CSV line {line_number}"
                )
            if not (
                math.isfinite(point["amplitude_energy_GeV"])
                and point["amplitude_energy_GeV"] > 0.0
                and math.isfinite(point["dipole_amplitude"])
                and point["dipole_amplitude"] > 0.0
            ):
                raise ValueError(
                    "Non-positive or missing amplitude coordinate "
                    f"on CSV line {line_number}"
                )
            grouped.setdefault(experiment, []).append(point)

    if not grouped:
        raise ValueError(f"CSV contains no data rows: {csv_path}")

    for experiment, points in grouped.items():
        marker_shapes = {point["marker_shape"] for point in points}
        colors = {point["color"] for point in points}
        if len(marker_shapes) != 1 or len(colors) != 1:
            raise ValueError(
                f"Inconsistent marker style within experiment {experiment!r}"
            )
        points.sort(key=lambda point: point["amplitude_energy_GeV"])
    return grouped

