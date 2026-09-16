"""Interactive explorer for the dipole response of a low-diffusion sphere.

Run this file directly in PyCharm.  The default ``history`` curve mode joins
successive valid slider updates in their update order.  ``parameter scan``
instead evaluates continuous curves along one selected input parameter while
holding all other inputs fixed.

The S-position controls describe the fixed centre-to-S displacement in the
Galactic basis by ``r/R, l, b``.  Its analytic Cartesian coordinates are
recomputed whenever the magnetic-field or background-gradient frame changes.

The displayed ``E(a)`` is inferred from ``a=D_low/D_parallel`` and sets the
absolute diffusion scale used by the dipole amplitude.  ``M_A`` remains an
independent parameter and continues to set
``D_perpendicular/D_parallel=M_A**4`` throughout the explorer.

Digitized Li et al. (2024) observations can be overlaid with the ``Li+2024``
checkbox.  They are shown only in parameter-scan mode when ``a`` is the scan
parameter, because that is the only curve mode with a defined energy x-axis.

The calculation uses the existing analytic convention: the magnetic field is
the analytic +z axis and the perpendicular background-gradient component is
the analytic +x axis.  Density-denominator perturbations are omitted, exactly
as in
:func:`models.analysis.low_diffusion_dipole.low_diffusion_dipole_at_point`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

from models import config
from models.analysis.low_diffusion_dipole import (
    low_diffusion_dipole_at_point,
    relative_position_from_galactic_lonlat,
)
from models.coordinates import (
    cartesian_direction_from_galactic_lonlat,
    galactic_lonlat_from_cartesian_direction,
)
from models.analytic.sphere import (
    arbitrary_spatial_amplification,
    perpendicular_spatial_amplification,
)
from models.physics.diffusion import (
    Anisotropic_ratio_to_energy,
    diffusion_coefficient_parallel,
    low_parallel_ratio_to_energy,
)
from models.observations import (
    DISPLAY_LABELS as LI2024_DISPLAY_LABELS,
    MARKERS as LI2024_MARKERS,
    load_digitized_data as load_li2024_digitized_data,
)


# These values are deliberately modest so slider updates remain responsive.
ANALYTIC_NZ = 61
ANALYTIC_NV = 21
ANALYTIC_ZMAX = 8.5
ANALYTIC_VMAX = 3.0
ANALYTIC_COLOR_MIN = 0.03
ANALYTIC_COLOR_MAX = 3.0
SCAN_SAMPLE_COUNT = 81

# Digitized experimental points from Li et al. (2024), Figure 3.  They can be
# compared with the model only when the curve x-axis has a valid E(a) mapping.
LI2024_DATA_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "Li_2024_Figure3_vector_digitized.csv"
)
SHOW_LI2024_DATA_DEFAULT = False

# Fixed y-axis limits shared by the total- and RA-projected dipole amplitudes.
DIPOLE_AMPLITUDE_Y_MIN = 1.0e-5
DIPOLE_AMPLITUDE_Y_MAX = 1.0e-2

# Window and plot-grid spacing.  Keep these near the other display controls so
# the PyCharm window can be resized without changing the plotting methods.
EXPLORER_FIGURE_SIZE = (20.5, 12.0)
PLOT_GRID_LEFT = 0.295
PLOT_GRID_RIGHT = 0.975
PLOT_GRID_BOTTOM = 0.080
PLOT_GRID_TOP = 0.935
PLOT_GRID_WSPACE = 0.90
PLOT_GRID_HSPACE = 0.62


def _signed_longitude(longitude_deg: float) -> float:
    """Wrap Galactic longitude to the interval [-180 deg, 180 deg]."""
    longitude = float(longitude_deg)
    wrapped = (longitude + 180.0) % 360.0 - 180.0
    if np.isclose(wrapped, -180.0) and longitude > 0.0:
        return 180.0
    return wrapped


def _low_ratio_to_energy_axis(d_low_over_d_parallel):
    """Matplotlib-safe ``D_low/D_parallel -> E(a)`` axis transform.

    Secondary axes probe zero while configuring their scale, so this display
    transform intentionally avoids the strict positive-value validation used
    by the public physical conversion helper.
    """
    ratio = np.asarray(d_low_over_d_parallel, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return 10.0 * (ratio / float(config.A_at_10TeV)) ** (
            1.0 / float(config.Delta_low_parallel)
        )


def _energy_to_low_ratio_axis(energy_tev):
    """Matplotlib-safe inverse transform for the secondary energy axis."""
    energy = np.asarray(energy_tev, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return float(config.A_at_10TeV) * (energy / 10.0) ** float(
            config.Delta_low_parallel
        )


def _energy_conversion_is_valid() -> bool:
    """Return whether ``a=D_low/D_parallel`` uniquely determines energy."""
    normalization = float(config.A_at_10TeV)
    exponent = float(config.Delta_low_parallel)
    return bool(
        np.isfinite(normalization)
        and normalization > 0.0
        and np.isfinite(exponent)
        and not np.isclose(exponent, 0.0)
    )


def _finite_errors(values: Iterable[float]) -> np.ndarray:
    """Return non-negative error lengths, replacing absent values by zero."""
    errors = np.asarray(tuple(values), dtype=np.float64)
    errors[~np.isfinite(errors)] = 0.0
    return np.maximum(errors, 0.0)


@dataclass(frozen=True)
class ExplorerState:
    """One complete set of physical inputs for the explorer."""

    gradient_longitude_deg: float = 0.0
    gradient_latitude_deg: float = 0.0
    gradient_magnitude_per_pc: float = 1.0e-4
    magnetic_longitude_deg: float = _signed_longitude(0)
    magnetic_latitude_deg: float = float(0)
    ma: float = float(config.anisotropic_ratio_at_10TeV) ** -0.25
    d_low_over_d_parallel: float = 0.03
    # Fixed Galactic centre-to-S displacement.  These defaults preserve the
    # physical point represented by the former analytic (-1, 0, -2) default.
    position_distance_over_radius: float = float(np.sqrt(3.0))
    position_longitude_deg: float = 90
    position_latitude_deg: float = 0

    @property
    def position_simulation_over_radius(self) -> np.ndarray:
        return relative_position_from_galactic_lonlat(
            self.position_distance_over_radius,
            self.position_longitude_deg,
            self.position_latitude_deg,
        )

    @property
    def relative_gradient_per_pc(self) -> np.ndarray:
        direction = cartesian_direction_from_galactic_lonlat(
            self.gradient_longitude_deg,
            self.gradient_latitude_deg,
        )
        return self.gradient_magnitude_per_pc * direction


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    axis_label: str
    minimum: float
    maximum: float
    logarithmic: bool = False

    @property
    def slider_minimum(self) -> float:
        return float(np.log10(self.minimum)) if self.logarithmic else self.minimum

    @property
    def slider_maximum(self) -> float:
        return float(np.log10(self.maximum)) if self.logarithmic else self.maximum

    def to_slider(self, physical_value: float) -> float:
        value = float(physical_value)
        self.validate(value)
        return float(np.log10(value)) if self.logarithmic else value

    def from_slider(self, slider_value: float) -> float:
        value = 10.0 ** float(slider_value) if self.logarithmic else float(slider_value)
        self.validate(value)
        return value

    def validate(self, value: float) -> None:
        if not np.isfinite(value) or not self.minimum <= value <= self.maximum:
            raise ValueError(
                f"{self.label} 必须位于 [{self.minimum:g}, {self.maximum:g}]"
            )

    def scan_values(
        self,
        sample_count: int = SCAN_SAMPLE_COUNT,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> np.ndarray:
        minimum = self.minimum if minimum is None else float(minimum)
        maximum = self.maximum if maximum is None else float(maximum)
        self.validate(minimum)
        self.validate(maximum)
        if minimum >= maximum:
            raise ValueError("扫描区间必须满足 min < max")
        slider_minimum = float(np.log10(minimum)) if self.logarithmic else minimum
        slider_maximum = float(np.log10(maximum)) if self.logarithmic else maximum
        slider_values = np.linspace(
            slider_minimum,
            slider_maximum,
            int(sample_count),
        )
        if self.logarithmic:
            return 10.0 ** slider_values
        return slider_values


PARAMETER_SPECS = (
    ParameterSpec(
        "gradient_longitude_deg", "grad l [deg]", "Gradient longitude l [deg]",
        -180.0, 180.0,
    ),
    ParameterSpec(
        "gradient_latitude_deg", "grad b [deg]", "Gradient latitude b [deg]",
        -90.0, 90.0,
    ),
    ParameterSpec(
        "gradient_magnitude_per_pc", "|grad n/n| [pc^-1]",
        "Background relative-gradient magnitude [pc^-1]", 1.0e-7, 1.0e-2,
        logarithmic=True,
    ),
    ParameterSpec(
        "magnetic_longitude_deg", "B l [deg]", "Magnetic longitude l [deg]",
        -180.0, 180.0,
    ),
    ParameterSpec(
        "magnetic_latitude_deg", "B b [deg]", "Magnetic latitude b [deg]",
        -90.0, 90.0,
    ),
    ParameterSpec("ma", "MA", "Alfven Mach number MA", 0.05, 1.0),
    ParameterSpec(
        "d_low_over_d_parallel", "Dlow / Dparallel",
        "Dlow / Dparallel", 1.0e-4, 1.0, logarithmic=True,
    ),
    ParameterSpec(
        "position_distance_over_radius", "S r/R", "S distance r/R",
        0.0, 12.0,
    ),
    ParameterSpec(
        "position_longitude_deg", "S l [deg]", "S longitude l [deg]",
        -180.0, 180.0,
    ),
    ParameterSpec(
        "position_latitude_deg", "S b [deg]", "S latitude b [deg]",
        -90.0, 90.0,
    ),
)
PARAMETER_BY_KEY = {spec.key: spec for spec in PARAMETER_SPECS}
SCAN_LABEL_TO_KEY = {spec.label: spec.key for spec in PARAMETER_SPECS}


def calculate_state(state: ExplorerState) -> dict[str, object]:
    """Calculate one independent-parameter explorer state.

    ``M_A`` and ``D_low/D_parallel`` remain independent analytic inputs.  The
    latter additionally supplies the energy used for the absolute diffusion
    scale and dipole amplitude; it never changes ``M_A``.
    """
    energy_from_low_ratio_tev = float(low_parallel_ratio_to_energy(
        state.d_low_over_d_parallel
    ))
    result = low_diffusion_dipole_at_point(
        magnetic_longitude_deg=state.magnetic_longitude_deg,
        magnetic_latitude_deg=state.magnetic_latitude_deg,
        relative_gradient_per_pc=state.relative_gradient_per_pc,
        d_perp_over_d_parallel=state.ma ** 4,
        d_low_over_d_parallel=state.d_low_over_d_parallel,
        diffusion_energy_tev=energy_from_low_ratio_tev,
        position_distance_over_radius=state.position_distance_over_radius,
        position_longitude_deg=state.position_longitude_deg,
        position_latitude_deg=state.position_latitude_deg,
    )
    d_parallel = float(diffusion_coefficient_parallel(
        energy_from_low_ratio_tev,
        delta=config.delta_Diffuse,
        D0=config.D_0_10TeV,
    ))
    d_perpendicular = state.ma ** 4 * d_parallel
    d_low = state.d_low_over_d_parallel * d_parallel
    coefficients = np.asarray(
        [d_parallel, d_perpendicular, d_low], dtype=np.float64
    )
    if np.any(~np.isfinite(coefficients)) or np.any(coefficients <= 0.0):
        raise ValueError("独立参数模式换算得到的扩散系数必须为有限正数")

    # Keep the separate energy implied by M_A as diagnostic metadata.  It does
    # not normalize the diffusion coefficients or the dipole amplitude here.
    result["energy_from_ma_tev"] = float(Anisotropic_ratio_to_energy(
        state.ma ** -4
    ))
    result.update({
        "energy_from_d_low_over_d_parallel_tev": energy_from_low_ratio_tev,
        "independent_d_parallel_cm2_s": d_parallel,
        "independent_d_perpendicular_cm2_s": d_perpendicular,
        "independent_d_low_cm2_s": d_low,
    })
    return result


def slice_plane_through_point(
    position_analytic: Iterable[float],
    plot_u: np.ndarray,
    plot_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return analytic (x,y,z) grids for the B-axis plane through ``S``.

    ``plot_u`` is the analytic z coordinate along B.  The positive ``plot_v``
    direction is chosen from the magnetic axis toward the xy projection of S,
    so S appears at ``(u, v) = (z_S, hypot(x_S, y_S))``.  ``psi_deg`` follows
    the sign convention of ``perpendicular_slice_coordinates``.
    """
    point = np.asarray(tuple(position_analytic), dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("position_analytic 必须包含三个有限分量")
    plot_u, plot_v = np.broadcast_arrays(
        np.asarray(plot_u, dtype=np.float64),
        np.asarray(plot_v, dtype=np.float64),
    )
    rho = float(np.hypot(point[0], point[1]))
    if rho == 0.0:
        transverse_x, transverse_y = 1.0, 0.0
    else:
        transverse_x = float(point[0] / rho)
        transverse_y = float(point[1] / rho)
    x = plot_v * transverse_x
    y = plot_v * transverse_y
    z = plot_u
    psi_deg = float(
        np.degrees(np.arctan2(-transverse_y, transverse_x)) % 180.0
    )
    return x, y, z, rho, psi_deg


def _mollweide_coordinates(longitude_deg, latitude_deg):
    wrapped_longitude = (
        np.asarray(longitude_deg, dtype=np.float64) + 180.0
    ) % 360.0 - 180.0
    return -np.deg2rad(wrapped_longitude), np.deg2rad(latitude_deg)


def _path_with_seam_breaks(longitude_deg, latitude_deg):
    x, y = _mollweide_coordinates(longitude_deg, latitude_deg)
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    if x.size <= 1:
        return x, y
    output_x = [x[0]]
    output_y = [y[0]]
    for index in range(1, x.size):
        if (
            not np.isfinite(x[index - 1:index + 1]).all()
            or not np.isfinite(y[index - 1:index + 1]).all()
            or abs(x[index] - x[index - 1]) > np.pi
        ):
            output_x.append(np.nan)
            output_y.append(np.nan)
        output_x.append(x[index])
        output_y.append(y[index])
    return np.asarray(output_x), np.asarray(output_y)


def _directions_to_galactic(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray([
        galactic_lonlat_from_cartesian_direction(vector)
        for vector in np.asarray(vectors, dtype=np.float64)
    ])
    return coordinates[:, 0], coordinates[:, 1]


def _perpendicular_great_circle(
    axis_simulation: Iterable[float], sample_count: int = 721
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(tuple(axis_simulation), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    reference = (
        np.asarray([0.0, 0.0, 1.0])
        if abs(float(axis[2])) < 0.9
        else np.asarray([1.0, 0.0, 0.0])
    )
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count)
    circle = (
        np.cos(angles)[:, None] * first
        + np.sin(angles)[:, None] * second
    )
    return _directions_to_galactic(circle)


def _right_ascension_great_circle(
    right_ascension_hour: float = 4.0,
    sample_count: int = 721,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full RA=4 h / RA=-8 h meridian in Galactic coordinates."""
    alpha = np.deg2rad(15.0 * float(right_ascension_hour))
    equatorial_axis = np.asarray([np.cos(alpha), np.sin(alpha), 0.0])
    north_celestial_pole = np.asarray([0.0, 0.0, 1.0])
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count)
    equatorial_circle = (
        np.cos(angles)[:, None] * equatorial_axis
        + np.sin(angles)[:, None] * north_celestial_pole
    )
    rotation = np.asarray(
        config.SIMULATION_TO_EQUATORIAL_J2000, dtype=np.float64
    )
    simulation_circle = np.linalg.solve(
        rotation, equatorial_circle.T
    ).T
    simulation_circle /= np.linalg.norm(
        simulation_circle, axis=1
    )[:, None]
    return _directions_to_galactic(simulation_circle)


def _celestial_poles_galactic() -> tuple[tuple[float, float], tuple[float, float]]:
    rotation = np.asarray(
        config.SIMULATION_TO_EQUATORIAL_J2000, dtype=np.float64
    )
    north = np.linalg.solve(rotation, np.asarray([0.0, 0.0, 1.0]))
    north /= np.linalg.norm(north)
    north_angles = galactic_lonlat_from_cartesian_direction(north)
    south_angles = galactic_lonlat_from_cartesian_direction(-north)
    return north_angles, south_angles


def _break_wrapped_curve(x, phase_hour):
    x = np.asarray(x, dtype=np.float64)
    phase = np.asarray(phase_hour, dtype=np.float64)
    if x.size <= 1:
        return x, phase
    output_x = [x[0]]
    output_phase = [phase[0]]
    for index in range(1, x.size):
        if (
            np.isfinite(phase[index - 1])
            and np.isfinite(phase[index])
            and abs(phase[index] - phase[index - 1]) > 12.0
        ):
            output_x.append(np.nan)
            output_phase.append(np.nan)
        output_x.append(x[index])
        output_phase.append(phase[index])
    return np.asarray(output_x), np.asarray(output_phase)


class LowDiffusionDipoleExplorer:
    """Matplotlib-based interactive comparison window."""

    def __init__(self, initial_state: ExplorerState | None = None):
        self.state = initial_state or ExplorerState()
        self._synchronizing_widgets = False
        self.curve_mode = "history"
        self.scan_parameter = "ma"
        self.scan_ranges = {
            spec.key: (spec.minimum, spec.maximum)
            for spec in PARAMETER_SPECS
        }
        self.last_result: dict[str, object] | None = None
        self.show_li2024_data = SHOW_LI2024_DATA_DEFAULT
        self.li2024_data_artists = []
        self.li2024_amplitude_handles = []
        self.li2024_phase_handles = []
        self.li2024_data_legend = None
        self.li2024_data_load_error: str | None = None

        self.history = {
            "step": [],
            "amplitude": [],
            "ra_amplitude": [],
            "ra_phase": [],
        }
        self.gradient_trajectory = {"longitude": [], "latitude": []}
        self.dipole_trajectory = {"longitude": [], "latitude": []}
        self._next_history_step = 0

        self.figure = plt.figure(figsize=EXPLORER_FIGURE_SIZE)
        plot_grid = self.figure.add_gridspec(
            2,
            6,
            left=PLOT_GRID_LEFT,
            right=PLOT_GRID_RIGHT,
            bottom=PLOT_GRID_BOTTOM,
            top=PLOT_GRID_TOP,
            wspace=PLOT_GRID_WSPACE,
            hspace=PLOT_GRID_HSPACE,
            height_ratios=(1.12, 1.0),
        )
        self.sky_axis = self.figure.add_subplot(
            plot_grid[0, :3], projection="mollweide"
        )
        self.slice_axis = self.figure.add_subplot(plot_grid[0, 3:])
        self.amplitude_axis = self.figure.add_subplot(plot_grid[1, 0:3])
        self.phase_axis = self.figure.add_subplot(plot_grid[1, 3:6])
        self.curve_axes = (self.amplitude_axis, self.phase_axis)
        self.energy_secondary_axes = []

        initial_result = calculate_state(self.state)
        self._create_sky_plot()
        self._create_slice_plot(initial_result)
        self._create_curve_plots()
        self._create_parameter_controls()
        self._refresh(record=True)

    def _create_sky_plot(self) -> None:
        axis = self.sky_axis
        self.gradient_track_line, = axis.plot(
            [np.nan], [np.nan], color="tab:green", linewidth=1.1, alpha=0.7,
            label="Gradient trajectory",
        )
        self.dipole_track_line, = axis.plot(
            [np.nan], [np.nan], color="tab:red", linewidth=1.1, alpha=0.7,
            label="Affected-dipole trajectory",
        )
        self.magnetic_circle_line, = axis.plot(
            [np.nan], [np.nan], color="tab:blue", linestyle="--", linewidth=1.1,
            alpha=0.8, label="Great circle perpendicular to B",
        )

        ra_longitude, ra_latitude = _right_ascension_great_circle(4.0)
        ra_x, ra_y = _path_with_seam_breaks(ra_longitude, ra_latitude)
        axis.plot(
            ra_x, ra_y, color="tab:purple", linestyle=":", linewidth=1.4,
            label="RA=4 h / -8 h great circle",
        )

        self.gradient_marker = axis.scatter(
            [], [], marker="^", s=80, color="tab:green", edgecolor="black",
            linewidth=0.6, zorder=5, label="Background gradient",
        )
        self.dipole_marker = axis.scatter(
            [], [], marker="o", s=75, color="tab:red", edgecolor="black",
            linewidth=0.6, zorder=5, label="Affected dipole",
        )
        self.magnetic_markers = axis.scatter(
            [], [], marker="*", s=150, color="tab:blue", edgecolor="black",
            linewidth=0.7, zorder=5, label="+/- B",
        )

        north, south = _celestial_poles_galactic()
        pole_x, pole_y = _mollweide_coordinates(
            [north[0], south[0]], [north[1], south[1]]
        )
        axis.scatter(
            pole_x, pole_y, marker="X", s=75, color="black", zorder=6,
            label="NCP / SCP",
        )
        axis.annotate("NCP", (pole_x[0], pole_y[0]), xytext=(4, 5),
                      textcoords="offset points", fontsize=8)
        axis.annotate("SCP", (pole_x[1], pole_y[1]), xytext=(4, 5),
                      textcoords="offset points", fontsize=8)

        ticks_deg = np.arange(-150.0, 180.0, 30.0)
        axis.set_xticks(np.deg2rad(ticks_deg))
        axis.set_xticklabels([f"{int(-tick)}°" for tick in ticks_deg])
        axis.grid(True, linewidth=0.6, alpha=0.45)
        axis.set_xlabel(
            "Galactic longitude l (increases to the left)", labelpad=9.0
        )
        axis.set_ylabel("Galactic latitude b", labelpad=8.0)
        axis.set_title(
            "Galactic sky (Galactic centre at the map centre)", pad=13.0
        )
        axis.legend(loc="lower left", fontsize=7, frameon=False, ncol=2)

    def _slice_field(self, result: dict[str, object]):
        point_analytic = np.asarray(
            result["position_analytic"], dtype=np.float64
        )
        x, y, z, point_v, psi_deg = slice_plane_through_point(
            point_analytic, self.slice_u, self.slice_v
        )
        if result["analytic_solution_case"] == "perpendicular":
            _, dipole_amplification = perpendicular_spatial_amplification(
                x,
                y,
                z,
                a=self.state.d_low_over_d_parallel,
                MA=self.state.ma,
            )
        else:
            _, dipole_amplification = arbitrary_spatial_amplification(
                x,
                y,
                z,
                a=self.state.d_low_over_d_parallel,
                MA=self.state.ma,
                perp_to_parallel_ratio=float(
                    result["gradient_perpendicular_over_parallel"]
                ),
            )
        return (
            np.asarray(dipole_amplification),
            point_v,
            psi_deg,
            point_analytic,
        )

    def _create_slice_plot(self, initial_result: dict[str, object]) -> None:
        u = np.linspace(-ANALYTIC_ZMAX, ANALYTIC_ZMAX, ANALYTIC_NZ)
        v = np.linspace(-ANALYTIC_VMAX, ANALYTIC_VMAX, ANALYTIC_NV)
        self.slice_u, self.slice_v = np.meshgrid(u, v)
        field, point_v, psi_deg, point_analytic = self._slice_field(
            initial_result
        )
        self.slice_mesh = self.slice_axis.pcolormesh(
            self.slice_u,
            self.slice_v,
            field,
            shading="auto",
            cmap="magma",
            norm=LogNorm(vmin=ANALYTIC_COLOR_MIN, vmax=ANALYTIC_COLOR_MAX),
        )
        colorbar = self.figure.colorbar(
            self.slice_mesh, ax=self.slice_axis, pad=0.02
        )
        colorbar.set_label(
            r"$A_\delta^*=|\mathbf{D}\nabla n|/|\mathbf{D}\nabla n_0|$",
            labelpad=8.0,
        )
        self.slice_axis.add_patch(plt.Circle(
            (0.0, 0.0), 1.0, fill=False, color="white", linewidth=0.8
        ))
        self.slice_point, = self.slice_axis.plot(
            [point_analytic[2]], [point_v], marker="o", markersize=7,
            color="cyan", markeredgecolor="black", linestyle="None",
            label="S",
        )
        self.slice_axis.set_xlim(-ANALYTIC_ZMAX, ANALYTIC_ZMAX)
        self.slice_axis.set_ylim(-ANALYTIC_VMAX, ANALYTIC_VMAX)
        self.slice_axis.set_aspect("equal", adjustable="box")
        self.slice_axis.set_xlabel(
            r"$z/R$  ($\parallel\mathbf{B}$)", labelpad=8.0
        )
        self.slice_axis.set_ylabel(
            r"$v/R$  (slice direction toward S)", labelpad=9.0
        )
        self.slice_axis.set_title(
            "Analytic dipole amplification; "
            f"slice psi={psi_deg:.1f} deg\n"
            f"S_ana=({point_analytic[0]:.2f}, {point_analytic[1]:.2f}, "
            f"{point_analytic[2]:.2f}) R",
            pad=12.0,
        )
        self.slice_axis.legend(loc="upper right", frameon=False)

    def _create_curve_plots(self) -> None:
        total_amplitude_line, = self.amplitude_axis.plot(
            [], [],
            color="tab:blue",
            linestyle="-",
            marker="o",
            markersize=2.8,
            linewidth=1.35,
            label="Total dipole amplitude",
        )
        ra_amplitude_line, = self.amplitude_axis.plot(
            [], [],
            color="tab:orange",
            linestyle="--",
            marker="s",
            markersize=2.5,
            linewidth=1.25,
            label="RA-projected amplitude",
        )
        phase_line, = self.phase_axis.plot(
            [], [],
            color="tab:green",
            linestyle="-",
            marker="o",
            markersize=2.8,
            linewidth=1.2,
            label="RA phase",
        )
        self.curve_lines = (
            total_amplitude_line,
            ra_amplitude_line,
            phase_line,
        )
        self.current_parameter_lines = []
        for axis in self.curve_axes:
            current_line = axis.axvline(
                0.0,
                color="0.45",
                linestyle="--",
                linewidth=0.9,
                visible=False,
            )
            self.current_parameter_lines.append(current_line)
            axis.grid(True, which="both", alpha=0.25)
            axis.set_xlim(0.0, 1.0)

        self.amplitude_axis.set_ylabel("Dipole amplitude", labelpad=9.0)
        self.amplitude_axis.set_yscale("log")
        self.amplitude_axis.set_ylim(
            DIPOLE_AMPLITUDE_Y_MIN,
            DIPOLE_AMPLITUDE_Y_MAX,
        )
        self.amplitude_curve_legend = self.amplitude_axis.legend(
            handles=(total_amplitude_line, ra_amplitude_line),
            loc="upper right",
            frameon=False,
            fontsize=8,
        )
        self.phase_axis.set_ylabel("RA phase [h]", labelpad=9.0)
        self.phase_axis.set_ylim(-12.0, 12.0)

    @staticmethod
    def _errorbar_artists(container) -> list:
        """Flatten a Matplotlib ErrorbarContainer into visible artists."""
        data_line, cap_lines, bar_collections = container.lines
        return [data_line, *cap_lines, *bar_collections]

    def _energy_scan_is_active(self) -> bool:
        """Return whether the current curve x-axis has the E(a) mapping."""
        return bool(
            self.curve_mode == "parameter scan"
            and self.scan_parameter == "d_low_over_d_parallel"
            and _energy_conversion_is_valid()
        )

    def _create_li2024_data_artists(self) -> bool:
        """Load the digitized observations and create initially hidden artists."""
        if self.li2024_data_artists:
            return True

        try:
            grouped_data = load_li2024_digitized_data(LI2024_DATA_CSV_PATH)
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            self.li2024_data_load_error = str(error)
            return False

        self.li2024_data_load_error = None
        for experiment, points in grouped_data.items():
            color = points[0]["color"]
            marker = LI2024_MARKERS[points[0]["marker_shape"]]
            label = LI2024_DISPLAY_LABELS.get(experiment, experiment)

            amplitude_energy_tev = np.asarray(
                [point["amplitude_energy_GeV"] / 1.0e3 for point in points],
                dtype=np.float64,
            )
            amplitude_x = np.asarray(
                _energy_to_low_ratio_axis(amplitude_energy_tev),
                dtype=np.float64,
            )
            amplitude = np.asarray(
                [point["dipole_amplitude"] for point in points],
                dtype=np.float64,
            )
            amplitude_error_minus = _finite_errors(
                point["amplitude_err_minus"] for point in points
            )
            amplitude_error_plus = _finite_errors(
                point["amplitude_err_plus"] for point in points
            )
            for index, point in enumerate(points):
                if point["amplitude_lower_clipped"]:
                    amplitude_error_minus[index] = max(
                        amplitude[index] - DIPOLE_AMPLITUDE_Y_MIN,
                        0.0,
                    )
            valid_amplitude = (
                np.isfinite(amplitude_x)
                & (amplitude_x > 0.0)
                & np.isfinite(amplitude)
                & (amplitude > 0.0)
            )
            amplitude_container = self.amplitude_axis.errorbar(
                amplitude_x[valid_amplitude],
                amplitude[valid_amplitude],
                yerr=np.vstack((
                    amplitude_error_minus[valid_amplitude],
                    amplitude_error_plus[valid_amplitude],
                )),
                fmt=marker,
                linestyle="none",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=4.0,
                elinewidth=0.9,
                capsize=1.8,
                capthick=0.8,
                alpha=0.82,
                label=label,
                zorder=5,
            )
            amplitude_artists = self._errorbar_artists(amplitude_container)
            self.li2024_data_artists.extend(amplitude_artists)
            self.li2024_amplitude_handles.append(amplitude_container.lines[0])

            phase_points = [
                point
                for point in points
                if not point["phase_missing"]
                and np.isfinite(point["phase_energy_GeV"])
                and np.isfinite(point["phase_hour"])
            ]
            if not phase_points:
                continue
            phase_energy_tev = np.asarray(
                [point["phase_energy_GeV"] / 1.0e3 for point in phase_points],
                dtype=np.float64,
            )
            phase_x = np.asarray(
                _energy_to_low_ratio_axis(phase_energy_tev),
                dtype=np.float64,
            )
            phase = np.asarray(
                [point["phase_hour"] for point in phase_points],
                dtype=np.float64,
            )
            phase_error_minus = _finite_errors(
                point["phase_err_minus_hour"] for point in phase_points
            )
            phase_error_plus = _finite_errors(
                point["phase_err_plus_hour"] for point in phase_points
            )
            valid_phase = (
                np.isfinite(phase_x)
                & (phase_x > 0.0)
                & np.isfinite(phase)
            )
            phase_container = self.phase_axis.errorbar(
                phase_x[valid_phase],
                phase[valid_phase],
                yerr=np.vstack((
                    phase_error_minus[valid_phase],
                    phase_error_plus[valid_phase],
                )),
                fmt=marker,
                linestyle="none",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=4.0,
                elinewidth=0.9,
                capsize=1.8,
                capthick=0.8,
                alpha=0.82,
                zorder=5,
            )
            phase_artists = self._errorbar_artists(phase_container)
            self.li2024_data_artists.extend(phase_artists)
            self.li2024_phase_handles.append(phase_container.lines[0])

        for artist in self.li2024_data_artists:
            artist.set_visible(False)

        # Keep the two model curves and the eleven experimental data sets in
        # separate legends so the observational labels remain readable.
        self.amplitude_axis.add_artist(self.amplitude_curve_legend)
        self.li2024_data_legend = self.amplitude_axis.legend(
            handles=self.li2024_amplitude_handles,
            labels=[
                LI2024_DISPLAY_LABELS.get(experiment, experiment)
                for experiment in grouped_data
            ],
            loc="lower left",
            ncol=2,
            frameon=False,
            fontsize=5.5,
            title="Li et al. (2024) observations",
            title_fontsize=6.0,
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.65,
            borderaxespad=0.25,
        )
        self.li2024_data_legend.set_visible(False)
        return True

    def _update_li2024_data_visibility(self) -> bool:
        """Show observations only when requested on a valid energy scan."""
        energy_scan_active = self._energy_scan_is_active()
        should_show = self.show_li2024_data and energy_scan_active
        if should_show and not self._create_li2024_data_artists():
            should_show = False

        for artist in self.li2024_data_artists:
            artist.set_visible(should_show)
        if self.li2024_data_legend is not None:
            self.li2024_data_legend.set_visible(should_show)
        if hasattr(self, "li2024_data_checkbox"):
            self.li2024_data_checkbox.labels[0].set_color(
                "0.15" if energy_scan_active else "0.55"
            )
        return should_show

    def _remove_energy_secondary_axes(self) -> None:
        """Remove the energy scales before using a non-positive x domain."""
        for axis in self.energy_secondary_axes:
            axis.remove()
        self.energy_secondary_axes.clear()

    def _ensure_energy_secondary_axes(self) -> None:
        """Add reusable top ``E(a)`` scales to both curve panels."""
        if self.energy_secondary_axes:
            return
        for axis in self.curve_axes:
            energy_axis = axis.secondary_xaxis(
                "top",
                functions=(
                    _low_ratio_to_energy_axis,
                    _energy_to_low_ratio_axis,
                ),
            )
            energy_axis.tick_params(axis="x", labelsize=7, pad=1.5)
            energy_axis.set_xlabel(
                r"Equivalent energy $E(a)$ [TeV]",
                fontsize=8,
                labelpad=5.0,
            )
            self.energy_secondary_axes.append(energy_axis)

    def _create_parameter_controls(self) -> None:
        self.figure.text(0.025, 0.968, "Parameters", fontsize=13, weight="bold")
        self.sliders: dict[str, Slider] = {}
        self.textboxes: dict[str, TextBox] = {}
        y_positions = np.linspace(0.925, 0.525, len(PARAMETER_SPECS))
        for spec, y_position in zip(PARAMETER_SPECS, y_positions):
            slider_axis = self.figure.add_axes([0.075, y_position, 0.115, 0.021])
            textbox_axis = self.figure.add_axes([0.205, y_position - 0.003, 0.06, 0.029])
            initial_value = float(getattr(self.state, spec.key))
            slider = Slider(
                slider_axis,
                spec.label,
                spec.slider_minimum,
                spec.slider_maximum,
                valinit=spec.to_slider(initial_value),
            )
            slider.valtext.set_visible(False)
            textbox = TextBox(
                textbox_axis, "", initial=self._format_value(initial_value)
            )
            slider.on_changed(
                lambda value, key=spec.key: self._slider_changed(key, value)
            )
            textbox.on_submit(
                lambda text, key=spec.key: self._textbox_submitted(key, text)
            )
            self.sliders[spec.key] = slider
            self.textboxes[spec.key] = textbox

        self.position_readout = self.figure.text(
            0.025,
            0.512,
            "",
            fontsize=6.4,
            va="top",
            family="monospace",
            linespacing=1.05,
        )

        data_axis = self.figure.add_axes([0.145, 0.953, 0.120, 0.027])
        self.li2024_data_checkbox = CheckButtons(
            data_axis,
            ("Li+2024 data (E scan)",),
            (self.show_li2024_data,),
        )
        for label in self.li2024_data_checkbox.labels:
            label.set_fontsize(7.2)
            label.set_color("0.55")
        self.li2024_data_checkbox.on_clicked(
            self._li2024_data_checkbox_changed
        )

        mode_axis = self.figure.add_axes([0.025, 0.355, 0.235, 0.055])
        mode_axis.set_title("Curve mode", fontsize=9, loc="left")
        self.mode_radio = RadioButtons(
            mode_axis, ("history", "parameter scan"), active=0
        )
        for label in self.mode_radio.labels:
            label.set_fontsize(8)
        self.mode_radio.on_clicked(self._mode_changed)

        scan_axis = self.figure.add_axes([0.025, 0.145, 0.235, 0.195])
        scan_axis.set_title("Scan parameter", fontsize=9, loc="left")
        scan_labels = tuple(spec.label for spec in PARAMETER_SPECS)
        initial_scan_index = scan_labels.index(PARAMETER_BY_KEY[self.scan_parameter].label)
        self.scan_radio = RadioButtons(
            scan_axis, scan_labels, active=initial_scan_index
        )
        for label in self.scan_radio.labels:
            label.set_fontsize(7.5)
        self.scan_radio.on_clicked(self._scan_parameter_changed)

        self.figure.text(0.025, 0.132, "Scan range", fontsize=8)
        self.figure.text(0.025, 0.111, "min", fontsize=7.5, va="center")
        self.figure.text(0.145, 0.111, "max", fontsize=7.5, va="center")
        scan_min_axis = self.figure.add_axes([0.052, 0.098, 0.075, 0.03])
        scan_max_axis = self.figure.add_axes([0.174, 0.098, 0.075, 0.03])
        initial_minimum, initial_maximum = self.scan_ranges[self.scan_parameter]
        self.scan_min_textbox = TextBox(
            scan_min_axis, "", initial=self._format_value(initial_minimum)
        )
        self.scan_max_textbox = TextBox(
            scan_max_axis, "", initial=self._format_value(initial_maximum)
        )
        self.scan_min_textbox.on_submit(self._scan_range_submitted)
        self.scan_max_textbox.on_submit(self._scan_range_submitted)

        reset_axis = self.figure.add_axes([0.025, 0.055, 0.105, 0.038])
        self.reset_button = Button(reset_axis, "Reset traces")
        self.reset_button.on_clicked(self.reset_traces)
        self.status_text = self.figure.text(
            0.14, 0.074, "", fontsize=7.5, va="center", color="0.2"
        )

    @staticmethod
    def _format_value(value: float) -> str:
        return f"{float(value):.7g}"

    def _slider_changed(self, key: str, slider_value: float) -> None:
        if self._synchronizing_widgets:
            return
        spec = PARAMETER_BY_KEY[key]
        value = spec.from_slider(slider_value)
        self.state = replace(self.state, **{key: value})
        self._synchronizing_widgets = True
        try:
            self.textboxes[key].set_val(self._format_value(value))
        finally:
            self._synchronizing_widgets = False
        self._refresh(record=True)

    def _textbox_submitted(self, key: str, text: str) -> None:
        if self._synchronizing_widgets:
            return
        spec = PARAMETER_BY_KEY[key]
        try:
            value = float(text)
            spec.validate(value)
        except (TypeError, ValueError) as error:
            self.textboxes[key].ax.set_facecolor("#ffe6e6")
            self._set_status(str(error), error=True)
            self.figure.canvas.draw_idle()
            return
        self.textboxes[key].ax.set_facecolor("white")
        self.set_parameter(key, value, record=True)

    def set_parameter(self, key: str, value: float, record: bool = True) -> None:
        """Set one physical parameter programmatically and refresh the window."""
        if key not in PARAMETER_BY_KEY:
            raise KeyError(f"未知参数 {key!r}")
        spec = PARAMETER_BY_KEY[key]
        value = float(value)
        spec.validate(value)
        self.state = replace(self.state, **{key: value})
        self._synchronizing_widgets = True
        try:
            self.sliders[key].set_val(spec.to_slider(value))
            self.textboxes[key].set_val(self._format_value(value))
            self.textboxes[key].ax.set_facecolor("white")
        finally:
            self._synchronizing_widgets = False
        self._refresh(record=record)

    def _mode_changed(self, label: str) -> None:
        if self._synchronizing_widgets:
            return
        self.curve_mode = str(label)
        if self.curve_mode == "parameter scan":
            self._draw_parameter_scan()
        else:
            self._draw_history_curves()
        self.figure.canvas.draw_idle()

    def _scan_parameter_changed(self, label: str) -> None:
        if self._synchronizing_widgets:
            return
        self.scan_parameter = SCAN_LABEL_TO_KEY[str(label)]
        self._sync_scan_range_textboxes()
        if self.curve_mode == "parameter scan":
            self._draw_parameter_scan()
            self.figure.canvas.draw_idle()

    def set_curve_mode(self, mode: str) -> None:
        """Switch between ``history`` and ``parameter scan`` modes."""
        normalized = str(mode).strip().lower().replace("_", " ")
        if normalized not in {"history", "parameter scan"}:
            raise ValueError("mode 必须为 'history' 或 'parameter scan'")
        self.curve_mode = normalized
        self._synchronizing_widgets = True
        try:
            self.mode_radio.set_active(0 if normalized == "history" else 1)
        finally:
            self._synchronizing_widgets = False
        if normalized == "parameter scan":
            self._draw_parameter_scan()
        else:
            self._draw_history_curves()
        self.figure.canvas.draw_idle()

    def set_scan_parameter(self, key: str) -> None:
        """Select the x-axis parameter used by parameter-scan mode."""
        if key not in PARAMETER_BY_KEY:
            raise KeyError(f"未知扫描参数 {key!r}")
        self.scan_parameter = key
        index = list(PARAMETER_BY_KEY).index(key)
        self._synchronizing_widgets = True
        try:
            self.scan_radio.set_active(index)
        finally:
            self._synchronizing_widgets = False
        self._sync_scan_range_textboxes()
        if self.curve_mode == "parameter scan":
            self._draw_parameter_scan()
            self.figure.canvas.draw_idle()

    def _sync_scan_range_textboxes(self) -> None:
        minimum, maximum = self.scan_ranges[self.scan_parameter]
        self._synchronizing_widgets = True
        try:
            self.scan_min_textbox.set_val(self._format_value(minimum))
            self.scan_max_textbox.set_val(self._format_value(maximum))
            self.scan_min_textbox.ax.set_facecolor("white")
            self.scan_max_textbox.ax.set_facecolor("white")
        finally:
            self._synchronizing_widgets = False

    def _scan_range_submitted(self, _text: str) -> None:
        if self._synchronizing_widgets:
            return
        try:
            minimum = float(self.scan_min_textbox.text)
            maximum = float(self.scan_max_textbox.text)
            self.set_scan_range(minimum, maximum)
        except (TypeError, ValueError) as error:
            self.scan_min_textbox.ax.set_facecolor("#ffe6e6")
            self.scan_max_textbox.ax.set_facecolor("#ffe6e6")
            self._set_status(str(error), error=True)
            self.figure.canvas.draw_idle()

    def set_scan_range(self, minimum: float, maximum: float) -> None:
        """Set the current scan parameter's runtime interval."""
        spec = PARAMETER_BY_KEY[self.scan_parameter]
        minimum = float(minimum)
        maximum = float(maximum)
        spec.validate(minimum)
        spec.validate(maximum)
        if minimum >= maximum:
            raise ValueError("扫描区间必须满足 min < max")
        self.scan_ranges[self.scan_parameter] = (minimum, maximum)
        self._sync_scan_range_textboxes()
        if self.curve_mode == "parameter scan":
            self._draw_parameter_scan()
        self._set_status(
            f"Scan range: {minimum:g} to {maximum:g}"
        )
        self.figure.canvas.draw_idle()

    def _li2024_data_checkbox_changed(self, _label: str) -> None:
        if self._synchronizing_widgets:
            return
        self.show_li2024_data = bool(
            self.li2024_data_checkbox.get_status()[0]
        )
        shown = self._update_li2024_data_visibility()
        if self.show_li2024_data and self.li2024_data_load_error:
            self._set_status(self.li2024_data_load_error, error=True)
        elif self.show_li2024_data and not shown:
            self._set_status(
                "Li+2024 points require parameter scan of Dlow / Dparallel"
            )
        elif shown:
            self._set_status("Li+2024 observational points shown")
        else:
            self._set_status("Li+2024 observational points hidden")
        self.figure.canvas.draw_idle()

    def set_li2024_data_visible(self, visible: bool) -> None:
        """Enable or disable the digitized Li et al. (2024) observations."""
        visible = bool(visible)
        self.show_li2024_data = visible
        checkbox_visible = bool(self.li2024_data_checkbox.get_status()[0])
        if checkbox_visible != visible:
            self._synchronizing_widgets = True
            try:
                self.li2024_data_checkbox.set_active(0)
            finally:
                self._synchronizing_widgets = False
        shown = self._update_li2024_data_visibility()
        if visible and self.li2024_data_load_error:
            self._set_status(self.li2024_data_load_error, error=True)
        elif visible and not shown:
            self._set_status(
                "Li+2024 points require parameter scan of Dlow / Dparallel"
            )
        elif shown:
            self._set_status("Li+2024 observational points shown")
        else:
            self._set_status("Li+2024 observational points hidden")
        self.figure.canvas.draw_idle()

    def _update_sky(self, result: dict[str, object], record: bool) -> None:
        gradient_longitude = _signed_longitude(
            self.state.gradient_longitude_deg
        )
        gradient_latitude = self.state.gradient_latitude_deg
        dipole_longitude = _signed_longitude(
            result["affected_dipole_galactic_longitude_deg"]
        )
        dipole_latitude = float(
            result["affected_dipole_galactic_latitude_deg"]
        )
        gradient_x, gradient_y = _mollweide_coordinates(
            [gradient_longitude], [gradient_latitude]
        )
        dipole_x, dipole_y = _mollweide_coordinates(
            [dipole_longitude], [dipole_latitude]
        )
        self.gradient_marker.set_offsets(np.column_stack((gradient_x, gradient_y)))
        self.dipole_marker.set_offsets(np.column_stack((dipole_x, dipole_y)))

        magnetic_axis = cartesian_direction_from_galactic_lonlat(
            self.state.magnetic_longitude_deg,
            self.state.magnetic_latitude_deg,
        )
        plus = galactic_lonlat_from_cartesian_direction(magnetic_axis)
        minus = galactic_lonlat_from_cartesian_direction(-magnetic_axis)
        magnetic_x, magnetic_y = _mollweide_coordinates(
            [plus[0], minus[0]], [plus[1], minus[1]]
        )
        self.magnetic_markers.set_offsets(
            np.column_stack((magnetic_x, magnetic_y))
        )
        circle_longitude, circle_latitude = _perpendicular_great_circle(
            magnetic_axis
        )
        circle_x, circle_y = _path_with_seam_breaks(
            circle_longitude, circle_latitude
        )
        self.magnetic_circle_line.set_data(circle_x, circle_y)

        if record:
            self.gradient_trajectory["longitude"].append(gradient_longitude)
            self.gradient_trajectory["latitude"].append(gradient_latitude)
            self.dipole_trajectory["longitude"].append(dipole_longitude)
            self.dipole_trajectory["latitude"].append(dipole_latitude)
        gradient_track_x, gradient_track_y = _path_with_seam_breaks(
            self.gradient_trajectory["longitude"],
            self.gradient_trajectory["latitude"],
        )
        dipole_track_x, dipole_track_y = _path_with_seam_breaks(
            self.dipole_trajectory["longitude"],
            self.dipole_trajectory["latitude"],
        )
        self.gradient_track_line.set_data(gradient_track_x, gradient_track_y)
        self.dipole_track_line.set_data(dipole_track_x, dipole_track_y)

    def _update_slice(self, result: dict[str, object]) -> None:
        field, point_v, psi_deg, point_analytic = self._slice_field(result)
        self.slice_mesh.set_array(field.ravel())
        self.slice_point.set_data([point_analytic[2]], [point_v])
        self.slice_axis.set_title(
            "Analytic dipole amplification; "
            f"slice psi={psi_deg:.1f} deg\n"
            f"S_ana=({point_analytic[0]:.2f}, {point_analytic[1]:.2f}, "
            f"{point_analytic[2]:.2f}) R",
            pad=12.0,
        )
        point_simulation = np.asarray(
            result["position_simulation_over_radius"], dtype=np.float64
        )
        self.position_readout.set_text(
            "S sim/R = "
            f"({point_simulation[0]:.3f}, {point_simulation[1]:.3f}, "
            f"{point_simulation[2]:.3f})\n"
            "S ana/R = "
            f"({point_analytic[0]:.3f}, {point_analytic[1]:.3f}, "
            f"{point_analytic[2]:.3f})\n"
            "E(a) = "
            f"{result['energy_from_d_low_over_d_parallel_tev']:.4g} TeV\n"
            "D_parallel(E(a)) = "
            f"{result['independent_d_parallel_cm2_s']:.3e} cm^2/s\n"
            "D_perp(MA) = "
            f"{result['independent_d_perpendicular_cm2_s']:.3e} cm^2/s\n"
            "D_low(a) = "
            f"{result['independent_d_low_cm2_s']:.3e} cm^2/s"
        )

    def _append_history(self, result: dict[str, object]) -> None:
        self.history["step"].append(self._next_history_step)
        self.history["amplitude"].append(
            float(result["affected_dipole_amplitude"])
        )
        self.history["ra_amplitude"].append(
            float(result["affected_right_ascension_amplitude"])
        )
        self.history["ra_phase"].append(
            float(result["affected_right_ascension_phase_hour"])
        )
        self._next_history_step += 1

    def _configure_curve_x_axis(self, label: str, logarithmic: bool) -> None:
        for axis in self.curve_axes:
            axis.set_xlabel(label, labelpad=8.0)
            axis.set_xscale("log" if logarithmic else "linear")
            if not logarithmic:
                axis.locator_params(axis="x", nbins=5)

    def _rescale_curve_axes(self) -> None:
        self.amplitude_axis.set_autoscaley_on(False)
        self.amplitude_axis.set_ylim(
            DIPOLE_AMPLITUDE_Y_MIN,
            DIPOLE_AMPLITUDE_Y_MAX,
        )
        self.phase_axis.relim()
        self.phase_axis.set_ylim(-12.0, 12.0)

    def _set_history_x_limits(self, x: np.ndarray) -> None:
        """Show every recorded update while leaving a small edge margin."""
        if x.size == 0:
            limits = (0.0, 1.0)
        elif x.size == 1:
            limits = (-0.5, 0.5)
        else:
            limits = (-0.5, float(x[-1]) + 0.5)
        for axis in self.curve_axes:
            axis.set_xlim(*limits)

    def _set_scan_x_limits(
        self, minimum: float, maximum: float
    ) -> None:
        """Use the complete selected-parameter range on all curve panels."""
        for axis in self.curve_axes:
            axis.set_xlim(minimum, maximum)

    def _draw_history_curves(self) -> None:
        self._remove_energy_secondary_axes()
        x = np.asarray(self.history["step"], dtype=np.float64)
        self.curve_lines[0].set_data(x, self.history["amplitude"])
        self.curve_lines[1].set_data(x, self.history["ra_amplitude"])
        phase_x, phase_y = _break_wrapped_curve(x, self.history["ra_phase"])
        self.curve_lines[2].set_data(phase_x, phase_y)
        for line in self.current_parameter_lines:
            line.set_visible(False)
        self._configure_curve_x_axis("Valid slider-update number", False)
        self._set_history_x_limits(x)
        self._rescale_curve_axes()
        self._update_li2024_data_visibility()

    def _draw_parameter_scan(self) -> None:
        spec = PARAMETER_BY_KEY[self.scan_parameter]
        scan_minimum, scan_maximum = self.scan_ranges[self.scan_parameter]
        x = spec.scan_values(
            minimum=scan_minimum,
            maximum=scan_maximum,
        )
        total_amplitude = np.full(x.shape, np.nan)
        ra_amplitude = np.full(x.shape, np.nan)
        ra_phase = np.full(x.shape, np.nan)
        for index, value in enumerate(x):
            scan_state = replace(self.state, **{spec.key: float(value)})
            try:
                result = calculate_state(scan_state)
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            total_amplitude[index] = float(result["affected_dipole_amplitude"])
            ra_amplitude[index] = float(
                result["affected_right_ascension_amplitude"]
            )
            ra_phase[index] = float(
                result["affected_right_ascension_phase_hour"]
            )
        self.curve_lines[0].set_data(x, total_amplitude)
        self.curve_lines[1].set_data(x, ra_amplitude)
        phase_x, phase_y = _break_wrapped_curve(x, ra_phase)
        self.curve_lines[2].set_data(phase_x, phase_y)
        current_value = float(getattr(self.state, spec.key))
        for line in self.current_parameter_lines:
            line.set_xdata([current_value, current_value])
            line.set_visible(True)
        self._configure_curve_x_axis(spec.axis_label, spec.logarithmic)
        self._set_scan_x_limits(scan_minimum, scan_maximum)
        if (
            self.scan_parameter == "d_low_over_d_parallel"
            and _energy_conversion_is_valid()
        ):
            self._ensure_energy_secondary_axes()
        else:
            self._remove_energy_secondary_axes()
        self._rescale_curve_axes()
        self._update_li2024_data_visibility()

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_text.set_text(message)
        self.status_text.set_color("tab:red" if error else "0.2")

    def _refresh(self, record: bool) -> None:
        try:
            result = calculate_state(self.state)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            self._set_status(str(error), error=True)
            self.figure.canvas.draw_idle()
            return
        self.last_result = result
        self._update_sky(result, record=record)
        self._update_slice(result)
        if self.curve_mode == "history":
            if record:
                self._append_history(result)
            self._draw_history_curves()
        else:
            self._draw_parameter_scan()
        self._set_status(
            f"eta={result['gradient_perpendicular_over_parallel']:.4g}; "
            f"A_delta={result['transport_amplification']:.4g}; "
            f"RA phase={result['affected_right_ascension_phase_hour']:.3g} h"
        )
        self.figure.canvas.draw_idle()

    def reset_traces(self, _event=None) -> None:
        """Clear all history curves and both Galactic-sky trajectories."""
        for values in self.history.values():
            values.clear()
        for trajectory in (self.gradient_trajectory, self.dipole_trajectory):
            trajectory["longitude"].clear()
            trajectory["latitude"].clear()
        self._next_history_step = 0
        self.gradient_track_line.set_data([np.nan], [np.nan])
        self.dipole_track_line.set_data([np.nan], [np.nan])
        for line in self.curve_lines:
            line.set_data([], [])
        for line in self.current_parameter_lines:
            line.set_visible(False)
        self._rescale_curve_axes()
        self._set_status("Curves and sky trajectories cleared")
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> LowDiffusionDipoleExplorer:
    explorer = LowDiffusionDipoleExplorer()
    explorer.show()
    return explorer


if __name__ == "__main__":
    main()
