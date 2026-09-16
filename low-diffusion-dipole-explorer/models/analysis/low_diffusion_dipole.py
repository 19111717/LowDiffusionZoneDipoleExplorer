"""Physical dipole response of a spherical low-diffusion region.

The existing analytic coordinates are retained: ``B || +z`` and the
perpendicular background-gradient component lies along ``+x``.  A physical
point is preferably supplied as a fixed Galactic centre-to-point direction
and a distance in sphere radii, then rotated into this analytic frame.

The supplied magnetic-field direction is treated as an unoriented diffusion
axis.  Its sign and the analytic x-axis are selected so that the transformed
relative background gradient is ``(-G_perp, 0, -G_parallel)``.  Changes to
the density denominator caused by the sphere are deliberately ignored.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from cygbubble import config
from cygbubble.coordinates import (
    cartesian_direction_from_galactic_lonlat,
    galactic_lonlat_from_cartesian_direction,
    simulation_to_equatorial_components,
)
from cygbubble.analytic.sphere import (
    arbitrary_density_and_flux,
    arbitrary_spatial_amplification,
    perpendicular_density_and_flux,
    perpendicular_spatial_amplification,
)
from cygbubble.physics.diffusion import (
    Anisotropic_ratio_to_energy,
    diffusion_coefficient_parallel,
)


def _finite_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须包含三个有限分量")
    return vector


def _finite_scalar(value, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} 必须是有限标量")
    return scalar


def _signed_longitude(longitude_deg: float) -> float:
    """Wrap one longitude to the closed display interval [-180, 180]."""
    longitude = float(longitude_deg)
    wrapped = (longitude + 180.0) % 360.0 - 180.0
    if np.isclose(wrapped, -180.0) and longitude > 0.0:
        return 180.0
    return wrapped


def relative_position_from_galactic_lonlat(
    distance_over_radius: float,
    longitude_deg: float,
    latitude_deg: float,
) -> np.ndarray:
    r"""Return the centre-to-S vector in simulation coordinates, divided by R.

    ``longitude_deg`` and ``latitude_deg`` describe the direction of the
    displacement from the low-diffusion-region centre to S in the project's
    fixed Galactic basis.  They are therefore independent of the magnetic and
    background-gradient directions.
    """
    distance = _finite_scalar(distance_over_radius, "distance_over_radius")
    if distance < 0.0:
        raise ValueError("distance_over_radius 必须大于或等于零")
    direction = cartesian_direction_from_galactic_lonlat(
        longitude_deg, latitude_deg
    )
    return distance * direction


def _fallback_perpendicular_axis(axis: np.ndarray) -> np.ndarray:
    """Choose a deterministic unit vector perpendicular to ``axis``."""
    coordinate_axes = np.eye(3, dtype=np.float64)
    reference = coordinate_axes[np.argmin(np.abs(coordinate_axes @ axis))]
    perpendicular = reference - float(np.dot(reference, axis)) * axis
    return perpendicular / np.linalg.norm(perpendicular)


def _equatorial_projection(dipole_simulation: Sequence[float]) -> dict[str, object]:
    r"""Project a dipole onto the J2000 equatorial plane.

    The right-ascension phase is measured from RA=0 h toward RA=6 h and is
    wrapped to the half-open interval [-12 h, 12 h).  A dipole parallel to the
    celestial-pole axis has zero RA amplitude and an undefined (NaN) phase.
    """
    dipole_equatorial = simulation_to_equatorial_components(
        _finite_vector(dipole_simulation, "dipole_simulation")
    )
    right_ascension_amplitude = float(np.hypot(
        dipole_equatorial[0], dipole_equatorial[1]
    ))
    if right_ascension_amplitude == 0.0:
        right_ascension_phase_deg = float("nan")
        right_ascension_phase_hour = float("nan")
    else:
        raw_phase_deg = float(np.degrees(np.arctan2(
            dipole_equatorial[1], dipole_equatorial[0]
        )))
        right_ascension_phase_deg = (
            (raw_phase_deg + 180.0) % 360.0 - 180.0
        )
        right_ascension_phase_hour = right_ascension_phase_deg / 15.0
    return {
        "dipole_equatorial": dipole_equatorial,
        "right_ascension_amplitude": right_ascension_amplitude,
        "right_ascension_phase_deg": right_ascension_phase_deg,
        "right_ascension_phase_hour": right_ascension_phase_hour,
    }


def analytic_frame_from_magnetic_field_and_gradient(
    magnetic_longitude_deg: float,
    magnetic_latitude_deg: float,
    relative_gradient_per_pc: Sequence[float],
) -> dict[str, object]:
    r"""Construct the native analytic frame from a simulated gradient.

    ``simulation_to_analytic`` acts on column vectors; its transpose is the
    inverse rotation.  For a purely perpendicular gradient, ``+z`` follows the
    supplied magnetic-field direction and the analytic solution is marked as
    ``perpendicular`` so callers can avoid the singular
    ``G_perp/G_parallel`` parametrization.
    """
    gradient_simulation = _finite_vector(
        relative_gradient_per_pc, "relative_gradient_per_pc"
    )
    gradient_norm = float(np.linalg.norm(gradient_simulation))
    if gradient_norm == 0.0:
        raise ValueError("relative_gradient_per_pc 不能是零矢量")

    magnetic_axis = cartesian_direction_from_galactic_lonlat(
        magnetic_longitude_deg, magnetic_latitude_deg
    )
    signed_parallel = float(np.dot(gradient_simulation, magnetic_axis))
    parallel_magnitude = abs(signed_parallel)
    zero_tolerance = 64.0 * np.finfo(np.float64).eps * gradient_norm
    perpendicular_case = parallel_magnitude <= zero_tolerance

    # D is invariant under B -> -B.  Select +z so the parallel gradient is -z.
    if perpendicular_case:
        # B and -B define the same diffusion tensor.  With no parallel
        # gradient to select one sign, retain the direction supplied by the
        # caller and explicitly remove round-off parallel to it.
        analytic_z_axis = magnetic_axis
        parallel_magnitude = 0.0
    else:
        analytic_z_axis = (
            -magnetic_axis if signed_parallel > 0.0 else magnetic_axis
        )
    perpendicular_gradient = (
        gradient_simulation - signed_parallel * magnetic_axis
    )
    perpendicular_magnitude = float(np.linalg.norm(perpendicular_gradient))
    if perpendicular_magnitude <= zero_tolerance:
        perpendicular_magnitude = 0.0
        analytic_x_axis = _fallback_perpendicular_axis(analytic_z_axis)
        x_axis_from_gradient = False
    else:
        analytic_x_axis = -perpendicular_gradient / perpendicular_magnitude
        x_axis_from_gradient = True

    # e_y = e_z x e_x makes (e_x, e_y, e_z) right handed.
    analytic_y_axis = np.cross(analytic_z_axis, analytic_x_axis)
    analytic_y_axis /= np.linalg.norm(analytic_y_axis)
    # Recompute x to remove the last round-off component along z.
    analytic_x_axis = np.cross(analytic_y_axis, analytic_z_axis)
    analytic_x_axis /= np.linalg.norm(analytic_x_axis)

    simulation_to_analytic = np.vstack((
        analytic_x_axis,
        analytic_y_axis,
        analytic_z_axis,
    ))
    analytic_to_simulation = simulation_to_analytic.T
    transformed_gradient = simulation_to_analytic @ gradient_simulation
    expected_gradient = np.asarray(
        [-perpendicular_magnitude, 0.0, -parallel_magnitude],
        dtype=np.float64,
    )
    frame_tolerance = 256.0 * np.finfo(np.float64).eps * gradient_norm
    if not np.allclose(
        transformed_gradient,
        expected_gradient,
        rtol=256.0 * np.finfo(np.float64).eps,
        atol=frame_tolerance,
    ):
        raise RuntimeError("解析坐标系构造失败：梯度分量与预期不一致")
    if not np.allclose(
        simulation_to_analytic @ analytic_to_simulation,
        np.eye(3),
        rtol=0.0,
        atol=256.0 * np.finfo(np.float64).eps,
    ) or not np.isclose(
        np.linalg.det(simulation_to_analytic), 1.0,
        rtol=0.0,
        atol=256.0 * np.finfo(np.float64).eps,
    ):
        raise RuntimeError("解析坐标系构造失败：旋转矩阵不是右手正交矩阵")

    background_gradient_ratio = (
        float("inf")
        if perpendicular_case
        else perpendicular_magnitude / parallel_magnitude
    )

    return {
        "magnetic_axis_simulation": magnetic_axis,
        "analytic_x_axis_simulation": analytic_x_axis,
        "analytic_y_axis_simulation": analytic_y_axis,
        "analytic_z_axis_simulation": analytic_z_axis,
        "magnetic_axis_was_flipped": bool(
            np.dot(analytic_z_axis, magnetic_axis) < 0.0
        ),
        "x_axis_from_gradient": x_axis_from_gradient,
        "analytic_solution_case": (
            "perpendicular" if perpendicular_case else "arbitrary"
        ),
        "simulation_to_analytic": simulation_to_analytic,
        "analytic_to_simulation": analytic_to_simulation,
        "relative_gradient_simulation_per_pc": gradient_simulation,
        "relative_gradient_analytic_per_pc": expected_gradient,
        "gradient_parallel_magnitude_per_pc": parallel_magnitude,
        "gradient_perpendicular_magnitude_per_pc": perpendicular_magnitude,
        "background_gradient_ratio": background_gradient_ratio,
    }


def low_diffusion_dipole_at_point(
    magnetic_longitude_deg: float,
    magnetic_latitude_deg: float,
    relative_gradient_per_pc: Sequence[float],
    d_perp_over_d_parallel: float,
    d_low_over_d_parallel: float,
    position_analytic: Sequence[float] | None = None,
    *,
    diffusion_energy_tev: float | None = None,
    position_distance_over_radius: float | None = None,
    position_longitude_deg: float | None = None,
    position_latitude_deg: float | None = None,
) -> dict[str, object]:
    r"""Return the physical CR dipole before and after a low-D sphere.

    ``relative_gradient_per_pc`` is the signed, unperturbed ``grad(n)/n`` in
    simulation coordinates.  Prefer specifying S by the centre-to-S distance
    in sphere-radius units and the displacement direction in the fixed
    Galactic basis.  The corresponding analytic coordinates are recalculated
    for every magnetic-field/gradient frame, keeping the physical point fixed.

    ``position_analytic`` remains available as a compatibility input.  It is
    mutually exclusive with the three Galactic-position arguments.
    ``diffusion_energy_tev`` can explicitly set the energy used to normalize
    the physical diffusion coefficients.  When omitted, the legacy behavior
    infers that energy from ``d_perp_over_d_parallel``.  This calculation
    deliberately omits the density-denominator perturbation caused by the
    sphere.
    """
    q = _finite_scalar(
        d_perp_over_d_parallel, "d_perp_over_d_parallel"
    )
    a = _finite_scalar(
        d_low_over_d_parallel, "d_low_over_d_parallel"
    )
    if not 0.0 < q <= 1.0:
        raise ValueError(
            "d_perp_over_d_parallel 必须位于 (0, 1]，以满足当前解析解"
        )
    if a <= 0.0:
        raise ValueError("d_low_over_d_parallel 必须大于零")
    frame = analytic_frame_from_magnetic_field_and_gradient(
        magnetic_longitude_deg,
        magnetic_latitude_deg,
        relative_gradient_per_pc,
    )
    galactic_position_values = (
        position_distance_over_radius,
        position_longitude_deg,
        position_latitude_deg,
    )
    has_any_galactic_position = any(
        value is not None for value in galactic_position_values
    )
    has_all_galactic_position = all(
        value is not None for value in galactic_position_values
    )
    if position_analytic is not None and has_any_galactic_position:
        raise ValueError(
            "position_analytic 与位置的 distance/l/b 参数不能同时传入"
        )
    if position_analytic is None and not has_all_galactic_position:
        raise ValueError(
            "请完整传入 position_distance_over_radius、"
            "position_longitude_deg 和 position_latitude_deg"
        )

    simulation_to_analytic = np.asarray(
        frame["simulation_to_analytic"], dtype=np.float64
    )
    analytic_to_simulation = np.asarray(
        frame["analytic_to_simulation"], dtype=np.float64
    )
    if position_analytic is None:
        position_simulation = relative_position_from_galactic_lonlat(
            position_distance_over_radius,
            position_longitude_deg,
            position_latitude_deg,
        )
        point = simulation_to_analytic @ position_simulation
        position_distance = float(position_distance_over_radius)
        position_longitude = _signed_longitude(position_longitude_deg)
        position_latitude = float(position_latitude_deg)
        position_input_mode = "galactic"
    else:
        point = _finite_vector(position_analytic, "position_analytic")
        position_simulation = analytic_to_simulation @ point
        position_distance = float(np.linalg.norm(position_simulation))
        position_longitude, position_latitude = (
            galactic_lonlat_from_cartesian_direction(position_simulation)
        )
        if np.isfinite(position_longitude):
            position_longitude = _signed_longitude(position_longitude)
        position_input_mode = "analytic"

    gradient_analytic = np.asarray(
        frame["relative_gradient_analytic_per_pc"], dtype=np.float64
    )
    eta = float(frame["background_gradient_ratio"])

    ma = q ** 0.25
    parallel_over_perpendicular = 1.0 / q
    if diffusion_energy_tev is None:
        delta_delta = float(config.Delta_delta)
        if not np.isfinite(delta_delta) or delta_delta == 0.0:
            raise ValueError(
                "config.Delta_delta 必须是有限非零数，"
                "才能由扩散系数比反推能量"
            )
        reference_ratio = float(config.anisotropic_ratio_at_10TeV)
        if not np.isfinite(reference_ratio) or reference_ratio <= 0.0:
            raise ValueError(
                "config.anisotropic_ratio_at_10TeV 必须为有限正数"
            )
        energy_tev = float(
            Anisotropic_ratio_to_energy(parallel_over_perpendicular)
        )
        diffusion_energy_source = "d_parallel_over_d_perpendicular"
    else:
        energy_tev = _finite_scalar(
            diffusion_energy_tev, "diffusion_energy_tev"
        )
        if energy_tev <= 0.0:
            raise ValueError("diffusion_energy_tev 必须大于零")
        diffusion_energy_source = "explicit"
    if not np.isfinite(energy_tev) or energy_tev <= 0.0:
        raise ValueError("由扩散系数比得到的能量不是有限正数")
    d_parallel = float(diffusion_coefficient_parallel(
        energy_tev,
        delta=config.delta_Diffuse,
        D0=config.D_0_10TeV,
    ))
    d_perpendicular = q * d_parallel
    d_low = a * d_parallel
    if not np.all(np.isfinite([d_parallel, d_perpendicular, d_low])) or min(
        d_parallel, d_perpendicular, d_low
    ) <= 0.0:
        raise ValueError("换算得到的扩散系数必须全部为有限正数")

    perpendicular_case = frame["analytic_solution_case"] == "perpendicular"
    if perpendicular_case:
        density_perturbation_unit, flux_x, flux_y, flux_z = (
            perpendicular_density_and_flux(
                *point,
                a=a,
                MA=ma,
                n0=0.0,
            )
        )
        _, transport_amplification_value = (
            perpendicular_spatial_amplification(
                *point,
                a=a,
                MA=ma,
            )
        )
        unit_gradient_scale_per_pc = float(gradient_analytic[0])
    else:
        density_perturbation_unit, flux_x, flux_y, flux_z = (
            arbitrary_density_and_flux(
                *point,
                a=a,
                MA=ma,
                perp_to_parallel_ratio=eta,
                n0=0.0,
            )
        )
        _, transport_amplification_value = arbitrary_spatial_amplification(
            *point,
            a=a,
            MA=ma,
            perp_to_parallel_ratio=eta,
        )
        unit_gradient_scale_per_pc = float(gradient_analytic[2])

    analytic_flux_unit = np.asarray(
        [flux_x, flux_y, flux_z], dtype=np.float64
    ).reshape(3)
    if not np.all(np.isfinite(analytic_flux_unit)):
        raise RuntimeError("解析解在 position_analytic 处返回了非有限通量")

    transport_amplification = float(transport_amplification_value)
    if not np.isfinite(transport_amplification) or (
        transport_amplification < 0.0
    ):
        raise RuntimeError("解析解返回了无效的偶极输运放大倍数")

    background_tensor_analytic = np.diag(
        [d_perpendicular, d_perpendicular, d_parallel]
    )
    relative_gradient_cm = gradient_analytic / config.PC_TO_CM
    background_dipole_analytic = (
        3.0 / config.C_LIGHT_CM_S
    ) * (background_tensor_analytic @ relative_gradient_cm)
    background_amplitude = float(np.linalg.norm(background_dipole_analytic))
    if not np.isfinite(background_amplitude) or background_amplitude <= 0.0:
        raise RuntimeError("未扰动背景偶极振幅无效")
    background_direction_analytic = (
        background_dipole_analytic / background_amplitude
    )

    # Each unit solution assumes a +1 gradient along its driving axis.  The
    # constructed analytic frame gives the physical component a negative sign.
    diffusive_flux_over_density_analytic_cm_s = (
        d_parallel
        * unit_gradient_scale_per_pc
        / config.PC_TO_CM
        * analytic_flux_unit
    )
    direct_affected_dipole_analytic = (
        -3.0 / config.C_LIGHT_CM_S
    ) * diffusive_flux_over_density_analytic_cm_s
    direct_affected_amplitude = float(
        np.linalg.norm(direct_affected_dipole_analytic)
    )

    affected_amplitude = background_amplitude * transport_amplification
    amplitude_tolerance = max(
        1.0e-30,
        5.0e-12 * max(affected_amplitude, direct_affected_amplitude),
    )
    if not np.isclose(
        affected_amplitude,
        direct_affected_amplitude,
        rtol=5.0e-12,
        atol=amplitude_tolerance,
    ):
        raise RuntimeError(
            "背景偶极乘放大倍数与解析通量的真实单位结果不一致"
        )

    if direct_affected_amplitude == 0.0:
        affected_direction_analytic = np.full(3, np.nan, dtype=np.float64)
        affected_dipole_analytic = np.zeros(3, dtype=np.float64)
    else:
        affected_direction_analytic = (
            direct_affected_dipole_analytic / direct_affected_amplitude
        )
        affected_dipole_analytic = (
            affected_amplitude * affected_direction_analytic
        )

    background_dipole_simulation = (
        analytic_to_simulation @ background_dipole_analytic
    )
    affected_dipole_simulation = (
        analytic_to_simulation @ affected_dipole_analytic
    )

    # Recalculate the background in simulation coordinates as a rotation check.
    magnetic_axis = np.asarray(
        frame["magnetic_axis_simulation"], dtype=np.float64
    )
    background_tensor_simulation = (
        d_perpendicular * np.eye(3)
        + (d_parallel - d_perpendicular)
        * np.outer(magnetic_axis, magnetic_axis)
    )
    # Use the gradient represented by the analytic frame.  In the strictly
    # perpendicular branch this removes only the floating-point residue of the
    # nominally zero parallel component (for example cos(90 deg)).
    gradient_simulation = analytic_to_simulation @ gradient_analytic
    direct_background_simulation = (
        3.0 / config.C_LIGHT_CM_S
    ) * (
        background_tensor_simulation
        @ (gradient_simulation / config.PC_TO_CM)
    )
    if not np.allclose(
        background_dipole_simulation,
        direct_background_simulation,
        rtol=1.0e-5,
        atol=1.0e-10,
    ):
        raise RuntimeError("旋转后的背景偶极与模拟坐标直接计算结果不一致")

    background_direction_simulation = (
        background_dipole_simulation / background_amplitude
    )
    (
        background_longitude_deg,
        background_latitude_deg,
    ) = galactic_lonlat_from_cartesian_direction(
        background_dipole_simulation
    )

    if affected_amplitude == 0.0:
        affected_direction_simulation = np.full(3, np.nan, dtype=np.float64)
    else:
        affected_direction_simulation = (
            affected_dipole_simulation / affected_amplitude
        )
    longitude_deg, latitude_deg = galactic_lonlat_from_cartesian_direction(
        affected_dipole_simulation
    )
    if affected_amplitude == 0.0:
        deflection_deg = float("nan")
    else:
        deflection_deg = float(np.degrees(np.arccos(np.clip(
            np.dot(
                background_direction_simulation,
                affected_direction_simulation,
            ),
            -1.0,
            1.0,
        ))))

    background_equatorial = _equatorial_projection(
        background_dipole_simulation
    )
    affected_equatorial = _equatorial_projection(
        affected_dipole_simulation
    )

    inside_sphere = bool(float(np.dot(point, point)) <= 1.0)
    local_tensor_analytic = (
        d_low * np.eye(3)
        if inside_sphere
        else background_tensor_analytic.copy()
    )

    result = dict(frame)
    result.update({
        "magnetic_longitude_deg": float(magnetic_longitude_deg),
        "magnetic_latitude_deg": float(magnetic_latitude_deg),
        "position_input_mode": position_input_mode,
        "position_distance_over_radius": position_distance,
        "position_galactic_longitude_deg": position_longitude,
        "position_galactic_latitude_deg": position_latitude,
        "position_simulation_over_radius": position_simulation,
        "position_analytic": point,
        "inside_low_diffusion_region": inside_sphere,
        "d_perp_over_d_parallel": q,
        "d_parallel_over_d_perp": parallel_over_perpendicular,
        "d_low_over_d_parallel": a,
        # Explicit user-facing aliases for the two quantities that set the
        # arbitrary-gradient analytic solution.  The gradient ratio uses
        # component magnitudes and is therefore non-negative.
        "gradient_perpendicular_over_parallel": eta,
        "equivalent_ma": ma,
        "ma": ma,
        "energy_tev": energy_tev,
        "diffusion_energy_source": diffusion_energy_source,
        "d_parallel_cm2_s": d_parallel,
        "d_perpendicular_cm2_s": d_perpendicular,
        "d_low_cm2_s": d_low,
        "background_diffusion_tensor_analytic_cm2_s": (
            background_tensor_analytic
        ),
        "background_diffusion_tensor_simulation_cm2_s": (
            background_tensor_simulation
        ),
        "local_diffusion_tensor_analytic_cm2_s": local_tensor_analytic,
        "analytic_density_perturbation_unit": float(
            density_perturbation_unit
        ),
        "analytic_diffusive_flux_unit": analytic_flux_unit,
        "diffusive_flux_over_density_analytic_cm_s": (
            diffusive_flux_over_density_analytic_cm_s
        ),
        "background_dipole_analytic": background_dipole_analytic,
        "background_dipole_simulation": background_dipole_simulation,
        "background_dipole_amplitude": background_amplitude,
        "background_dipole_direction_analytic": (
            background_direction_analytic
        ),
        "background_dipole_direction_simulation": (
            background_direction_simulation
        ),
        "background_dipole_galactic_longitude_deg": (
            background_longitude_deg
        ),
        "background_dipole_galactic_latitude_deg": background_latitude_deg,
        "background_dipole_equatorial": (
            background_equatorial["dipole_equatorial"]
        ),
        "background_right_ascension_amplitude": (
            background_equatorial["right_ascension_amplitude"]
        ),
        "background_right_ascension_phase_deg": (
            background_equatorial["right_ascension_phase_deg"]
        ),
        "background_right_ascension_phase_hour": (
            background_equatorial["right_ascension_phase_hour"]
        ),
        "transport_amplification": transport_amplification,
        "affected_dipole_direction_analytic": affected_direction_analytic,
        "affected_dipole_direction_simulation": (
            affected_direction_simulation
        ),
        "affected_dipole_analytic": affected_dipole_analytic,
        "affected_dipole_simulation": affected_dipole_simulation,
        "affected_dipole_amplitude": affected_amplitude,
        "affected_dipole_galactic_longitude_deg": longitude_deg,
        "affected_dipole_galactic_latitude_deg": latitude_deg,
        "affected_dipole_equatorial": (
            affected_equatorial["dipole_equatorial"]
        ),
        "affected_right_ascension_amplitude": (
            affected_equatorial["right_ascension_amplitude"]
        ),
        "affected_right_ascension_phase_deg": (
            affected_equatorial["right_ascension_phase_deg"]
        ),
        "affected_right_ascension_phase_hour": (
            affected_equatorial["right_ascension_phase_hour"]
        ),
        "dipole_deflection_deg": deflection_deg,
        "density_correction_applied": False,
    })
    return result


__all__ = [
    "analytic_frame_from_magnetic_field_and_gradient",
    "low_diffusion_dipole_at_point",
    "relative_position_from_galactic_lonlat",
]
