"""Spherical low-diffusion region in a uniform anisotropic background.

Dimensionless convention: R=1, G0=1; flux is in D_parallel*G0 units.
The magnetic field is along z. a=D_low/D_parallel and D_perp/D_parallel=MA**4.
These infinite-background analytic solutions are distinct from the finite
Galactic box/cylinder simulations, whose coordinates use pc.
"""
from __future__ import annotations
import numpy as np

def L_z_function(MA):
    """Depolarization factor along B for 0 < MA <= 1."""
    if not (0 < MA <= 1):
        raise ValueError("This implementation assumes 0 < MA <= 1.")

    xi = np.sqrt(MA ** -4 - 1.0)

    # Avoid 0/0 and cancellation in the isotropic limit.
    if xi < 1.0e-4:
        return 1.0 / 3.0 + 2.0 * xi ** 2 / 15.0 - 2.0 * xi ** 4 / 35.0

    return (1.0 + xi ** 2) / xi ** 3 * (xi - np.arctan(xi))


def Nabla_rho_ratio_parallel(MA, a):
    """beta_parallel = |grad n_in| / |grad n_0|."""
    Lz = L_z_function(MA)
    return 1.0 / (1.0 + (a - 1.0) * Lz)


def Nabla_rho_ratio_perp(MA, a):
    """Perpendicular result retained for comparison with the earlier model."""
    L_perp = (1.0 - L_z_function(MA)) / 2.0
    return 1.0 / (1.0 + (a / MA ** 4 - 1.0) * L_perp)


def dipole_anisotropy_ratio_parallel(MA, a):
    """Ratio of D|grad n| inside/outside, assuming the density factor cancels.

    For delta = 3 D |grad n| / (c n), this is exactly the anisotropy ratio
    at the sphere centre and is also the ratio throughout the sphere when the
    fractional density variation across the sphere is small.
    """
    return a * Nabla_rho_ratio_parallel(MA, a)


def lambda_function(rho, z, MA):
    """Positive confocal coordinate lambda outside the unit sphere.

    It solves
        rho^2/(1 + MA^4 lambda) + z^2/(1 + lambda) = 1.
    The function should only be evaluated for rho^2 + z^2 >= 1.
    """
    rho, z = np.broadcast_arrays(
        np.asarray(rho, dtype=float), np.asarray(z, dtype=float)
    )
    m4 = MA ** 4
    r2 = rho ** 2 + z ** 2
    B = 1.0 + m4 - rho ** 2 - m4 * z ** 2
    discriminant = B ** 2 + 4.0 * m4 * (r2 - 1.0)
    discriminant = np.maximum(discriminant, 0.0)
    root = np.sqrt(discriminant)

    # The rationalized expression avoids loss of precision when lambda -> 0
    # and B > 0. The direct positive root is stable for B <= 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(
            B > 0.0,
            2.0 * (r2 - 1.0) / (B + root),
            (-B + root) / (2.0 * m4),
        )
    return lam


def F_lambda(lam, MA):
    """Exterior shape function F(lambda), normalized so F(0)=1."""
    lam = np.asarray(lam, dtype=float)
    xi = np.sqrt(MA ** -4 - 1.0)

    # Exact MA -> 1 limit: F = (1 + lambda)^(-3/2).
    if xi < 1.0e-4:
        return (1.0 + lam) ** (-1.5)

    s = np.sqrt(1.0 + lam)
    denominator = 1.0 - np.arctan(xi) / xi
    numerator = 1.0 / s - np.arctan(xi / s) / xi
    return numerator / denominator


def dF_dlambda(lam, MA):
    """Analytic derivative dF/dlambda for the exterior flux field."""
    lam = np.asarray(lam, dtype=float)
    xi = np.sqrt(MA ** -4 - 1.0)

    if xi < 1.0e-4:
        return -1.5 * (1.0 + lam) ** (-2.5)

    s = np.sqrt(1.0 + lam)
    denominator = 1.0 - np.arctan(xi) / xi
    return -xi ** 2 / (
            2.0 * s ** 3 * (s ** 2 + xi ** 2) * denominator
    )


def parallel_density_and_flux(rho, z, a, MA, n0=0.0):
    """Density and diffusive CR flux for B || grad(n0).

    R = G0 = 1.  Returned fluxes are J/D_parallel, so outside
        J_rho/D_parallel = -MA^4 d n/d rho,
        J_z/D_parallel   = -d n/d z,
    while inside J/D_parallel = -a grad(n).
    """
    rho_arr, z_arr = np.broadcast_arrays(
        np.asarray(rho, dtype=float), np.asarray(z, dtype=float)
    )
    shape = rho_arr.shape
    rho_flat = rho_arr.ravel()
    z_flat = z_arr.ravel()
    r2 = rho_flat ** 2 + z_flat ** 2

    density = np.empty_like(rho_flat)
    J_rho = np.empty_like(rho_flat)
    J_z = np.empty_like(rho_flat)

    beta = Nabla_rho_ratio_parallel(MA, a)
    inside = r2 <= 1.0
    outside = ~inside

    # Uniform internal gradient: grad(n)_in = beta e_z.
    density[inside] = n0 + beta * z_flat[inside]
    J_rho[inside] = 0.0
    J_z[inside] = -a * beta

    if np.any(outside):
        ro = rho_flat[outside]
        zo = z_flat[outside]
        lam = lambda_function(ro, zo, MA)
        F = F_lambda(lam, MA)
        Fp = dF_dlambda(lam, MA)
        q = beta - 1.0

        density[outside] = n0 + zo * (1.0 + q * F)

        m4 = MA ** 4
        denom = (
                m4 * ro ** 2 / (1.0 + m4 * lam) ** 2
                + zo ** 2 / (1.0 + lam) ** 2
        )
        dlambda_drho = (2.0 * ro / (1.0 + m4 * lam)) / denom
        dlambda_dz = (2.0 * zo / (1.0 + lam)) / denom

        dn_drho = zo * q * Fp * dlambda_drho
        dn_dz = 1.0 + q * F + zo * q * Fp * dlambda_dz

        J_rho[outside] = -m4 * dn_drho
        J_z[outside] = -dn_dz

    return (
        density.reshape(shape),
        J_rho.reshape(shape),
        J_z.reshape(shape),
    )


def N_parallel(rho, z, a, MA, n0=0.0):
    """Convenience wrapper returning only the parallel-case density."""
    density, _, _ = parallel_density_and_flux(rho, z, a, MA, n0=n0)
    return density


def parallel_density_and_flux_cartesian(x, y, z, a, MA, n0=0.0):
    """Cartesian form of the unit parallel-gradient density and CR flux.

    The underlying parallel solution is axisymmetric about ``B || e_z`` and
    returns a cylindrical radial flux.  This wrapper converts that radial
    component into ``(J_x, J_y)`` so it can be superposed component by
    component with the perpendicular-gradient solution.
    """
    x_arr, y_arr, z_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    rho = np.hypot(x_arr, y_arr)
    density, J_rho, J_z = parallel_density_and_flux(
        rho, z_arr, a, MA, n0=n0
    )

    radial_x = np.divide(
        x_arr,
        rho,
        out=np.zeros_like(rho),
        where=rho > 0.0,
    )
    radial_y = np.divide(
        y_arr,
        rho,
        out=np.zeros_like(rho),
        where=rho > 0.0,
    )
    J_x = J_rho * radial_x
    J_y = J_rho * radial_y
    return density, J_x, J_y, J_z


def parallel_spatial_amplification(rho, z, a, MA):
    """Spatial gradient and dipole-transport amplification factors.

    With R = G0 = 1,

        gradient_ratio = |grad n| / |grad n_0| = |grad n|,

    and

        dipole_ratio = |D grad n| / (D_parallel |grad n_0|).

    The latter is the dipole-anisotropy ratio when the density factor in
    delta = 3 |D grad n|/(c n) cancels.  It is also exactly |J|/|J_0|.
    """
    rho_arr, z_arr = np.broadcast_arrays(
        np.asarray(rho, dtype=float), np.asarray(z, dtype=float)
    )
    density, J_rho, J_z = parallel_density_and_flux(rho_arr, z_arr, a, MA)

    inside = rho_arr ** 2 + z_arr ** 2 <= 1.0
    dn_drho = np.empty_like(density)
    dn_dz = np.empty_like(density)

    # Recover grad(n) from J = -D grad(n).  G0 = 1.
    dn_drho[inside] = -J_rho[inside] / a
    dn_dz[inside] = -J_z[inside] / a

    outside = ~inside
    dn_drho[outside] = -J_rho[outside] / MA ** 4
    dn_dz[outside] = -J_z[outside]

    gradient_ratio = np.hypot(dn_drho, dn_dz)
    dipole_ratio = np.hypot(J_rho, J_z)
    return gradient_ratio, dipole_ratio


def L_perp_function(MA):
    """Depolarization factor along the imposed perpendicular gradient."""
    return (1.0 - L_z_function(MA)) / 2.0


def F_lambda_perp(lam, MA):
    """Exterior shape function for B perpendicular to the background gradient."""
    lam = np.asarray(lam, dtype=float)
    xi = np.sqrt(MA ** -4 - 1.0)

    # Isotropic limit: the parallel and perpendicular shape functions coincide.
    if xi < 1.0e-4:
        return (1.0 + lam) ** (-1.5)

    s = np.sqrt(1.0 + lam)
    denominator = np.arctan(xi) - xi / (1.0 + xi ** 2)
    numerator = np.arctan(xi / s) - xi * s / (s ** 2 + xi ** 2)
    return numerator / denominator


def dF_dlambda_perp(lam, MA):
    """Analytic derivative of F_lambda_perp."""
    lam = np.asarray(lam, dtype=float)
    xi = np.sqrt(MA ** -4 - 1.0)

    if xi < 1.0e-4:
        return -1.5 * (1.0 + lam) ** (-2.5)

    s = np.sqrt(1.0 + lam)
    denominator = np.arctan(xi) - xi / (1.0 + xi ** 2)
    return -xi ** 3 / (s * (s ** 2 + xi ** 2) ** 2 * denominator)


def perpendicular_density_and_flux(x, y, z, a, MA, n0=0.0):
    """Density and 3-D diffusive CR flux for B perpendicular to grad(n_0).

    Coordinates are chosen so that
        B || e_z,  grad(n_0) || e_x,
    with R = G0 = 1.  Returned J components are in units of
    D_parallel * G0.
    """
    x_arr, y_arr, z_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    shape = x_arr.shape
    xf = x_arr.ravel()
    yf = y_arr.ravel()
    zf = z_arr.ravel()
    r2 = xf ** 2 + yf ** 2 + zf ** 2

    density = np.empty_like(xf)
    J_x = np.empty_like(xf)
    J_y = np.empty_like(xf)
    J_z = np.empty_like(xf)

    beta = Nabla_rho_ratio_perp(MA, a)
    inside = r2 <= 1.0
    outside = ~inside

    # Inside: n = n0 + beta_perp * x and grad(n) = beta_perp e_x.
    density[inside] = n0 + beta * xf[inside]
    J_x[inside] = -a * beta
    J_y[inside] = 0.0
    J_z[inside] = 0.0

    if np.any(outside):
        xo = xf[outside]
        yo = yf[outside]
        zo = zf[outside]
        rho = np.hypot(xo, yo)
        lam = lambda_function(rho, zo, MA)
        F = F_lambda_perp(lam, MA)
        Fp = dF_dlambda_perp(lam, MA)
        q = beta - 1.0
        H = 1.0 + q * F

        density[outside] = n0 + xo * H

        m4 = MA ** 4
        rho2 = xo ** 2 + yo ** 2
        denom = (
                m4 * rho2 / (1.0 + m4 * lam) ** 2
                + zo ** 2 / (1.0 + lam) ** 2
        )
        dlambda_dx = (2.0 * xo / (1.0 + m4 * lam)) / denom
        dlambda_dy = (2.0 * yo / (1.0 + m4 * lam)) / denom
        dlambda_dz = (2.0 * zo / (1.0 + lam)) / denom

        dn_dx = H + xo * q * Fp * dlambda_dx
        dn_dy = xo * q * Fp * dlambda_dy
        dn_dz = xo * q * Fp * dlambda_dz

        J_x[outside] = -m4 * dn_dx
        J_y[outside] = -m4 * dn_dy
        J_z[outside] = -dn_dz

    return (
        density.reshape(shape),
        J_x.reshape(shape),
        J_y.reshape(shape),
        J_z.reshape(shape),
    )


def N_perpendicular(x, y, z, a, MA, n0=0.0):
    """Convenience wrapper returning only the perpendicular-case density."""
    density, _, _, _ = perpendicular_density_and_flux(x, y, z, a, MA, n0=n0)
    return density


def perpendicular_spatial_amplification(x, y, z, a, MA):
    """Full 3-D gradient and dipole-anisotropy ratios at (x, y, z).

    gradient_ratio = |grad n| / |grad n_0|.

    dipole_ratio is normalized to the *original perpendicular* dipole:

        |D grad n| / (D_perp |grad n_0|)
        = |J| / (MA^4 D_parallel G0).

    Thus both ratios approach 1 in the undisturbed far field.
    """
    x_arr, y_arr, z_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    _, J_x, J_y, J_z = perpendicular_density_and_flux(
        x_arr, y_arr, z_arr, a, MA
    )

    inside = x_arr ** 2 + y_arr ** 2 + z_arr ** 2 <= 1.0
    dn_dx = np.empty_like(J_x)
    dn_dy = np.empty_like(J_y)
    dn_dz = np.empty_like(J_z)

    dn_dx[inside] = -J_x[inside] / a
    dn_dy[inside] = -J_y[inside] / a
    dn_dz[inside] = -J_z[inside] / a

    outside = ~inside
    dn_dx[outside] = -J_x[outside] / MA ** 4
    dn_dy[outside] = -J_y[outside] / MA ** 4
    dn_dz[outside] = -J_z[outside]

    gradient_ratio = np.sqrt(dn_dx ** 2 + dn_dy ** 2 + dn_dz ** 2)
    dipole_ratio = np.sqrt(J_x ** 2 + J_y ** 2 + J_z ** 2) / MA ** 4
    return gradient_ratio, dipole_ratio


def _validate_perp_to_parallel_ratio(perp_to_parallel_ratio):
    """Return a finite scalar eta = G_perp/G_parallel."""
    eta = float(perp_to_parallel_ratio)
    if not np.isfinite(eta):
        raise ValueError("perp_to_parallel_ratio must be finite.")
    return eta


def arbitrary_background_norms(MA, perp_to_parallel_ratio):
    """Undisturbed total-gradient and total-flux norms.

    The dimensionless convention is G_parallel = D_parallel = 1, while
    D_perp/D_parallel = MA**4.  Hence

        |grad n_0| = sqrt(1 + eta**2),
        |J_0|      = sqrt(1 + MA**8 eta**2).
    """
    if not (0 < MA <= 1):
        raise ValueError("This implementation assumes 0 < MA <= 1.")
    eta = _validate_perp_to_parallel_ratio(perp_to_parallel_ratio)
    gradient_norm = np.hypot(1.0, eta)
    flux_norm = np.hypot(1.0, MA ** 4 * eta)
    return gradient_norm, flux_norm


def arbitrary_density_and_flux(
        x,
        y,
        z,
        a,
        MA,
        perp_to_parallel_ratio=1.0,
        n0=0.0,
):
    """Density and 3-D CR flux for an arbitrary background-gradient angle.

    Let eta = G_perp/G_parallel and use G_parallel = 1.  Linearity of the
    diffusion equation and of both interface conditions gives the exact
    superposition

        n = n0 + Phi_parallel + eta Phi_perpendicular,
        J = J_parallel + eta J_perpendicular.

    The perpendicular direction is chosen as +e_x without loss of generality,
    because the unperturbed transport tensor is axisymmetric about B || e_z.
    Returned flux components are in units of D_parallel G_parallel.
    """
    eta = _validate_perp_to_parallel_ratio(perp_to_parallel_ratio)
    x_arr, y_arr, z_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )

    n_parallel, Jpx, Jpy, Jpz = parallel_density_and_flux_cartesian(
        x_arr, y_arr, z_arr, a, MA, n0=0.0
    )
    n_perp, Jtx, Jty, Jtz = perpendicular_density_and_flux(
        x_arr, y_arr, z_arr, a, MA, n0=0.0
    )

    density = n0 + n_parallel + eta * n_perp
    J_x = Jpx + eta * Jtx
    J_y = Jpy + eta * Jty
    J_z = Jpz + eta * Jtz
    return density, J_x, J_y, J_z


def N_arbitrary(
        x,
        y,
        z,
        a,
        MA,
        perp_to_parallel_ratio=1.0,
        n0=0.0,
):
    """Convenience wrapper returning only the arbitrary-direction density."""
    density, _, _, _ = arbitrary_density_and_flux(
        x,
        y,
        z,
        a,
        MA,
        perp_to_parallel_ratio=perp_to_parallel_ratio,
        n0=n0,
    )
    return density


def arbitrary_spatial_amplification(
        x,
        y,
        z,
        a,
        MA,
        perp_to_parallel_ratio=1.0,
):
    """Total gradient and dipole-transport amplification factors.

    Vector components are superposed first.  The norms are then divided by the
    undisturbed *total* magnitudes, rather than by either directional component:

        A_grad  = |grad n| / sqrt(1 + eta**2),
        A_delta = |J| / sqrt(1 + MA**8 eta**2).

    ``A_delta`` is the transport factor A_delta^*.  The exact anisotropy ratio
    including the density denominator is A_delta^* n_infinity/n.
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    eta = _validate_perp_to_parallel_ratio(perp_to_parallel_ratio)
    x_arr, y_arr, z_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    _, J_x, J_y, J_z = arbitrary_density_and_flux(
        x_arr,
        y_arr,
        z_arr,
        a,
        MA,
        perp_to_parallel_ratio=eta,
    )

    inside = x_arr ** 2 + y_arr ** 2 + z_arr ** 2 <= 1.0
    dn_dx = np.empty_like(J_x)
    dn_dy = np.empty_like(J_y)
    dn_dz = np.empty_like(J_z)

    # Recover the complete gradient vector from J = -D grad(n).
    dn_dx[inside] = -J_x[inside] / a
    dn_dy[inside] = -J_y[inside] / a
    dn_dz[inside] = -J_z[inside] / a

    outside = ~inside
    dn_dx[outside] = -J_x[outside] / MA ** 4
    dn_dy[outside] = -J_y[outside] / MA ** 4
    dn_dz[outside] = -J_z[outside]

    background_gradient_norm, background_flux_norm = arbitrary_background_norms(
        MA, eta
    )
    gradient_ratio = (
            np.sqrt(dn_dx ** 2 + dn_dy ** 2 + dn_dz ** 2)
            / background_gradient_norm
    )
    dipole_ratio = (
            np.sqrt(J_x ** 2 + J_y ** 2 + J_z ** 2)
            / background_flux_norm
    )
    return gradient_ratio, dipole_ratio


def delta_angle(a=0.01, MA=0.3, gradient_ratio=1):
    beta_perp = Nabla_rho_ratio_perp(MA, a)
    beta_parallel = Nabla_rho_ratio_parallel(MA, a)
    return np.arctan(beta_perp / beta_parallel * gradient_ratio) - np.arctan(MA ** 4 * gradient_ratio)
