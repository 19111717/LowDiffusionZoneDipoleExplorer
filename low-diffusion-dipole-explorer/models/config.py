"""Physical constants and model parameters used by the standalone explorer."""

from __future__ import annotations


PC_TO_CM = 3.0856775814913673e18
C_LIGHT_CM_S = 2.99792458e10

# Project Cartesian basis -> J2000 equatorial Cartesian basis.  The output
# axes point toward RA=0 h, RA=6 h, and the north celestial pole.
SIMULATION_TO_EQUATORIAL_J2000 = (
    (0.0548755604162154, 0.4941094278755837, -0.8676661490190047),
    (0.8734370902348850, -0.4448296299600112, -0.1980763734312015),
    (0.4838350155487132, 0.7469822444972189, 0.4559837761750669),
)

# Parallel diffusion coefficient at 10 TeV and its energy index.
D_0_10TeV = 1.0e30  # cm^2/s
delta_Diffuse = 0.24

# D_parallel/D_perpendicular energy relation.
Delta_delta = 0.3
anisotropic_ratio_at_10TeV = 0.25 ** (-4)

# a(E) = D_low/D_parallel at 10 TeV and its relative energy index.
Delta_low_parallel = 0.7
A_at_10TeV = 1.0e-2

# IBEX local interstellar magnetic-field direction, retained for experiments
# that want to use it as the initial field direction.
b_IBEX = -57.1
l_IBEX = 210.5

