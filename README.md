# Low-Diffusion Dipole Explorer

This standalone interactive application was extracted from the CygBubble project. It is designed to investigate how a spherical low-diffusion region embedded in a uniform ordered magnetic field changes the amplitude, direction, and right-ascension phase of the background cosmic-ray dipole.

This directory can be used directly as the root of a GitHub repository. It does not depend on the original CygBubble project and does not require numerical simulation grids, NPZ files, magnetic-field archives, or native solvers.

![Explorer preview](assets/explorer_preview.png)

## Features

- Interactively adjust the Galactic longitude, Galactic latitude, and magnitude of the background relative gradient.
- Interactively adjust the ordered magnetic-field direction, Alfvénic Mach number, and low-diffusion coefficient ratio.
- Specify the observation point using fixed Galactic coordinates `r/R, l, b`.
- Display a Galactic sky map, an analytic two-dimensional slice, the dipole amplitude, and the right-ascension phase.
- Record successive slider updates or continuously scan one selected parameter.
- Display an equivalent-energy secondary x-axis when scanning `D_low/D_parallel`.
- Optionally overlay the experimental amplitude and phase measurements digitized from Figure 3 of Li et al. (2024).

## Requirements

- Python 3.10 or later
- NumPy
- Matplotlib

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running the Explorer

From the repository root, run:

```bash
python LowDiffusionDipoleExplorer.py
```

The file can also be run directly from PyCharm. The application requires a graphical desktop. A headless Linux server therefore needs X11 forwarding, VNC, or another graphical session.

## Overlaying Observational Data

To display the digitized observations:

1. Set `Curve mode` to `parameter scan`.
2. Select `Dlow / Dparallel` as the scan parameter.
3. Enable `Li+2024 data (E scan)` at the top of the window.

The observational energies are mapped to the primary x-axis through

```text
a(E) = A_at_10TeV * (E / 10 TeV) ** Delta_low_parallel
```

The data points are displayed only when this energy conversion is valid and `Dlow / Dparallel` is the active scan parameter. The measured amplitude is the first harmonic in right ascension and should therefore be compared primarily with the `RA-projected amplitude` curve.

## Configuration

- Window dimensions, analytic-slice resolution, scan sample count, and amplitude-axis limits are defined near the top of `LowDiffusionDipoleExplorer.py`.
- Diffusion coefficients, energy power-law indices, and the coordinate-projection matrix are defined in `cygbubble/config.py`.
- The digitized Li et al. data are stored in `data/Li_2024_Figure3_vector_digitized.csv`.

## Repository Structure

```text
.
├── LowDiffusionDipoleExplorer.py       # Interactive application entry point
├── cygbubble/
│   ├── config.py                       # Physical constants and model parameters
│   ├── coordinates.py                  # Galactic, equatorial, and Cartesian transforms
│   ├── observations.py                 # Li et al. CSV loading and validation
│   ├── analysis/
│   │   └── low_diffusion_dipole.py     # Physical dipole direction and amplitude
│   ├── analytic/
│   │   └── sphere.py                   # Analytic spherical low-diffusion solution
│   └── physics/
│       └── diffusion.py                # Energy and diffusion-coefficient conversions
├── data/
│   └── Li_2024_Figure3_vector_digitized.csv
├── tests/
│   └── test_smoke.py
├── assets/
│   └── explorer_preview.png
└── requirements.txt
```

## Validation

Run the included tests with:

```bash
python -m unittest discover -s tests
```

## Observational Data Source

The observational CSV was extracted from the PDF vector objects in Figure 3 of Li et al. (2024), *The Astrophysical Journal*, **962**, 43. It contains 44 amplitude measurements and 43 phase measurements from 11 experiments. These values are figure-digitization results rather than official data tables released by the experimental collaborations.

Please cite the corresponding experimental publications when using these data.
