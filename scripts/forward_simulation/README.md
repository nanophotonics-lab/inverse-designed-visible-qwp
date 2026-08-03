# QWP forward simulations

This directory contains the Meep forward-simulation scripts used to verify the
final inverse-designed visible quarter-wave plate (QWP) after optimization.
The optimization and adjoint-gradient code is not included.

## Files

- `qwp_model.py`: shared TiO2 dispersion, geometry, snapped z-stack, source,
  and `MaterialGrid` definitions.
- `calculate_jones_matrix.py`: Jones-matrix extraction and optional
  polarization-conversion checks.
- `calculate_field_profiles.py`: y=0 electric-field amplitude maps for x- and
  y-polarized incidence.

Both simulation scripts import `qwp_model.py` and therefore use the same:

- 200 nm x 200 nm unit cell;
- 600 nm design thickness;
- 500 nm SiO2 substrate with n = 1.45;
- x/y periodic boundaries and z-directed PMLs;
- two-pole dispersive TiO2 Lorentz model;
- `MaterialGrid(grid_type="U_MEAN", do_averaging=False)`;
- center-first z-grid snapping used in the final verification.

## Input structure

Run the commands below from the repository root. The released free-form
structure is:

```text
data/optimized_qwp_binary_structure_3d.npy
```

Its expected shape is `(11, 11, 31)` with axis order `(x, y, z)`, C-order
reshape convention, `0 = air`, and `1 = TiO2`.

Because this file is already binary, do not add `--binarize` unless an
intentional second thresholding operation is desired.

## Software environment

The manuscript calculations used:

```text
Python 3.10.20
Meep 1.33.0-beta
NumPy 1.26.4
MPICH 4.3.2 (HYDRA)
```

A parallel Meep installation with `mpi4py`, NumPy, and Matplotlib is required.
Meep should be installed through a compatible Conda environment or from source;
it is not treated as an ordinary pip-only dependency here.

## Jones-matrix verification

Run the full 500-600 nm calculation in 1 nm increments:

```bash
mpirun -np 14 python scripts/forward_simulation/calculate_jones_matrix.py \
  --rho data/optimized_qwp_binary_structure_3d.npy \
  --output outputs/forward_simulation/jones_matrix \
  --mode jones
```

A single-wavelength test at 550 nm:

```bash
mpirun -np 14 python scripts/forward_simulation/calculate_jones_matrix.py \
  --rho data/optimized_qwp_binary_structure_3d.npy \
  --output outputs/forward_simulation/jones_test_550nm \
  --mode jones \
  --wavelengths 550
```

Available modes:

```text
jones  complex Jones matrix from x/y incidence
lcp    LCP to +45-degree linear-polarization metrics
p45    +45-degree linear to circular-polarization metrics
all    run all three blocks
```

The extracted matrix is stored as

```text
J_xy = [[t_xx, t_xy],
        [t_yx, t_yy]]
```

The coefficients are normalized by separate air-only reference simulations.
The circular-polarization convention used in the optimization is
`Ex = 1, Ey = -i` for LCP incidence, and the +45-degree linear basis is
`(x + y)/sqrt(2)`.

## Field-profile calculation

```bash
mpirun -np 14 python scripts/forward_simulation/calculate_field_profiles.py \
  --rho data/optimized_qwp_binary_structure_3d.npy \
  --output outputs/forward_simulation/field_profiles \
  --wavelengths 500 550 600
```

For each wavelength, the script calculates:

```text
|Ex(x, y=0, z)| under x-polarized incidence
|Ey(x, y=0, z)| under y-polarized incidence
```

The displayed z coordinate is shifted so that the design region spans
0-0.6 um. The x-polarized maps share one normalization scale across
wavelengths, while the y-polarized maps share a separate scale. Raw arrays are
saved in the output NPZ file.

## Reproducibility note

`qwp_model.py` is the single source of truth for the physical model. The Jones
and field-profile scripts should remain in the same directory as
`qwp_model.py`.
