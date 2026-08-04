# QWP forward simulations

This directory contains the Meep forward-simulation scripts used to verify the
final inverse-designed visible quarter-wave plate (QWP) after optimization.
The optimization and adjoint-gradient code is not included.

## Included scripts

- `qwp_model.py` — shared TiO2 dispersion, geometry, z-stack, source, and
  `MaterialGrid` definitions.
- `calculate_jones_matrix.py` — Jones-matrix extraction and optional
  polarization-conversion checks.
- `calculate_field_profiles.py` — Figure 6(a) real-field profiles and optional
  electric-field amplitude profiles.

Both forward scripts use the same physical model:

- unit cell: `0.2 um x 0.2 um`;
- design thickness: `0.6 um`;
- finite SiO2 substrate: `0.5 um`, `n = 1.45`;
- x/y periodic boundaries and z-directed PMLs;
- final two-pole dispersive TiO2 Lorentz model;
- `MaterialGrid(grid_type="U_MEAN", do_averaging=False)`;
- the center-first z-grid snapping used in the final verification.

## Input structure

Run the commands below from the repository root. The released structure is:

```text
data/optimized_qwp_binary_structure_3d.npy
```

The expected array shape is `(11, 11, 31)` with axis order `(x, y, z)`,
C-order reshape convention, `0 = air`, and `1 = TiO2`.

## Software environment

The manuscript calculations used:

```text
Python 3.10.20
Meep 1.33.0-beta
NumPy 1.26.4
MPICH 4.3.2 (HYDRA)
```

A working parallel Meep installation with `mpi4py` and Matplotlib is required.

## Jones-matrix verification

Single-wavelength test at `550 nm`:

```bash
mpirun -np 14 python scripts/forward_simulation/calculate_jones_matrix.py \
  --rho data/optimized_qwp_binary_structure_3d.npy \
  --output outputs/forward_simulation/jones_test_550nm \
  --mode jones \
  --wavelengths 550
```

Full `500–600 nm` calculation at `1 nm` intervals:

```bash
mpirun -np 14 python scripts/forward_simulation/calculate_jones_matrix.py \
  --rho data/optimized_qwp_binary_structure_3d.npy \
  --output outputs/forward_simulation/jones_matrix \
  --mode jones
```

Available modes:

```text
jones  complex Jones matrix from x/y incidence
lcp    LCP -> +45-degree linear-polarization metrics
p45    +45-degree linear -> circular-polarization metrics
all    run all three blocks
```

The extracted matrix is

```text
J_xy = [[t_xx, t_xy],
        [t_yx, t_yy]]
```

The x/y transmission coefficients are normalized by separate air-only
reference simulations. The LCP source convention is `Ex = 1, Ey = -i`, and the
`+45-degree` linear basis is `(x + y)/sqrt(2)`.

## Figure 6(a) field-profile calculation

The default `real` mode calculates the field definition used in Figure 6(a):

```text
Re(Ex(x,y=0,z)) under x-polarized incidence
Re(Ey(x,y=0,z)) under y-polarized incidence
```

Run:

```bash
mpirun -np 14 python scripts/forward_simulation/calculate_field_profiles.py \
  --rho data/optimized_qwp_binary_structure_3d.npy \
  --output outputs/forward_simulation/field_profiles_real \
  --wavelengths 500 550 600 \
  --field-quantity real
```

The real-field maps use a symmetric zero-centered scale and the `RdBu_r`
colormap by default. The x-polarized maps share one scale across wavelengths;
the y-polarized maps share a separate scale.

To calculate the previous amplitude representation instead:

```bash
mpirun -np 14 python scripts/forward_simulation/calculate_field_profiles.py \
  --rho data/optimized_qwp_binary_structure_3d.npy \
  --output outputs/forward_simulation/field_profiles_amplitude \
  --wavelengths 500 550 600 \
  --field-quantity amplitude
```

The script saves quantity-specific PNG, PDF, SVG, and NPZ files together with
`simulation_info.json` and the rho arrays used for the simulation.

## Reproducibility note

`qwp_model.py` is the single source of truth for the physical model. Both
forward scripts import the same TiO2 Lorentz parameters, snapped z-stack, and
MaterialGrid settings to prevent silent differences between Jones-matrix and
field-profile calculations.
