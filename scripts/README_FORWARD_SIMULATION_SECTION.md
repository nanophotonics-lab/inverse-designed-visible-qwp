## Forward-simulation verification

The `scripts/forward_simulation/` directory contains the Meep forward codes
used after optimization to verify the released free-form QWP structure. The
provided scripts calculate:

- the complex x/y Jones matrix and QWP phase retardance;
- LCP-to-+45-degree polarization-conversion metrics; and
- y=0 electric-field amplitude profiles under x- and y-polarized incidence.

The Jones-matrix and field-profile scripts share the same dispersive TiO2
model, finite SiO2 substrate, snapped simulation geometry, and
`MaterialGrid(grid_type="U_MEAN", do_averaging=False)` definition through
`scripts/forward_simulation/qwp_model.py`.

Run a 550 nm Jones-matrix test from the repository root:

```bash
mpirun -np 14 python scripts/forward_simulation/calculate_jones_matrix.py \
  --rho data/optimized_qwp_binary_structure_3d.npy \
  --output outputs/forward_simulation/jones_test_550nm \
  --mode jones \
  --wavelengths 550
```

Run the field profiles used for representative 500, 550, and 600 nm maps:

```bash
mpirun -np 14 python scripts/forward_simulation/calculate_field_profiles.py \
  --rho data/optimized_qwp_binary_structure_3d.npy \
  --output outputs/forward_simulation/field_profiles \
  --wavelengths 500 550 600
```

Detailed options and output definitions are provided in
`scripts/forward_simulation/README.md`.
