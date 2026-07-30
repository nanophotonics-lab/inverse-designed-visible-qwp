# Visible QWP design structures

This repository provides the final structures and visualization scripts for
three visible-light quarter-wave-plate (QWP) design spaces evaluated in the
associated manuscript:

1. fully free-form inverse design;
2. five-layer constrained inverse design;
3. single-layer rectangular nanopillar baseline.

## Repository contents

```text
inverse-designed-visible-qwp/
├── README.md
├── requirements.txt
├── data/
│   ├── optimized_qwp_binary_structure_3d.npy
│   ├── optimized_qwp_binary_structure_3d.txt
│   ├── optimized_qwp_binary_structure_3d_indexed.csv
│   ├── five_layer/
│   │   ├── five_layer_qwp_binary_structure_3d.npy
│   │   ├── five_layer_qwp_binary_structure_3d.txt
│   │   ├── five_layer_qwp_binary_structure_3d_indexed.csv
│   │   └── five_layer_stack_definition.csv
│   └── single_layer/
│       ├── README.md
│       └── single_layer_nanopillar_parameters.csv
└── scripts/
    ├── plot_optimized_qwp_structure.py
    ├── plot_five_layer_qwp_structure.py
    └── plot_single_layer_nanopillar.py
```

## Common simulation geometry

- Unit-cell period: `200 nm × 200 nm`
- Total device height: `600 nm`
- Substrate: `500-nm-thick SiO2`
- High-index material: dispersive `TiO2`
- Target wavelength range: `500–600 nm`

## 1. Fully free-form inverse design

The files in the top level of `data/` store the final binary free-form
structure.

- Physical size: `0.2 µm × 0.2 µm × 0.6 µm`
- Resolution: `50 pixels/µm`
- NumPy shape: `(11, 11, 31)`
- Axis order: `(x, y, z)`
- Flatten/reshape convention: C-order
- Values: `0 = air`, `1 = TiO2`

Plot it from the repository root:

```bash
python scripts/plot_optimized_qwp_structure.py
```

## 2. Five-layer constrained inverse design

The five-layer structure is stored in `data/five_layer/` as a binary
three-dimensional array.

- NumPy shape: `(11, 11, 31)`
- Axis order: `(x, y, z)`
- Flatten/reshape convention: C-order
- Values: `0 = air`, `1 = TiO2`

The bottom-to-top stack is:

```text
P1 / I1 / P2 / I2 / P3 / I3 / P4 / I4 / P5
```

Patterned-layer thicknesses are `100, 100, 120, 100, and 100 nm`.
Each fixed TiO2 interlayer is `20 nm` thick and `160 nm` wide along x.
The detailed index ranges are provided in
`data/five_layer/five_layer_stack_definition.csv`.

Plot a cross-section:

```bash
python scripts/plot_five_layer_qwp_structure.py
```

Example x–z cross-section:

```bash
python scripts/plot_five_layer_qwp_structure.py --axis y --index 5
```

## 3. Single-layer rectangular nanopillar

The single-layer baseline is defined analytically by:

- `wx = 60 nm`
- `wy = 140 nm`
- height: `600 nm`
- period: `200 nm × 200 nm`

Its authoritative geometry definition is:

```text
data/single_layer/single_layer_nanopillar_parameters.csv
```

No voxelized `.npy` file is provided for this structure because an exact
parameter definition avoids grid-dependent geometric approximation.

Plot its top view:

```bash
python scripts/plot_single_layer_nanopillar.py
```

Plot an x–z side view:

```bash
python scripts/plot_single_layer_nanopillar.py --view xz
```

## Plain-text and indexed formats

For each voxelized structure, the TXT file stores consecutive constant-z
cross-sections. In every block:

- columns correspond to x indices;
- rows correspond to y indices;
- the block equals `structure[:, :, z_index].T`.

The indexed CSV files contain explicit `ix`, `iy`, `iz`, and `rho` columns.

## Python environment

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Citation

When using these structures or scripts, please cite the associated manuscript:

> [Manuscript citation to be added after publication.]

## Contact

Electromagnetic and Intelligence Design Laboratory (EIDL), Hanyang University.
