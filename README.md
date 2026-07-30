# Visible QWP design structures

This repository provides the final structure data and a three-dimensional
rendering script for the visible-light quarter-wave-plate (QWP) design spaces
evaluated in the associated manuscript:

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
    └── render_qwp_from_npy.py
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

The nominal patterned-layer thicknesses are `100, 100, 120, 100, and
100 nm`, and each fixed TiO2 interlayer has a nominal thickness of `20 nm`.

In the released 31-sample z-directed representation, the grid weights are
allocated as `5 / 1 / 5 / 1 / 7 / 1 / 5 / 1 / 5` for
`P1 / I1 / P2 / I2 / P3 / I3 / P4 / I4 / P5`, respectively.

Each interlayer is `160 nm` wide along x. Detailed index ranges are provided
in `data/five_layer/five_layer_stack_definition.csv`. The binary `.npy` file
is the authoritative numerical representation used for visualization and
reproduction.

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

No voxelized `.npy` file is provided for this structure because the exact
parameter definition avoids grid-dependent geometric approximation. The
NPY-based renderer described below therefore applies to the fully free-form
and five-layer structures.

## 3D unit-cell rendering

Install the required packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

Render the fully free-form structure:

```bash
python scripts/render_qwp_from_npy.py
```

Render the five-layer structure:

```bash
python scripts/render_qwp_from_npy.py \
  --input data/five_layer/five_layer_qwp_binary_structure_3d.npy \
  --prefix five_layer_qwp
```

Rendered images are saved in:

```text
outputs/renders/
```

For each camera azimuth, the script saves:

- `*_raw.png`: direct PyVista rendering;
- `*_tight.png`: rendering after white-margin removal;
- `*_final.png`: rendering after contrast, sharpening, and graphite-gray
  structure post-processing.

The physical SiO2 substrate thickness is `500 nm`. In the rendered image,
the displayed substrate thickness is reduced to `30%` of its physical value
solely for visual clarity. This visual scaling does not modify the released
structure data or the simulation geometry.

The default camera azimuths are `225°` and `315°`. Alternative views can be
specified using `--azimuths`, for example:

```bash
python scripts/render_qwp_from_npy.py --azimuths 225
```

Rendering details can vary slightly with the PyVista, VTK, and graphics
backend versions.

## Plain-text and indexed formats

For each voxelized structure, the TXT file stores consecutive constant-z
cross-sections. In every block:

- columns correspond to x indices;
- rows correspond to y indices;
- the block equals `structure[:, :, z_index].T`.

The indexed CSV files contain explicit `ix`, `iy`, `iz`, and `rho` columns.

## Citation

When using these structures or scripts, please cite the associated manuscript:

> [Manuscript citation to be added after publication.]

## Contact

Electromagnetic and Intelligence Design Laboratory (EIDL), Hanyang University.
