# Visible QWP design structures

This repository provides the final structure data, source data underlying
Figures 2--4, and a three-dimensional rendering script for the
visible-light quarter-wave-plate (QWP) design spaces evaluated in the
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
│   ├── source_data/
│   │   ├── source_data_Figure_2_optimization_history.csv
│   │   ├── source_data_Figure_3_LCP_response.csv
│   │   └── source_data_Figure_4_Jones_matrix.csv
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

## Source data for Figures 2--4

The `data/source_data/` directory contains the numerical data underlying the
optimization-history and spectral-analysis plots in Figures 2--4.

### Figure 2(a): Optimization history

```text
data/source_data/source_data_Figure_2_optimization_history.csv
```

This file contains the recorded objective-function evaluations and related
optimization metrics over 140 iterations. The columns `global_iteration` and
`normalized_FOM` are included explicitly for reproducing the blue normalized
FOM curve in Figure 2(a), where

```text
normalized_FOM = F_avg / max(F_avg)
```

across the 140 recorded evaluations. The remaining columns preserve the
original optimization log, including the beta-continuation stage, evaluation
number within each stage, objective value, gradient norm, step norm,
transmission-related metric, leakage-related metric, minimum objective value,
and average absolute phase error.

The black binarization-degree curve and the intermediate structure snapshots
shown in Figure 2(a) are not contained in this CSV.

### Figures 3 and 4: Spectral source data

The Figure 3 and Figure 4 files cover `500–600 nm` at `1 nm` intervals.

### Figure 3: LCP response

```text
data/source_data/source_data_Figure_3_LCP_response.csv
```

This file contains the spectral response under left-circularly polarized
(LCP) incidence, including:

- total transmitted power normalized to the air-reference flux;
- projected transmission into the target `+45°` linear-polarization channel;
- overlaps with linear and circular analyzer bases;
- signed phase mismatch between the transmitted `Ex` and `Ey` components.

Columns ending in `_percent` are reported in percent. `delta_deg` is the
signed phase mismatch in degrees. The columns `retardance_deg` and
`retardance_deg_plus90` are auxiliary quantities derived from the LCP-output
phase mismatch and should not be confused with the Jones-matrix retardance
reported for Figure 4.

### Figure 4: Jones-matrix analysis

```text
data/source_data/source_data_Figure_4_Jones_matrix.csv
```

This file contains the normalized complex Jones-matrix coefficients,

```text
t_xx, t_xy, t_yx, t_yy
```

stored as separate real and imaginary parts, together with their squared
magnitudes, diagonal phases, phase retardance, quarter-wave-plate error, and
the total transmitted powers for x- and y-polarized incidence.

The transmitted powers are evaluated as

```text
T_x = |t_xx|^2 + |t_yx|^2
T_y = |t_xy|^2 + |t_yy|^2
```

The Jones coefficients and squared magnitudes are dimensionless.
`phase_txx_rad` and `phase_tyy_rad` are in radians, whereas
`retardance_deg` and `qwp_error_deg` are in degrees.

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

When using these structures, source data, or scripts, please cite the
associated manuscript:

> [Manuscript citation to be added after publication.]

## Contact

Electromagnetic and Intelligence Design Laboratory (EIDL), Hanyang University.
