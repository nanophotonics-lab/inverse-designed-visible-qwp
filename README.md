# Inverse-designed visible QWP structure

This repository provides the final binary optimized structure and the Python
visualization script associated with the inverse-designed visible-light
quarter-wave plate (QWP).

## Repository contents

```text
inverse-designed-visible-qwp/
├── README.md
├── requirements.txt
├── data/
│   ├── optimized_qwp_binary_structure_3d.npy
│   ├── optimized_qwp_binary_structure_3d.txt
│   └── optimized_qwp_binary_structure_3d_indexed.csv
└── scripts/
    └── plot_optimized_qwp_structure.py
```

## Structure data

The final binary structure has the following definition:

- Physical size: `0.2 µm × 0.2 µm × 0.6 µm`
- Resolution: `50 pixels/µm`
- NumPy array shape: `(11, 11, 31)`
- NumPy axis order: `(x, y, z)`
- NumPy flatten/reshape convention: C-order
- Material values:
  - `0`: air
  - `1`: TiO2

### NumPy file

`data/optimized_qwp_binary_structure_3d.npy` stores the structure as a
three-dimensional array with axis order `(x, y, z)`.

### Plain-text file

`data/optimized_qwp_binary_structure_3d.txt` stores consecutive constant-z
cross-sections. In each block:

- columns correspond to x indices;
- rows correspond to y indices;
- the first row is `y_index = 0`;
- the last row is `y_index = 10`;
- the block is equivalent to `structure[:, :, z_index].T`.

Thus, the horizontal and vertical directions represent x and y, respectively.

### Indexed CSV file

`data/optimized_qwp_binary_structure_3d_indexed.csv` provides the same data as
an explicit table with columns `ix`, `iy`, `iz`, and `rho`.

## Python environment

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Plotting the structure

Run the script from the repository root:

```bash
python scripts/plot_optimized_qwp_structure.py
```

Plot a selected constant-z cross-section:

```bash
python scripts/plot_optimized_qwp_structure.py --axis z --index 15
```

Load the plain-text file instead:

```bash
python scripts/plot_optimized_qwp_structure.py \
  --file data/optimized_qwp_binary_structure_3d.txt
```

The script saves the plotted cross-section as a vector PDF.

## Citation

When using these data or scripts, please cite the associated manuscript:

> [Manuscript citation to be added after publication.]

## Contact

Electromagnetic and Intelligence Design Laboratory (EIDL), Hanyang University.
