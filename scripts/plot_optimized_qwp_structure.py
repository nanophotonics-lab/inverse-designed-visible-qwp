#!/usr/bin/env python3
"""Plot cross-sections of the final optimized QWP structure."""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

SHAPE = (11, 11, 31)
LX_UM = 0.2
LY_UM = 0.2
LZ_UM = 0.6


def load_txt_slices(path: Path) -> np.ndarray:
    """Load TXT stored as transposed constant-z slices.

    In each TXT block:
      columns -> x index
      rows    -> y index
    """
    stacked = np.loadtxt(path, comments="#")
    expected_shape = (SHAPE[2] * SHAPE[1], SHAPE[0])

    if stacked.shape != expected_shape:
        raise ValueError(
            f"Expected stacked TXT shape {expected_shape}, got {stacked.shape}."
        )

    # Stored block order:
    # z = 0..30, each block is structure[:, :, z].T with shape (Ny, Nx)
    slices_zyx = stacked.reshape(SHAPE[2], SHAPE[1], SHAPE[0], order="C")

    # Convert (z, y, x) -> (x, y, z)
    return np.transpose(slices_zyx, (2, 1, 0))


def load_structure(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        structure = np.load(path, allow_pickle=False)
        if structure.shape == (np.prod(SHAPE),):
            structure = structure.reshape(SHAPE, order="C")
    elif suffix == ".txt":
        structure = load_txt_slices(path)
    else:
        raise ValueError("The input file must be .npy or .txt.")

    if structure.shape != SHAPE:
        raise ValueError(f"Expected structure shape {SHAPE}, got {structure.shape}.")

    return structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/optimized_qwp_binary_structure_3d.npy"),
        help="Path to the released .npy or .txt structure file.",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Axis normal to the plotted cross-section.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Slice index. The center slice is used by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optimized_qwp_structure_cross_section.pdf"),
    )
    args = parser.parse_args()

    structure = load_structure(args.file)
    axis_number = {"x": 0, "y": 1, "z": 2}[args.axis]
    index = structure.shape[axis_number] // 2 if args.index is None else args.index

    if not 0 <= index < structure.shape[axis_number]:
        raise IndexError(
            f"Index {index} is outside 0 to {structure.shape[axis_number] - 1}."
        )

    section = np.take(structure, index, axis=axis_number)

    if args.axis == "z":
        # section shape = (Nx, Ny); transpose for plotting as rows=y, columns=x
        image_data = section.T
        extent = [-LX_UM / 2, LX_UM / 2, -LY_UM / 2, LY_UM / 2]
        xlabel = "x (µm)"
        ylabel = "y (µm)"
    elif args.axis == "y":
        # section shape = (Nx, Nz); transpose for plotting as rows=z, columns=x
        image_data = section.T
        extent = [-LX_UM / 2, LX_UM / 2, -LZ_UM / 2, LZ_UM / 2]
        xlabel = "x (µm)"
        ylabel = "z (µm)"
    else:
        # section shape = (Ny, Nz); transpose for plotting as rows=z, columns=y
        image_data = section.T
        extent = [-LY_UM / 2, LY_UM / 2, -LZ_UM / 2, LZ_UM / 2]
        xlabel = "y (µm)"
        ylabel = "z (µm)"

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    image = ax.imshow(
        1.0 - image_data,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="gray",
    )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8.5)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("1 - material density", fontsize=9)
    colorbar.ax.tick_params(labelsize=8.5)

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded structure shape: {structure.shape}")
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
