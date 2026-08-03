#!/usr/bin/env python3
"""Calculate y=0 electric-field amplitude profiles of the final QWP.

For each wavelength, the script calculates:
  - |Ex| at y=0 under x-polarized incidence, and
  - |Ey| at y=0 under y-polarized incidence.

The figure uses the design-region coordinate z=0...0.6 um and overlays the
voxelized TiO2/air boundary without contour interpolation. Raw and normalized
arrays are also saved for reuse.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from matplotlib.collections import LineCollection
from mpi4py import MPI

from qwp_model import (
    DESIGN_RESOLUTION,
    LX_UM,
    LY_UM,
    LZ_UM,
    PML_BOTTOM_UM,
    PML_TOP_UM,
    TIO2_EPS_INF,
    TIO2_LORENTZ_PARAMS,
    build_device_geometry,
    build_layout,
    build_material_grid,
    build_tio2_medium,
    choose_decay_component,
    load_rho,
    make_plane_sources,
    make_simulation,
)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
is_master = rank == 0
mp.verbosity(1 if is_master else 0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate y=0 field-amplitude profiles of the final QWP."
    )
    parser.add_argument(
        "--rho",
        type=Path,
        required=True,
        help="Full 3D rho .npy file, normally with shape (11, 11, 31).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/field_profiles"),
        help="Directory for figures, arrays, and metadata.",
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=[500.0, 550.0, 600.0],
        metavar="NM",
        help="Wavelengths in nm. Default: 500 550 600.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=50,
        help="Meep simulation resolution in pixels/um. Default: 50.",
    )
    parser.add_argument(
        "--design-resolution",
        type=int,
        default=DESIGN_RESOLUTION,
        help="Resolution used to define the stored rho grid. Default: 50.",
    )
    parser.add_argument(
        "--decay-tol",
        type=float,
        default=1e-5,
        help="Field-decay tolerance. Default: 1e-5.",
    )
    parser.add_argument(
        "--fwidth-frac",
        type=float,
        default=0.10,
        help="Gaussian source fwidth/frequency. Default: 0.10.",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Hard-threshold rho before simulation. Omit for an already binary rho.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used with --binarize. Default: 0.5.",
    )
    parser.add_argument(
        "--cmap",
        default="hot",
        help="Matplotlib colormap. Default: hot.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "bilinear"),
        default="bilinear",
        help="Field-map interpolation used only for display. Default: bilinear.",
    )
    parser.add_argument(
        "--vmin-percentile",
        type=float,
        default=0.0,
        help="Lower display percentile. Default: 0.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=100.0,
        help="Upper display percentile. Default: 100.",
    )
    parser.add_argument(
        "--no-structure-overlay",
        action="store_true",
        help="Do not draw the voxelized structure boundary.",
    )
    parser.add_argument(
        "--structure-fill",
        action="store_true",
        help="Add a faint fill inside the TiO2 voxels.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster output resolution. Default: 600 dpi.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> np.ndarray:
    wavelengths_nm = np.asarray(args.wavelengths, dtype=float)
    if wavelengths_nm.ndim != 1 or wavelengths_nm.size == 0:
        raise ValueError("at least one wavelength is required")
    if np.any(~np.isfinite(wavelengths_nm)) or np.any(wavelengths_nm <= 0):
        raise ValueError("wavelengths must be finite and positive")
    if args.resolution <= 0 or args.design_resolution <= 0:
        raise ValueError("resolutions must be positive")
    if args.decay_tol <= 0 or args.fwidth_frac <= 0:
        raise ValueError("decay tolerance and source bandwidth must be positive")
    if not 0 <= args.vmin_percentile < args.vmax_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= vmin < vmax <= 100")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    return wavelengths_nm


def safe_reset_sim(simulation: mp.Simulation) -> None:
    try:
        simulation.reset_meep()
    except Exception:
        pass


def squeeze_field(array: np.ndarray) -> np.ndarray:
    field = np.squeeze(np.asarray(array))
    if field.ndim != 3:
        raise ValueError(f"expected a 3D field array, received {field.shape}")
    return field


def x_sources(frequency: float, layout, fwidth_fraction: float) -> list[mp.Source]:
    return make_plane_sources(
        frequency,
        layout,
        1.0,
        0.0,
        fwidth_fraction=fwidth_fraction,
    )


def y_sources(frequency: float, layout, fwidth_fraction: float) -> list[mp.Source]:
    return make_plane_sources(
        frequency,
        layout,
        0.0,
        1.0,
        fwidth_fraction=fwidth_fraction,
    )


def run_field_volume(
    *,
    label: str,
    wavelength_nm: float,
    sources: list[mp.Source],
    layout,
    resolution: int,
    decay_tolerance: float,
    tio2: mp.Medium,
    material_grid: mp.MaterialGrid,
) -> dict[str, np.ndarray]:
    frequency = 1000.0 / float(wavelength_nm)
    if is_master:
        print(f"[FIELD] {label}, wavelength={wavelength_nm:g} nm")

    simulation = make_simulation(
        resolution=resolution,
        layout=layout,
        sources=sources,
        geometry=build_device_geometry(material_grid, layout),
        tio2=tio2,
    )
    dft_volume = simulation.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez],
        frequency,
        0,
        1,
        where=mp.Volume(
            center=mp.Vector3(0, 0, layout.z_structure_center),
            size=mp.Vector3(LX_UM, LY_UM, LZ_UM),
        ),
    )
    decay_component = choose_decay_component(sources)
    condition = mp.stop_when_fields_decayed(
        50,
        decay_component,
        mp.Vector3(0, 0, layout.z_monitor),
        decay_tolerance,
    )
    simulation.run(until_after_sources=lambda sim: condition(sim))

    fields = {
        "Ex": squeeze_field(simulation.get_dft_array(dft_volume, mp.Ex, 0)),
        "Ey": squeeze_field(simulation.get_dft_array(dft_volume, mp.Ey, 0)),
        "Ez": squeeze_field(simulation.get_dft_array(dft_volume, mp.Ez, 0)),
    }
    safe_reset_sim(simulation)
    del simulation
    gc.collect()
    comm.Barrier()
    return fields


def center_y_amplitude(field: np.ndarray) -> np.ndarray:
    """Return |E(x,y=0,z)| from a complex E(x,y,z) array."""

    y_index = field.shape[1] // 2
    return np.abs(field[:, y_index, :])


def calculate_maps_for_wavelength(
    wavelength_nm: float,
    *,
    layout,
    resolution: int,
    decay_tolerance: float,
    fwidth_fraction: float,
    tio2: mp.Medium,
    material_grid: mp.MaterialGrid,
) -> dict[str, np.ndarray | float]:
    frequency = 1000.0 / float(wavelength_nm)
    x_fields = run_field_volume(
        label="x-polarized incidence",
        wavelength_nm=wavelength_nm,
        sources=x_sources(frequency, layout, fwidth_fraction),
        layout=layout,
        resolution=resolution,
        decay_tolerance=decay_tolerance,
        tio2=tio2,
        material_grid=material_grid,
    )
    y_fields = run_field_volume(
        label="y-polarized incidence",
        wavelength_nm=wavelength_nm,
        sources=y_sources(frequency, layout, fwidth_fraction),
        layout=layout,
        resolution=resolution,
        decay_tolerance=decay_tolerance,
        tio2=tio2,
        material_grid=material_grid,
    )
    return {
        "wavelength_nm": float(wavelength_nm),
        "xpol_ex_y0_abs_xz": center_y_amplitude(x_fields["Ex"]),
        "ypol_ey_y0_abs_xz": center_y_amplitude(y_fields["Ey"]),
    }


def structure_center_y(rho_3d: np.ndarray) -> np.ndarray:
    y_index = rho_3d.shape[1] // 2
    return np.asarray(rho_3d[:, y_index, :], dtype=float)


def percentile_limits(arrays: list[np.ndarray], lower: float, upper: float) -> tuple[float, float]:
    values = [
        np.asarray(array, dtype=float)[np.isfinite(array)].ravel()
        for array in arrays
        if np.asarray(array).size
    ]
    values = [value for value in values if value.size]
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    vmin = float(np.percentile(merged, lower))
    vmax = float(np.percentile(merged, upper))
    if vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def normalize(array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((np.asarray(array, dtype=float) - vmin) / (vmax - vmin), 0.0, 1.0)


def normalize_maps(
    results: dict[float, dict[str, np.ndarray | float]],
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[dict[float, dict[str, np.ndarray]], tuple[float, float], tuple[float, float]]:
    x_limits = percentile_limits(
        [result["xpol_ex_y0_abs_xz"] for result in results.values()],
        lower_percentile,
        upper_percentile,
    )
    y_limits = percentile_limits(
        [result["ypol_ey_y0_abs_xz"] for result in results.values()],
        lower_percentile,
        upper_percentile,
    )
    normalized = {
        wavelength_nm: {
            "xpol_ex_y0_abs_xz": normalize(result["xpol_ex_y0_abs_xz"], *x_limits),
            "ypol_ey_y0_abs_xz": normalize(result["ypol_ey_y0_abs_xz"], *y_limits),
        }
        for wavelength_nm, result in results.items()
    }
    return normalized, x_limits, y_limits


def centers_to_edges(vmin: float, vmax: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.array([vmin, vmax], dtype=float)
    centers = np.linspace(vmin, vmax, count)
    edges = np.empty(count + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = vmin
    edges[-1] = vmax
    return edges


def voxel_boundary_segments(mask: np.ndarray, x_edges: np.ndarray, z_edges: np.ndarray) -> list[list[tuple[float, float]]]:
    nx, nz = mask.shape
    segments: list[list[tuple[float, float]]] = []
    for i in range(nx):
        for j in range(nz):
            if not mask[i, j]:
                continue
            x0, x1 = x_edges[i], x_edges[i + 1]
            z0, z1 = z_edges[j], z_edges[j + 1]
            if i == 0 or not mask[i - 1, j]:
                segments.append([(x0, z0), (x0, z1)])
            if i == nx - 1 or not mask[i + 1, j]:
                segments.append([(x1, z0), (x1, z1)])
            if j == 0 or not mask[i, j - 1]:
                segments.append([(x0, z0), (x1, z0)])
            if j == nz - 1 or not mask[i, j + 1]:
                segments.append([(x0, z1), (x1, z1)])
    return segments


def overlay_structure(
    axis,
    structure_xz: np.ndarray,
    *,
    show_boundary: bool,
    show_fill: bool,
    threshold: float,
) -> None:
    if not show_boundary and not show_fill:
        return
    mask = np.asarray(structure_xz) >= threshold
    if not np.any(mask):
        return
    nx, nz = mask.shape
    x_edges = centers_to_edges(-LX_UM / 2.0, LX_UM / 2.0, nx)
    z_edges = centers_to_edges(0.0, LZ_UM, nz)

    if show_fill:
        fill = np.where(mask, 1.0, np.nan).T
        axis.pcolormesh(
            x_edges,
            z_edges,
            fill,
            shading="flat",
            cmap="Greys",
            vmin=0,
            vmax=1,
            alpha=0.10,
            zorder=3,
            rasterized=True,
        )
    if show_boundary:
        segments = voxel_boundary_segments(mask, x_edges, z_edges)
        if segments:
            axis.add_collection(
                LineCollection(
                    segments,
                    colors="white",
                    linewidths=0.65,
                    alpha=0.99,
                    zorder=5,
                )
            )


def setup_matplotlib() -> None:
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    font_name = "Arial" if "Arial" in available_fonts else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 5.8,
            "axes.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def plot_profiles(
    output_dir: Path,
    results: dict[float, dict[str, np.ndarray | float]],
    normalized: dict[float, dict[str, np.ndarray]],
    structure_xz: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    setup_matplotlib()
    wavelengths = sorted(results)
    row_count = len(wavelengths)
    figure, axes = plt.subplots(
        row_count,
        2,
        figsize=(6.30, max(1.45 * row_count, 2.4)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.125,
        right=0.875,
        bottom=0.125,
        top=0.905,
        wspace=0.130,
        hspace=0.140,
    )
    figure.text(0.030, 0.955, "(a)", fontsize=10.0, fontweight="bold", ha="left", va="top")
    axes[0, 0].set_title(r"x-pol, $|E_x|$", fontsize=8.2, pad=5)
    axes[0, 1].set_title(r"y-pol, $|E_y|$", fontsize=8.2, pad=5)

    image = None
    for row, wavelength_nm in enumerate(wavelengths):
        for column, key in enumerate(("xpol_ex_y0_abs_xz", "ypol_ey_y0_abs_xz")):
            axis = axes[row, column]
            image = axis.imshow(
                normalized[wavelength_nm][key].T,
                origin="lower",
                extent=[-LX_UM / 2.0, LX_UM / 2.0, 0.0, LZ_UM],
                aspect="auto",
                cmap=args.cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation=args.interpolation,
                rasterized=True,
                zorder=1,
            )
            overlay_structure(
                axis,
                structure_xz,
                show_boundary=not args.no_structure_overlay,
                show_fill=args.structure_fill,
                threshold=args.threshold,
            )
            axis.set_xlim(-LX_UM / 2.0, LX_UM / 2.0)
            axis.set_ylim(0.0, LZ_UM)
            axis.set_yticks([0.0, 0.2, 0.4, 0.6])
            axis.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
            axis.tick_params(top=False, right=False, labelsize=5.8)
            if row != row_count - 1:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(r"$x$ ($\mu$m)", fontsize=8.0, labelpad=2)
            if column == 0:
                axis.set_ylabel(r"$z$ ($\mu$m)", fontsize=8.0, labelpad=5.0)
            else:
                axis.tick_params(labelleft=False)

        axes[row, 0].text(
            -0.205,
            0.50,
            rf"$\lambda={wavelength_nm:g}\,\mathrm{{nm}}$",
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=6.5,
        )

    if image is None:
        raise RuntimeError("no field maps were generated")
    color_axis = figure.add_axes([0.895, 0.120, 0.015, 0.785])
    colorbar = figure.colorbar(image, cax=color_axis, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    colorbar.set_label(r"Normalized $|E|$", rotation=90, labelpad=8, fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.8, width=0.65, length=2.5, direction="in")

    png_path = output_dir / "field_profiles_y0.png"
    pdf_path = output_dir / "field_profiles_y0.pdf"
    svg_path = output_dir / "field_profiles_y0.svg"
    figure.savefig(png_path, dpi=args.dpi)
    figure.savefig(pdf_path, dpi=args.dpi)
    figure.savefig(svg_path, dpi=args.dpi)
    plt.close(figure)
    return png_path, pdf_path, svg_path


def save_arrays(
    output_dir: Path,
    results: dict[float, dict[str, np.ndarray | float]],
    normalized: dict[float, dict[str, np.ndarray]],
    structure_xz: np.ndarray,
    *,
    args: argparse.Namespace,
) -> Path:
    arrays: dict[str, np.ndarray] = {
        "wavelength_nm": np.array(sorted(results), dtype=float),
        "design_region_um": np.array([LX_UM, LY_UM, LZ_UM]),
        "simulation_resolution": np.array([args.resolution]),
        "design_resolution": np.array([args.design_resolution]),
        "structure_y0_xz": structure_xz,
    }
    for wavelength_nm, result in results.items():
        tag = f"{wavelength_nm:g}nm".replace(".", "p")
        arrays[f"xpol_ex_y0_abs_xz_raw_{tag}"] = result["xpol_ex_y0_abs_xz"]
        arrays[f"ypol_ey_y0_abs_xz_raw_{tag}"] = result["ypol_ey_y0_abs_xz"]
        arrays[f"xpol_ex_y0_abs_xz_normalized_{tag}"] = normalized[wavelength_nm]["xpol_ex_y0_abs_xz"]
        arrays[f"ypol_ey_y0_abs_xz_normalized_{tag}"] = normalized[wavelength_nm]["ypol_ey_y0_abs_xz"]
    path = output_dir / "field_profiles_y0_arrays.npz"
    np.savez_compressed(path, **arrays)
    return path


def save_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    wavelengths_nm: np.ndarray,
    layout,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> Path:
    metadata = {
        "source_rho": str(args.rho.expanduser().resolve()),
        "wavelengths_nm": wavelengths_nm.tolist(),
        "simulation_resolution_px_per_um": args.resolution,
        "design_resolution_px_per_um": args.design_resolution,
        "binarize": args.binarize,
        "threshold": args.threshold,
        "material_grid": {"grid_type": "U_MEAN", "do_averaging": False},
        "tio2_eps_inf": TIO2_EPS_INF,
        "tio2_lorentz_params": list(TIO2_LORENTZ_PARAMS),
        "field_quantities": [
            "|Ex(x,y=0,z)| under x-polarized incidence",
            "|Ey(x,y=0,z)| under y-polarized incidence",
        ],
        "field_volume": "design region only",
        "display_z_coordinate_um": [0.0, LZ_UM],
        "normalization": {
            "x_polarization_common_raw_limits": list(x_limits),
            "y_polarization_common_raw_limits": list(y_limits),
            "vmin_percentile": args.vmin_percentile,
            "vmax_percentile": args.vmax_percentile,
            "gamma_correction": False,
        },
        "display": {
            "colormap": args.cmap,
            "interpolation": args.interpolation,
            "voxel_boundary": not args.no_structure_overlay,
            "voxel_fill": args.structure_fill,
        },
        "pml_um": [PML_BOTTOM_UM, PML_TOP_UM],
        "layout": layout.to_dict(),
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = output_dir / "simulation_info.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_arguments()
    wavelengths_nm = validate_arguments(args)
    layout = build_layout(args.resolution)
    rho_raw, rho_used = load_rho(
        args.rho,
        design_resolution=args.design_resolution,
        binarize=args.binarize,
        threshold=args.threshold,
    )
    tio2 = build_tio2_medium()
    material_grid = build_material_grid(rho_used, tio2=tio2)
    structure_xz = structure_center_y(rho_used)

    output_dir = args.output.expanduser().resolve()
    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "rho_raw_loaded.npy", rho_raw)
        np.save(output_dir / "rho_used_for_simulation.npy", rho_used)
        np.save(output_dir / "rho_structure_y0_xz.npy", structure_xz)
        print("=" * 72)
        print("QWP FIELD-PROFILE CALCULATION")
        print(f"rho              : {args.rho.expanduser().resolve()}")
        print(f"output           : {output_dir}")
        print(f"wavelengths (nm) : {wavelengths_nm.tolist()}")
        print(f"simulation res.  : {args.resolution} px/um")
        print("MaterialGrid     : U_MEAN, do_averaging=False")
        print("TiO2 model       : final two-pole Lorentz fit")
        print("field maps       : |Ex|/|Ey| at y=0 inside design region")
        print("=" * 72)
    comm.Barrier()

    results: dict[float, dict[str, np.ndarray | float]] = {}
    for wavelength_nm in wavelengths_nm:
        results[float(wavelength_nm)] = calculate_maps_for_wavelength(
            float(wavelength_nm),
            layout=layout,
            resolution=args.resolution,
            decay_tolerance=args.decay_tol,
            fwidth_fraction=args.fwidth_frac,
            tio2=tio2,
            material_grid=material_grid,
        )

    if is_master:
        normalized, x_limits, y_limits = normalize_maps(
            results,
            args.vmin_percentile,
            args.vmax_percentile,
        )
        png_path, pdf_path, svg_path = plot_profiles(
            output_dir,
            results,
            normalized,
            structure_xz,
            args=args,
        )
        arrays_path = save_arrays(
            output_dir,
            results,
            normalized,
            structure_xz,
            args=args,
        )
        info_path = save_metadata(
            output_dir,
            args=args,
            wavelengths_nm=wavelengths_nm,
            layout=layout,
            x_limits=x_limits,
            y_limits=y_limits,
        )
        print(f"[DONE] PNG: {png_path}")
        print(f"[DONE] PDF: {pdf_path}")
        print(f"[DONE] SVG: {svg_path}")
        print(f"[DONE] Arrays: {arrays_path}")
        print(f"[DONE] Metadata: {info_path}")
    comm.Barrier()


if __name__ == "__main__":
    main()
