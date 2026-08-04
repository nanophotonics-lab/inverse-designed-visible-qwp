"""Forward verification of the final inverse-designed visible QWP.

The script loads the full 3D rho array and can calculate:
  1. the complex Jones matrix under x- and y-polarized incidence,
  2. LCP-to-+45-degree linear-polarization metrics, and
  3. +45-degree linear-to-circular-polarization metrics.

The default mode is Jones-matrix extraction. The air-only reference produces
an air-referenced transmission coefficient for the complete substrate/device
stack, consistent with the reported final verification.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from mpi4py import MPI

from qwp_model import (
    BACKGROUND_INDEX,
    DESIGN_RESOLUTION,
    LX_UM,
    LY_UM,
    LZ_UM,
    SUBSTRATE_INDEX,
    SUBSTRATE_THICKNESS_UM,
    TIO2_EPS_INF,
    TIO2_LORENTZ_PARAMS,
    binarization_degree,
    build_device_geometry,
    build_layout,
    build_material_grid,
    build_tio2_medium,
    choose_decay_component,
    complex_eps_tio2,
    design_grid_shape,
    load_rho,
    make_plane_sources,
    make_simulation,
)


EPS = 1e-12
FIGSIZE = (7.4, 4.6)
LINEWIDTH = 2.2
REFERENCE_LINEWIDTH = 1.4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
is_master = rank == 0
mp.verbosity(1 if is_master else 0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward verification of the final inverse-designed QWP."
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
        default=Path("results/jones_matrix"),
        help="Directory for CSV, plots, and metadata.",
    )
    parser.add_argument(
        "--mode",
        choices=("jones", "lcp", "p45", "all"),
        default="jones",
        help="Verification block to execute.",
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=None,
        metavar="NM",
        help="Explicit wavelengths in nm. Default: 500 to 600 nm in 1 nm steps.",
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
        help="Hard-threshold rho before simulation. Omit for an already saved binary rho.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used with --binarize. Default: 0.5.",
    )
    return parser.parse_args()


def wavelength_array_nm(values: list[float] | None) -> np.ndarray:
    if values is None:
        return np.arange(500.0, 601.0, 1.0)
    wavelengths = np.asarray(values, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("at least one wavelength is required")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths must be finite and positive")
    return wavelengths


def safe_reset_sim(sim: mp.Simulation) -> None:
    try:
        sim.reset_meep()
    except Exception:
        pass


def wrap_deg_180(value_deg: float) -> float:
    return (value_deg + 180.0) % 360.0 - 180.0


def circular_distance_rad(a: float, b: float) -> float:
    return float(abs(np.angle(np.exp(1j * (a - b)))))


def phase_delta_deg_corr(ex: np.ndarray, ey: np.ndarray) -> float:
    correlation = np.sum(ey * np.conj(ex))
    return float(np.degrees(np.arctan2(np.imag(correlation), np.real(correlation))))


def sz_from_fields(
    ex: np.ndarray,
    ey: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
) -> np.ndarray:
    return np.real(ex * np.conj(hy) - ey * np.conj(hx))


def field_coefficients(ex: np.ndarray, ey: np.ndarray) -> tuple[complex, complex]:
    return complex(np.mean(ex)), complex(np.mean(ey))


def complex_pair(value: complex) -> tuple[float, float]:
    return float(np.real(value)), float(np.imag(value))


def source_pair(
    frequency: float,
    layout,
    ex_amplitude: complex,
    ey_amplitude: complex,
    fwidth_fraction: float,
) -> list[mp.Source]:
    return make_plane_sources(
        frequency,
        layout,
        ex_amplitude,
        ey_amplitude,
        fwidth_fraction=fwidth_fraction,
    )


def x_sources(frequency: float, layout, fwidth_fraction: float) -> list[mp.Source]:
    return source_pair(frequency, layout, 1.0, 0.0, fwidth_fraction)


def y_sources(frequency: float, layout, fwidth_fraction: float) -> list[mp.Source]:
    return source_pair(frequency, layout, 0.0, 1.0, fwidth_fraction)


def lcp_sources(frequency: float, layout, fwidth_fraction: float) -> list[mp.Source]:
    # Convention used in the optimization: Ex = 1, Ey = -i.
    return source_pair(frequency, layout, 1.0, -1.0j, fwidth_fraction)


def p45_sources(frequency: float, layout, fwidth_fraction: float) -> list[mp.Source]:
    amplitude = 1.0 / np.sqrt(2.0)
    return source_pair(frequency, layout, amplitude, amplitude, fwidth_fraction)


def run_one_polarization(
    *,
    frequency: float,
    sources: list[mp.Source],
    layout,
    resolution: int,
    decay_tolerance: float,
    tio2: mp.Medium,
    material_grid: mp.MaterialGrid | None,
) -> tuple[complex, complex, float]:
    geometry = (
        build_device_geometry(material_grid, layout)
        if material_grid is not None
        else []
    )
    simulation = make_simulation(
        resolution=resolution,
        layout=layout,
        sources=sources,
        geometry=geometry,
        tio2=tio2,
    )
    monitor_volume = mp.Volume(
        center=mp.Vector3(0, 0, layout.z_monitor),
        size=mp.Vector3(LX_UM, LY_UM, 0),
    )
    dft = simulation.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Hx, mp.Hy],
        frequency,
        0,
        1,
        where=monitor_volume,
    )
    decay_component = choose_decay_component(sources)
    condition = mp.stop_when_fields_decayed(
        50,
        decay_component,
        mp.Vector3(0, 0, layout.z_monitor),
        decay_tolerance,
    )
    simulation.run(until_after_sources=lambda sim: condition(sim))

    ex, ey, hx, hy = [
        simulation.get_dft_array(dft, component, 0)
        for component in (mp.Ex, mp.Ey, mp.Hx, mp.Hy)
    ]
    ex_coefficient, ey_coefficient = field_coefficients(ex, ey)
    flux = float(np.sum(sz_from_fields(ex, ey, hx, hy)))

    safe_reset_sim(simulation)
    del simulation
    gc.collect()
    comm.Barrier()
    return ex_coefficient, ey_coefficient, flux


def compute_reference_and_device_xy(
    wavelengths_nm: np.ndarray,
    *,
    layout,
    resolution: int,
    decay_tolerance: float,
    fwidth_fraction: float,
    tio2: mp.Medium,
    material_grid: mp.MaterialGrid,
) -> list[dict]:
    rows: list[dict] = []
    for wavelength_nm in wavelengths_nm:
        frequency = 1000.0 / float(wavelength_nm)
        if is_master:
            print(f"[JONES] {wavelength_nm:g} nm: air reference")
        ex_x_ref, ey_x_ref, flux_x_ref = run_one_polarization(
            frequency=frequency,
            sources=x_sources(frequency, layout, fwidth_fraction),
            layout=layout,
            resolution=resolution,
            decay_tolerance=decay_tolerance,
            tio2=tio2,
            material_grid=None,
        )
        ex_y_ref, ey_y_ref, flux_y_ref = run_one_polarization(
            frequency=frequency,
            sources=y_sources(frequency, layout, fwidth_fraction),
            layout=layout,
            resolution=resolution,
            decay_tolerance=decay_tolerance,
            tio2=tio2,
            material_grid=None,
        )

        if is_master:
            print(f"[JONES] {wavelength_nm:g} nm: device")
        ex_x, ey_x, flux_x = run_one_polarization(
            frequency=frequency,
            sources=x_sources(frequency, layout, fwidth_fraction),
            layout=layout,
            resolution=resolution,
            decay_tolerance=decay_tolerance,
            tio2=tio2,
            material_grid=material_grid,
        )
        ex_y, ey_y, flux_y = run_one_polarization(
            frequency=frequency,
            sources=y_sources(frequency, layout, fwidth_fraction),
            layout=layout,
            resolution=resolution,
            decay_tolerance=decay_tolerance,
            tio2=tio2,
            material_grid=material_grid,
        )

        x_denominator = ex_x_ref if abs(ex_x_ref) > EPS else ex_x_ref + EPS
        y_denominator = ey_y_ref if abs(ey_y_ref) > EPS else ey_y_ref + EPS

        t_xx = ex_x / x_denominator
        t_yx = ey_x / x_denominator
        t_xy = ex_y / y_denominator
        t_yy = ey_y / y_denominator
        jones_xy = np.array([[t_xx, t_xy], [t_yx, t_yy]], dtype=complex)

        phase_difference = float(np.angle(t_yy * np.conj(t_xx)))
        displayed_phase = 90.0 + np.degrees(
            np.angle(np.exp(1j * (phase_difference - np.pi / 2.0)))
        )
        qwp_error = np.degrees(
            min(
                circular_distance_rad(phase_difference, np.pi / 2.0),
                circular_distance_rad(phase_difference, -np.pi / 2.0),
            )
        )

        rows.append(
            {
                "wavelength_nm": float(wavelength_nm),
                "J_xy": jones_xy,
                "phase_difference_rad": phase_difference,
                "phase_difference_display_deg": float(displayed_phase),
                "qwp_error_deg": float(qwp_error),
                "flux_x_reference": flux_x_ref,
                "flux_y_reference": flux_y_ref,
                "flux_x_device": flux_x,
                "flux_y_device": flux_y,
            }
        )
    return rows


def polarization_overlaps(ex: np.ndarray, ey: np.ndarray) -> dict[str, float]:
    total = float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2)) + EPS
    e_p45 = (ex + ey) / np.sqrt(2.0)
    e_m45 = (ex - ey) / np.sqrt(2.0)
    e_r = (ex - 1j * ey) / np.sqrt(2.0)
    e_l = (ex + 1j * ey) / np.sqrt(2.0)
    return {
        "total": total,
        "p45": float(np.sum(np.abs(e_p45) ** 2)) / total,
        "m45": float(np.sum(np.abs(e_m45) ** 2)) / total,
        "x": float(np.sum(np.abs(ex) ** 2)) / total,
        "y": float(np.sum(np.abs(ey) ** 2)) / total,
        "r": float(np.sum(np.abs(e_r) ** 2)) / total,
        "l": float(np.sum(np.abs(e_l) ** 2)) / total,
    }


def run_lcp_verification(
    wavelengths_nm: np.ndarray,
    *,
    layout,
    resolution: int,
    decay_tolerance: float,
    fwidth_fraction: float,
    tio2: mp.Medium,
    material_grid: mp.MaterialGrid,
) -> list[dict]:
    rows: list[dict] = []
    for wavelength_nm in wavelengths_nm:
        frequency = 1000.0 / float(wavelength_nm)
        reference_sources = lcp_sources(frequency, layout, fwidth_fraction)
        _, _, flux_reference = run_one_polarization(
            frequency=frequency,
            sources=reference_sources,
            layout=layout,
            resolution=resolution,
            decay_tolerance=decay_tolerance,
            tio2=tio2,
            material_grid=None,
        )

        device_sources = lcp_sources(frequency, layout, fwidth_fraction)
        simulation = make_simulation(
            resolution=resolution,
            layout=layout,
            sources=device_sources,
            geometry=build_device_geometry(material_grid, layout),
            tio2=tio2,
        )
        monitor_volume = mp.Volume(
            center=mp.Vector3(0, 0, layout.z_monitor),
            size=mp.Vector3(LX_UM, LY_UM, 0),
        )
        dft = simulation.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Hx, mp.Hy],
            frequency,
            0,
            1,
            where=monitor_volume,
        )
        decay_component = choose_decay_component(device_sources)
        condition = mp.stop_when_fields_decayed(
            50,
            decay_component,
            mp.Vector3(0, 0, layout.z_monitor),
            decay_tolerance,
        )
        simulation.run(until_after_sources=lambda sim: condition(sim))
        ex, ey, hx, hy = [
            simulation.get_dft_array(dft, component, 0)
            for component in (mp.Ex, mp.Ey, mp.Hx, mp.Hy)
        ]

        flux_device = float(np.sum(sz_from_fields(ex, ey, hx, hy)))
        transmission = flux_device / (flux_reference if abs(flux_reference) > EPS else EPS)
        overlaps = polarization_overlaps(ex, ey)
        delta_deg = phase_delta_deg_corr(ex, ey)
        rows.append(
            {
                "wavelength_nm": float(wavelength_nm),
                "transmission": transmission,
                "p45_projected_transmission": transmission * overlaps["p45"],
                "p45_purity": overlaps["p45"],
                "m45_overlap": overlaps["m45"],
                "r_overlap": overlaps["r"],
                "l_overlap": overlaps["l"],
                "phase_delta_deg": delta_deg,
                "phase_error_to_p45_deg": abs(wrap_deg_180(delta_deg)),
                "flux_reference": flux_reference,
                "flux_device": flux_device,
            }
        )

        safe_reset_sim(simulation)
        del simulation
        gc.collect()
        comm.Barrier()
    return rows


def stokes_parameters(field: np.ndarray) -> tuple[float, float, float, float]:
    ex, ey = field
    s0 = float((abs(ex) ** 2 + abs(ey) ** 2).real)
    s1 = float((abs(ex) ** 2 - abs(ey) ** 2).real)
    s2 = float((2.0 * np.real(ex * np.conj(ey))).real)
    s3 = float((-2.0 * np.imag(ex * np.conj(ey))).real)
    return s0, s1, s2, s3


def run_p45_verification(
    wavelengths_nm: np.ndarray,
    *,
    layout,
    resolution: int,
    decay_tolerance: float,
    fwidth_fraction: float,
    tio2: mp.Medium,
    material_grid: mp.MaterialGrid,
) -> list[dict]:
    rows: list[dict] = []
    for wavelength_nm in wavelengths_nm:
        frequency = 1000.0 / float(wavelength_nm)
        reference = run_one_polarization(
            frequency=frequency,
            sources=p45_sources(frequency, layout, fwidth_fraction),
            layout=layout,
            resolution=resolution,
            decay_tolerance=decay_tolerance,
            tio2=tio2,
            material_grid=None,
        )
        device = run_one_polarization(
            frequency=frequency,
            sources=p45_sources(frequency, layout, fwidth_fraction),
            layout=layout,
            resolution=resolution,
            decay_tolerance=decay_tolerance,
            tio2=tio2,
            material_grid=material_grid,
        )
        reference_field = np.array(reference[:2], dtype=complex)
        device_field = np.array(device[:2], dtype=complex)
        reference_norm = float(np.linalg.norm(reference_field))
        if reference_norm < EPS:
            reference_norm = EPS
        output = device_field / reference_norm
        ex, ey = output
        e_p45 = (ex + ey) / np.sqrt(2.0)
        e_m45 = (ex - ey) / np.sqrt(2.0)
        e_r = (ex - 1j * ey) / np.sqrt(2.0)
        e_l = (ex + 1j * ey) / np.sqrt(2.0)
        p_p45 = float(abs(e_p45) ** 2)
        p_m45 = float(abs(e_m45) ** 2)
        p_r = float(abs(e_r) ** 2)
        p_l = float(abs(e_l) ** 2)
        total = float(abs(ex) ** 2 + abs(ey) ** 2) + EPS
        s0, s1, s2, s3 = stokes_parameters(output)
        chi = 0.0 if abs(s0) < EPS else 0.5 * np.arcsin(np.clip(s3 / s0, -1, 1))

        rows.append(
            {
                "wavelength_nm": float(wavelength_nm),
                "eta_p45": p_p45 / total,
                "eta_m45": p_m45 / total,
                "eta_r": p_r / (p_r + p_l + EPS),
                "eta_l": p_l / (p_r + p_l + EPS),
                "S0": s0,
                "S1": s1,
                "S2": s2,
                "S3": s3,
                "ellipticity_angle_deg": float(np.degrees(chi)),
                "ellipticity_ratio": float(np.tan(chi)),
            }
        )
    return rows


def save_jones_csv(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "jones_xy_extracted.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wavelength_nm",
                "t_xx_re",
                "t_xx_im",
                "t_xy_re",
                "t_xy_im",
                "t_yx_re",
                "t_yx_im",
                "t_yy_re",
                "t_yy_im",
                "T_xx",
                "T_xy",
                "T_yx",
                "T_yy",
                "phi_xx_deg",
                "phi_yy_deg",
                "phase_difference_display_deg",
                "qwp_error_deg",
            ]
        )
        for row in rows:
            matrix = row["J_xy"]
            t_xx, t_xy = matrix[0, 0], matrix[0, 1]
            t_yx, t_yy = matrix[1, 0], matrix[1, 1]
            writer.writerow(
                [
                    f"{row['wavelength_nm']:.8g}",
                    *[f"{value:.12e}" for value in complex_pair(t_xx)],
                    *[f"{value:.12e}" for value in complex_pair(t_xy)],
                    *[f"{value:.12e}" for value in complex_pair(t_yx)],
                    *[f"{value:.12e}" for value in complex_pair(t_yy)],
                    f"{abs(t_xx) ** 2:.12e}",
                    f"{abs(t_xy) ** 2:.12e}",
                    f"{abs(t_yx) ** 2:.12e}",
                    f"{abs(t_yy) ** 2:.12e}",
                    f"{np.degrees(np.angle(t_xx)):.8f}",
                    f"{np.degrees(np.angle(t_yy)):.8f}",
                    f"{row['phase_difference_display_deg']:.8f}",
                    f"{row['qwp_error_deg']:.8f}",
                ]
            )
    return path


def save_lcp_csv(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "lcp_to_p45_metrics.csv"
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_p45_csv(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "p45_to_circular_metrics.csv"
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_jones_plots(output_dir: Path, rows: list[dict]) -> None:
    wavelengths = np.array([row["wavelength_nm"] for row in rows])
    t_xx = np.array([row["J_xy"][0, 0] for row in rows])
    t_xy = np.array([row["J_xy"][0, 1] for row in rows])
    t_yx = np.array([row["J_xy"][1, 0] for row in rows])
    t_yy = np.array([row["J_xy"][1, 1] for row in rows])
    phase = np.array([row["phase_difference_display_deg"] for row in rows])
    error = np.array([row["qwp_error_deg"] for row in rows])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(wavelengths, abs(t_xx) ** 2, linewidth=LINEWIDTH, label=r"$|t_{xx}|^2$")
    ax.plot(wavelengths, abs(t_yy) ** 2, linewidth=LINEWIDTH, label=r"$|t_{yy}|^2$")
    ax.set(xlabel="Wavelength (nm)", ylabel="Transmission", ylim=(0, 1.05))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "jones_copolarized_transmission.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogy(wavelengths, abs(t_xy) ** 2 + 1e-30, linewidth=LINEWIDTH, label=r"$|t_{xy}|^2$")
    ax.semilogy(wavelengths, abs(t_yx) ** 2 + 1e-30, linewidth=LINEWIDTH, label=r"$|t_{yx}|^2$")
    ax.set(xlabel="Wavelength (nm)", ylabel="Cross-polarized transmission")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "jones_cross_polarized_transmission.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(wavelengths, phase, linewidth=LINEWIDTH)
    ax.axhline(90, linestyle="--", linewidth=REFERENCE_LINEWIDTH)
    ax.set(xlabel="Wavelength (nm)", ylabel="Phase difference (deg)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "jones_phase_difference.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(wavelengths, error, linewidth=LINEWIDTH)
    ax.set(xlabel="Wavelength (nm)", ylabel="Nearest QWP phase error (deg)")
    ax.set_ylim(0, max(5.0, float(np.max(error)) * 1.1))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "jones_qwp_phase_error.png", dpi=240)
    plt.close(fig)


def save_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    wavelengths_nm: np.ndarray,
    layout,
    rho_raw: np.ndarray,
    rho_used: np.ndarray,
) -> Path:
    metadata = {
        "source_rho": str(args.rho.expanduser().resolve()),
        "mode": args.mode,
        "wavelengths_nm": wavelengths_nm.tolist(),
        "simulation_resolution_px_per_um": args.resolution,
        "design_resolution_px_per_um": args.design_resolution,
        "rho_shape": list(rho_used.shape),
        "binarize": args.binarize,
        "threshold": args.threshold,
        "raw_rho_min": float(np.min(rho_raw)),
        "raw_rho_max": float(np.max(rho_raw)),
        "used_rho_min": float(np.min(rho_used)),
        "used_rho_max": float(np.max(rho_used)),
        "raw_binarization_degree": binarization_degree(rho_raw),
        "used_binarization_degree": binarization_degree(rho_used),
        "material_grid": {"grid_type": "U_MEAN", "do_averaging": False},
        "background_index": BACKGROUND_INDEX,
        "substrate_index": SUBSTRATE_INDEX,
        "substrate_thickness_um": SUBSTRATE_THICKNESS_UM,
        "design_region_um": [LX_UM, LY_UM, LZ_UM],
        "tio2_eps_inf": TIO2_EPS_INF,
        "tio2_lorentz_params": list(TIO2_LORENTZ_PARAMS),
        "decay_tolerance": args.decay_tol,
        "source_fwidth_fraction": args.fwidth_frac,
        "normalization": "air-only reference for the complete substrate/device stack",
        "lcp_convention": "Ex=1, Ey=-i",
        "layout": layout.to_dict(),
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = output_dir / "simulation_info.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_arguments()
    wavelengths_nm = wavelength_array_nm(args.wavelengths)
    if args.resolution <= 0 or args.design_resolution <= 0:
        raise ValueError("resolutions must be positive")
    if args.decay_tol <= 0 or args.fwidth_frac <= 0:
        raise ValueError("decay tolerance and source bandwidth must be positive")

    layout = build_layout(args.resolution)
    rho_raw, rho_used = load_rho(
        args.rho,
        design_resolution=args.design_resolution,
        binarize=args.binarize,
        threshold=args.threshold,
    )
    tio2 = build_tio2_medium()
    material_grid = build_material_grid(rho_used, tio2=tio2)

    output_dir = args.output.expanduser().resolve()
    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "rho_raw_loaded.npy", rho_raw)
        np.save(output_dir / "rho_used_for_simulation.npy", rho_used)
        print("=" * 72)
        print("QWP FORWARD VERIFICATION")
        print(f"rho              : {args.rho.expanduser().resolve()}")
        print(f"output           : {output_dir}")
        print(f"mode             : {args.mode}")
        print(f"wavelengths (nm) : {wavelengths_nm.tolist()}")
        print(f"rho shape        : {design_grid_shape(args.design_resolution)}")
        print(f"simulation res.  : {args.resolution} px/um")
        print("MaterialGrid     : U_MEAN, do_averaging=False")
        print("TiO2 model       : final two-pole Lorentz fit")
        for wavelength_nm in (500.0, 550.0, 600.0):
            eps = complex_eps_tio2(1000.0 / wavelength_nm)
            print(
                f"  {wavelength_nm:g} nm: eps={eps.real:.8f}{eps.imag:+.3e}j, "
                f"n~{np.sqrt(eps.real):.6f}"
            )
        print("=" * 72)
    comm.Barrier()

    if args.mode in ("jones", "all"):
        jones_rows = compute_reference_and_device_xy(
            wavelengths_nm,
            layout=layout,
            resolution=args.resolution,
            decay_tolerance=args.decay_tol,
            fwidth_fraction=args.fwidth_frac,
            tio2=tio2,
            material_grid=material_grid,
        )
        if is_master:
            save_jones_csv(output_dir, jones_rows)
            save_jones_plots(output_dir, jones_rows)
    comm.Barrier()

    if args.mode in ("lcp", "all"):
        lcp_rows = run_lcp_verification(
            wavelengths_nm,
            layout=layout,
            resolution=args.resolution,
            decay_tolerance=args.decay_tol,
            fwidth_fraction=args.fwidth_frac,
            tio2=tio2,
            material_grid=material_grid,
        )
        if is_master:
            save_lcp_csv(output_dir, lcp_rows)
    comm.Barrier()

    if args.mode in ("p45", "all"):
        p45_rows = run_p45_verification(
            wavelengths_nm,
            layout=layout,
            resolution=args.resolution,
            decay_tolerance=args.decay_tol,
            fwidth_fraction=args.fwidth_frac,
            tio2=tio2,
            material_grid=material_grid,
        )
        if is_master:
            save_p45_csv(output_dir, p45_rows)
    comm.Barrier()

    if is_master:
        info_path = save_metadata(
            output_dir,
            args=args,
            wavelengths_nm=wavelengths_nm,
            layout=layout,
            rho_raw=rho_raw,
            rho_used=rho_used,
        )
        print(f"[DONE] Results: {output_dir}")
        print(f"[DONE] Metadata: {info_path}")
    comm.Barrier()


if __name__ == "__main__":
    main()
