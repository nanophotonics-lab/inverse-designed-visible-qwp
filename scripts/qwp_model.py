#!/usr/bin/env python3
"""Shared geometry and material definitions for QWP forward simulations.

All lengths are expressed in micrometers, following Meep's dimensionless-unit
convention with 1 simulation length unit = 1 um.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import meep as mp
import numpy as np


# Device dimensions used for the final inverse-designed QWP.
LX_UM = 0.2
LY_UM = 0.2
LZ_UM = 0.6
SUBSTRATE_THICKNESS_UM = 0.5
SUBSTRATE_INDEX = 1.45
BACKGROUND_INDEX = 1.0
DESIGN_RESOLUTION = 50

# Source, monitor, and PML spacing used in the optimization/verification setup.
PML_BOTTOM_UM = 0.5
PML_TOP_UM = 0.5
GAP_PML_TO_SOURCE_UM = 0.4
GAP_SOURCE_TO_SUBSTRATE_UM = 0.8
GAP_STRUCTURE_TO_MONITOR_UM = 1.0
GAP_MONITOR_TO_PML_UM = 0.4

# Lossless two-pole TiO2 Lorentz fit used in the final optimization.
TIO2_EPS_INF = 1.0000000000000002
TIO2_LORENTZ_PARAMS = (
    {"frequency": 3.17296032821, "gamma": 0.0, "sigma": 0.637757693334},
    {"frequency": 5.35783194651, "gamma": 0.0, "sigma": 3.52503406897},
)


@dataclass(frozen=True)
class Layout:
    """Snapped z-stack coordinates for one simulation resolution."""

    resolution: int
    sz: float
    z_bottom: float
    z_top: float
    z_after_bottom_pml: float
    z_source: float
    z_substrate_start: float
    z_substrate_end: float
    z_substrate_center: float
    z_structure_start: float
    z_structure_end: float
    z_structure_center: float
    z_monitor: float
    z_top_pml_start: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _snap_z(value: float, resolution: int) -> float:
    dz = 1.0 / float(resolution)
    return round(value / dz) * dz


def build_layout(resolution: int = 50) -> Layout:
    """Build the center-first snapped layout used in the final verification."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")

    z_bottom_raw = 0.0
    z_after_bottom_pml_raw = z_bottom_raw + PML_BOTTOM_UM
    z_source_raw = z_after_bottom_pml_raw + GAP_PML_TO_SOURCE_UM
    z_substrate_start_raw = z_source_raw + GAP_SOURCE_TO_SUBSTRATE_UM
    z_substrate_center_raw = z_substrate_start_raw + 0.5 * SUBSTRATE_THICKNESS_UM
    z_substrate_end_raw = z_substrate_center_raw + 0.5 * SUBSTRATE_THICKNESS_UM

    # Direct substrate/design contact: no pad, overlap, or air gap.
    z_structure_start_raw = z_substrate_end_raw
    z_structure_center_raw = z_structure_start_raw + 0.5 * LZ_UM
    z_structure_end_raw = z_structure_center_raw + 0.5 * LZ_UM

    z_monitor_raw = z_structure_end_raw + GAP_STRUCTURE_TO_MONITOR_UM
    z_top_pml_start_raw = z_monitor_raw + GAP_MONITOR_TO_PML_UM
    z_top_raw = z_top_pml_start_raw + PML_TOP_UM
    z_shift = 0.5 * (z_top_raw + z_bottom_raw)

    z_after_bottom_pml = _snap_z(z_after_bottom_pml_raw - z_shift, resolution)
    z_source = _snap_z(z_source_raw - z_shift, resolution)
    z_substrate_center = _snap_z(z_substrate_center_raw - z_shift, resolution)

    # Derive the structure position from the snapped substrate center so that
    # the interface remains exactly contiguous.
    z_structure_center = (
        z_substrate_center + 0.5 * SUBSTRATE_THICKNESS_UM + 0.5 * LZ_UM
    )
    z_substrate_start = z_substrate_center - 0.5 * SUBSTRATE_THICKNESS_UM
    z_substrate_end = z_substrate_center + 0.5 * SUBSTRATE_THICKNESS_UM
    z_structure_start = z_structure_center - 0.5 * LZ_UM
    z_structure_end = z_structure_center + 0.5 * LZ_UM

    z_monitor = _snap_z(z_monitor_raw - z_shift, resolution)
    z_top_pml_start = _snap_z(z_top_pml_start_raw - z_shift, resolution)
    z_bottom = _snap_z(z_after_bottom_pml - PML_BOTTOM_UM, resolution)
    z_top = _snap_z(z_top_pml_start + PML_TOP_UM, resolution)
    sz = z_top - z_bottom

    layout = Layout(
        resolution=resolution,
        sz=sz,
        z_bottom=z_bottom,
        z_top=z_top,
        z_after_bottom_pml=z_after_bottom_pml,
        z_source=z_source,
        z_substrate_start=z_substrate_start,
        z_substrate_end=z_substrate_end,
        z_substrate_center=z_substrate_center,
        z_structure_start=z_structure_start,
        z_structure_end=z_structure_end,
        z_structure_center=z_structure_center,
        z_monitor=z_monitor,
        z_top_pml_start=z_top_pml_start,
    )
    validate_layout(layout)
    return layout


def validate_layout(layout: Layout) -> None:
    """Raise an error when the snapped stack no longer matches the paper setup."""

    substrate_thickness = layout.z_substrate_end - layout.z_substrate_start
    structure_thickness = layout.z_structure_end - layout.z_structure_start
    interface_gap = layout.z_structure_start - layout.z_substrate_end

    if abs(substrate_thickness - SUBSTRATE_THICKNESS_UM) > 1e-12:
        raise ValueError("substrate thickness changed after snapping")
    if abs(structure_thickness - LZ_UM) > 1e-12:
        raise ValueError("design-region thickness changed after snapping")
    if abs(interface_gap) > 1e-12:
        raise ValueError("substrate/design interface is not contiguous")
    if layout.z_monitor >= layout.z_top_pml_start:
        raise ValueError("monitor lies inside or beyond the upper PML boundary")


def build_tio2_medium() -> mp.Medium:
    """Return the dispersive TiO2 model used for optimization and validation."""

    return mp.Medium(
        epsilon=TIO2_EPS_INF,
        E_susceptibilities=[
            mp.LorentzianSusceptibility(
                frequency=params["frequency"],
                gamma=params["gamma"],
                sigma=params["sigma"],
            )
            for params in TIO2_LORENTZ_PARAMS
        ],
    )


def complex_eps_tio2(frequency: float) -> complex:
    """Evaluate the analytic Lorentz permittivity at a Meep frequency."""

    eps_value = complex(TIO2_EPS_INF)
    for params in TIO2_LORENTZ_PARAMS:
        omega_0 = params["frequency"]
        gamma = params["gamma"]
        sigma = params["sigma"]
        eps_value += (
            sigma
            * omega_0**2
            / (omega_0**2 - frequency**2 - 1j * gamma * frequency)
        )
    return eps_value


def design_grid_shape(design_resolution: int = DESIGN_RESOLUTION) -> tuple[int, int, int]:
    """Return the stored MaterialGrid node count."""

    if design_resolution <= 0:
        raise ValueError("design_resolution must be positive")
    return (
        int(round(LX_UM * design_resolution)) + 1,
        int(round(LY_UM * design_resolution)) + 1,
        int(round(LZ_UM * design_resolution)) + 1,
    )


def load_rho(
    rho_path: str | Path,
    *,
    design_resolution: int = DESIGN_RESOLUTION,
    binarize: bool = False,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a full 3D rho array and optionally apply a hard threshold.

    Returns
    -------
    rho_raw_3d, rho_used_3d
        Arrays with shape ``design_grid_shape(design_resolution)``.
    """

    path = Path(rho_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rho file not found: {path}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")

    expected_shape = design_grid_shape(design_resolution)
    expected_size = int(np.prod(expected_shape))
    rho_flat = np.asarray(np.load(path), dtype=float).ravel()

    if rho_flat.size != expected_size:
        raise ValueError(
            f"rho size mismatch: got {rho_flat.size}, expected {expected_size} "
            f"for shape {expected_shape}"
        )
    if not np.all(np.isfinite(rho_flat)):
        raise ValueError("rho contains NaN or infinite values")

    rho_raw_3d = rho_flat.reshape(expected_shape)
    if binarize:
        rho_used_3d = np.where(rho_raw_3d >= threshold, 1.0, 0.0)
    else:
        rho_used_3d = rho_raw_3d.copy()

    return rho_raw_3d, rho_used_3d


def binarization_degree(rho: np.ndarray) -> float:
    """Compute the 0-to-1 binarization degree used in the original script."""

    values = np.asarray(rho, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    return float(np.sum(np.abs(values - 0.5)) / (0.5 * values.size))


def build_material_grid(
    rho_used_3d: np.ndarray,
    *,
    background: mp.Medium | None = None,
    tio2: mp.Medium | None = None,
) -> mp.MaterialGrid:
    """Create the final U_MEAN MaterialGrid with averaging disabled."""

    background = background or mp.Medium(index=BACKGROUND_INDEX)
    tio2 = tio2 or build_tio2_medium()
    nx, ny, nz = rho_used_3d.shape

    material_grid = mp.MaterialGrid(
        mp.Vector3(nx, ny, nz),
        medium1=background,
        medium2=tio2,
        grid_type="U_MEAN",
        do_averaging=False,
    )
    try:
        material_grid.update_weights(rho_used_3d)
    except Exception:
        material_grid.weights = np.array(rho_used_3d, dtype=float, order="C")
    return material_grid


def build_device_geometry(
    material_grid: mp.MaterialGrid,
    layout: Layout,
    *,
    substrate: mp.Medium | None = None,
) -> list[mp.GeometricObject]:
    """Return the finite SiO2 substrate and inverse-designed layer."""

    substrate = substrate or mp.Medium(index=SUBSTRATE_INDEX)
    return [
        mp.Block(
            center=mp.Vector3(0, 0, layout.z_substrate_center),
            size=mp.Vector3(LX_UM, LY_UM, SUBSTRATE_THICKNESS_UM),
            material=substrate,
        ),
        mp.Block(
            center=mp.Vector3(0, 0, layout.z_structure_center),
            size=mp.Vector3(LX_UM, LY_UM, LZ_UM),
            material=material_grid,
        ),
    ]


def make_plane_sources(
    frequency: float,
    layout: Layout,
    ex_amplitude: complex,
    ey_amplitude: complex,
    *,
    fwidth_fraction: float = 0.10,
) -> list[mp.Source]:
    """Create a normally incident periodic plane-wave source pair."""

    source_time = mp.GaussianSource(
        frequency=frequency,
        fwidth=fwidth_fraction * frequency,
    )
    center = mp.Vector3(0, 0, layout.z_source)
    size = mp.Vector3(LX_UM, LY_UM, 0)
    return [
        mp.Source(
            source_time,
            component=mp.Ex,
            center=center,
            size=size,
            amplitude=ex_amplitude,
        ),
        mp.Source(
            source_time,
            component=mp.Ey,
            center=center,
            size=size,
            amplitude=ey_amplitude,
        ),
    ]


def choose_decay_component(sources: Sequence[mp.Source]) -> int:
    """Choose a nonzero electric component for field-decay termination."""

    for source in sources:
        if source.component == mp.Ex and abs(complex(source.amplitude)) > 0:
            return mp.Ex
    for source in sources:
        if source.component == mp.Ey and abs(complex(source.amplitude)) > 0:
            return mp.Ey
    return mp.Ex


def make_simulation(
    *,
    resolution: int,
    layout: Layout,
    sources: Sequence[mp.Source],
    geometry: Sequence[mp.GeometricObject] | None = None,
    tio2: mp.Medium | None = None,
) -> mp.Simulation:
    """Create the periodic-x/y, PML-z Meep simulation."""

    tio2 = tio2 or build_tio2_medium()
    return mp.Simulation(
        cell_size=mp.Vector3(LX_UM, LY_UM, layout.sz),
        boundary_layers=[
            mp.PML(PML_BOTTOM_UM, direction=mp.Z, side=mp.Low),
            mp.PML(PML_TOP_UM, direction=mp.Z, side=mp.High),
        ],
        k_point=mp.Vector3(0, 0, 0),
        geometry=list(geometry or []),
        sources=list(sources),
        default_material=mp.Medium(index=BACKGROUND_INDEX),
        resolution=resolution,
        extra_materials=[tio2],
    )
