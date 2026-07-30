"""Generate publication-style 3D unit-cell renderings from binary QWP structure arrays."""

import argparse
import math
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = REPO_ROOT / "data" / "optimized_qwp_binary_structure_3d.npy"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "renders"

RHO_SHAPE = (11, 11, 31)
RESHAPE_ORDER = "C"
THRESHOLD = 0.5

PERIOD_X_UM = 0.2
PERIOD_Y_UM = 0.2
STRUCTURE_THICKNESS_UM = 0.6
PHYSICAL_SUBSTRATE_THICKNESS_UM = 0.5

# The substrate thickness is reduced only in the rendered image for visual clarity.
SUBSTRATE_VISUAL_Z_SCALE = 0.3
SUBSTRATE_XY_MARGIN_RATIO = 0.10

TRANSPOSE_XY = False
FLIP_X = False
FLIP_Y = False
FLIP_Z = False

WINDOW_SIZE = (1800, 1800)
OFF_SCREEN = True
BACKGROUND_COLOR = "white"
USE_PARALLEL_PROJECTION = True
CAMERA_DISTANCE_FACTOR = 1.90
CAMERA_ZOOM = 2.10

AZIMUTH_DEG_LIST = (225, 315)
VIEW_Z_COMPONENT = 0.38
VIEW_XY_RADIUS = 1.0

STRUCTURE_COLOR = "#4f4f4f"
STRUCTURE_OPACITY = 1.0
STRUCTURE_AMBIENT = 0.42
STRUCTURE_DIFFUSE = 0.65
STRUCTURE_SPECULAR = 0.06

SUBSTRATE_COLOR = "#dceef8"
SUBSTRATE_OPACITY = 0.60
SUBSTRATE_AMBIENT = 0.45
SUBSTRATE_DIFFUSE = 0.42
SUBSTRATE_SPECULAR = 0.02

SHOW_VOXEL_EDGES = False
EDGE_COLOR = "#5a5a5a"
EDGE_LINE_WIDTH = 0.20
SHOW_AXES = False

WHITE_THRESHOLD = 248
CROP_MARGIN_PX = 4

PREPROCESS_TARGET_HEIGHT = 1100
AUTO_CONTRAST_BLEND = 0.78
SHADOW_LIFT = 12.0
SHADOW_PIVOT = 150.0
SHADOW_POWER = 1.45
LOCAL_RADIUS = 7.0
LOCAL_AMOUNT = 0.22
UNSHARP_RADIUS = 0.80
UNSHARP_PERCENT = 55
UNSHARP_THRESHOLD = 3

TIO2_MASK_Y_MAX = 125.0
GRAPHITE_GRAY_MIN = 58.0
GRAPHITE_GRAY_MAX = 102.0
GRAPHITE_GAMMA = 0.92


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a QWP unit cell directly from a binary NumPy array."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input .npy file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered images.",
    )
    parser.add_argument(
        "--prefix",
        default="qwp_unitcell",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--azimuths",
        type=float,
        nargs="+",
        default=list(AZIMUTH_DEG_LIST),
        help="Camera azimuth angles in degrees.",
    )
    return parser.parse_args()


def apply_orientation(rho):
    if TRANSPOSE_XY:
        rho = np.transpose(rho, (1, 0, 2))
    if FLIP_X:
        rho = np.flip(rho, axis=0)
    if FLIP_Y:
        rho = np.flip(rho, axis=1)
    if FLIP_Z:
        rho = np.flip(rho, axis=2)
    return rho


def load_rho(path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rho = np.asarray(np.load(path, allow_pickle=False))

    if rho.ndim == 3:
        pass
    elif rho.ndim == 2:
        rho = rho[:, :, np.newaxis]
    elif rho.ndim == 1:
        expected_size = int(np.prod(RHO_SHAPE))
        if rho.size != expected_size:
            raise ValueError(
                f"Flattened array size is {rho.size}; expected "
                f"{expected_size} for shape {RHO_SHAPE}."
            )
        rho = rho.reshape(RHO_SHAPE, order=RESHAPE_ORDER)
    else:
        raise ValueError(f"Unsupported array shape: {rho.shape}")

    rho = apply_orientation(rho)

    print(f"[Input] {path}")
    print(f"[Shape] {rho.shape}")
    print(f"[Range] {rho.min():.6f} to {rho.max():.6f}")
    print(f"[Occupied fraction] {np.mean(rho > THRESHOLD):.6f}")

    return rho


def make_box(bounds):
    return pv.Box(bounds=bounds, quads=True)


def build_structure_mesh(rho):
    nx, ny, nz = rho.shape

    x_edges = np.linspace(-PERIOD_X_UM / 2, PERIOD_X_UM / 2, nx + 1)
    y_edges = np.linspace(-PERIOD_Y_UM / 2, PERIOD_Y_UM / 2, ny + 1)
    z_edges = np.linspace(0, STRUCTURE_THICKNESS_UM, nz + 1)

    voxel_meshes = []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if rho[ix, iy, iz] > THRESHOLD:
                    voxel_meshes.append(
                        make_box(
                            (
                                x_edges[ix],
                                x_edges[ix + 1],
                                y_edges[iy],
                                y_edges[iy + 1],
                                z_edges[iz],
                                z_edges[iz + 1],
                            )
                        )
                    )

    if not voxel_meshes:
        raise RuntimeError("No occupied voxels were found.")

    merged = voxel_meshes[0]
    for mesh in voxel_meshes[1:]:
        merged = merged.merge(mesh, merge_points=True)

    merged = merged.clean(tolerance=1e-12)
    merged = merged.triangulate()
    merged = merged.compute_normals(
        cell_normals=False,
        point_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False,
    )

    print(f"[Occupied voxels] {len(voxel_meshes)}")
    return merged


def build_substrate_mesh():
    mx = SUBSTRATE_XY_MARGIN_RATIO * PERIOD_X_UM
    my = SUBSTRATE_XY_MARGIN_RATIO * PERIOD_Y_UM

    substrate = make_box(
        (
            -PERIOD_X_UM / 2 - mx,
            PERIOD_X_UM / 2 + mx,
            -PERIOD_Y_UM / 2 - my,
            PERIOD_Y_UM / 2 + my,
            -PHYSICAL_SUBSTRATE_THICKNESS_UM,
            0.0,
        )
    )

    z_top = substrate.bounds[5]
    points = substrate.points.copy()
    points[:, 2] = z_top - (
        z_top - points[:, 2]
    ) * SUBSTRATE_VISUAL_Z_SCALE
    substrate.points = points

    substrate = substrate.triangulate()
    substrate = substrate.compute_normals(
        cell_normals=False,
        point_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False,
    )
    return substrate


def get_scene_geometry(meshes):
    bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=float)

    combined = (
        bounds[:, 0].min(),
        bounds[:, 1].max(),
        bounds[:, 2].min(),
        bounds[:, 3].max(),
        bounds[:, 4].min(),
        bounds[:, 5].max(),
    )

    center = np.array(
        [
            0.5 * (combined[0] + combined[1]),
            0.5 * (combined[2] + combined[3]),
            0.5 * (combined[4] + combined[5]),
        ]
    )

    extent = np.array(
        [
            combined[1] - combined[0],
            combined[3] - combined[2],
            combined[5] - combined[4],
        ]
    )

    return center, float(np.linalg.norm(extent))


def make_view_vector(azimuth_deg):
    theta = math.radians(azimuth_deg)
    return np.array(
        [
            VIEW_XY_RADIUS * math.cos(theta),
            VIEW_XY_RADIUS * math.sin(theta),
            VIEW_Z_COMPONENT,
        ],
        dtype=float,
    )


def set_camera(plotter, view_vector, center, diagonal):
    view_direction = view_vector / np.linalg.norm(view_vector)
    camera_position = center + view_direction * (
        CAMERA_DISTANCE_FACTOR * diagonal
    )

    plotter.camera_position = [
        tuple(camera_position),
        tuple(center),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = USE_PARALLEL_PROJECTION

    if not USE_PARALLEL_PROJECTION:
        plotter.camera.view_angle = 22

    plotter.camera.zoom(CAMERA_ZOOM)
    plotter.camera.clipping_range = (0.001, 1000)


def make_plotter(structure, substrate):
    plotter = pv.Plotter(
        window_size=WINDOW_SIZE,
        off_screen=OFF_SCREEN,
    )
    plotter.set_background(BACKGROUND_COLOR)

    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        try:
            plotter.enable_anti_aliasing()
        except Exception:
            pass

    plotter.add_mesh(
        substrate,
        color=SUBSTRATE_COLOR,
        opacity=SUBSTRATE_OPACITY,
        smooth_shading=False,
        show_edges=False,
        ambient=SUBSTRATE_AMBIENT,
        diffuse=SUBSTRATE_DIFFUSE,
        specular=SUBSTRATE_SPECULAR,
    )

    plotter.add_mesh(
        structure,
        color=STRUCTURE_COLOR,
        opacity=STRUCTURE_OPACITY,
        smooth_shading=False,
        show_edges=SHOW_VOXEL_EDGES,
        edge_color=EDGE_COLOR,
        line_width=EDGE_LINE_WIDTH,
        ambient=STRUCTURE_AMBIENT,
        diffuse=STRUCTURE_DIFFUSE,
        specular=STRUCTURE_SPECULAR,
    )

    if SHOW_AXES:
        plotter.add_axes(xlabel="x", ylabel="y", zlabel="z")

    return plotter


def crop_white_margin(image):
    array = np.asarray(image.convert("RGB"))
    mask = np.any(array < WHITE_THRESHOLD, axis=2)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return image

    left = max(0, int(xs.min()) - CROP_MARGIN_PX)
    right = min(image.width, int(xs.max()) + 1 + CROP_MARGIN_PX)
    upper = max(0, int(ys.min()) - CROP_MARGIN_PX)
    lower = min(image.height, int(ys.max()) + 1 + CROP_MARGIN_PX)

    return image.crop((left, upper, right, lower))


def resize_by_height(image, target_height):
    scale = target_height / image.height
    width = max(1, int(round(image.width * scale)))
    return image.resize(
        (width, target_height),
        Image.Resampling.LANCZOS,
    )


def object_mask(rgb):
    rgb = rgb.astype(np.float32)
    luminance = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    return np.any(rgb < WHITE_THRESHOLD, axis=2) & (luminance > 18)


def smooth_mask(mask):
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    return np.asarray(image, dtype=np.float32) / 255.0


def apply_contrast(image):
    image = image.convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)

    ycbcr = image.convert("YCbCr")
    y_image, cb, cr = ycbcr.split()
    source_y = np.asarray(y_image, dtype=np.float32)

    mask_bool = object_mask(rgb)
    if np.count_nonzero(mask_bool) < 100:
        return image

    blend_mask = smooth_mask(mask_bool)
    object_y = source_y[mask_bool]

    low = np.percentile(object_y, 3)
    high = np.percentile(object_y, 97)

    if high <= low + 1:
        low = np.percentile(object_y, 1)
        high = np.percentile(object_y, 99)

    stretched = np.clip(
        (source_y - low) / max(high - low, 1e-6),
        0,
        1,
    )
    stretched = stretched ** 0.88 * 255.0

    y1 = source_y * (
        1 - AUTO_CONTRAST_BLEND * blend_mask
    ) + stretched * (
        AUTO_CONTRAST_BLEND * blend_mask
    )

    shadow_weight = np.clip(
        (SHADOW_PIVOT - y1) / SHADOW_PIVOT,
        0,
        1,
    ) ** SHADOW_POWER

    y2 = y1 + SHADOW_LIFT * shadow_weight * blend_mask

    blurred = np.asarray(
        Image.fromarray(
            np.clip(y2, 0, 255).astype(np.uint8),
            mode="L",
        ).filter(ImageFilter.GaussianBlur(radius=LOCAL_RADIUS)),
        dtype=np.float32,
    )

    y3 = y2 + LOCAL_AMOUNT * (y2 - blurred) * blend_mask

    sharpened = Image.fromarray(
        np.clip(y3, 0, 255).astype(np.uint8),
        mode="L",
    ).filter(
        ImageFilter.UnsharpMask(
            radius=UNSHARP_RADIUS,
            percent=UNSHARP_PERCENT,
            threshold=UNSHARP_THRESHOLD,
        )
    )

    output_y = source_y * (1 - blend_mask) + np.asarray(
        sharpened,
        dtype=np.float32,
    ) * blend_mask

    return Image.merge(
        "YCbCr",
        (
            Image.fromarray(
                np.clip(output_y, 0, 255).astype(np.uint8),
                mode="L",
            ),
            cb,
            cr,
        ),
    ).convert("RGB")


def recolor_structure(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )

    mask = (
        np.any(rgb < WHITE_THRESHOLD, axis=2)
        & (luminance <= TIO2_MASK_Y_MAX)
    )

    if np.count_nonzero(mask) < 50:
        return image

    values = luminance[mask]
    low = np.percentile(values, 2)
    high = np.percentile(values, 98)

    if high <= low + 1:
        low = float(values.min())
        high = float(values.max() + 1)

    normalized = np.clip(
        (luminance - low) / max(high - low, 1e-6),
        0,
        1,
    ) ** GRAPHITE_GAMMA

    gray = GRAPHITE_GRAY_MIN + normalized * (
        GRAPHITE_GRAY_MAX - GRAPHITE_GRAY_MIN
    )

    output = rgb.copy()
    output[..., 0][mask] = gray[mask]
    output[..., 1][mask] = gray[mask]
    output[..., 2][mask] = gray[mask]

    return Image.fromarray(
        np.clip(output, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def process_render(raw_path, tight_path, final_path):
    image = Image.open(raw_path).convert("RGB")
    image = crop_white_margin(image)
    image.save(tight_path, dpi=(600, 600))

    image = resize_by_height(image, PREPROCESS_TARGET_HEIGHT)
    image = apply_contrast(image)
    image = recolor_structure(image)
    image.save(final_path, dpi=(600, 600))


def render_view(
    structure,
    substrate,
    center,
    diagonal,
    azimuth,
    output_dir,
    prefix,
):
    raw_path = output_dir / f"{prefix}_az{azimuth:03.0f}_raw.png"
    tight_path = output_dir / f"{prefix}_az{azimuth:03.0f}_tight.png"
    final_path = output_dir / f"{prefix}_az{azimuth:03.0f}_final.png"

    plotter = make_plotter(structure, substrate)
    set_camera(
        plotter,
        make_view_vector(azimuth),
        center,
        diagonal,
    )
    plotter.show(
        screenshot=str(raw_path),
        auto_close=True,
    )

    process_render(raw_path, tight_path, final_path)

    print(f"[Saved] {final_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rho = load_rho(args.input)
    structure = build_structure_mesh(rho)
    substrate = build_substrate_mesh()
    center, diagonal = get_scene_geometry([structure, substrate])

    for azimuth in args.azimuths:
        render_view(
            structure,
            substrate,
            center,
            diagonal,
            azimuth,
            args.output_dir,
            args.prefix,
        )


if __name__ == "__main__":
    main()
