"""
Visualize the first N frames per camera as 3D camera frustums using Open3D.

Each camera pose is drawn as a small pyramid: the apex sits at the GPS
position, and the rectangular base shows where the camera "sees". One color
per camera, so you can read which cam0..cam5 frustum is which.

Reads EXIF (GPS lat/lon/alt) and XMP (PoseHeadingDegrees, PosePitchDegrees,
PoseRollDegrees) for each JPG.

Run:
    python show_poses_o3d.py <path-to-georeferenced_frames>
    python show_poses_o3d.py <path> --first 5 --size 0.3 --hfov 90 --vfov 60
    python show_poses_o3d.py <path> --range 10:50          # frames [10, 50)
    python show_poses_o3d.py <path> --range :              # all frames
    python show_poses_o3d.py <path> --range 0:200 --skip 3 # every 3rd frame in [0, 200)

Requires (one extra install on top of the existing venv):
    pip install open3d
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pyexiv2


XMP_HEADING = "Xmp.GPano.PoseHeadingDegrees"
XMP_PITCH = "Xmp.GPano.PosePitchDegrees"
XMP_ROLL = "Xmp.GPano.PoseRollDegrees"


CAM_COLORS = [
    [1.00, 0.00, 0.00],  # cam0  red
    [0.00, 0.60, 0.00],  # cam1  green
    [0.00, 0.40, 1.00],  # cam2  blue
    [1.00, 0.55, 0.00],  # cam3  orange
    [0.65, 0.00, 0.85],  # cam4  purple
    [0.00, 0.75, 0.75],  # cam5  cyan
]


def _rational_to_float(rat) -> float:
    if isinstance(rat, (int, float)):
        return float(rat)
    s = str(rat).strip()
    if "/" in s:
        num, den = s.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(s)


def _dms_to_decimal(dms_str: str, ref: str) -> float:
    parts = str(dms_str).split()
    d = _rational_to_float(parts[0])
    m = _rational_to_float(parts[1]) if len(parts) > 1 else 0.0
    s = _rational_to_float(parts[2]) if len(parts) > 2 else 0.0
    decimal = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _safe_float(val):
    if val is None:
        return None
    try:
        return _rational_to_float(val)
    except Exception:
        try:
            return float(val)
        except Exception:
            return None


def read_pose(jpg_path: Path) -> dict | None:
    try:
        with pyexiv2.Image(str(jpg_path)) as img:
            exif = img.read_exif()
            try:
                xmp = img.read_xmp()
            except Exception:
                xmp = {}
    except Exception:
        return None

    lat_str = exif.get("Exif.GPSInfo.GPSLatitude")
    lat_ref = exif.get("Exif.GPSInfo.GPSLatitudeRef")
    lon_str = exif.get("Exif.GPSInfo.GPSLongitude")
    lon_ref = exif.get("Exif.GPSInfo.GPSLongitudeRef")
    if not (lat_str and lat_ref and lon_str and lon_ref):
        return None

    try:
        lat = _dms_to_decimal(lat_str, lat_ref)
        lon = _dms_to_decimal(lon_str, lon_ref)
    except Exception:
        return None

    alt = _safe_float(exif.get("Exif.GPSInfo.GPSAltitude", "0/1"))
    heading = _safe_float(xmp.get(XMP_HEADING))
    if heading is None:
        heading = _safe_float(exif.get("Exif.GPSInfo.GPSImgDirection"))
    pitch = _safe_float(xmp.get(XMP_PITCH))
    roll = _safe_float(xmp.get(XMP_ROLL))

    if heading is None or pitch is None or roll is None:
        return None

    return {
        "file": jpg_path.name,
        "lat": lat, "lon": lon,
        "alt": alt if alt is not None else 0.0,
        "heading": heading, "pitch": pitch, "roll": roll,
    }


def find_cam_dirs(path: Path) -> dict[str, list[Path]]:
    if not path.is_dir():
        return {}
    direct = sorted(path.glob("*.jpg")) + sorted(path.glob("*.JPG"))
    if direct:
        return {path.name: direct}
    out: dict[str, list[Path]] = {}
    for sub in sorted(p for p in path.iterdir() if p.is_dir()):
        jpgs = sorted(sub.glob("*.jpg")) + sorted(sub.glob("*.JPG"))
        if jpgs:
            out[sub.name] = jpgs
    return out


def lonlat_to_local_xy(lon, lat, lon_ref, lat_ref):
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat_ref))
    x = (lon - lon_ref) * m_per_deg_lon
    y = (lat - lat_ref) * m_per_deg_lat
    return x, y


def euler_to_rotmat(heading_deg, pitch_deg, roll_deg) -> np.ndarray:
    """
    Decode (heading, pitch, roll) from the XMP/EXIF tags back into the
    body-to-world rotation matrix, reversing exactly what the pipeline does in
    calculate_exif_meta.py:

        to_euler     = inverse(R_body_to_world).as_euler("ZYX")
        yaw_to_heading = [euler[2], euler[1], euler[0] + 90]

    So to recover R_body_to_world we:
        1. yaw_math = heading - 90       (undo the +90 offset)
        2. build R_inv via scipy "ZYX" intrinsic = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        3. R_body_to_world = R_inv.T     (undo the inversion)
    """
    yaw = np.deg2rad(heading_deg - 90.0)
    p = np.deg2rad(pitch_deg)
    r = np.deg2rad(roll_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    cp, sp = np.cos(p), np.sin(p)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    R_inv = Rz @ Ry @ Rx
    return R_inv.T


FORWARD_AXIS_MAP = {
    "y":  ("y", +1),  "+y": ("y", +1),
    "-y": ("y", -1),
    "x":  ("x", +1),  "+x": ("x", +1),
    "-x": ("x", -1),
    "z":  ("z", +1),  "+z": ("z", +1),
    "-z": ("z", -1),
}


def _build_body_corners(forward_axis: str, depth: float, hw: float, hh: float):
    """Return 5 body-frame points (apex + 4 base corners) for a frustum whose
    optical axis is the given body axis ('+y', '-x', etc.)."""
    axis, sign = FORWARD_AXIS_MAP[forward_axis]
    d = sign * depth

    if axis == "y":
        return np.array([
            [0, 0, 0],
            [+hw, d, +hh], [-hw, d, +hh], [-hw, d, -hh], [+hw, d, -hh],
        ])
    if axis == "x":
        return np.array([
            [0, 0, 0],
            [d, +hw, +hh], [d, -hw, +hh], [d, -hw, -hh], [d, +hw, -hh],
        ])
    # z
    return np.array([
        [0, 0, 0],
        [+hw, +hh, d], [-hw, +hh, d], [-hw, -hh, d], [+hw, -hh, d],
    ])


def make_frustum(position, R, depth, hfov_deg, vfov_deg, color, forward_axis):
    """
    Build a SOLID camera frustum as an Open3D TriangleMesh (open at the back).

    `forward_axis` is which body-frame axis is the optical axis: one of
    'y', '+y', '-y', 'x', '+x', '-x', 'z', '+z', '-z'.
    """
    hw = depth * np.tan(np.deg2rad(hfov_deg) / 2)
    hh = depth * np.tan(np.deg2rad(vfov_deg) / 2)
    body_pts = _build_body_corners(forward_axis, depth, hw, hh)

    world_pts = (R @ body_pts.T).T + np.asarray(position)

    # 4 side triangles only (no base) so you can see "through" the back of
    # the frustum and it still reads as a camera shape.
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(world_pts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def make_dot(position, radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    sphere.translate(np.asarray(position))
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def make_trajectory(positions, color):
    """LineSet connecting consecutive camera positions, showing the path."""
    if len(positions) < 2:
        return None
    pts = np.asarray(positions, dtype=float)
    lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=int)
    colors = np.tile(np.asarray(color, dtype=float), (len(lines), 1))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def parse_range(s: str, total: int) -> tuple[int, int]:
    """Parse '10:50', ':50', '10:', or ':' against a total list length."""
    if ":" not in s:
        raise ValueError(f"--range must contain ':' (got {s!r})")
    a, b = s.split(":", 1)
    start = int(a) if a.strip() else 0
    end = int(b) if b.strip() else total
    if start < 0:
        start = max(total + start, 0)
    if end < 0:
        end = max(total + end, 0)
    return start, end


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", type=Path,
                    help="Parent folder containing cam0..cam5 (or a single cam folder)")
    ap.add_argument("--first", type=int, default=5,
                    help="Leading frames per camera (default: 5). Ignored if --range is given.")
    ap.add_argument("--range", dest="frame_range", type=str, default=None,
                    help="Frame index range per camera, e.g. '10:50', ':' for all, "
                         "':50' for first 50, '10:' from 10 to end. Overrides --first.")
    ap.add_argument("--skip", type=int, default=1,
                    help="Step between frames (default: 1). E.g. --skip 3 keeps every 3rd frame.")
    ap.add_argument("--no-trajectory", action="store_true",
                    help="Disable the polyline connecting consecutive camera positions.")
    ap.add_argument("--size", type=float, default=0.12,
                    help="Frustum depth in metres (default: 0.12)")
    ap.add_argument("--hfov", type=float, default=90.0,
                    help="Horizontal FOV in degrees, for drawing only (default: 90)")
    ap.add_argument("--vfov", type=float, default=60.0,
                    help="Vertical FOV in degrees, for drawing only (default: 60)")
    ap.add_argument("--forward-axis", default="y",
                    choices=list(FORWARD_AXIS_MAP.keys()),
                    help="Which body axis is the camera optical axis "
                         "(try: y, x, z, -y, -x, -z). Default: y")
    args = ap.parse_args()

    if args.skip < 1:
        print(f"Error: --skip must be >= 1 (got {args.skip})", file=sys.stderr)
        sys.exit(1)

    if not args.path.exists():
        print(f"Error: path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    cam_files = find_cam_dirs(args.path)
    if not cam_files:
        print(f"No .jpg files found at {args.path}", file=sys.stderr)
        sys.exit(1)

    if args.frame_range is not None:
        sel_desc = f"range {args.frame_range} (skip {args.skip})"
    else:
        sel_desc = f"first {args.first} (skip {args.skip})"
    print(f"Reading {sel_desc} poses per camera from {args.path}")

    cam_poses: dict[str, list[dict]] = {}
    for cam, files in cam_files.items():
        if args.frame_range is not None:
            start, end = parse_range(args.frame_range, len(files))
        else:
            start, end = 0, min(args.first, len(files))
        chosen = files[start:end:args.skip]
        poses = []
        for jpg in chosen:
            p = read_pose(jpg)
            if p is not None:
                poses.append(p)
        if poses:
            cam_poses[cam] = poses
        print(f"  {cam}: {len(poses)}/{len(chosen)} with full pose "
              f"(from {start}:{end}:{args.skip} of {len(files)} files)")

    if not cam_poses:
        print("No poses with full lat/lon/yaw/pitch/roll — nothing to draw.",
              file=sys.stderr)
        sys.exit(1)

    first_cam = sorted(cam_poses.keys())[0]
    ref = cam_poses[first_cam][0]
    lon_ref, lat_ref, alt_ref = ref["lon"], ref["lat"], ref["alt"]

    geometries: list = []

    # Single black dot at the world origin (was the big ENU triad).
    origin_dot = make_dot(
        position=[0, 0, 0],
        radius=args.size * 0.25,
        color=[0.0, 0.0, 0.0],
    )
    geometries.append(origin_dot)

    apex_radius = args.size * 0.12

    sorted_cams = sorted(cam_poses.keys())
    for cam_idx, cam in enumerate(sorted_cams):
        color = CAM_COLORS[cam_idx % len(CAM_COLORS)]
        cam_positions: list[list[float]] = []
        for r in cam_poses[cam]:
            x, y = lonlat_to_local_xy(r["lon"], r["lat"], lon_ref, lat_ref)
            z = r["alt"] - alt_ref
            position = [x, y, z]
            cam_positions.append(position)

            R = euler_to_rotmat(r["heading"], r["pitch"], r["roll"])
            frustum = make_frustum(
                position=position,
                R=R,
                depth=args.size,
                hfov_deg=args.hfov,
                vfov_deg=args.vfov,
                color=color,
                forward_axis=args.forward_axis,
            )
            geometries.append(frustum)
            geometries.append(make_dot(position, apex_radius, color))

        if not args.no_trajectory:
            traj = make_trajectory(cam_positions, color)
            if traj is not None:
                geometries.append(traj)

    print("\nCamera colors:")
    for cam_idx, cam in enumerate(sorted_cams):
        c = CAM_COLORS[cam_idx % len(CAM_COLORS)]
        print(f"  {cam}: RGB ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")

    print("\nFirst-frame raw pose values per camera (XMP / EXIF):")
    print(f"  {'cam':<6} {'heading':>10} {'pitch':>10} {'roll':>10}   file")
    for cam in sorted_cams:
        r = cam_poses[cam][0]
        print(
            f"  {cam:<6} {r['heading']:>10.3f} {r['pitch']:>10.3f} "
            f"{r['roll']:>10.3f}   {r['file']}"
        )
    print("Adjacent cams on a 360 rig should differ by ~60 deg in heading.")
    print("Identical numbers between two cams = a duplicate-trajectory bug.")

    # For each camera, show where each body axis ends up pointing in the world,
    # so you can tell which body axis is actually the optical axis.
    #   bearing = compass heading (0=N, 90=E, 180=S, 270=W)
    #   elev    = degrees above horizon (+ up, - down)
    def world_dir(R, axis):
        v = R @ axis
        bearing = (np.degrees(np.arctan2(v[0], v[1])) + 360) % 360  # X=E, Y=N
        elev = np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
        return bearing, elev

    print("\nWorld-frame direction of each body axis per camera "
          "(bearing deg from north / elevation deg above horizon):")
    print(f"  {'cam':<6}   {'body +X':>20}   {'body +Y':>20}   {'body +Z':>20}")
    for cam in sorted_cams:
        r = cam_poses[cam][0]
        R = euler_to_rotmat(r["heading"], r["pitch"], r["roll"])
        bx, ex = world_dir(R, np.array([1.0, 0.0, 0.0]))
        by, ey = world_dir(R, np.array([0.0, 1.0, 0.0]))
        bz, ez = world_dir(R, np.array([0.0, 0.0, 1.0]))
        print(f"  {cam:<6}   {bx:>7.1f}/{ex:+6.1f}     "
              f"{by:>7.1f}/{ey:+6.1f}     {bz:>7.1f}/{ez:+6.1f}")
    print("Pick the column where the values look most like a sensible camera "
          "layout, and pass it as --forward-axis (y/x/z, or with - prefix).")
    print("\nBlack dot = world origin (first frame, first camera).")
    print("Each colored dot = a camera position; the colored pyramid points where it looks.")
    if not args.no_trajectory:
        print("Colored polyline = trajectory through consecutive frames of that camera.")
    print("Left-drag = rotate, scroll = zoom, "
          "Ctrl+left-drag or middle-mouse-drag = pan/move. Press Q to quit.")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera frustums (first frames per cam)",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()
