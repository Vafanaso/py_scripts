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


def make_frustum(position, R, depth, hfov_deg, vfov_deg, color):
    """
    Build a SOLID camera frustum as an Open3D TriangleMesh (open at the back).

    The apex is at `position`. The forward axis (body Y) points down the
    optical axis. Base is at distance `depth` ahead, sized by hfov/vfov.
    """
    hw = depth * np.tan(np.deg2rad(hfov_deg) / 2)
    hh = depth * np.tan(np.deg2rad(vfov_deg) / 2)

    body_pts = np.array([
        [0.0, 0.0, 0.0],          # 0 apex
        [+hw, depth, +hh],        # 1 top-right
        [-hw, depth, +hh],        # 2 top-left
        [-hw, depth, -hh],        # 3 bottom-left
        [+hw, depth, -hh],        # 4 bottom-right
    ])

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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", type=Path,
                    help="Parent folder containing cam0..cam5 (or a single cam folder)")
    ap.add_argument("--first", type=int, default=5,
                    help="Leading frames per camera (default: 5)")
    ap.add_argument("--size", type=float, default=0.12,
                    help="Frustum depth in metres (default: 0.12)")
    ap.add_argument("--hfov", type=float, default=90.0,
                    help="Horizontal FOV in degrees, for drawing only (default: 90)")
    ap.add_argument("--vfov", type=float, default=60.0,
                    help="Vertical FOV in degrees, for drawing only (default: 60)")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"Error: path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    cam_files = find_cam_dirs(args.path)
    if not cam_files:
        print(f"No .jpg files found at {args.path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading first {args.first} poses per camera from {args.path}")
    cam_poses: dict[str, list[dict]] = {}
    for cam, files in cam_files.items():
        poses = []
        for jpg in files[: args.first]:
            p = read_pose(jpg)
            if p is not None:
                poses.append(p)
        if poses:
            cam_poses[cam] = poses
        print(f"  {cam}: {len(poses)}/{min(args.first, len(files))} with full pose")

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
        for r in cam_poses[cam]:
            x, y = lonlat_to_local_xy(r["lon"], r["lat"], lon_ref, lat_ref)
            z = r["alt"] - alt_ref
            position = [x, y, z]

            R = euler_to_rotmat(r["heading"], r["pitch"], r["roll"])
            frustum = make_frustum(
                position=position,
                R=R,
                depth=args.size,
                hfov_deg=args.hfov,
                vfov_deg=args.vfov,
                color=color,
            )
            geometries.append(frustum)
            geometries.append(make_dot(position, apex_radius, color))

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
    print("\nBlack dot = world origin (first frame, first camera).")
    print("Each colored dot = a camera position; the colored pyramid points where it looks.")
    print("Drag to rotate, scroll to zoom, right-drag to pan. Press Q to quit.")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera frustums (first frames per cam)",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()
