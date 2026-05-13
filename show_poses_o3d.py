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


def euler_to_rotmat(yaw_deg, pitch_deg, roll_deg) -> np.ndarray:
    """
    body x = right, body y = forward (optical axis), body z = up
    yaw  : clockwise from north (GPSImgDirection convention)
    pitch: about body x (nose up positive)
    roll : about body y (right wing down positive)
    """
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    r = np.deg2rad(roll_deg)

    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    cp, sp = np.cos(p), np.sin(p)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    cr, sr = np.cos(r), np.sin(r)
    Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])

    return Rz @ Rx @ Ry


def make_frustum(position, R, depth, hfov_deg, vfov_deg, color):
    """
    Build a wireframe camera frustum as an Open3D LineSet.

    The apex is at `position`. The forward axis (body Y) points down the
    optical axis. Base is at distance `depth` ahead, sized by hfov/vfov.
    """
    hw = depth * np.tan(np.deg2rad(hfov_deg) / 2)  # half width  along body x
    hh = depth * np.tan(np.deg2rad(vfov_deg) / 2)  # half height along body z

    # 5 points in body frame: apex + 4 base corners
    body_pts = np.array([
        [0.0, 0.0, 0.0],          # 0 apex
        [+hw, depth, +hh],        # 1 top-right
        [-hw, depth, +hh],        # 2 top-left
        [-hw, depth, -hh],        # 3 bottom-left
        [+hw, depth, -hh],        # 4 bottom-right
    ])

    world_pts = (R @ body_pts.T).T + np.asarray(position)

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],   # apex to corners
        [1, 2], [2, 3], [3, 4], [4, 1],   # base rectangle
        [1, 2],                           # (top edge highlighted by repeat)
    ]

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(world_pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", type=Path,
                    help="Parent folder containing cam0..cam5 (or a single cam folder)")
    ap.add_argument("--first", type=int, default=5,
                    help="Leading frames per camera (default: 5)")
    ap.add_argument("--size", type=float, default=0.3,
                    help="Frustum depth in metres (default: 0.3)")
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

    # World ENU axes at origin: red=East, green=North, blue=Up.
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=args.size * 3.0, origin=[0, 0, 0]
    )
    geometries.append(world_axes)

    sorted_cams = sorted(cam_poses.keys())
    for cam_idx, cam in enumerate(sorted_cams):
        color = CAM_COLORS[cam_idx % len(CAM_COLORS)]
        for r in cam_poses[cam]:
            x, y = lonlat_to_local_xy(r["lon"], r["lat"], lon_ref, lat_ref)
            z = r["alt"] - alt_ref

            R = euler_to_rotmat(r["heading"], r["pitch"], r["roll"])
            frustum = make_frustum(
                position=[x, y, z],
                R=R,
                depth=args.size,
                hfov_deg=args.hfov,
                vfov_deg=args.vfov,
                color=color,
            )
            geometries.append(frustum)

    print("\nCamera colors:")
    for cam_idx, cam in enumerate(sorted_cams):
        c = CAM_COLORS[cam_idx % len(CAM_COLORS)]
        print(f"  {cam}: RGB ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
    print("\nLarge triad at origin = world ENU (red=E, green=N, blue=Up).")
    print("Drag to rotate, scroll to zoom, right-drag to pan. Press Q to quit.")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera frustums (first frames per cam)",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()
