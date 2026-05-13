"""
Visualize per-camera pose from EXIF + XMP of JPG frames.

Pass either a single .jpg, a single cam folder, or a parent folder containing
cam0..cam5 subfolders.

Two plots:
1. 2D lon/lat trajectory per camera (no clutter)
2. 3D pose triads for the first N frames per camera (default 10)
   Each pose is drawn as three short arrows in ENU local meters:
       red   = camera forward (yaw/pitch/roll applied)
       green = camera right
       blue  = camera up

Reads:
- EXIF: GPSLatitude/Longitude/Altitude, GPSImgDirection (yaw fallback)
- XMP : Xmp.GPano.PoseHeadingDegrees/PosePitchDegrees/PoseRollDegrees

Uses only matplotlib + pyexiv2 + stdlib (already in the venv).

Run:
    python show_exif.py <path>
    python show_exif.py <path> --first 20 --axis-len 0.3
    python show_exif.py <path> --save out.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyexiv2


XMP_HEADING = "Xmp.GPano.PoseHeadingDegrees"
XMP_PITCH = "Xmp.GPano.PosePitchDegrees"
XMP_ROLL = "Xmp.GPano.PoseRollDegrees"


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

    return {
        "file": jpg_path.name,
        "lat": lat, "lon": lon,
        "alt": alt if alt is not None else float("nan"),
        "heading": heading, "pitch": pitch, "roll": roll,
    }


def find_cam_dirs(path: Path) -> dict[str, list[Path]]:
    if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg"):
        return {"frame": [path]}
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


def collect(cam_files: dict[str, list[Path]]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for cam, files in cam_files.items():
        rows = [r for jpg in files if (r := read_pose(jpg)) is not None]
        if rows:
            result[cam] = rows
        nh = sum(1 for r in rows if r["heading"] is not None)
        npi = sum(1 for r in rows if r["pitch"] is not None)
        nr = sum(1 for r in rows if r["roll"] is not None)
        print(f"  {cam}: {len(rows)}/{len(files)} with GPS  "
              f"(yaw:{nh}  pitch:{npi}  roll:{nr})")
    return result


def lonlat_to_local_xy(lon, lat, lon_ref, lat_ref):
    """Approximate lon/lat to local ENU meters relative to (lon_ref, lat_ref)."""
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat_ref))
    x = (np.asarray(lon) - lon_ref) * m_per_deg_lon
    y = (np.asarray(lat) - lat_ref) * m_per_deg_lat
    return x, y


def euler_to_rotmat(yaw_deg, pitch_deg, roll_deg):
    """
    Build a body-to-world rotation matrix for ENU world frame.

    Convention:
        body x = right, body y = forward (optical axis), body z = up
        yaw   = heading, degrees clockwise from north (GPSImgDirection convention)
        pitch = rotation about body x (nose up positive)
        roll  = rotation about body y (right wing down positive)
    """
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    r = np.deg2rad(roll_deg)

    # Yaw: clockwise from north when viewed from above (Z up).
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    # Pitch around body X (after yaw)
    cp, sp = np.cos(p), np.sin(p)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    # Roll around body Y
    cr, sr = np.cos(r), np.sin(r)
    Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])

    return Rz @ Rx @ Ry


def plot(cam_data: dict[str, list[dict]], first_n: int, axis_len: float, save_path):
    # establish a common ENU origin from the first available frame across all cams
    all_lons = [r["lon"] for rows in cam_data.values() for r in rows]
    all_lats = [r["lat"] for rows in cam_data.values() for r in rows]
    all_alts = [r["alt"] for rows in cam_data.values() for r in rows if np.isfinite(r["alt"])]
    lon_ref, lat_ref = all_lons[0], all_lats[0]
    alt_ref = all_alts[0] if all_alts else 0.0

    fig = plt.figure(figsize=(14, 7))
    ax_map = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, max(6, len(cam_data))))

    for i, (cam, rows) in enumerate(sorted(cam_data.items())):
        c = colors[i]

        # --- left: full trajectory in lon/lat, just a line ---
        lons = np.array([r["lon"] for r in rows])
        lats = np.array([r["lat"] for r in rows])
        ax_map.plot(lons, lats, color=c, linewidth=1.2, label=cam)

        # --- right: first N poses as 3D triads ---
        head = rows[:first_n]
        for r in head:
            if r["heading"] is None or r["pitch"] is None or r["roll"] is None:
                continue
            x, y = lonlat_to_local_xy(r["lon"], r["lat"], lon_ref, lat_ref)
            z = (r["alt"] - alt_ref) if np.isfinite(r["alt"]) else 0.0

            R = euler_to_rotmat(r["heading"], r["pitch"], r["roll"])
            # body-frame axes -> world ENU
            ex = R @ np.array([1.0, 0.0, 0.0]) * axis_len  # body X = right
            ey = R @ np.array([0.0, 1.0, 0.0]) * axis_len  # body Y = forward
            ez = R @ np.array([0.0, 0.0, 1.0]) * axis_len  # body Z = up

            # forward (red), right (green), up (blue)
            ax_3d.quiver(x, y, z, ey[0], ey[1], ey[2], color="red", linewidth=1.5)
            ax_3d.quiver(x, y, z, ex[0], ex[1], ex[2], color="green", linewidth=1.5)
            ax_3d.quiver(x, y, z, ez[0], ez[1], ez[2], color="blue", linewidth=1.5)
            ax_3d.scatter([x], [y], [z], color=c, s=15)

    ax_map.set_xlabel("Longitude (deg)")
    ax_map.set_ylabel("Latitude (deg)")
    ax_map.set_title("Full trajectory from EXIF (lon / lat)")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.legend(loc="best", fontsize=8)
    ax_map.grid(True, alpha=0.3)

    ax_3d.set_xlabel("East (m, local)")
    ax_3d.set_ylabel("North (m, local)")
    ax_3d.set_zlabel("Up (m)")
    ax_3d.set_title(
        f"First {first_n} poses per camera\n"
        "red=forward  green=right  blue=up"
    )
    # rough equal-aspect for 3D
    try:
        ax_3d.set_box_aspect((1, 1, 0.5))
    except Exception:
        pass

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", type=Path,
                    help="A .jpg file, a cam folder, or a parent of cam0..cam5")
    ap.add_argument("--first", type=int, default=10,
                    help="How many leading frames per camera to draw as pose triads (default: 10)")
    ap.add_argument("--axis-len", type=float, default=0.3,
                    help="Length of each pose-axis arrow in metres (default: 0.3)")
    ap.add_argument("--save", type=Path, default=None,
                    help="Optional PNG output path; otherwise just shows the figure")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"Error: path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    cam_files = find_cam_dirs(args.path)
    if not cam_files:
        print(f"No .jpg files found at {args.path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading EXIF+XMP from {args.path}")
    cam_data = collect(cam_files)

    if not cam_data:
        print("No frames had GPS EXIF — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    plot(cam_data, args.first, args.axis_len, args.save)


if __name__ == "__main__":
    main()
