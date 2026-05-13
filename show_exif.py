"""
Visualize EXIF + XMP camera pose from JPG frames.

Pass either a single .jpg, a single cam folder, or a parent folder containing
cam0..cam5 subfolders. The script reads:

- EXIF GPS  : GPSLatitude, GPSLongitude, GPSAltitude, GPSImgDirection (yaw)
- XMP GPano : PoseHeadingDegrees, PosePitchDegrees, PoseRollDegrees

and plots, per camera:

- Top:    lon/lat scatter with yaw arrows
- Bottom: three stacked line plots (yaw, pitch, roll) vs. frame index

Uses only matplotlib + pyexiv2 + stdlib (already in the venv).

Run:
    python show_exif.py <path>
    python show_exif.py <path> --arrow-every 50
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
    """Return per-frame pose dict or None if no GPS in EXIF."""
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

    # Heading priority: XMP PoseHeadingDegrees > EXIF GPSImgDirection
    heading = _safe_float(xmp.get(XMP_HEADING))
    if heading is None:
        heading = _safe_float(exif.get("Exif.GPSInfo.GPSImgDirection"))

    pitch = _safe_float(xmp.get(XMP_PITCH))
    roll = _safe_float(xmp.get(XMP_ROLL))

    return {
        "file": jpg_path.name,
        "lat": lat,
        "lon": lon,
        "alt": alt if alt is not None else float("nan"),
        "heading": heading,
        "pitch": pitch,
        "roll": roll,
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


def plot(cam_data: dict[str, list[dict]], arrow_every: int, save_path):
    fig = plt.figure(figsize=(11, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.4)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_yaw = fig.add_subplot(gs[1, 0])
    ax_pitch = fig.add_subplot(gs[2, 0], sharex=ax_yaw)
    ax_roll = fig.add_subplot(gs[3, 0], sharex=ax_yaw)

    colors = plt.cm.tab10(np.linspace(0, 1, max(6, len(cam_data))))
    arrow_len_deg = 5e-5

    for i, (cam, rows) in enumerate(sorted(cam_data.items())):
        lons = np.array([r["lon"] for r in rows])
        lats = np.array([r["lat"] for r in rows])
        headings = np.array([np.nan if r["heading"] is None else r["heading"] for r in rows])
        pitches = np.array([np.nan if r["pitch"] is None else r["pitch"] for r in rows])
        rolls = np.array([np.nan if r["roll"] is None else r["roll"] for r in rows])
        c = colors[i]

        ax_map.scatter(lons, lats, s=8, color=c, label=cam, zorder=2)

        if np.isfinite(headings).any():
            idx = np.arange(0, len(rows), max(1, arrow_every))
            valid = idx[np.isfinite(headings[idx])]
            if len(valid):
                hd_rad = np.deg2rad(headings[valid])
                dx = np.sin(hd_rad) * arrow_len_deg
                dy = np.cos(hd_rad) * arrow_len_deg
                ax_map.quiver(
                    lons[valid], lats[valid], dx, dy,
                    angles="xy", scale_units="xy", scale=1,
                    color=c, width=0.003, alpha=0.7, zorder=3,
                )

        x = np.arange(len(rows))
        ax_yaw.plot(x, headings, color=c, linewidth=1, label=cam)
        ax_pitch.plot(x, pitches, color=c, linewidth=1, label=cam)
        ax_roll.plot(x, rolls, color=c, linewidth=1, label=cam)

    ax_map.set_xlabel("Longitude (deg)")
    ax_map.set_ylabel("Latitude (deg)")
    ax_map.set_title("Lon/Lat from EXIF (arrows = yaw / GPSImgDirection)")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.legend(loc="best", fontsize=8)
    ax_map.grid(True, alpha=0.3)

    for ax, title, ylabel in (
        (ax_yaw, "Yaw (heading)", "deg"),
        (ax_pitch, "Pitch", "deg"),
        (ax_roll, "Roll", "deg"),
    ):
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    ax_roll.set_xlabel("Frame index (within camera)")

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
    ap.add_argument("--arrow-every", type=int, default=20,
                    help="Draw a heading arrow every N frames (default: 20)")
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

    plot(cam_data, args.arrow_every, args.save)


if __name__ == "__main__":
    main()
