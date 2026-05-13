"""
Visualize EXIF GPS data from JPG frames.

Pass either a single .jpg, a single cam folder, or a parent folder containing
cam0..cam5 subfolders. The script reads EXIF GPS from every JPG it finds and
plots:

- Top    : 2D scatter of lon/lat per camera, with heading arrows (if available)
- Bottom : altitude vs. frame index per camera

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


def read_exif_gps(jpg_path: Path) -> dict | None:
    try:
        with pyexiv2.Image(str(jpg_path)) as img:
            exif = img.read_exif()
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

    try:
        alt = _rational_to_float(exif.get("Exif.GPSInfo.GPSAltitude", "0/1"))
    except Exception:
        alt = float("nan")

    heading = None
    dir_str = exif.get("Exif.GPSInfo.GPSImgDirection")
    if dir_str is not None:
        try:
            heading = _rational_to_float(dir_str)
        except Exception:
            heading = None

    return {"lat": lat, "lon": lon, "alt": alt, "heading": heading}


def find_cam_dirs(path: Path) -> dict[str, list[Path]]:
    """Return {cam_label: [jpg paths]} based on what `path` points to.

    - Single .jpg                       -> {"frame": [path]}
    - A folder of .jpgs (e.g. cam0)     -> {<folder name>: [jpgs]}
    - A parent containing cam* folders  -> {cam0: [...], cam1: [...], ...}
    """
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
    """Read EXIF GPS for every JPG, grouped by cam label. Skips files without GPS."""
    result: dict[str, list[dict]] = {}
    for cam, files in cam_files.items():
        rows = []
        for jpg in files:
            gps = read_exif_gps(jpg)
            if gps is not None:
                rows.append({"file": jpg.name, **gps})
        if rows:
            result[cam] = rows
        print(f"  {cam}: {len(rows)}/{len(files)} files with GPS"
              + (f" ({sum(1 for r in rows if r['heading'] is not None)} with heading)"
                 if rows else ""))
    return result


def plot(cam_data: dict[str, list[dict]], arrow_every: int, save_path):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, max(6, len(cam_data))))
    arrow_len_deg = 5e-5

    for i, (cam, rows) in enumerate(sorted(cam_data.items())):
        lons = np.array([r["lon"] for r in rows])
        lats = np.array([r["lat"] for r in rows])
        alts = np.array([r["alt"] for r in rows])
        c = colors[i]

        ax_top.scatter(lons, lats, s=8, color=c, label=cam, zorder=2)

        headings = np.array(
            [r["heading"] if r["heading"] is not None else np.nan for r in rows]
        )
        if np.isfinite(headings).any():
            idx = np.arange(0, len(rows), max(1, arrow_every))
            valid = idx[np.isfinite(headings[idx])]
            if len(valid):
                hd_rad = np.deg2rad(headings[valid])
                dx = np.sin(hd_rad) * arrow_len_deg
                dy = np.cos(hd_rad) * arrow_len_deg
                ax_top.quiver(
                    lons[valid], lats[valid], dx, dy,
                    angles="xy", scale_units="xy", scale=1,
                    color=c, width=0.003, alpha=0.7, zorder=3,
                )

        ax_bot.plot(np.arange(len(rows)), alts, color=c, linewidth=1, label=cam)

    ax_top.set_xlabel("Longitude (deg)")
    ax_top.set_ylabel("Latitude (deg)")
    ax_top.set_title("EXIF GPS positions (arrows = GPSImgDirection)")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.legend(loc="best", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    ax_bot.set_xlabel("Frame index (within camera)")
    ax_bot.set_ylabel("Altitude (m)")
    ax_bot.set_title("EXIF altitude per camera")
    ax_bot.legend(loc="best", fontsize=8)
    ax_bot.grid(True, alpha=0.3)

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

    print(f"Reading EXIF from {args.path}")
    cam_data = collect(cam_files)

    if not cam_data:
        print("No frames had GPS EXIF — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    plot(cam_data, args.arrow_every, args.save)


if __name__ == "__main__":
    main()
