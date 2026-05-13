"""
Visualize georeferenced frames against a trajectory CSV and a GPX track.

Reads:
- traj.csv: local SLAM trajectory with columns
  timestamp_ns, x, y, z, qx, qy, qz, qw
- A .gpx file: GPS ground truth in WGS84
- georeferenced_frames/cam0..cam5/*.jpg: frames with GPS written into EXIF

Produces a single figure with two subplots:
- Top    (WGS84): GPX track + per-camera EXIF GPS, with heading arrows
- Bottom (local): traj.csv x/y trajectory, start (green) and end (red) marked

If the georeferencing is working, the SHAPE of the bottom curve should match
the shape of the GPX/EXIF tracks in the top panel.

Run:
    pixi run python scripts/visualize_georef.py \\
        --frames-dir /path/to/georeferenced_frames \\
        --traj-csv   /path/to/traj.csv \\
        --gpx        /path/to/route.gpx
"""

import argparse
import sys
from pathlib import Path

import gpxpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyexiv2


TRAJ_COLS = ["timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw"]


def parse_traj_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=TRAJ_COLS)


def parse_gpx(path: Path) -> pd.DataFrame:
    with open(path) as f:
        gpx = gpxpy.parse(f)
    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                rows.append({"lat": p.latitude, "lon": p.longitude, "alt": p.elevation})
    return pd.DataFrame(rows)


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


def collect_cam_data(frames_dir: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for cam_id in range(6):
        cam_dir = frames_dir / f"cam{cam_id}"
        if not cam_dir.is_dir():
            continue
        rows = []
        for jpg in sorted(cam_dir.glob("*.jpg")):
            gps = read_exif_gps(jpg)
            if gps is not None:
                rows.append({"file": jpg.name, **gps})
        if rows:
            result[f"cam{cam_id}"] = pd.DataFrame(rows)
    return result


def plot(traj_df, gpx_df, cam_dfs, arrow_every, save_path):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 13))

    if not gpx_df.empty:
        ax_top.plot(
            gpx_df["lon"], gpx_df["lat"],
            color="grey", linewidth=2.0, label="GPX track", zorder=1,
        )

    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    arrow_len_deg = 5e-5  # ~5 m at the equator; visible at typical zoom

    for i, (cam, df) in enumerate(sorted(cam_dfs.items())):
        ax_top.scatter(
            df["lon"], df["lat"], s=8, color=colors[i], label=cam, zorder=2,
        )
        sub = df.iloc[::arrow_every]
        if "heading" in sub.columns:
            valid = sub.dropna(subset=["heading"])
            if not valid.empty:
                hd_rad = np.deg2rad(valid["heading"].to_numpy())
                dx = np.sin(hd_rad) * arrow_len_deg
                dy = np.cos(hd_rad) * arrow_len_deg
                ax_top.quiver(
                    valid["lon"], valid["lat"], dx, dy,
                    angles="xy", scale_units="xy", scale=1,
                    color=colors[i], width=0.003, alpha=0.7, zorder=3,
                )

    ax_top.set_xlabel("Longitude (deg)")
    ax_top.set_ylabel("Latitude (deg)")
    ax_top.set_title("WGS84 view: GPX track vs. per-camera EXIF GPS (arrows = heading)")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.legend(loc="best", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    if not traj_df.empty:
        ax_bot.plot(
            traj_df["x"], traj_df["y"],
            color="blue", linewidth=1.2, label="traj.csv", zorder=1,
        )
        ax_bot.scatter(
            traj_df["x"].iloc[0], traj_df["y"].iloc[0],
            color="green", s=90, label="start", zorder=5,
        )
        ax_bot.scatter(
            traj_df["x"].iloc[-1], traj_df["y"].iloc[-1],
            color="red", s=90, label="end", zorder=5,
        )
    ax_bot.set_xlabel("X (m, local SLAM)")
    ax_bot.set_ylabel("Y (m, local SLAM)")
    ax_bot.set_title("Local SLAM trajectory — shape should match the GPX track above")
    ax_bot.set_aspect("equal", adjustable="datalim")
    ax_bot.legend(loc="best")
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
    ap.add_argument("--frames-dir", required=True, type=Path,
                    help="Folder containing cam0..cam5 subfolders of .jpg frames")
    ap.add_argument("--traj-csv", required=True, type=Path,
                    help="Path to traj.csv (timestamp_ns,x,y,z,qx,qy,qz,qw)")
    ap.add_argument("--gpx", required=True, type=Path,
                    help="Path to .gpx track")
    ap.add_argument("--arrow-every", type=int, default=20,
                    help="Draw a heading arrow every N frames (default: 20)")
    ap.add_argument("--save", type=Path, default=None,
                    help="Optional PNG output path; otherwise just shows the figure")
    args = ap.parse_args()

    for label, p in (("frames-dir", args.frames_dir),
                     ("traj-csv", args.traj_csv),
                     ("gpx", args.gpx)):
        if not p.exists():
            print(f"Error: --{label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Reading traj.csv  from {args.traj_csv}")
    traj_df = parse_traj_csv(args.traj_csv)
    print(f"  {len(traj_df)} rows")

    print(f"Reading GPX       from {args.gpx}")
    gpx_df = parse_gpx(args.gpx)
    print(f"  {len(gpx_df)} points")

    print(f"Reading frames    from {args.frames_dir}")
    cam_dfs = collect_cam_data(args.frames_dir)
    for cam, df in sorted(cam_dfs.items()):
        with_heading = df["heading"].notna().sum() if "heading" in df.columns else 0
        print(f"  {cam}: {len(df)} frames with GPS ({with_heading} with heading)")

    if not cam_dfs:
        print("WARNING: no camera GPS data found — frames may not have EXIF GPS tags.",
              file=sys.stderr)

    plot(traj_df, gpx_df, cam_dfs, args.arrow_every, args.save)


if __name__ == "__main__":
    main()
