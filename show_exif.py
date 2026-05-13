"""
Print EXIF info from JPG frames.

Pass either a single .jpg or a directory of .jpgs (e.g. a cam0 folder, or the
parent georeferenced_frames folder containing cam0..cam5 subfolders).

By default prints GPS-related tags only (lat, lon, alt, heading, timestamp).
Use --all to dump every EXIF tag.

Run:
    python show_exif.py <path>
    python show_exif.py <path> --all
    python show_exif.py <path> --limit 3
"""

import argparse
import sys
from pathlib import Path

import pyexiv2


GPS_TAGS_OF_INTEREST = [
    "Exif.GPSInfo.GPSLatitude",
    "Exif.GPSInfo.GPSLatitudeRef",
    "Exif.GPSInfo.GPSLongitude",
    "Exif.GPSInfo.GPSLongitudeRef",
    "Exif.GPSInfo.GPSAltitude",
    "Exif.GPSInfo.GPSAltitudeRef",
    "Exif.GPSInfo.GPSImgDirection",
    "Exif.GPSInfo.GPSImgDirectionRef",
    "Exif.GPSInfo.GPSTimeStamp",
    "Exif.GPSInfo.GPSDateStamp",
    "Exif.GPSInfo.GPSSpeed",
    "Exif.GPSInfo.GPSSpeedRef",
    "Exif.GPSInfo.GPSTrack",
    "Exif.GPSInfo.GPSTrackRef",
]


def find_jpgs(path: Path) -> list[Path]:
    """Return a sorted list of JPGs found at `path`.

    - If `path` is a .jpg/.JPG file, returns just that file.
    - If `path` is a directory, returns all .jpg files in it and its
      immediate subdirectories (handles both a single cam folder and a
      parent folder containing cam0..cam5).
    """
    if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg"):
        return [path]
    if path.is_dir():
        jpgs = list(path.glob("*.jpg")) + list(path.glob("*.JPG"))
        if not jpgs:
            for sub in sorted(p for p in path.iterdir() if p.is_dir()):
                jpgs.extend(sub.glob("*.jpg"))
                jpgs.extend(sub.glob("*.JPG"))
        return sorted(jpgs)
    return []


def read_exif(jpg_path: Path) -> dict:
    with pyexiv2.Image(str(jpg_path)) as img:
        return img.read_exif()


def print_file(jpg_path: Path, root: Path, show_all: bool):
    try:
        exif = read_exif(jpg_path)
    except Exception as err:
        print(f"\n[{jpg_path.relative_to(root) if root in jpg_path.parents else jpg_path}]")
        print(f"  ERROR reading EXIF: {err}")
        return

    try:
        rel = jpg_path.relative_to(root)
    except ValueError:
        rel = jpg_path
    print(f"\n[{rel}]")

    if not exif:
        print("  (no EXIF data)")
        return

    if show_all:
        for key in sorted(exif.keys()):
            print(f"  {key} = {exif[key]}")
        return

    gps = {k: exif[k] for k in GPS_TAGS_OF_INTEREST if k in exif}
    if not gps:
        print("  (no GPS EXIF tags)")
        return
    width = max(len(k) for k in gps)
    for key in GPS_TAGS_OF_INTEREST:
        if key in gps:
            print(f"  {key:<{width}} = {gps[key]}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", type=Path,
                    help="A .jpg file, or a folder of .jpgs (or a parent of cam0..cam5)")
    ap.add_argument("--all", action="store_true",
                    help="Dump every EXIF tag (default: GPS tags only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only print the first N files (default: all)")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"Error: path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    jpgs = find_jpgs(args.path)
    if not jpgs:
        print(f"No .jpg files found at {args.path}", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        jpgs = jpgs[: args.limit]

    root = args.path if args.path.is_dir() else args.path.parent
    print(f"Reading EXIF from {len(jpgs)} file(s) under {args.path}")

    for jpg in jpgs:
        print_file(jpg, root, args.all)


if __name__ == "__main__":
    main()
