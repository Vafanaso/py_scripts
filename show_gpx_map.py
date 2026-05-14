"""
Render a GPX route on an interactive web map (OpenStreetMap basemap).

Reads a GPX file, draws the track as a blue line on a real map, marks the
start (green) and end (red), and writes a self-contained .html you can open
in any browser to pan/zoom.

Run:
    python show_gpx_map.py route.gpx
    python show_gpx_map.py route.gpx --out my_map.html --open

Requires (one extra install on top of the existing venv):
    pip install folium
"""

import argparse
import sys
import webbrowser
from pathlib import Path

import folium
import gpxpy


def parse_gpx(path: Path) -> list[tuple[float, float, float | None]]:
    """Return list of (lat, lon, elevation) tuples from every track point."""
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    pts = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                pts.append((p.latitude, p.longitude, p.elevation))
    return pts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("gpx", type=Path, help="Path to .gpx file")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output HTML path (default: <gpx-stem>_map.html next to the script)")
    ap.add_argument("--open", action="store_true",
                    help="Open the resulting HTML in the default browser")
    args = ap.parse_args()

    if not args.gpx.exists():
        print(f"Error: GPX not found: {args.gpx}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading GPX from {args.gpx}")
    points = parse_gpx(args.gpx)
    if not points:
        print("No track points found in this GPX.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(points)} track points")

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    elevs = [p[2] for p in points if p[2] is not None]

    print(f"  bounding box: lat {min(lats):.6f}..{max(lats):.6f}, "
          f"lon {min(lons):.6f}..{max(lons):.6f}")
    if elevs:
        print(f"  elevation:    {min(elevs):.1f} .. {max(elevs):.1f} m")

    center = [sum(lats) / len(lats), sum(lons) / len(lons)]

    # Build the map. zoom_start gets auto-corrected by fit_bounds below.
    m = folium.Map(location=center, zoom_start=16, control_scale=True)

    # Draw the route polyline (lat, lon pairs).
    line_coords = [(la, lo) for la, lo, _ in points]
    folium.PolyLine(
        line_coords,
        color="blue",
        weight=4,
        opacity=0.85,
        tooltip=f"Route ({len(points)} points)",
    ).add_to(m)

    # Start marker (green) and end marker (red).
    folium.Marker(
        line_coords[0],
        tooltip="Start",
        popup=f"Start<br>lat {lats[0]:.6f}, lon {lons[0]:.6f}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        line_coords[-1],
        tooltip="End",
        popup=f"End<br>lat {lats[-1]:.6f}, lon {lons[-1]:.6f}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Auto-fit the viewport to the route's bounding box.
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    # Optional: tile-layer switcher so the user can toggle to satellite etc.
    folium.LayerControl().add_to(m)

    out = args.out or (args.gpx.parent / f"{args.gpx.stem}_map.html")
    m.save(str(out))
    print(f"\nWrote map to {out}")

    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
