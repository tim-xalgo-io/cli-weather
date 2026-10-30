"""One-off: rasterise Natural Earth 10m land + lakes onto the 1 km radar grid.

    python -I tools/build_basemap.py [--download-dir DIR] [--size 400]

Writes data/basemap.npz with `land` (uint8, 1 = land) and `origin` (row0, col0).
Needs shapely (pip install 'cli-weather[basemap]').
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cli_weather import grid  # noqa: E402

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/{}.geojson"
LAYERS = {"land": "ne_10m_land", "lakes": "ne_10m_lakes", "lakes_europe": "ne_10m_lakes_europe"}


def load_layer(name: str, cache: Path) -> list:
    path = cache / f"{name}.geojson"
    if not path.exists():
        print(f"downloading {name} …", file=sys.stderr)
        req = urllib.request.Request(NE.format(name), headers={"User-Agent": "cli-weather basemap build"})
        with urllib.request.urlopen(req) as r:
            path.write_bytes(r.read())
    return [shape(f["geometry"]) for f in json.loads(path.read_text())["features"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", type=Path, default=ROOT / ".cache" / "naturalearth")
    ap.add_argument("--size", type=int, default=400, help="box size in km (square)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "basemap.npz")
    args = ap.parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)

    row0 = grid.STOCKHOLM[0] - args.size // 2
    col0 = grid.STOCKHOLM[1] - args.size // 2
    rows, cols = np.mgrid[row0:row0 + args.size, col0:col0 + args.size]
    latlon = np.array([grid.rc_to_latlon(r, c) for r, c in zip(rows.ravel(), cols.ravel())])
    lat, lon = latlon[:, 0], latlon[:, 1]

    pad = 0.5
    bbox = shapely.box(lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)

    def union(name: str):
        geoms = [g.intersection(bbox) for g in load_layer(name, args.download_dir) if g.intersects(bbox)]
        return shapely.union_all(geoms) if geoms else shapely.GeometryCollection()

    land = union(LAYERS["land"])
    water = shapely.union_all([union(LAYERS["lakes"]), union(LAYERS["lakes_europe"])])
    shapely.prepare(land)
    shapely.prepare(water)

    pts = shapely.points(lon, lat)
    mask = shapely.contains(land, pts) & ~shapely.contains(water, pts)
    mask = mask.reshape(args.size, args.size).astype(np.uint8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, land=mask, origin=np.array([row0, col0]))
    print(f"wrote {args.out}: {mask.shape}, land {mask.mean():.0%}, origin {row0},{col0}", file=sys.stderr)


if __name__ == "__main__":
    main()
