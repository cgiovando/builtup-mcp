"""Export Sentinel-2 true-color before/after PNGs per AOI for the web map.

For each region's hotspot AOI (read from docs/data/<region>_meta.json), find the
least-cloudy Sentinel-2 scene near a "before" and "after" epoch, read true-color
(B04/B03/B02) onto the AOI grid, stretch to 8-bit, and write a PNG plus the
lon/lat corner coordinates (for a MapLibre image source). Metadata is merged
into the existing <region>_meta.json under an "imagery" key.

Network required. Run with:
    uv run python scripts/export_web_imagery.py [region ...]   (default: all 4)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer

from builtup_mcp import config
from builtup_mcp.geo import BBox, utm_epsg
from builtup_mcp.sources import stac
from builtup_mcp.sources.raster_io import read_on_grid, target_grid

DATA_DIR = Path("docs/data")
IMG_DIR = DATA_DIR / "imagery"
RGB_ASSETS = ("B04", "B03", "B02")  # red, green, blue (10 m)
# (epoch label, representative date, look-back window days)
EPOCHS = [("before", "2018-12-31", 420), ("after", "2026-05-15", 400)]


def _aoi_cloud_fraction(scl: np.ndarray) -> float:
    return float(np.isin(scl, config.SCL_CLOUD_CLASSES).mean())


def _pick_scene(bbox: BBox, target_date: str, window_days: int):
    items = stac.search_items(
        config.S2_COLLECTION, bbox, target_date, window_days=window_days,
        query={"eo:cloud_cover": {"lt": config.MAX_SCENE_CLOUD_PCT}},
    )
    items = stac.covering_items(items, bbox)
    if not items:
        raise RuntimeError(f"No S2 scene near {target_date} for {bbox.as_tuple()}")
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    best = None
    for item in items[:6]:  # check the 6 least-cloudy by metadata
        grid = target_grid(bbox, stac.item_epsg(item), config.TARGET_RESOLUTION_M)
        scl = read_on_grid(item.assets[config.S2_ASSETS["scl"]].href, grid, "nearest")
        frac = _aoi_cloud_fraction(scl)
        if best is None or frac < best[0]:
            best = (frac, item, grid)
        if frac < 0.02:
            break
    return best  # (cloud_fraction, item, grid)


def _stretch(band: np.ndarray) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros(band.shape, dtype="uint8")
    lo, hi = np.percentile(valid, [2, 98])
    out = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype("uint8")


def _corners_lonlat(grid) -> list[list[float]]:
    t = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    left, bottom, right, top = grid.bounds
    pts = [(left, top), (right, top), (right, bottom), (left, bottom)]  # TL,TR,BR,BL
    return [list(t.transform(x, y)) for x, y in pts]


def export_region(region: str) -> None:
    meta_path = DATA_DIR / f"{region}_meta.json"
    meta = json.loads(meta_path.read_text())
    bbox = BBox(*meta["bounds"])
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    imagery = {}

    for label, date, window in EPOCHS:
        frac, item, grid = _pick_scene(bbox, date, window)
        bands = [read_on_grid(item.assets[a].href, grid, "bilinear") for a in RGB_ASSETS]
        rgb = np.dstack([_stretch(b) for b in bands])
        alpha = np.where(np.dstack(bands).max(axis=2) > 0, 255, 0).astype("uint8")
        rgba = np.dstack([rgb, alpha])
        png = IMG_DIR / f"{region}_{label}.png"
        Image.fromarray(rgba, "RGBA").save(png)
        imagery[label] = {
            "png": f"data/imagery/{png.name}",
            "date": str(item.datetime.date()),
            "cloud_fraction_aoi": round(frac, 3),
            "coordinates": _corners_lonlat(grid),
        }
        print(f"[{region}] {label}: {item.datetime.date()} cloud_aoi={frac:.2f} -> {png.name}")

    meta["imagery"] = imagery
    meta_path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    regions = sys.argv[1:] or ["nairobi_ne", "greater_dhaka", "bengaluru_east", "lagos_lekki"]
    for r in regions:
        export_region(r)


if __name__ == "__main__":
    main()
