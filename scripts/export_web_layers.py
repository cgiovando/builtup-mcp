"""Export per-city GeoJSON layers for the interactive web map (docs/v0.1/data/).

For each scan region: find the change hotspot, run the detector, compare to Open
Buildings, and write classified points (agreement / detected-only / reference-only) plus a
small metadata JSON. The MapLibre app in docs/ loads these.

Network required. Run with:
    uv run python scripts/export_web_layers.py [region_name ...]   (default: all)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.transform

from builtup_mcp import config
from builtup_mcp.geo import BBox, utm_epsg
from builtup_mcp.pipeline import run_change_detection
from builtup_mcp.processing.align import grid_transform
from builtup_mcp.processing.hotspot import find_change_hotspot
from builtup_mcp.sources.open_buildings import fetch_building_presence
from builtup_mcp.sources.raster_io import target_grid
from builtup_mcp.validation import _dilate, compare_gain_to_open_buildings

DATA_DIR = Path("docs/v0.1/data")
Y1, Y2 = 2018, 2023

PRETTY = {
    "nairobi_ne": "Nairobi NE, Kenya",
    "greater_dhaka": "Greater Dhaka, Bangladesh",
    "bengaluru_east": "Bengaluru East, India",
    "lagos_lekki": "Lagos (Lekki), Nigeria",
}


def _mask_points(mask: np.ndarray, grid, cls: str) -> gpd.GeoDataFrame:
    rows, cols = np.where(mask)
    if not rows.size:
        return gpd.GeoDataFrame({"class": []}, geometry=[], crs="EPSG:4326")
    xs, ys = rasterio.transform.xy(grid_transform(grid), rows.tolist(), cols.tolist())
    gdf = gpd.GeoDataFrame(
        {"class": [cls] * len(xs)},
        geometry=gpd.points_from_xy(xs, ys),
        crs=f"EPSG:{grid.epsg}",
    )
    return gdf.to_crs("EPSG:4326")


def export_region(region: str, tmp: Path) -> None:
    bbox = BBox(*config.SCAN_REGIONS[region])
    print(f"[{region}] scanning {Y1}->{Y2} ...")
    scan_grid = target_grid(bbox, utm_epsg(bbox), 40.0)
    p1 = fetch_building_presence(scan_grid, Y1, mosaic=True)
    p2 = fetch_building_presence(scan_grid, Y2, mosaic=True)
    aoi, _ = find_change_hotspot(bbox, p1, p2, scan_grid, target_km2=4.0)

    print(f"[{region}] detecting on hotspot {tuple(round(v,4) for v in aoi.as_tuple())} ...")
    run = run_change_detection(list(aoi.as_tuple()), f"{Y2}-06-15", f"{Y1}-06-15", output_dir=tmp)
    cmp = compare_gain_to_open_buildings(gpd.read_file(run.geojson_path), aoi, Y1, Y2)

    tol = cmp.metrics.get("tolerance_px", 1)
    new_buf = _dilate(cmp.google_new, tol)
    gain_buf = _dilate(cmp.gain, tol)
    agree = cmp.gain & new_buf
    gain_only = cmp.gain & ~new_buf
    new_only = cmp.google_new & ~gain_buf

    layers = [
        _mask_points(agree, cmp.grid, "agreement"),
        _mask_points(gain_only, cmp.grid, "gain_only"),
        _mask_points(new_only, cmp.grid, "ob_new"),
    ]
    points = gpd.GeoDataFrame(
        gpd.pd.concat(layers, ignore_index=True), crs="EPSG:4326"
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    points.to_file(DATA_DIR / f"{region}.geojson", driver="GeoJSON")

    w, s, e, n = aoi.as_tuple()
    meta = {
        "id": region,
        "name": PRETTY.get(region, region),
        "bounds": [w, s, e, n],
        "center": [(w + e) / 2, (s + n) / 2],
        "years": [Y1, Y2],
        "metrics": {
            "gain_px": cmp.metrics.get("n_gain"),
            "precision_ob_new": round(cmp.metrics.get("prec_new", 0) * 100, 1),
            "recall_ob_new": round(cmp.metrics.get("recall_new", 0) * 100, 1),
            "enrichment_ob_new": round(cmp.metrics.get("enrichment_new", 0), 1),
        },
        "counts": {
            "agreement": int(agree.sum()),
            "gain_only": int(gain_only.sum()),
            "ob_new": int(new_only.sum()),
        },
    }
    (DATA_DIR / f"{region}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{region}] wrote {len(points)} points; "
          f"precision {meta['metrics']['precision_ob_new']}% recall {meta['metrics']['recall_ob_new']}%")


def main() -> None:
    regions = sys.argv[1:] or ["nairobi_ne", "greater_dhaka", "bengaluru_east", "lagos_lekki"]
    tmp = Path("/tmp/builtup-mcp-web")
    for region in regions:
        export_region(region, tmp)


if __name__ == "__main__":
    main()
