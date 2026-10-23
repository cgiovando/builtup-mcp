"""What the published static layers say about an AOI (v0.2).

Reads the newest epoch of each open global built-up layer that has a cloud-native
endpoint, on the same grid as the change product, so a request can state how
old the static picture is and how much built-up appeared since. Each entry
records the layer's own latest year; nothing here is interpolated.
"""

from __future__ import annotations

import numpy as np

from . import config
from .sources import stac
from .sources.raster_io import Grid, read_warped

# (collection, built-up class value, label, licence) on Planetary Computer.
_PC_LAYERS = (
    ("io-lulc-annual-v02", 7, "Esri / Impact Observatory 10 m land cover (built area)", "CC-BY-4.0"),
    ("esa-worldcover", 50, "ESA WorldCover 10 m (built-up)", "CC-BY-4.0"),
)


def _latest_pc_items(collection: str, bbox):
    """All items of the newest year intersecting the bbox (a year can span tiles)."""
    items = stac.search_range(collection, bbox, "2000-01-01", "2100-01-01")
    if not items:
        return []

    def year(it):
        return str(it.datetime or it.properties.get("start_datetime") or "")[:4]

    newest = max(year(it) for it in items)
    return [it for it in items if year(it) == newest]


def pc_layer_fraction(collection: str, value: int, bbox, grid: Grid, aoi: np.ndarray):
    items = _latest_pc_items(collection, bbox)
    if not items:
        return None
    arr = np.full(grid.shape, np.nan, dtype="float32")
    for item in items:
        asset = item.assets.get("data") or item.assets.get("map")
        part = read_warped(asset.href, grid, "nearest")
        fill = np.isnan(arr) & np.isfinite(part) & (part > 0)
        arr[fill] = part[fill]
    ok = np.isfinite(arr) & aoi
    built = (arr == value) & ok
    year = str(items[0].datetime or items[0].properties.get("start_datetime"))[:4]
    return year, built, ok


def static_snapshot(bbox, grid: Grid, aoi: np.ndarray, out_dir=None) -> list[dict]:
    """Latest-epoch built-up area per static layer over the AOI."""
    px_ha = (grid.resolution_m**2) / 1e4
    out: list[dict] = []
    for collection, value, label, lic in _PC_LAYERS:
        try:
            res = pc_layer_fraction(collection, value, bbox, grid, aoi)
        except Exception as exc:  # keep the product even if a layer is down
            out.append({"layer": label, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if res is None:
            continue
        year, built, ok = res
        if out_dir is not None:
            from .product import write_mask_png

            write_mask_png(out_dir / f"static_{collection}.png", built, (0, 191, 255, 200))
        out.append(
            {
                "layer": label,
                "source": f"planetarycomputer:{collection}",
                "latest_year": int(year),
                "built_ha": round(float(built.sum()) * px_ha, 1),
                "built_fraction": round(float(built.sum() / max(ok.sum(), 1)), 4),
                "license": lic,
            }
        )
    try:
        from .sources.open_buildings import fetch_building_presence

        pres = fetch_building_presence(grid, config.OPEN_BUILDINGS_LATEST_YEAR)
        ok = np.isfinite(pres) & aoi
        built = (pres >= 0.5) & ok
        if out_dir is not None:
            from .product import write_mask_png

            write_mask_png(out_dir / "static_open-buildings.png", built, (0, 191, 255, 200))
        out.append(
            {
                "layer": "Google Open Buildings 2.5D Temporal (presence >= 0.5)",
                "source": "gs://open-buildings-temporal-data/v1",
                "latest_year": config.OPEN_BUILDINGS_LATEST_YEAR,
                "built_ha": round(float(built.sum()) * px_ha, 1),
                "built_fraction": round(float(built.sum() / max(ok.sum(), 1)), 4),
                "license": "CC-BY-4.0",
            }
        )
    except Exception as exc:
        out.append({"layer": "Google Open Buildings 2.5D Temporal", "error": str(exc)})
    return out
