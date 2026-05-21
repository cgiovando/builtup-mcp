"""Vectorize a change raster into GeoJSON point centroids (Phase 4).

Each changed pixel becomes a point at its cell center, carrying the change type,
confidence, and the contributing signal values. Points (not polygons) per the
prototype's coarse-signal goal; the schema is open so attributes can be added.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio.transform

from ..sources.raster_io import Grid
from .align import grid_transform
from .change import GAIN, ChangeResult

_LABEL = {GAIN: "gain"}


def change_to_points(result: ChangeResult, grid: Grid) -> gpd.GeoDataFrame:
    """Build a WGS84 GeoDataFrame of changed-pixel centroids with attributes."""
    rows, cols = np.where(result.change_code != 0)
    crs = f"EPSG:{grid.epsg}"
    if rows.size == 0:
        return gpd.GeoDataFrame(
            {"change_type": [], "confidence": []}, geometry=[], crs=crs
        ).to_crs("EPSG:4326")

    xs, ys = rasterio.transform.xy(
        grid_transform(grid), rows.tolist(), cols.tolist(), offset="center"
    )
    codes = result.change_code[rows, cols]
    data = {
        "change_type": ["gain" if c == GAIN else "loss" for c in codes],
        "confidence": np.round(result.confidence[rows, cols], 3),
    }
    for name, layer in result.signals.items():
        vals = layer[rows, cols]
        data[name] = np.round(vals, 3) if np.issubdtype(vals.dtype, np.floating) else vals

    gdf = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(xs, ys), crs=crs
    )
    return gdf.to_crs("EPSG:4326")
