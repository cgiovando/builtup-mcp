"""Rasterize the Overture building baseline onto the analysis grid (Phase 4)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from ..sources.raster_io import Grid
from .align import grid_transform


def rasterize_buildings(gdf: gpd.GeoDataFrame, grid: Grid) -> np.ndarray:
    """Burn building footprints into a boolean mask on ``grid`` (True = building).

    Reprojects footprints into the grid CRS first. ``all_touched`` is used so
    small footprints relative to the 10m pixel still register.
    """
    if gdf is None or gdf.empty:
        return np.zeros(grid.shape, dtype=bool)
    geoms = gdf.to_crs(epsg=grid.epsg).geometry
    shapes = [(geom, 1) for geom in geoms if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(grid.shape, dtype=bool)
    burned = rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid_transform(grid),
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return burned.astype(bool)
