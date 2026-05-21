"""Grid alignment: reproject rasters onto a canonical analysis grid (Phase 4).

The Sentinel-2 grid is treated as canonical. Sentinel-1 (and any baseline-date
scene that lands in a different UTM zone) is reprojected onto it so all bands
share one pixel grid before change math.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

from ..sources.base import RasterStack
from ..sources.raster_io import Grid


def grid_transform(grid: Grid) -> Affine:
    """North-up affine transform for a :class:`Grid`."""
    left, _bottom, _right, top = grid.bounds
    return Affine(grid.resolution_m, 0, left, 0, -grid.resolution_m, top)


def reproject_array(
    arr: np.ndarray, src_grid: Grid, dst_grid: Grid, resampling: str = "bilinear"
) -> np.ndarray:
    """Reproject a single 2D array from ``src_grid`` to ``dst_grid`` (NaN nodata)."""
    src = arr.astype("float32")
    dst = np.full(dst_grid.shape, np.nan, dtype="float32")
    reproject(
        source=src,
        destination=dst,
        src_transform=grid_transform(src_grid),
        src_crs=CRS.from_epsg(src_grid.epsg),
        dst_transform=grid_transform(dst_grid),
        dst_crs=CRS.from_epsg(dst_grid.epsg),
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling[resampling],
    )
    return dst


def reproject_stack(stack: RasterStack, dst_grid: Grid) -> RasterStack:
    """Reproject every band of a stack onto ``dst_grid``."""
    if stack.grid == dst_grid:
        return stack
    bands = {
        name: reproject_array(arr, stack.grid, dst_grid)
        for name, arr in stack.bands.items()
    }
    return RasterStack(
        bands=bands,
        grid=dst_grid,
        item_id=stack.item_id,
        datetime=stack.datetime,
        collection=stack.collection,
        meta=dict(stack.meta),
    )
