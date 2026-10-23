"""Windowed raster reads onto a common analysis grid (Phase 3).

Whole Sentinel tiles are never downloaded. Given a WGS84 BBox and a target UTM
grid, the pixel window on each COG asset is computed and only that window is read,
resampling to the target shape in one rasterio call. This keeps reads tiny
(a few hundred pixels per side for a sub-5 sq km AOI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from ..geo import BBox

_RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


@dataclass(frozen=True)
class Grid:
    """A target analysis grid in a projected CRS."""

    epsg: int
    bounds: tuple[float, float, float, float]  # (left, bottom, right, top) in CRS units
    width: int
    height: int
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def target_grid(bbox: BBox, epsg: int, resolution_m: float) -> Grid:
    """Build the analysis grid for ``bbox`` in the given UTM ``epsg``.

    The BBox (WGS84) is reprojected to the target CRS and snapped to an integer
    number of ``resolution_m`` pixels.
    """
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", f"EPSG:{epsg}", *bbox.as_tuple()
    )
    width = max(1, math.ceil((right - left) / resolution_m))
    height = max(1, math.ceil((top - bottom) / resolution_m))
    # Re-snap the far edge so bounds match width*res exactly (north-up).
    right = left + width * resolution_m
    bottom = top - height * resolution_m
    return Grid(epsg, (left, bottom, right, top), width, height, resolution_m)


def aoi_grid(bbox: BBox, resolution_m: float) -> Grid:
    """Analysis grid for ``bbox`` in the UTM zone of its center.

    Used for multi-scene composites, where scenes may come from different UTM
    zones or orbits: every scene is warped onto this one grid.
    """
    from ..geo import utm_epsg

    return target_grid(bbox, utm_epsg(bbox), resolution_m)


def read_warped(
    href: str, grid: Grid, resampling: str = "bilinear", band: int = 1
) -> np.ndarray:
    """Read one band of a COG onto ``grid``, reprojecting on the fly if needed.

    Same-CRS assets take the fast windowed path (:func:`read_on_grid`); others go
    through a ``WarpedVRT`` so GDAL fetches only the source blocks that overlap the
    grid. Pixels outside the asset footprint (or source nodata) come back as NaN.
    """
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
        with rasterio.open(href) as src:
            same_crs = src.crs is not None and src.crs.to_epsg() == grid.epsg
            nodata = src.nodata
            if same_crs:
                window = from_bounds(*grid.bounds, transform=src.transform)
                data = src.read(
                    band,
                    window=window,
                    out_shape=grid.shape,
                    resampling=_RESAMPLING[resampling],
                    boundless=True,
                    masked=True,
                )
            else:
                left, _bottom, _right, top = grid.bounds
                transform = Affine(
                    grid.resolution_m, 0, left, 0, -grid.resolution_m, top
                )
                with WarpedVRT(
                    src,
                    crs=CRS.from_epsg(grid.epsg),
                    transform=transform,
                    width=grid.width,
                    height=grid.height,
                    resampling=_RESAMPLING[resampling],
                    src_nodata=nodata,
                    nodata=nodata if nodata is not None else 0,
                ) as vrt:
                    data = vrt.read(band, masked=True)
    out = data.astype("float32").filled(np.nan)
    return out


def read_on_grid(
    href: str, grid: Grid, resampling: str = "bilinear", band: int = 1
) -> np.ndarray:
    """Windowed read of one band of a COG asset onto ``grid``.

    Reads only the pixel window overlapping the grid bounds and resamples to the
    grid shape. ``boundless`` fills any out-of-extent margin with NaN-safe zeros.
    The asset must already be in the grid's CRS (no reprojection here).
    """
    # Retry transient HTTP errors when reading remote COGs.
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
        with rasterio.open(href) as src:
            src_epsg = src.crs.to_epsg() if src.crs else None
            if src_epsg != grid.epsg:
                raise ValueError(
                    f"Asset CRS EPSG:{src_epsg} does not match grid EPSG:{grid.epsg}; "
                    "reproject the asset onto the grid CRS before reading."
                )
            window = from_bounds(*grid.bounds, transform=src.transform)
            data = src.read(
                band,
                window=window,
                out_shape=grid.shape,
                resampling=_RESAMPLING[resampling],
                boundless=True,
                fill_value=0,
            )
    return data
