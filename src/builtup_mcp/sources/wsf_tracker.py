"""DLR / MindEarth WSF Tracker reader (comparison layer).

The WSF Tracker is the closest published product to this one: a global 10 m
layer whose pixels hold the six-month period in which built-up was first
detected (1 = already built by 2016-07, 20 = first seen Jul 2025 - Jan 2026).
It is read straight from the public GeoZarr on Source Cooperative and resampled
(nearest) onto the analysis grid, so the two products can be compared pixel by
pixel.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from ..sources.raster_io import Grid

_BUCKET = "us-west-2.opendata.source.coop"
_PREFIX = "mindearth/wsf/World_WSF_20160701-20260101.zarr"

# Index -> first day of the six-month period that ENDS the detection interval.
EPOCHS: dict[int, str] = {
    i: (date(2016, 7, 1) if i == 1 else date(2016 + i // 2, 1 if i % 2 == 0 else 7, 1)).isoformat()
    for i in range(1, 21)
}


def _open():
    import obstore
    import xarray as xr
    from zarr.storage import ObjectStore

    store = obstore.store.S3Store(
        bucket=_BUCKET,
        prefix=_PREFIX,
        config={"aws_skip_signature": "true", "aws_region": "us-west-2"},
    )
    return xr.open_zarr(ObjectStore(store, read_only=True), group="0", decode_coords="all")


def fetch_wsf_tracker(grid: Grid, bbox) -> np.ndarray:
    """WSF Tracker index (0..20) on ``grid``; 0 = not built / nodata."""
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from ..processing.align import grid_transform

    ds = _open()
    pad = 0.002
    lo0, la0, lo1, la1 = bbox.as_tuple()
    sub = ds["wsf_tracker"].sel(x=slice(lo0 - pad, lo1 + pad), y=slice(la1 + pad, la0 - pad))
    arr = np.asarray(sub.values, dtype="uint8")
    xs, ys = sub["x"].values, sub["y"].values
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    src_t = Affine(dx, 0, float(xs[0]) - dx / 2, 0, dy, float(ys[0]) - dy / 2)
    dst = np.zeros(grid.shape, dtype="uint8")
    reproject(
        source=arr, destination=dst, src_transform=src_t, src_crs=CRS.from_epsg(4326),
        dst_transform=grid_transform(grid), dst_crs=CRS.from_epsg(grid.epsg),
        resampling=Resampling.nearest, src_nodata=None, dst_nodata=None,
    )
    return dst
