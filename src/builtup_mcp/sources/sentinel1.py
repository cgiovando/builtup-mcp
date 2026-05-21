"""Sentinel-1 RTC SAR source (Phase 3, primary all-weather signal).

Windowed reads of radiometrically terrain-corrected backscatter (VV, VH) onto a
common 10m UTM grid. SAR is cloud-independent, so the acquisition closest in
time to the target date is selected rather than filtering on cloud cover.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .. import config
from ..geo import BBox
from . import stac
from .base import RasterStack
from .raster_io import read_on_grid, target_grid


class NoUsableSceneError(RuntimeError):
    """Raised when no Sentinel-1 RTC acquisition is found in the window."""


def _to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear backscatter power to decibels (10*log10), guarding zeros."""
    out = np.full(linear.shape, np.nan, dtype="float32")
    pos = linear > 0
    out[pos] = 10.0 * np.log10(linear[pos])
    return out


def fetch_sar(
    bbox: BBox,
    target_date: str,
    *,
    window_days: int = config.SEARCH_WINDOW_DAYS,
    resolution_m: float = config.TARGET_RESOLUTION_M,
) -> RasterStack:
    """Fetch the Sentinel-1 RTC acquisition closest in time to ``target_date``.

    Returns VV/VH backscatter in both linear and dB (``vv``/``vh`` linear,
    ``vv_db``/``vh_db``). dB is the usual domain for built-up backscatter work.
    """
    items = stac.search_items(
        config.S1_COLLECTION, bbox, target_date, window_days=window_days
    )
    if not items:
        raise NoUsableSceneError(
            f"No Sentinel-1 RTC acquisitions within {window_days} days before "
            f"{target_date} for bbox {bbox.as_tuple()}."
        )

    # Prefer acquisitions that fully cover the AOI (avoid swath-edge nodata),
    # then take the one closest in time to the target date.
    items = stac.covering_items(items, bbox)
    ref = datetime.fromisoformat(target_date).date()
    items.sort(key=lambda it: abs((it.datetime.date() - ref).days))
    item = items[0]

    grid = target_grid(bbox, stac.item_epsg(item), resolution_m)
    vv = read_on_grid(item.assets[config.S1_ASSETS["vv"]].href, grid, "bilinear")
    vh = read_on_grid(item.assets[config.S1_ASSETS["vh"]].href, grid, "bilinear")

    return RasterStack(
        bands={
            "vv": vv.astype("float32"),
            "vh": vh.astype("float32"),
            "vv_db": _to_db(vv),
            "vh_db": _to_db(vh),
        },
        grid=grid,
        item_id=item.id,
        datetime=str(item.datetime),
        collection=config.S1_COLLECTION,
        meta={
            "orbit_state": item.properties.get("sat:orbit_state"),
            "days_from_target": abs((item.datetime.date() - ref).days),
            "aoi_coverage": stac.aoi_coverage(item, bbox),
        },
    )
