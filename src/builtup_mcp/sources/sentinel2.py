"""Sentinel-2 L2A optical source (Phase 3).

Finds the least-cloudy usable scene over the AOI on/before a target date and
performs windowed reads of NIR (B08, 10m), SWIR (B11, 20m), and the SCL scene
classification (20m), all resampled onto a common 10m UTM grid. NDBI (Phase 4)
needs NIR + SWIR; SCL drives cloud masking and AOI cloud-fraction selection.
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..geo import BBox
from . import stac
from .base import RasterStack
from .raster_io import read_on_grid, target_grid


class NoUsableSceneError(RuntimeError):
    """Raised when no Sentinel-2 scene meets the AOI cloud threshold in window."""


def harmonize(arr: np.ndarray, item) -> np.ndarray:
    """Remove the processing-baseline 04.00 reflectance offset (DN - 1000)."""
    arr = arr.astype("float32")
    if str(item.properties.get("s2:processing_baseline", "00.00")) >= "04.00":
        arr = np.clip(arr - 1000.0, 0.0, None)
    return arr


def _aoi_cloud_fraction(scl: np.ndarray) -> float:
    """Fraction of AOI pixels flagged as cloud/shadow/invalid by SCL."""
    cloudy = np.isin(scl, config.SCL_CLOUD_CLASSES)
    return float(cloudy.mean())


def fetch_optical(
    bbox: BBox,
    target_date: str,
    *,
    window_days: int = config.SEARCH_WINDOW_DAYS,
    resolution_m: float = config.TARGET_RESOLUTION_M,
    max_aoi_cloud: float = config.MAX_AOI_CLOUD_FRACTION,
) -> RasterStack:
    """Fetch a low-cloud Sentinel-2 stack (NIR, SWIR, SCL) over ``bbox``.

    Scans candidates by ascending scene cloud cover, reads each candidate's SCL
    over the AOI to measure *local* cloudiness, and accepts the first scene under
    ``max_aoi_cloud`` (else the least-cloudy-over-AOI scene found). Raises if no
    scenes exist in the window at all.
    """
    items = stac.search_items(
        config.S2_COLLECTION,
        bbox,
        target_date,
        window_days=window_days,
        query={"eo:cloud_cover": {"lt": config.MAX_SCENE_CLOUD_PCT}},
    )
    if not items:
        raise NoUsableSceneError(
            f"No Sentinel-2 scenes (cloud < {config.MAX_SCENE_CLOUD_PCT}%) within "
            f"{window_days} days before {target_date} for bbox {bbox.as_tuple()}."
        )

    # Restrict to scenes that fully cover the AOI, then try least-cloudy first.
    items = stac.covering_items(items, bbox)
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))

    best = None  # (aoi_cloud_fraction, item, grid, scl)
    for item in items:
        grid = target_grid(bbox, stac.item_epsg(item), resolution_m)
        scl = read_on_grid(item.assets[config.S2_ASSETS["scl"]].href, grid, "nearest")
        frac = _aoi_cloud_fraction(scl)
        if best is None or frac < best[0]:
            best = (frac, item, grid, scl)
        if frac <= max_aoi_cloud:
            break

    frac, item, grid, scl = best
    nir = read_on_grid(item.assets[config.S2_ASSETS["nir"]].href, grid, "bilinear")
    swir = read_on_grid(item.assets[config.S2_ASSETS["swir"]].href, grid, "bilinear")
    # Processing baseline 04.00+ (from 25 Jan 2022) adds +1000 DN to reflectance.
    # Without removing it, NDBI shrinks after 2022 and pre/post pairs are biased.
    nir, swir = harmonize(nir, item), harmonize(swir, item)
    valid = ~np.isin(scl, config.SCL_CLOUD_CLASSES)

    return RasterStack(
        bands={
            "nir": nir.astype("float32"),
            "swir": swir.astype("float32"),
            "scl": scl.astype("uint8"),
        },
        grid=grid,
        item_id=item.id,
        datetime=str(item.datetime),
        collection=config.S2_COLLECTION,
        meta={
            "scene_cloud_cover": item.properties.get("eo:cloud_cover"),
            "aoi_cloud_fraction": frac,
            "valid_mask": valid,
        },
    )
