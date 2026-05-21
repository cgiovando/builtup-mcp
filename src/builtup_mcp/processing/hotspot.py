"""Find the AOI within a region with the most built-up change (Phase 4 / validation).

Uses a temporal reference (Google Open Buildings presence) to locate, by evidence
rather than by guesswork, the fixed-size sub-window where built-up increased most
between two years. That window becomes a test AOI with confirmed change.
"""

from __future__ import annotations

import math

import numpy as np
from rasterio.warp import transform_bounds
from scipy.ndimage import uniform_filter

from ..geo import BBox
from ..sources.raster_io import Grid
from .align import grid_transform


def best_window(new_mask: np.ndarray, side: int) -> tuple[int, int]:
    """Top-left (row, col) of the ``side`` x ``side`` window with most True cells."""
    side = max(1, min(side, *new_mask.shape))
    # Mean over each window * area = count; argmax of the mean locates the densest.
    dens = uniform_filter(new_mask.astype("float32"), size=side, mode="constant")
    r, c = np.unravel_index(int(np.argmax(dens)), dens.shape)
    half = side // 2
    r0 = int(np.clip(r - half, 0, new_mask.shape[0] - side))
    c0 = int(np.clip(c - half, 0, new_mask.shape[1] - side))
    return r0, c0


def window_to_bbox(grid: Grid, r0: int, c0: int, side: int) -> BBox:
    """Convert a pixel window on ``grid`` to a WGS84 :class:`BBox`."""
    transform = grid_transform(grid)
    left, top = transform * (c0, r0)
    right, bottom = transform * (c0 + side, r0 + side)
    lon0, lat0, lon1, lat1 = transform_bounds(
        f"EPSG:{grid.epsg}", "EPSG:4326", left, bottom, right, top
    )
    return BBox(lon0, lat0, lon1, lat1)


def find_change_hotspot(
    region: BBox,
    presence_year1: np.ndarray,
    presence_year2: np.ndarray,
    grid: Grid,
    *,
    target_km2: float = 4.0,
    appear_thr: float = 0.5,
    pre_thr: float = 0.3,
) -> tuple[BBox, dict]:
    """Locate the ~``target_km2`` window with the most newly-built pixels.

    ``new`` = presence rose from below ``pre_thr`` to at/above ``appear_thr``
    between the two years. Returns the window BBox and summary stats.
    """
    new = (presence_year2 >= appear_thr) & (presence_year1 < pre_thr)
    side = max(1, round(math.sqrt(target_km2) * 1000.0 / grid.resolution_m))
    r0, c0 = best_window(new, side)
    win = new[r0 : r0 + side, c0 : c0 + side]
    bbox = window_to_bbox(grid, r0, c0, side)
    stats = {
        "new_px_in_window": int(win.sum()),
        "window_px": int(side * side),
        "new_fraction": float(win.mean()),
        "region_new_px": int(new.sum()),
        "side_px": side,
    }
    return bbox, stats
