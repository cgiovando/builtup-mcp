"""Tests for the windowed-read grid math (Phase 3, no network)."""

from __future__ import annotations

import math

from builtup_mcp import config
from builtup_mcp.geo import BBox, bbox_area_km2
from builtup_mcp.sources.raster_io import target_grid

# Demo AOI, in UTM zone 37S (Kenya) -> EPSG:32737.
AOI = BBox(*config.DEMO_BBOX)


def test_grid_shape_matches_bbox_dimensions():
    grid = target_grid(AOI, epsg=32737, resolution_m=10.0)
    # ~2.1 km x ~2.0 km at 10 m -> ~210 x 200 px. Allow a 2 px rounding margin.
    assert 195 <= grid.width <= 220
    assert 185 <= grid.height <= 210
    assert grid.shape == (grid.height, grid.width)


def test_grid_bounds_consistent_with_shape():
    res = 10.0
    grid = target_grid(AOI, epsg=32737, resolution_m=res)
    left, bottom, right, top = grid.bounds
    assert math.isclose(right - left, grid.width * res, rel_tol=1e-9)
    assert math.isclose(top - bottom, grid.height * res, rel_tol=1e-9)


def test_grid_area_close_to_bbox_area():
    grid = target_grid(AOI, epsg=32737, resolution_m=10.0)
    grid_area_km2 = (grid.width * 10.0) * (grid.height * 10.0) / 1e6
    assert math.isclose(grid_area_km2, bbox_area_km2(AOI), rel_tol=0.05)
