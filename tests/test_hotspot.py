"""Unit tests for the change-hotspot finder (no network)."""

from __future__ import annotations

import numpy as np

from builtup_mcp.geo import BBox
from builtup_mcp.processing.hotspot import best_window, window_to_bbox
from builtup_mcp.sources.raster_io import target_grid


def test_best_window_locates_dense_block():
    new = np.zeros((20, 20), dtype=bool)
    new[5:9, 5:9] = True  # a 4x4 cluster of change
    r0, c0 = best_window(new, side=4)
    # The chosen 4x4 window should capture most of the cluster.
    assert new[r0 : r0 + 4, c0 : c0 + 4].sum() >= 9


def test_best_window_prefers_denser_of_two():
    new = np.zeros((30, 30), dtype=bool)
    new[2:4, 2:4] = True       # small cluster (4 px)
    new[20:26, 20:26] = True   # big cluster (36 px)
    r0, c0 = best_window(new, side=6)
    assert r0 >= 15 and c0 >= 15  # window lands on the big cluster


def test_window_to_bbox_within_region():
    region = BBox(36.88, -1.20, 36.99, -1.10)
    grid = target_grid(region, 32737, 40.0)
    bbox = window_to_bbox(grid, 10, 10, 50)
    # Result must lie inside the region.
    assert region.min_lon <= bbox.min_lon <= bbox.max_lon <= region.max_lon
    assert region.min_lat <= bbox.min_lat <= bbox.max_lat <= region.max_lat
