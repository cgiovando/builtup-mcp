"""Live STAC integration tests for the EO sources (Phase 3).

Marked ``network`` and skipped by default. Run with: ``uv run pytest -m network``.
"""

from __future__ import annotations

import numpy as np
import pytest

from builtup_mcp import config
from builtup_mcp.geo import BBox
from builtup_mcp.sources import sentinel1, sentinel2
from builtup_mcp.sources.raster_io import target_grid

AOI = BBox(*config.DEMO_BBOX)
TARGET_DATE = "2024-06-15"


@pytest.mark.network
def test_optical_window_shapes_match_grid():
    stack = sentinel2.fetch_optical(AOI, TARGET_DATE)
    expected = target_grid(AOI, stack.grid.epsg, config.TARGET_RESOLUTION_M).shape
    assert stack["nir"].shape == expected
    assert stack["swir"].shape == expected
    assert stack["scl"].shape == expected
    assert 0.0 <= stack.meta["aoi_cloud_fraction"] <= 1.0
    assert np.isfinite(stack["nir"]).any()


@pytest.mark.network
def test_sar_window_shapes_match_grid():
    stack = sentinel1.fetch_sar(AOI, TARGET_DATE)
    expected = target_grid(AOI, stack.grid.epsg, config.TARGET_RESOLUTION_M).shape
    assert stack["vv"].shape == expected
    assert stack["vh"].shape == expected
    assert stack["vv_db"].shape == expected
