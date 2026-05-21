"""Tests for the Overture buildings source (Phase 2).

Fast unit tests cover query construction and the height fallback without the
network. The live S3 integration test is marked ``network`` and skipped by
default (run with: ``uv run pytest -m network``).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box as shp_box

from builtup_mcp import config
from builtup_mcp.geo import BBox
from builtup_mcp.sources import overture

AOI = BBox(*config.DEMO_BBOX)


def test_build_query_has_bbox_pushdown_predicates():
    sql = overture.build_bbox_query(AOI, release_path="s3://example/*")
    # All four bbox-struct predicates must be present for predicate pushdown.
    assert "bbox.xmin <=" in sql
    assert "bbox.xmax >=" in sql
    assert "bbox.ymin <=" in sql
    assert "bbox.ymax >=" in sql
    assert "read_parquet('s3://example/*'" in sql
    # The numeric bounds appear in the SQL.
    assert str(AOI.max_lon) in sql
    assert str(AOI.min_lat) in sql


def test_effective_height_prefers_height_then_floors():
    gdf = gpd.GeoDataFrame(
        {
            "height": [12.0, np.nan, np.nan],
            "num_floors": [np.nan, 4, np.nan],
            "geometry": [shp_box(0, 0, 1, 1)] * 3,
        },
        crs="EPSG:4326",
    )
    eff = overture._effective_height(gdf)
    assert eff.iloc[0] == 12.0  # explicit height wins
    assert eff.iloc[1] == 4 * config.FLOOR_HEIGHT_M  # fallback from floors
    assert np.isnan(eff.iloc[2])  # neither -> NaN


@pytest.mark.network
def test_fetch_buildings_live_dense_bbox():
    # Dense central Nairobi CBD box should return many buildings.
    dense = BBox(36.815, -1.288, 36.820, -1.284)
    gdf = overture.fetch_buildings(dense)
    assert len(gdf) > 20
    assert gdf.crs.to_epsg() == config.WGS84_EPSG
    assert "effective_height_m" in gdf.columns
