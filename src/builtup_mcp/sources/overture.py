"""Overture Maps buildings source (Phase 2).

Queries the Overture ``buildings`` GeoParquet on AWS S3 with DuckDB, fetching
only the building polygons intersecting a small BBox. Used as the "known
built-up" baseline prior for change detection.

Design notes:
- Filtering uses Overture's top-level ``bbox`` STRUCT (xmin/xmax/ymin/ymax) so
  DuckDB can use Parquet row-group statistics for predicate pushdown and skip
  files/row-groups without reading them. This is the fast path; filtering on the
  decoded geometry would force a full scan.
- ``geometry`` is stored as WKB; the raw bytes are selected and decoded with
  shapely in Python, which avoids needing the DuckDB ``spatial`` extension just
  to read. Results are then optionally clipped to the exact BBox with geopandas.
- The bucket is public/anonymous; no AWS credentials are required.
"""

from __future__ import annotations

import duckdb
import geopandas as gpd

from .. import config
from ..geo import BBox

# Attribute columns pulled alongside geometry. All are 2.5D-relevant or
# provenance fields (Overture height attributes are sparse, so num_floors is
# kept for the height fallback and `sources` to identify the origin).
_ATTR_COLUMNS = (
    "id",
    "subtype",
    "class",
    "height",
    "num_floors",
    "min_height",
    "roof_height",
)


def _connect() -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection configured for anonymous Overture S3 reads."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_region = '{config.OVERTURE_S3_REGION}';")
    # Force unsigned/anonymous access so ambient AWS credentials in the
    # environment don't cause signed requests against this public bucket.
    try:
        con.execute(
            "CREATE OR REPLACE SECRET overture_anon "
            "(TYPE s3, PROVIDER config, REGION '%s');" % config.OVERTURE_S3_REGION
        )
    except duckdb.Error:
        # Older DuckDB without the secrets manager: fall back to defaults.
        pass
    return con


def build_bbox_query(bbox: BBox, release_path: str = config.OVERTURE_BUILDINGS_PATH) -> str:
    """Build the DuckDB SQL that fetches buildings overlapping ``bbox``.

    Filters on Overture's ``bbox`` struct (xmin/xmax/ymin/ymax) so DuckDB can use
    Parquet statistics for predicate pushdown. Exposed separately so it can be
    unit-tested without a network round-trip.
    """
    min_lon, min_lat, max_lon, max_lat = bbox.as_tuple()
    cols = ", ".join(_ATTR_COLUMNS)
    # Most-recent source update_time per building = data freshness of the baseline.
    update_expr = "list_max(list_transform(sources, s -> s.update_time)) AS source_update_time"
    return f"""
        SELECT {cols}, {update_expr}, geometry
        FROM read_parquet('{release_path}', hive_partitioning => true)
        WHERE bbox.xmin <= {max_lon}
          AND bbox.xmax >= {min_lon}
          AND bbox.ymin <= {max_lat}
          AND bbox.ymax >= {min_lat}
    """


def fetch_buildings(
    bbox: BBox,
    *,
    release_path: str = config.OVERTURE_BUILDINGS_PATH,
    clip: bool = True,
) -> gpd.GeoDataFrame:
    """Fetch Overture building footprints intersecting ``bbox``.

    Args:
        bbox: The WGS84 query box.
        release_path: Overture buildings GeoParquet glob (S3). Defaults to the
            pinned release in :mod:`builtup_mcp.config`.
        clip: If True, clip results to the exact BBox; otherwise return any
            building whose own bbox overlaps (cheaper, includes edge spillover).

    Returns:
        A GeoDataFrame in EPSG:4326 with building geometries, the selected
        attribute columns, and an ``effective_height_m`` column (``height`` when
        present, else ``num_floors * FLOOR_HEIGHT_M``, else NaN).
    """
    min_lon, min_lat, max_lon, max_lat = bbox.as_tuple()
    query = build_bbox_query(bbox, release_path)

    con = _connect()
    try:
        df = con.execute(query).fetch_df()
    finally:
        con.close()

    # DuckDB returns WKB as bytearray; shapely.from_wkb wants bytes.
    wkb = df["geometry"].apply(lambda b: bytes(b) if b is not None else None)
    geometry = gpd.GeoSeries.from_wkb(wkb, crs=f"EPSG:{config.WGS84_EPSG}")
    gdf = gpd.GeoDataFrame(
        df.drop(columns="geometry"), geometry=geometry, crs=f"EPSG:{config.WGS84_EPSG}"
    )

    if clip and not gdf.empty:
        gdf = gdf.clip((min_lon, min_lat, max_lon, max_lat))

    gdf["effective_height_m"] = _effective_height(gdf)
    return gdf.reset_index(drop=True)


def _effective_height(gdf: gpd.GeoDataFrame):
    """Best-available per-building height: ``height`` else ``num_floors * 3 m``."""
    height = gdf["height"]
    floors_height = gdf["num_floors"] * config.FLOOR_HEIGHT_M
    return height.fillna(floors_height)


def baseline_age(gdf: gpd.GeoDataFrame, reference_date: str) -> dict:
    """Summarize the Overture baseline's data freshness relative to a date.

    Overture buildings carry per-source ``update_time``; the most recent is taken
    per feature. The result lets a change statement be time-qualified, e.g.
    "median baseline update 2017 -> detected gain ~ change over up to 7 years".
    """
    import pandas as pd

    if gdf is None or gdf.empty or "source_update_time" not in gdf:
        return {"n_dated": 0}
    ts = pd.to_datetime(gdf["source_update_time"], errors="coerce", utc=True).dropna()
    if ts.empty:
        return {"n_dated": 0}
    ref = pd.Timestamp(reference_date, tz="UTC")
    ages = (ref - ts).dt.days / 365.25
    return {
        "n_dated": int(len(ts)),
        "n_total": int(len(gdf)),
        "median_update": str(ts.median().date()),
        "oldest_update": str(ts.min().date()),
        "newest_update": str(ts.max().date()),
        "median_age_years": round(float(ages.median()), 1),
        "max_age_years": round(float(ages.max()), 1),
    }
