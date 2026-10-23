"""Overture Maps buildings source (Phase 2).

Queries the Overture ``buildings`` GeoParquet on AWS S3 with DuckDB, fetching
only the building polygons intersecting a small BBox. Used as the "known
built-up" baseline prior for change detection.

Design notes:
- The release is resolved through Overture's STAC catalog, and the catalog's
  per-file bboxes are used to read only the GeoParquet files that intersect the
  AOI (usually one or two of ~500). This replaces a glob over the whole release
  and cuts a typical query from ~100 s to a few seconds.
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

import json
import urllib.request
from functools import lru_cache

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


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as fh:
        return json.load(fh)


_latest_cache: dict[str, tuple[float, str]] = {}


def resolve_release(release: str = config.OVERTURE_RELEASE) -> str:
    """Return a concrete Overture release id ("latest" -> newest in the catalog).

    The lookup is cached for an hour so a long-running MCP server picks up new
    monthly releases.
    """
    import time

    if release != "latest":
        return release
    hit = _latest_cache.get("latest")
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    rel = _get_json(f"{config.OVERTURE_STAC_URL}/catalog.json")["latest"]
    _latest_cache["latest"] = (time.time(), rel)
    return rel


def _intersects(a, b) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


@lru_cache(maxsize=32)
def release_files(bbox_tuple: tuple, release: str) -> tuple[str, ...]:
    """S3 paths of the building files whose extent intersects the bbox.

    The collection's ``extent.spatial.bbox`` lists the union first and then one
    bbox per item, in the same order as its ``item`` links.
    """
    base = f"{config.OVERTURE_STAC_URL}/{release}/buildings/building"
    coll = _get_json(f"{base}/collection.json")
    file_bboxes = coll["extent"]["spatial"]["bbox"][1:]
    items = [lk["href"] for lk in coll["links"] if lk["rel"] == "item"]
    paths = []
    for fb, href in zip(file_bboxes, items):
        if _intersects(fb, bbox_tuple):
            item = _get_json(href)
            paths.append(item["assets"]["aws"]["alternate"]["s3"]["href"])
    return tuple(paths)


def release_path_for(bbox: BBox, release: str = config.OVERTURE_RELEASE) -> list[str]:
    """Concrete file list for ``bbox`` (falls back to the release-wide glob)."""
    rel = resolve_release(release)
    try:
        files = list(release_files(bbox.as_tuple(), rel))
    except Exception:  # catalog unreachable: glob the release
        files = []
    return files or [
        f"s3://overturemaps-{config.OVERTURE_S3_REGION}/release/{rel}"
        "/theme=buildings/type=building/*"
    ]


def _connect() -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection configured for anonymous Overture S3 reads."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar = false;")  # stdout belongs to MCP stdio
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


def build_bbox_query(bbox: BBox, release_path: str | list[str]) -> str:
    """Build the DuckDB SQL that fetches buildings overlapping ``bbox``.

    Filters on Overture's ``bbox`` struct (xmin/xmax/ymin/ymax) so DuckDB can use
    Parquet statistics for predicate pushdown. Exposed separately so it can be
    unit-tested without a network round-trip.
    """
    min_lon, min_lat, max_lon, max_lat = bbox.as_tuple()
    cols = ", ".join(_ATTR_COLUMNS)
    paths = [release_path] if isinstance(release_path, str) else list(release_path)
    source = "[" + ", ".join(f"'{p}'" for p in paths) + "]"
    # Most-recent source update_time per building = data freshness of the baseline.
    update_expr = "list_max(list_transform(sources, s -> s.update_time)) AS source_update_time"
    return f"""
        SELECT {cols}, {update_expr}, geometry
        FROM read_parquet({source}, hive_partitioning => true)
        WHERE bbox.xmin <= {max_lon}
          AND bbox.xmax >= {min_lon}
          AND bbox.ymin <= {max_lat}
          AND bbox.ymax >= {min_lat}
    """


def fetch_buildings(
    bbox: BBox,
    *,
    release_path: str | list[str] | None = None,
    clip: bool = True,
) -> gpd.GeoDataFrame:
    """Fetch Overture building footprints intersecting ``bbox``.

    Args:
        bbox: The WGS84 query box.
        release_path: Overture buildings GeoParquet path(s). Defaults to the
            files of the configured release that intersect ``bbox``.
        clip: If True, clip results to the exact BBox; otherwise return any
            building whose own bbox overlaps (cheaper, includes edge spillover).

    Returns:
        A GeoDataFrame in EPSG:4326 with building geometries, the selected
        attribute columns, and an ``effective_height_m`` column (``height`` when
        present, else ``num_floors * FLOOR_HEIGHT_M``, else NaN).
    """
    min_lon, min_lat, max_lon, max_lat = bbox.as_tuple()
    query = build_bbox_query(bbox, release_path or release_path_for(bbox))

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
