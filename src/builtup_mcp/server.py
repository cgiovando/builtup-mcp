"""MCP server for on-demand built-up change products.

Tools, in the order an agent typically uses them:

- ``plan_city_request``: expand a city request into composite windows and report
  the Sentinel-1/-2 scenes available for each (seconds; no processing).
- ``run_city_request``: produce the change product (COGs, GeoParquet, H3 stats,
  footprint age, summary) and register it as a STAC item in the local catalog.
- ``list_products`` / ``get_product_summary``: browse the catalog.
- ``rank_neighbourhoods``: answer "where did it grow most" from a product's H3 stats.
- ``compare_products``: rank several city products by growth.
- ``static_layers_report``: what the newest static global layers say for an AOI.
- ``detect_builtup_change``: the original small-bbox, two-date tool (v0.1).

Nothing here writes to stdout: the stdio transport owns it, so progress goes to
stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import config
from .geo import BBoxValidationError, validate_bbox, validate_change_dates

mcp = FastMCP(
    "builtup-mcp",
    instructions=(
        "On-demand built-up change products from Sentinel-1/-2 composites. Plan a city "
        "request first (fast), then run it; runs take minutes on first use and seconds "
        "when composites are cached. Report accuracy and data gaps from the product "
        "summary alongside any area figure, and say which window a number refers to."
    ),
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _spec(
    city: str,
    bbox: list[float] | None,
    start_year: int,
    end_year: int,
    season_months: list[int] | None,
    latest_window_days: int | None,
    country: str,
    detect_loss: bool,
    request_path: str | None,
) -> dict:
    if request_path:
        return json.loads(Path(request_path).read_text(encoding="utf-8"))
    if not bbox:
        raise BBoxValidationError("Give either bbox or request_path.")
    return {
        "city": city,
        "country": country,
        "aoi": {"bbox": bbox},
        "period": {
            "start_year": start_year,
            "end_year": end_year,
            "season_months": season_months or [1, 2, 3, 4],
        },
        "latest_window_days": latest_window_days,
        "options": {"detect_loss": detect_loss},
    }


@mcp.tool()
def plan_city_request(
    city: str = "AOI",
    bbox: list[float] | None = None,
    start_year: int = 2019,
    end_year: int = 2026,
    season_months: list[int] | None = None,
    latest_window_days: int | None = 90,
    country: str = "",
    request_path: str | None = None,
) -> str:
    """Plan a built-up change request without processing anything.

    Expands the period into one composite window per year over the same calendar
    months (``season_months``, e.g. [12, 1, 2, 3] for a December-March dry season),
    plus an optional rolling window of the last ``latest_window_days`` days, and
    counts usable Sentinel-2 (cloud < 60%) and Sentinel-1 scenes in each window.
    Use it to choose a season and to see Sentinel-1 gaps before running.

    Args:
        city: Name used in the product id and titles.
        bbox: WGS84 [min_lon, min_lat, max_lon, max_lat] (cap: CITY_MAX_AREA_KM2).
        start_year / end_year: first and last year compared.
        season_months: calendar months composited each year.
        latest_window_days: add a rolling "latest" window ending today (None to skip).
        country: Optional country name.
        request_path: Alternatively, a request JSON file (overrides the other args).
    """
    try:
        from .request import build_request
        from .sources import stac

        req = build_request(
            _spec(city, bbox, start_year, end_year, season_months, latest_window_days, country, False, request_path)
        )
        lines = [f"{req.city}: {req.area_km2:.1f} sq km bbox, {len(req.windows)} windows."]
        for label, (a, b) in zip(req.window_labels, req.windows):
            s2 = stac.search_range(config.S2_COLLECTION, req.bbox, a, b, query={"eo:cloud_cover": {"lt": config.MAX_SCENE_CLOUD_PCT}})
            s1 = stac.search_range(config.S1_COLLECTION, req.bbox, a, b)
            lines.append(f"- {label} ({a}..{b}): Sentinel-2 {len(s2)} scenes, Sentinel-1 {len(s1)} scenes")
        lines.append(
            "Windows with no Sentinel-1 fall back to an optical-only model; windows with "
            "few Sentinel-2 scenes may leave cloud gaps."
        )
        return "\n".join(lines)
    except Exception as exc:
        return f"Planning failed: {type(exc).__name__}: {exc}"


@mcp.tool()
def run_city_request(
    city: str = "AOI",
    bbox: list[float] | None = None,
    start_year: int = 2019,
    end_year: int = 2026,
    season_months: list[int] | None = None,
    latest_window_days: int | None = 90,
    country: str = "",
    detect_loss: bool = False,
    request_path: str | None = None,
) -> str:
    """Produce a built-up change product for a city and add it to the STAC catalog.

    Builds cloud-masked Sentinel-2 and Sentinel-1 composites per window, trains a
    classifier on the AOI's own open building footprints (Overture), scores every
    window, dates each new built-up pixel to the first window it appears in, and
    writes COGs, GeoParquet polygons, H3 neighbourhood statistics, footprint ages,
    a static-layer comparison and a STAC item. Returns a text summary including
    held-out accuracy and Sentinel-1 gaps. Same arguments as ``plan_city_request``;
    ``detect_loss`` adds an experimental built-up loss layer (crisis use).
    """
    try:
        from .city_pipeline import run_city_request as _run
        from .request import build_request

        req = build_request(
            _spec(city, bbox, start_year, end_year, season_months, latest_window_days, country, detect_loss, request_path)
        )
        return _run(req, log=_log).text()
    except Exception as exc:
        return f"Run failed: {type(exc).__name__}: {exc}"


def _items(all_versions: bool = False) -> list[dict]:
    """Products in the catalog; by default only the newest run per request."""
    root = config.PRODUCTS_DIR / "builtup-change"
    newest: dict[str, dict] = {}
    out = []
    for p in sorted(root.glob("*/summary.json")):
        s = json.loads(p.read_text())
        it = {"item_id": s["item_id"], "city": s["city"], "dir": str(p.parent), "summary": s}
        out.append(it)
        newest[s["request"]["request_id"]] = it  # item ids sort by run time
    return out if all_versions else list(newest.values())


def _find(item_id: str) -> dict:
    for it in _items(all_versions=True):
        if it["item_id"] == item_id:
            return it
    items = _items()
    # Allow a city name or request id: newest matching item wins.
    matches = [it for it in items if item_id.lower() in (it["item_id"].lower(), it["city"].lower()) or it["item_id"].startswith(item_id)]
    if matches:
        return matches[-1]
    raise KeyError(f"No product {item_id!r}. Known: {[i['item_id'] for i in items]}")


@mcp.tool()
def list_products() -> str:
    """List built-up change products in the local STAC catalog (newest last)."""
    items = _items()
    if not items:
        return f"No products yet in {config.PRODUCTS_DIR}."
    rows = [
        f"- {it['item_id']}: {it['city']}, {it['summary']['windows'][0]['start']}..{it['summary']['windows'][-1]['end']}, "
        f"new built-up {it['summary']['gain_ha_total']} ha"
        for it in items
    ]
    return f"Catalog: {config.PRODUCTS_DIR / 'catalog.json'}\n" + "\n".join(rows)


@mcp.tool()
def get_product_summary(item_id: str) -> str:
    """Key figures, accuracy, data gaps and static-layer comparison for a product.

    ``item_id`` may also be a city name; the newest product for it is used.
    """
    try:
        s = _find(item_id)["summary"]
    except KeyError as exc:
        return str(exc)
    keep = {
        k: s[k]
        for k in (
            "item_id", "city", "aoi_km2", "window_labels", "built_first_ha", "built_latest_ha",
            "gain_ha_total", "gain_ha_by_window", "provisional_gain_ha", "accuracy",
            "accuracy_note", "footprint_baseline", "static_layers",
        )
        if k in s
    }
    keep["windows"] = [
        {k: w[k] for k in ("label", "start", "end", "sentinel-2", "sentinel-1", "built_ha", "scored_by")}
        for w in s["windows"]
    ]
    return json.dumps(keep, indent=1)


@mcp.tool()
def rank_neighbourhoods(item_id: str, metric: str = "gain_ha_total", top_n: int = 10) -> str:
    """Rank H3 (res 8, ~0.74 sq km) cells of a product by a metric.

    Metrics: ``gain_ha_total`` (new built-up), ``growth_pct`` (built-up growth from
    the first to the latest window), ``built_latest_ha``, or ``gain_ha_e<k>`` for
    window k. Returns cell id, centre lat/lon and the cell's figures.
    """
    import geopandas as gpd
    import h3

    try:
        it = _find(item_id)
    except KeyError as exc:
        return str(exc)
    df = gpd.read_parquet(Path(it["dir"]) / "h3_stats.parquet")
    if metric not in df:
        return f"Unknown metric {metric!r}. Options: {[c for c in df.columns if c not in ('h3', 'geometry')]}"
    top = df[df["area_ha"] > 20].sort_values(metric, ascending=False).head(top_n)
    labels = [it["summary"]["window_labels"][0], it["summary"].get("latest_confirmed_window", it["summary"]["window_labels"][-1])]
    rows = []
    for _, r in top.iterrows():
        lat, lon = h3.cell_to_latlng(r["h3"])
        rows.append(
            f"- {r['h3']} ({lat:.4f}, {lon:.4f}): new built-up {r['gain_ha_total']:.1f} ha, "
            f"built {r['built_first_ha']:.1f} -> {r['built_latest_ha']:.1f} ha "
            f"({labels[0]} -> {labels[1]}), growth {r['growth_pct']}%"
        )
    return f"Top {len(rows)} cells by {metric} in {it['city']}:\n" + "\n".join(rows)


@mcp.tool()
def compare_products(item_ids: list[str] | None = None) -> str:
    """Rank city products by built-up growth (all products if none given).

    Growth is reported per year of the compared period so cities with different
    periods stay comparable.
    """
    try:
        items = [_find(i) for i in item_ids] if item_ids else _items()
    except KeyError as exc:
        return str(exc)
    rows = []
    for it in items:
        s = it["summary"]
        y0 = int(s["windows"][0]["end"][:4])
        conf = s.get("latest_confirmed_window", s["window_labels"][-1])
        y1 = int(next(w["end"] for w in s["windows"] if w["label"] == conf)[:4])
        years = max(y1 - y0, 1)
        b0, b1 = s["built_first_ha"], s["built_latest_ha"]
        rate = 100 * ((b1 / b0) ** (1 / years) - 1) if b0 else float("nan")
        rows.append((rate, f"- {s['city']}: built-up {b0} -> {b1} ha ({y0}-{y1}), {rate:.1f}%/yr, new {s['gain_ha_total']} ha"))
    rows.sort(key=lambda r: (r[0] != r[0], -r[0] if r[0] == r[0] else 0))  # NaN last
    return "\n".join(r for _, r in rows) or "No products."


@mcp.tool()
def static_layers_report(bbox: list[float]) -> str:
    """Newest epoch and built-up area of the open static layers over a bbox.

    Covers Esri/Impact Observatory 10 m land cover, ESA WorldCover and Google Open
    Buildings 2.5D Temporal (all read from cloud-native endpoints).
    """
    try:
        from .geo import parse_bbox
        from .product import aoi_mask
        from .sources.raster_io import aoi_grid
        from .static_layers import static_snapshot
        from shapely.geometry import box, mapping

        b = parse_bbox(bbox)
        grid = aoi_grid(b, config.TARGET_RESOLUTION_M)
        rows = static_snapshot(b, grid, aoi_mask(mapping(box(*b.as_tuple())), grid))
        return json.dumps(rows, indent=1)
    except Exception as exc:
        return f"Failed: {type(exc).__name__}: {exc}"


@mcp.tool()
def detect_builtup_change(
    bbox: list[float],
    target_date: str,
    baseline_date: str | None = None,
) -> str:
    """Two-date built-up change over a small bbox (v0.1 heuristic detector).

    Fuses single Sentinel-2 (NDBI) and Sentinel-1 (VV) scenes near each date with
    fixed thresholds; with no ``baseline_date`` it compares against Overture
    footprints. Kept for quick checks; prefer ``run_city_request`` for products.

    Args:
        bbox: WGS84 [min_lon, min_lat, max_lon, max_lat], under 5 sq km.
        target_date: The "after" date, ISO 'YYYY-MM-DD'.
        baseline_date: Optional "before" date, ISO 'YYYY-MM-DD'.
    """
    try:
        box = validate_bbox(bbox)
        target, baseline = validate_change_dates(target_date, baseline_date)
    except BBoxValidationError as exc:
        return f"Input validation failed: {exc}"
    try:
        from .pipeline import run_change_detection
    except ImportError as exc:
        return (
            "Validation passed, but the geospatial pipeline dependencies are not "
            f"installed ({exc}). Install with: uv sync --extra vector --extra raster."
        )
    try:
        run = run_change_detection(list(box.as_tuple()), target, baseline)
    except Exception as exc:
        return f"Pipeline error: {type(exc).__name__}: {exc}"
    return run.summary()


def run() -> None:
    """Run the MCP server over stdio transport."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mcp.run()
