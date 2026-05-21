"""MCP server exposing the ``detect_builtup_change`` tool.

Phase 1 wires up the server, the tool schema, and full input validation.
The geospatial change-detection pipeline (Phases 2-4) is stubbed and will be
filled in later.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import config
from .geo import BBoxValidationError, validate_bbox, validate_change_dates

mcp = FastMCP("builtup-mcp")


@mcp.tool()
def detect_builtup_change(
    bbox: list[float],
    target_date: str,
    baseline_date: str | None = None,
) -> str:
    """Detect on-demand built-up change (gain/loss) over a small bbox.

    Produces a built-up CHANGE signal: where built-up area has increased
    (new construction) or decreased (e.g. disaster-driven destruction) within
    the bbox, at higher temporal cadence than published epoch products
    (GHSL, WSF). Fuses SAR (Sentinel-1, all-weather) with optical (Sentinel-2)
    and uses Overture buildings as a baseline prior. Writes a GeoJSON of the
    change signal and returns a concise text summary.

    The output schema is intentionally extensible: future sources (e.g. NISAR)
    may add attributes such as height, footprint geometry, or roughness.

    Args:
        bbox: WGS84 bounding box as [min_lon, min_lat, max_lon, max_lat].
            Must be smaller than the configured cap (default 5 square kilometers).
        target_date: The "after" acquisition date, ISO 'YYYY-MM-DD'.
        baseline_date: Optional "before" date, ISO 'YYYY-MM-DD'. If given it
            defines a pre/post pair (must be before target_date); if omitted,
            change is measured against a static reference baseline dataset.

    Returns:
        A concise human-readable summary: counts of built-up gain/loss pixels,
        the comparison mode used, and the path to the saved GeoJSON file.
    """
    # --- Input validation (Phase 1) ---------------------------------------
    try:
        box = validate_bbox(bbox)
        target, baseline = validate_change_dates(target_date, baseline_date)
    except BBoxValidationError as exc:
        return f"Input validation failed: {exc}"

    # --- Pipeline (Phases 2-4) ---------------------------------------------
    # Imported lazily so the tool schema is available without the geospatial
    # extras (vector/raster) installed.
    try:
        from .pipeline import run_change_detection
    except ImportError as exc:  # missing optional extras
        return (
            "Validation passed, but the geospatial pipeline dependencies are not "
            f"installed ({exc}). Install with: uv sync --extra vector --extra raster."
        )

    try:
        run = run_change_detection(list(box.as_tuple()), target, baseline)
    except Exception as exc:  # surface runtime/network errors to the LLM
        return f"Pipeline error: {type(exc).__name__}: {exc}"

    return run.summary()


def run() -> None:
    """Run the MCP server over stdio transport."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mcp.run()
