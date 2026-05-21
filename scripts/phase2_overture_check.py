"""Phase 2 live check: query Overture buildings for the demo AOI and report.

Hits the public Overture S3 bucket (network required). Run with:
    uv run python scripts/phase2_overture_check.py
"""

from __future__ import annotations

import sys
import time

from builtup_mcp import config
from builtup_mcp.geo import validate_bbox
from builtup_mcp.sources.overture import fetch_buildings


def main() -> None:
    if len(sys.argv) == 5:
        coords = [float(x) for x in sys.argv[1:5]]
        box = validate_bbox(coords)
        name = "(custom bbox from argv)"
    else:
        coords = list(config.DEMO_BBOX)
        box = validate_bbox(coords)
        name = config.DEMO_AOI_NAME
    print(f"BBox: {box.as_tuple()}")
    print(f"Overture release: {config.OVERTURE_RELEASE}")

    t0 = time.perf_counter()
    gdf = fetch_buildings(box)
    dt = time.perf_counter() - t0

    print(f"\nRetrieved {len(gdf)} building(s) in {dt:.1f}s")
    if gdf.empty:
        print("No buildings in this BBox (undeveloped area or coverage gap).")
        return

    with_height = gdf["effective_height_m"].notna().sum()
    print(f"Buildings with an effective height: {with_height}/{len(gdf)}")
    print("Columns:", list(gdf.columns))
    print("\nSample rows:")
    print(
        gdf[["id", "subtype", "class", "height", "num_floors", "effective_height_m"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
