"""Evidence-driven AOI test: find the highest-change sub-AOI in a region using
Google Open Buildings, then run the detector there and validate against OB.

This removes coordinate guessing: the AOI is chosen because OB confirms built-up
increased there between two years.

Network required (several minutes). Run with:
    uv run python scripts/find_and_validate_aoi.py [region_name] [year1] [year2]
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

from builtup_mcp import config
from builtup_mcp.geo import BBox, utm_epsg
from builtup_mcp.pipeline import run_change_detection
from builtup_mcp.processing.hotspot import find_change_hotspot
from builtup_mcp.sources.open_buildings import fetch_building_presence
from builtup_mcp.sources.raster_io import target_grid
from builtup_mcp.validation import compare_gain_to_open_buildings, plot_comparison

OUT_DIR = Path("outputs")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    region_name = sys.argv[1] if len(sys.argv) > 1 else "nairobi_ne"
    if region_name not in config.SCAN_REGIONS:
        sys.exit(f"Unknown region '{region_name}'. Choices: {', '.join(config.SCAN_REGIONS)}")
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2018
    y2 = int(sys.argv[3]) if len(sys.argv) > 3 else 2023
    region = BBox(*config.SCAN_REGIONS[region_name])

    # 1) Scan the region with Open Buildings to find the highest-change AOI.
    print(f"Scanning region {region_name} {region.as_tuple()} for {y1}->{y2} change...")
    scan_grid = target_grid(region, utm_epsg(region), 40.0)
    p1 = fetch_building_presence(scan_grid, y1, mosaic=True)
    p2 = fetch_building_presence(scan_grid, y2, mosaic=True)
    aoi, stats = find_change_hotspot(region, p1, p2, scan_grid, target_km2=4.0)
    print(f"Hotspot AOI: {tuple(round(v, 4) for v in aoi.as_tuple())}")
    print(f"  OB-confirmed new pixels in window: {stats['new_px_in_window']} "
          f"({stats['new_fraction']:.1%} of window); region total {stats['region_new_px']}")

    # 2) Run the detector on the evidence-selected AOI (pair mode, same years).
    print(f"\nRunning detector {y1}-06-15 -> {y2}-06-15 on the hotspot...")
    run = run_change_detection(
        list(aoi.as_tuple()), f"{y2}-06-15", f"{y1}-06-15", output_dir=OUT_DIR
    )
    print(run.summary())

    # 3) Validate gains against Open Buildings on the same AOI/years.
    cmp = compare_gain_to_open_buildings(gpd.read_file(run.geojson_path), aoi, y1, y2)
    m = cmp.metrics
    print(f"\n=== Validation vs Open Buildings ({y1}->{y2}) ===")
    if m["n_gain"] == 0:
        print("No gains detected.")
    else:
        print(f"Detected gain pixels: {m['n_gain']}  (tolerance {m['tolerance_px']}px)")
        print(f"  built-up in OB {y2}: {m['prec_builtup_2023']:.1%} "
              f"(base {m['base_builtup_2023']:.1%}, enrichment x{m['enrichment_builtup']:.1f})")
        print(f"  OB-new {y1}->{y2} (precision): {m['prec_new']:.1%} "
              f"(base {m['base_new']:.1%}, enrichment x{m['enrichment_new']:.1f})")
        print(f"  OB-new recall: {m['recall_new']:.1%} of {m['n_google_new']} OB-new px detected")
        print(f"  mean OB presence delta under gains {m['mean_presence_delta_gain']:+.3f} "
              f"vs background {m['mean_presence_delta_bg']:+.3f}")

    out = OUT_DIR / f"hotspot_{region_name}_{y1}_{y2}.png"
    plot_comparison(
        cmp,
        f"Evidence-selected AOI in {region_name} ({y1}->{y2})\n"
        "yellow=agreement, red=detected-only, cyan=reference-only; faint=ESRI imagery",
        out,
    )
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
