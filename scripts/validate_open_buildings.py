"""Validate Sentinel-derived built-up GAIN against Google Open Buildings 2.5D
for a single AOI (pair mode 2021 -> 2023), with an ESRI imagery basemap.

Network required (several minutes). Run with:
    uv run python scripts/validate_open_buildings.py [aoi_name]
    uv run python scripts/validate_open_buildings.py <min_lon> <min_lat> <max_lon> <max_lat>
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

from builtup_mcp import config
from builtup_mcp.geo import BBox, utm_epsg
from builtup_mcp.pipeline import run_change_detection
from builtup_mcp.validation import compare_gain_to_open_buildings, plot_comparison

OUT_DIR = Path("outputs")
BASELINE, TARGET = "2021-06-15", "2023-06-15"
Y1, Y2 = 2021, 2023


def _resolve_aoi() -> tuple[str, tuple[float, float, float, float]]:
    if len(sys.argv) == 5:
        return "(custom)", tuple(float(x) for x in sys.argv[1:5])
    if len(sys.argv) == 2 and sys.argv[1] in config.TEST_AOIS:
        return sys.argv[1], config.TEST_AOIS[sys.argv[1]]
    return config.DEMO_AOI_NAME, config.DEMO_BBOX


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name, coords = _resolve_aoi()
    box = BBox(*coords)
    print(f"AOI: {name}  bbox={coords}  UTM EPSG:{utm_epsg(box)}")

    print(f"Running pipeline {BASELINE} -> {TARGET} ...")
    run = run_change_detection(list(coords), TARGET, BASELINE, output_dir=OUT_DIR)
    print(run.summary())

    cmp = compare_gain_to_open_buildings(gpd.read_file(run.geojson_path), box, Y1, Y2)
    m = cmp.metrics
    print(f"\n=== Validation vs Open Buildings ({Y1}->{Y2}) ===")
    if m["n_gain"]:
        print(f"Gain px: {m['n_gain']} (tol {m['tolerance_px']}px) | "
              f"built-up {Y2}: {m['prec_builtup_2023']:.1%} (x{m['enrichment_builtup']:.1f}) | "
              f"OB-new: {m['prec_new']:.1%} (x{m['enrichment_new']:.1f}) | "
              f"recall {m['recall_new']:.1%} of {m['n_google_new']}")
        ev = "STRONG" if m["base_new"] > 0.05 else "WEAK"
        print(f"Google-confirmed change base rate {m['base_new']:.1%} -> {ev} evidence in this AOI.")
    else:
        print("No gains detected.")

    slug = name.split("/")[0].strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
    out = OUT_DIR / f"validation_{slug}.png"
    plot_comparison(cmp, f"{name} ({Y1}->{Y2})\nyellow=agreement, red=detected-only, cyan=reference-only; faint=ESRI imagery", out)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
