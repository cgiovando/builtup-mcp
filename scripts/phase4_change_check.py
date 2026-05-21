"""Phase 4 live end-to-end: run change detection and render the result.

Runs the full pipeline (Overture + windowed S1/S2 + change detection) over the
demo AOI and writes a change GeoJSON, a KML for Google Earth, and a preview PNG.

Network required (several minutes). Run with:
    uv run python scripts/phase4_change_check.py [baseline_date] [target_date]
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from builtup_mcp import config
from builtup_mcp.pipeline import run_change_detection

OUT_DIR = Path("outputs")


def main() -> None:
    baseline = sys.argv[1] if len(sys.argv) > 1 else "2021-06-15"
    target = sys.argv[2] if len(sys.argv) > 2 else "2024-06-15"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"AOI: {config.DEMO_AOI_NAME}")
    print(f"Baseline -> Target: {baseline} -> {target}\n")

    run = run_change_detection(
        list(config.DEMO_BBOX), target, baseline, output_dir=OUT_DIR
    )
    print(run.summary())

    gdf = gpd.read_file(run.geojson_path)
    if gdf.empty:
        print("No change features to plot.")
        return

    # Export KML for Google Earth.
    kml = gdf.copy()
    kml["Name"] = kml["change_type"]
    kml["Description"] = "confidence=" + kml["confidence"].astype(str)
    kml[["Name", "Description", "geometry"]].to_file(
        OUT_DIR / "builtup_change.kml", driver="KML"
    )

    # Preview: change points over Overture footprints (if available).
    fig, ax = plt.subplots(figsize=(9, 9))
    footprints = OUT_DIR / "buildings.geojson"
    if footprints.exists():
        gpd.read_file(footprints).plot(
            ax=ax, facecolor="#dddddd", edgecolor="#bbbbbb", linewidth=0.2
        )
    for kind, color in [("gain", "#d73f3f"), ("loss", "#2c6fbb")]:
        sub = gdf[gdf["change_type"] == kind]
        if not sub.empty:
            sub.plot(ax=ax, color=color, markersize=6, label=f"{kind} ({len(sub)})")
    ax.set_title(
        f"Built-up change {baseline} -> {target}\n{config.DEMO_AOI_NAME}", fontsize=11
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "builtup_change_preview.png", dpi=150)
    plt.close(fig)
    print("\nWrote: outputs/builtup_change.kml, outputs/builtup_change_preview.png")


if __name__ == "__main__":
    main()
