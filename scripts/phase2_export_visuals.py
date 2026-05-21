"""Export Phase 2 visuals for inspection (Google Earth + quick PNG preview).

Queries Overture buildings for the demo AOI (or a bbox passed as argv) and writes
to ./outputs/:
  - buildings.geojson / buildings.kml  (footprints; open the .kml in Google Earth)
  - aoi_bbox.geojson / aoi_bbox.kml    (the AOI rectangle)
  - buildings_preview.png              (matplotlib footprint preview)

Network required. Run with:
    uv run python scripts/phase2_export_visuals.py
    uv run python scripts/phase2_export_visuals.py <min_lon> <min_lat> <max_lon> <max_lat>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box as shp_box

from builtup_mcp import config
from builtup_mcp.geo import bbox_area_km2, validate_bbox
from builtup_mcp.sources.overture import fetch_buildings

OUT_DIR = Path("outputs")


def main() -> None:
    if len(sys.argv) == 5:
        coords = [float(x) for x in sys.argv[1:5]]
        name = "(custom bbox)"
    else:
        coords = list(config.DEMO_BBOX)
        name = config.DEMO_AOI_NAME
    box = validate_bbox(coords)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"AOI: {name}")
    print(f"BBox: {box.as_tuple()}  (~{bbox_area_km2(box):.2f} sq km)")
    print(f"Overture release: {config.OVERTURE_RELEASE}")

    t0 = time.perf_counter()
    gdf = fetch_buildings(box)
    print(f"Retrieved {len(gdf)} building(s) in {time.perf_counter() - t0:.0f}s")

    # AOI rectangle as its own layer.
    aoi = gpd.GeoDataFrame(
        {"name": [name]}, geometry=[shp_box(*box.as_tuple())], crs="EPSG:4326"
    )

    # GeoJSON outputs (lossless, all attributes).
    gdf.to_file(OUT_DIR / "buildings.geojson", driver="GeoJSON")
    aoi.to_file(OUT_DIR / "aoi_bbox.geojson", driver="GeoJSON")

    # KML outputs for Google Earth. KML carries a single description field, so
    # keep a compact attribute string per building.
    if not gdf.empty:
        kml = gdf.copy()
        kml["Name"] = "building"
        kml["Description"] = (
            "height(m)="
            + gdf["effective_height_m"].round(1).astype("string").fillna("n/a")
        )
        kml[["Name", "Description", "geometry"]].to_file(
            OUT_DIR / "buildings.kml", driver="KML"
        )
    aoi.rename(columns={"name": "Name"}).to_file(OUT_DIR / "aoi_bbox.kml", driver="KML")

    # PNG preview: footprints (filled) inside the AOI outline.
    fig, ax = plt.subplots(figsize=(8, 8))
    aoi.boundary.plot(ax=ax, color="#d73f3f", linewidth=2, label="AOI")
    if not gdf.empty:
        gdf.plot(ax=ax, facecolor="#3b7dd8", edgecolor="#1f3f6e", linewidth=0.3)
    ax.set_title(f"Overture buildings - {name}\n{len(gdf)} footprints", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "buildings_preview.png", dpi=150)
    plt.close(fig)

    print("\nWrote:")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"  {f}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
