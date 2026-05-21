"""Phase 3 live check: windowed Sentinel-1 + Sentinel-2 reads over the AOI.

Verifies array shapes match the expected pixel dimensions, reports scene
selection / cloud info, and writes quick preview PNGs to ./outputs/.

Network required. Run with:
    uv run python scripts/phase3_eo_check.py [target_date]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from builtup_mcp import config
from builtup_mcp.geo import BBox, bbox_area_km2
from builtup_mcp.sources import sentinel1, sentinel2
from builtup_mcp.sources.raster_io import target_grid

OUT_DIR = Path("outputs")


def _save(arr, title, path, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6, 6))
    finite = arr[np.isfinite(arr)]
    vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0, 1))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    target_date = sys.argv[1] if len(sys.argv) > 1 else "2024-06-15"
    box = BBox(*config.DEMO_BBOX)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"AOI: {config.DEMO_AOI_NAME}  (~{bbox_area_km2(box):.2f} sq km)")
    print(f"Target date: {target_date}\n")

    print("Sentinel-2 (optical)...")
    s2 = sentinel2.fetch_optical(box, target_date)
    expected = target_grid(box, s2.grid.epsg, config.TARGET_RESOLUTION_M).shape
    print(f"  item: {s2.item_id}")
    print(f"  datetime: {s2.datetime}")
    print(f"  scene cloud %: {s2.meta['scene_cloud_cover']}")
    print(f"  AOI cloud fraction: {s2.meta['aoi_cloud_fraction']:.3f}")
    print(f"  UTM EPSG: {s2.grid.epsg}")
    print(f"  shapes: nir={s2['nir'].shape} swir={s2['swir'].shape} scl={s2['scl'].shape}")
    print(f"  expected shape from grid: {expected}  -> match: {s2['nir'].shape == expected}\n")

    print("Sentinel-1 (SAR)...")
    s1 = sentinel1.fetch_sar(box, target_date)
    expected_s1 = target_grid(box, s1.grid.epsg, config.TARGET_RESOLUTION_M).shape
    print(f"  item: {s1.item_id}")
    print(f"  datetime: {s1.datetime}  (|days from target|={s1.meta['days_from_target']})")
    print(f"  orbit: {s1.meta['orbit_state']}")
    print(f"  UTM EPSG: {s1.grid.epsg}")
    print(f"  shapes: vv={s1['vv'].shape} vh={s1['vh'].shape}")
    print(f"  expected shape from grid: {expected_s1}  -> match: {s1['vv'].shape == expected_s1}\n")

    # Quick previews (Phase 3 sanity; NDBI etc. is Phase 4).
    _save(s2["scl"].astype(float), f"S2 SCL (cloud mask classes)\n{s2.item_id}", OUT_DIR / "s2_scl.png", cmap="tab20")
    _save(s1["vv_db"], f"S1 VV backscatter (dB)\n{s1.item_id}", OUT_DIR / "s1_vv_db.png", cmap="gray")
    print("Wrote previews: outputs/s2_scl.png, outputs/s1_vv_db.png")


if __name__ == "__main__":
    main()
