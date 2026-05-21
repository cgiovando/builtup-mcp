"""Validate detected built-up gain against Google Open Buildings 2.5D."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio.transform
from scipy.ndimage import binary_dilation

from . import config
from .geo import BBox, utm_epsg
from .processing.align import grid_transform
from .sources.open_buildings import fetch_building_presence
from .sources.raster_io import Grid, target_grid


def rasterize_points(gdf: gpd.GeoDataFrame, grid: Grid) -> np.ndarray:
    """Burn point centroids onto a boolean grid mask."""
    mask = np.zeros(grid.shape, dtype=bool)
    if gdf is None or gdf.empty:
        return mask
    utm = gdf.to_crs(epsg=grid.epsg)
    rows, cols = rasterio.transform.rowcol(
        grid_transform(grid), utm.geometry.x.values, utm.geometry.y.values
    )
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    mask[rows[ok], cols[ok]] = True
    return mask


@dataclass
class OBComparison:
    grid: Grid
    gain: np.ndarray
    presence_y2: np.ndarray
    google_new: np.ndarray
    metrics: dict


def plot_comparison(cmp: "OBComparison", title: str, out_path, *, basemap: bool = True):
    """Render gain vs Open Buildings over a faint ESRI satellite basemap.

    Pixels are split three ways (using the same tolerance as the metrics):
      - agreement (yellow): detected gain that matches an Open Buildings new pixel
      - detected only (red): detected gain not flagged new by Open Buildings
      - reference only (cyan): Open Buildings new that was not detected (recall gap)
    Everything is drawn in the AOI's UTM CRS; contextily warps ESRI World Imagery
    tiles to that CRS so the pixels sit on real geography.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio.transform

    from .processing.align import grid_transform

    grid = cmp.grid
    tol = cmp.metrics.get("tolerance_px", 1)
    gain, new = cmp.gain, cmp.google_new
    new_buf = _dilate(new, tol)
    gain_buf = _dilate(gain, tol)
    agree = gain & new_buf
    gain_only = gain & ~new_buf
    new_only = new & ~gain_buf

    transform = grid_transform(grid)

    def scatter(mask, color, label, *, size, marker, zorder):
        rows, cols = np.where(mask)
        if not rows.size:
            return
        xs, ys = rasterio.transform.xy(transform, rows.tolist(), cols.tolist())
        ax.scatter(xs, ys, s=size, c=color, marker=marker, edgecolors="none",
                   zorder=zorder, label=f"{label} ({int(mask.sum())}px)")

    left, bottom, right, top = grid.bounds
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")

    if basemap:
        try:
            import contextily as ctx

            ctx.add_basemap(
                ax, crs=f"EPSG:{grid.epsg}", source=ctx.providers.Esri.WorldImagery,
                alpha=0.55, attribution_size=5, zorder=0,
            )
        except Exception as exc:  # offline / tile error -> plain background
            print(f"(basemap skipped: {exc})")

    scatter(new_only, "#00bfff", "reference only (undetected)", size=9, marker="s", zorder=2)
    scatter(gain_only, "#d73f3f", "detected only", size=10, marker="o", zorder=3)
    scatter(agree, "#ffe400", "agreement", size=16, marker="o", zorder=4)

    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _dilate(mask: np.ndarray, tolerance_px: int) -> np.ndarray:
    """Grow a mask by ``tolerance_px`` cells (to absorb sub-pixel georeferencing
    offset between the 10m detection grid and Open Buildings' ~4m grid)."""
    if tolerance_px <= 0:
        return mask
    return binary_dilation(mask, iterations=tolerance_px)


def compare_gain_to_open_buildings(
    gain_gdf: gpd.GeoDataFrame,
    bbox: BBox,
    year1: int,
    year2: int,
    *,
    presence_thr: float = 0.5,
    res_m: float | None = None,
    tolerance_px: int = 1,
) -> OBComparison:
    """Compare detected gain pixels to Open Buildings presence change ``year1`` -> ``year2``.

    A ``tolerance_px`` buffer treats a gain pixel as agreeing if it is within that
    many cells of an Open Buildings target pixel (and vice-versa for recall).
    Precision and the matching base rate both use the buffered target, so the
    enrichment ratio stays apples-to-apples.
    """
    grid = target_grid(bbox, utm_epsg(bbox), res_m or config.TARGET_RESOLUTION_M)
    gain = rasterize_points(gain_gdf, grid)
    p1 = fetch_building_presence(grid, year1)
    p2 = fetch_building_presence(grid, year2)

    valid = np.isfinite(p1) & np.isfinite(p2)
    g1 = (p1 >= presence_thr) & valid
    g2 = (p2 >= presence_thr) & valid
    google_new = g2 & ~g1
    gain = gain & valid

    g2_t = _dilate(g2, tolerance_px) & valid
    new_t = _dilate(google_new, tolerance_px) & valid
    gain_t = _dilate(gain, tolerance_px) & valid

    n_gain = int(gain.sum())
    n_valid = int(valid.sum())
    metrics: dict = {"n_gain": n_gain, "n_valid": n_valid, "tolerance_px": tolerance_px}
    if n_gain and n_valid:
        base_built = g2_t.sum() / n_valid
        base_new = new_t.sum() / n_valid
        prec_built = (gain & g2_t).sum() / n_gain
        prec_new = (gain & new_t).sum() / n_gain
        n_new = int(google_new.sum())
        recall_new = (google_new & gain_t).sum() / n_new if n_new else float("nan")
        metrics.update(
            prec_builtup_2023=float(prec_built),
            base_builtup_2023=float(base_built),
            enrichment_builtup=float(prec_built / base_built) if base_built else float("nan"),
            prec_new=float(prec_new),
            base_new=float(base_new),
            enrichment_new=float(prec_new / base_new) if base_new else float("nan"),
            recall_new=float(recall_new),
            n_google_new=n_new,
            mean_presence_delta_gain=float(np.nanmean((p2 - p1)[gain])),
            mean_presence_delta_bg=float(np.nanmean((p2 - p1)[valid])),
        )
    return OBComparison(grid, gain, p2, google_new, metrics)
