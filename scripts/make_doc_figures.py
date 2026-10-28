"""Figures for METHODOLOGY.md, drawn from the published Malakal product.

    uv run python scripts/make_doc_figures.py [product_dir]

Reads the product (default: the Malakal 66 sq km box in the products cache)
and WSF Tracker for the same grid, and writes three JPEGs to docs/assets/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
from pyproj import Transformer
from scipy import ndimage

from builtup_mcp import config
from builtup_mcp.geo import BBox
from builtup_mcp.processing.classify import building_fraction, weak_labels
from builtup_mcp.sources.raster_io import Grid
from builtup_mcp.sources.wsf_tracker import EPOCHS, fetch_wsf_tracker

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
DEFAULT = config.PRODUCTS_DIR / "builtup-change" / "ssd-malakal-builtup-change-20261023T043456Z"
# Zoom centre: the fastest-growing H3 cell in Malakal (rank_neighbourhoods).
ZOOM_LONLAT = (31.6608, 9.5108)
ZOOM_PX = 140  # 1.4 km square

SEASON_COLOURS = ["#fde725", "#a0da39", "#4ac16d", "#1fa187", "#277f8e", "#365c8d", "#46327e"]


def load(product: Path):
    s = json.loads((product / "summary.json").read_text())
    with rasterio.open(product / "gain_epoch.tif") as r:
        gain, crs, tf, bounds = r.read(1), r.crs, r.transform, r.bounds
    with rasterio.open(product / "builtup_probability.tif") as r:
        prob = r.read()
    grid = Grid(crs.to_epsg(), tuple(bounds), gain.shape[1], gain.shape[0], 10.0)
    fp = gpd.read_parquet(product / "footprints_age.parquet").to_crs(crs)
    return s, gain, prob, grid, tf, fp


def zoom_window(grid: Grid, tf, size: int = ZOOM_PX) -> tuple[slice, slice, tuple]:
    x, y = Transformer.from_crs(4326, grid.epsg, always_xy=True).transform(*ZOOM_LONLAT)
    row, col = rasterio.transform.rowcol(tf, x, y)
    h = size // 2
    r0, c0 = max(row - h, 0), max(col - h, 0)
    rs, cs = slice(r0, r0 + size), slice(c0, c0 + size)
    left, top = tf * (c0, r0)
    return rs, cs, (left, left + size * 10, top - size * 10, top)


def rgb(product: Path, k: int) -> np.ndarray:
    return np.asarray(Image.open(product / f"quicklook_{k:02d}.png").convert("RGB"))


def tidy(ax, title):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])


def fig_labels(product, s, prob, grid, tf, fp):
    """Footprints -> weak labels -> probability, on one zoom window."""
    rs, cs, ext = zoom_window(grid, tf)
    frac = building_fraction(fp, grid)
    lab = weak_labels(frac, grid.resolution_m)[rs, cs]
    last = len(s["window_labels"]) - 2  # last dry season
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    axes[0].imshow(rgb(product, last)[rs, cs], extent=ext)
    fp.cx[ext[0]:ext[1], ext[2]:ext[3]].boundary.plot(ax=axes[0], color="#ffcc00", linewidth=0.5)
    axes[0].set_xlim(ext[:2]), axes[0].set_ylim(ext[2:])
    tidy(axes[0], "1. Open footprints (Overture) on the\nSentinel-2 dry-season composite")
    axes[1].imshow(lab, cmap=ListedColormap(["#f0f0f0", "#3b6fb6", "#d7301f"]), vmin=-1, vmax=1, extent=ext, interpolation="nearest")
    tidy(axes[1], "2. Labels per 10 m pixel")
    axes[1].legend(
        handles=[Patch(color="#d7301f", label="built-up (>= 25% roof cover)"),
                 Patch(color="#3b6fb6", label="not built (> 40 m from any footprint)"),
                 Patch(facecolor="#f0f0f0", edgecolor="#999", label="ignored")],
        loc="lower left", fontsize=7.5, framealpha=0.9,
    )
    im = axes[2].imshow(prob[last][rs, cs], cmap="magma", vmin=0, vmax=100, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="probability (%)")
    tidy(axes[2], f"3. Model output: built-up probability,\n{s['window_labels'][last]}")
    fig.savefig(ASSETS / "method_labels.jpg", dpi=130, pil_kwargs={"quality": 85}, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fig_vs_wsf(product, s, gain, grid, tf):
    """Where this product and WSF Tracker agree on new built-up since the first season."""
    wb = s["benchmark"]["wsf_tracker"]
    lo, la = Transformer.from_crs(grid.epsg, 4326, always_xy=True).transform(
        [grid.bounds[0], grid.bounds[2]], [grid.bounds[1], grid.bounds[3]]
    )
    wsf = fetch_wsf_tracker(grid, BBox(min(lo), min(la), max(lo), max(la)))
    windows = [tuple(w) if isinstance(w, list) else (w["start"], w["end"]) for w in s["windows"]]
    first_end = windows[0][1]
    fi = max((i for i, d in EPOCHS.items() if d <= first_end), default=0)
    j = max(i for i, d in EPOCHS.items() if d <= wb["extent"]["our_window"][1])
    k = max(i for i, (_a, b) in enumerate(windows) if b <= wb["extent"]["our_window"][1])
    ours = (gain >= 1) & (gain <= k)
    theirs = (wsf > fi) & (wsf <= j)
    near_t = ndimage.binary_dilation(theirs)
    near_o = ndimage.binary_dilation(ours)
    cls = np.zeros(gain.shape, "uint8")
    cls[(wsf >= 1) & (wsf <= fi)] = 1  # built before the first season (WSF)
    cls[ours & ~near_t] = 2
    cls[theirs & ~near_o] = 3
    cls[(ours & near_t) | (theirs & near_o)] = 4
    cmap = ListedColormap(["#ffffff", "#d9d9d9", "#e66101", "#5e3c99", "#1a9641"])
    fig, ax = plt.subplots(figsize=(6.2, 8.4), constrained_layout=True)
    ax.imshow(cls, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    n = wb["new_builtup"]
    tidy(ax, f"New built-up, {s['window_labels'][0]} to Jan 2026\nthis product {n['ours_ha']:.0f} ha, WSF Tracker {n['reference_ha']:.0f} ha, agreement F1 {n['f1']:.2f}")
    ax.legend(
        handles=[Patch(color="#1a9641", label="new in both (1-pixel tolerance)"),
                 Patch(color="#e66101", label="new in this product only"),
                 Patch(color="#5e3c99", label="new in WSF Tracker only"),
                 Patch(color="#d9d9d9", label="built before 2019 (WSF Tracker)")],
        loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8, frameon=False,
    )
    ax.set_xlabel("Malakal, South Sudan, 66 sq km, 10 m grid", fontsize=7, color="#555", labelpad=40)
    fig.savefig(ASSETS / "method_vs_wsf.jpg", dpi=130, pil_kwargs={"quality": 85}, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fig_footprints(product, s, gain, grid, tf, fp):
    """Footprints coloured by the season they first read as built-up, plus unmapped gain."""
    rs, cs, ext = zoom_window(grid, tf, 70)
    last = len(s["window_labels"]) - 2
    frac = building_fraction(fp, grid)
    unmapped = ((gain >= 1) & (gain <= last) & (frac <= 0))[rs, cs]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    ax.imshow(rgb(product, last)[rs, cs], extent=ext, alpha=0.6)
    ax.imshow(np.ma.masked_where(~unmapped, unmapped), cmap=ListedColormap(["#ff00aa"]), extent=ext, alpha=0.45, interpolation="nearest")
    sub = fp.cx[ext[0]:ext[1], ext[2]:ext[3]]
    before = sub.builtup_since.str.startswith("before")
    sub[before].plot(ax=ax, facecolor="none", edgecolor="#444444", linewidth=0.6)
    missed = sub[(sub.gain_epoch < 1) & ~before]
    if len(missed):
        missed.plot(ax=ax, facecolor="none", edgecolor="#d7301f", linewidth=0.8)
    handles = [Patch(facecolor="none", edgecolor="#444444", label=f"built-up already in {s['window_labels'][0]}"),
               Patch(facecolor="none", edgecolor="#d7301f", label=f"never read as built-up ({len(missed)})")]
    for e in range(1, last + 1):
        g = sub[sub.gain_epoch == e]
        if not len(g):
            continue
        g.plot(ax=ax, facecolor=SEASON_COLOURS[(e - 1) % len(SEASON_COLOURS)], edgecolor="#000000", linewidth=0.6)
        handles.append(Patch(color=SEASON_COLOURS[(e - 1) % len(SEASON_COLOURS)], label=f"first built-up {s['window_labels'][e]} ({len(g)})"))
    handles.append(Patch(color="#ff00aa", alpha=0.45, label="new built-up with no footprint"))
    ax.set_xlim(ext[:2]), ax.set_ylim(ext[2:])
    tidy(ax, "Footprints dated by the season their pixel first read as built-up,\nand new built-up the footprint map does not have yet")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(ASSETS / "method_footprints.jpg", dpi=130, pil_kwargs={"quality": 85}, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    product = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    s, gain, prob, grid, tf, fp = load(product)
    fig_labels(product, s, prob, grid, tf, fp)
    fig_vs_wsf(product, s, gain, grid, tf)
    fig_footprints(product, s, gain, grid, tf, fp)
    print("wrote", *sorted(p.name for p in ASSETS.glob("method_*.jpg")))


if __name__ == "__main__":
    main()
