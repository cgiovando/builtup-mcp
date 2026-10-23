"""Multi-scene seasonal composites from Sentinel-2 and Sentinel-1 (v0.2).

A single scene per date is noisy: clouds and haze on the optical side, speckle
and incidence effects on the radar side, and season (green-up vs dry bare soil)
on both. Composites fix most of that. For each time window every usable scene
is warped onto one analysis grid and reduced per pixel:

- Sentinel-2: cloud-masked (SCL) median of each band, plus the count of clear
  observations. Processing baseline 04.00+ scenes (from 25 Jan 2022) carry a
  +1000 DN offset that is removed here, otherwise indices drift across 2022.
- Sentinel-1: median and standard deviation of VV/VH backscatter in dB over all
  acquisitions from one orbit direction. Built-up is bright AND temporally stable
  in C-band; this is the same temporal-statistics idea behind the World
  Settlement Footprint, applied on demand to an arbitrary window.

Reads run in a thread pool; each is a windowed COG read, so even a 100 sq km
AOI stays at a few hundred MB of network traffic per window.
"""

from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pystac

from .. import config
from ..geo import BBox
from . import stac
from .raster_io import Grid, read_warped

# Sentinel-2 bands kept in the composite (PC asset keys).
S2_BANDS: dict[str, str] = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "swir1": "B11",
    "swir2": "B12",
}
_S2_OFFSET_BASELINE = "04.00"
_FAILED = {"__failed__": True}  # sentinel: read failed (not "too cloudy")
_S2_OFFSET_DN = 1000.0


@dataclass
class Composite:
    """Per-window composite bands on the analysis grid, plus provenance."""

    window: tuple[str, str]
    grid: Grid
    bands: dict[str, np.ndarray]
    scenes: dict[str, list[str]] = field(default_factory=dict)  # sensor -> item ids
    meta: dict = field(default_factory=dict)

    @property
    def has_sar(self) -> bool:
        # Temporal statistics from one or two acquisitions are not comparable
        # with those from ten, so sparse windows are treated as SAR gaps.
        return "vv_med" in self.bands and self.meta.get("s1_used", 0) >= config.COMPOSITE_MIN_S1_FOR_MODEL

    @property
    def has_optical(self) -> bool:
        return "nir" in self.bands


def _harmonize(arr: np.ndarray, item: pystac.Item) -> np.ndarray:
    """Remove the processing-baseline 04.00 reflectance offset (DN - 1000)."""
    baseline = str(item.properties.get("s2:processing_baseline", "00.00"))
    if baseline >= _S2_OFFSET_BASELINE:
        arr = arr - _S2_OFFSET_DN
        arr[arr < 0] = 0
    return arr


def _read_s2_scene(item: pystac.Item, grid: Grid) -> dict[str, np.ndarray] | None:
    """Read SCL first; skip the scene if the AOI is mostly cloud or unreadable."""
    try:
        return _read_s2_scene_unsafe(item, grid)
    except Exception as exc:  # throttling or a corrupt asset: drop the scene, keep the window
        print(f"    skipped {item.id}: {type(exc).__name__}", file=sys.stderr)
        return _FAILED


def _read_s2_scene_unsafe(item: pystac.Item, grid: Grid) -> dict[str, np.ndarray] | None:
    scl = read_warped(item.assets["SCL"].href, grid, "nearest")
    covered = np.isfinite(scl)
    clear = covered & ~np.isin(scl, config.SCL_CLOUD_CLASSES) & (scl != 11)
    # Clear fraction of the part of the AOI this tile covers, so a cloud-free
    # tile that clips only a corner of the AOI is still used.
    if not covered.any() or clear.sum() < config.MIN_SCENE_CLEAR_FRACTION * covered.sum():
        return None
    out = {}
    for name, key in S2_BANDS.items():
        arr = read_warped(item.assets[key].href, grid, "bilinear")
        arr[(arr <= 0) | ~clear] = np.nan
        out[name] = _harmonize(arr, item)
    return out


def _read_s1_scene(item: pystac.Item, grid: Grid) -> dict[str, np.ndarray] | None:
    try:
        return _read_s1_scene_unsafe(item, grid)
    except Exception as exc:
        print(f"    skipped {item.id}: {type(exc).__name__}", file=sys.stderr)
        return _FAILED


def _read_s1_scene_unsafe(item: pystac.Item, grid: Grid) -> dict[str, np.ndarray]:
    out = {}
    for pol in ("vv", "vh"):
        lin = read_warped(item.assets[pol].href, grid, "bilinear")
        db = np.full(lin.shape, np.nan, dtype="float32")
        pos = np.isfinite(lin) & (lin > 0)
        db[pos] = 10.0 * np.log10(lin[pos])
        out[pol] = db
    return out


def _nanmedian_stack(arrs: list[np.ndarray]) -> np.ndarray:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.stack(arrs), axis=0).astype("float32")


def search_window(collection: str, bbox: BBox, start: str, end: str, query=None):
    return stac.search_range(collection, bbox, start, end, query=query)


def optical_composite(
    bbox: BBox, grid: Grid, start: str, end: str, *, max_scenes: int | None = None
) -> tuple[dict[str, np.ndarray], list[str], dict]:
    """Cloud-masked per-band median of Sentinel-2 L2A scenes in [start, end]."""
    max_scenes = max_scenes or config.COMPOSITE_MAX_S2_SCENES
    items = search_window(
        config.S2_COLLECTION,
        bbox,
        start,
        end,
        query={"eo:cloud_cover": {"lt": config.MAX_SCENE_CLOUD_PCT}},
    )
    # Clearest scenes first, capped per MGRS tile so an AOI that straddles two
    # tiles gets observations on both sides.
    n_candidates = len(items)
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    per_tile: dict[str, int] = {}
    capped = []
    for it in items:
        tile = str(it.properties.get("s2:mgrs_tile", ""))
        if per_tile.get(tile, 0) < max_scenes:
            per_tile[tile] = per_tile.get(tile, 0) + 1
            capped.append(it)
    items = capped
    if not items:
        return {}, [], {"s2_candidates": 0}

    with ThreadPoolExecutor(max_workers=config.READ_WORKERS) as pool:
        reads = list(pool.map(lambda it: _read_s2_scene(it, grid), items))
    n_failed = sum(r is _FAILED for r in reads)
    used = [(it, r) for it, r in zip(items, reads) if r is not None and r is not _FAILED]
    if not used:
        return {}, [], {"s2_candidates": n_candidates, "s2_used": 0, "s2_failed": n_failed}

    bands = {
        name: _nanmedian_stack([r[name] for _, r in used]) for name in S2_BANDS
    }
    bands["n_clear"] = np.sum(
        [np.isfinite(r["nir"]) for _, r in used], axis=0
    ).astype("float32")
    meta = {"s2_candidates": n_candidates, "s2_used": len(used), "s2_failed": n_failed}
    return bands, [it.id for it, _ in used], meta


def dominant_orbit_state(bbox: BBox, start: str, end: str) -> str | None:
    """Most common Sentinel-1 orbit direction over a period (for consistency)."""
    items = search_window(config.S1_COLLECTION, bbox, start, end)
    items = stac.covering_items(items, bbox, min_coverage=0.5)
    counts = Counter(it.properties.get("sat:orbit_state") for it in items)
    counts.pop(None, None)
    return counts.most_common(1)[0][0] if counts else None


def sar_composite(
    bbox: BBox,
    grid: Grid,
    start: str,
    end: str,
    *,
    orbit_state: str | None = None,
    max_scenes: int | None = None,
) -> tuple[dict[str, np.ndarray], list[str], dict]:
    """Median and temporal std of Sentinel-1 RTC backscatter (dB) in [start, end]."""
    max_scenes = max_scenes or config.COMPOSITE_MAX_S1_SCENES
    items = search_window(config.S1_COLLECTION, bbox, start, end)
    items = stac.covering_items(items, bbox, min_coverage=0.5)
    if orbit_state:
        same = [it for it in items if it.properties.get("sat:orbit_state") == orbit_state]
        items = same or items
    # Spread the sample over the window rather than taking the first N.
    items.sort(key=lambda it: it.datetime)
    if len(items) > max_scenes:
        idx = np.linspace(0, len(items) - 1, max_scenes).round().astype(int)
        items = [items[i] for i in sorted(set(idx))]
    n_candidates = len(items)
    if len(items) < config.COMPOSITE_MIN_S1_SCENES:
        return {}, [], {"s1_candidates": n_candidates}

    with ThreadPoolExecutor(max_workers=config.READ_WORKERS) as pool:
        reads = list(pool.map(lambda it: _read_s1_scene(it, grid), items))
    n_failed = sum(r is _FAILED for r in reads)
    pairs = [(it, r) for it, r in zip(items, reads) if r is not None and r is not _FAILED]
    if len(pairs) < config.COMPOSITE_MIN_S1_SCENES:
        return {}, [], {"s1_candidates": n_candidates, "s1_failed": n_failed}
    items, reads = [p[0] for p in pairs], [p[1] for p in pairs]

    import warnings

    vv = np.stack([r["vv"] for r in reads])
    vh = np.stack([r["vh"] for r in reads])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bands = {
            "vv_med": np.nanmedian(vv, axis=0).astype("float32"),
            "vh_med": np.nanmedian(vh, axis=0).astype("float32"),
            "vv_std": np.nanstd(vv, axis=0).astype("float32"),
            "vh_std": np.nanstd(vh, axis=0).astype("float32"),
        }
    meta = {
        "s1_candidates": n_candidates,
        "s1_failed": n_failed,
        "s1_used": len(items),
        "s1_orbit_state": orbit_state,
    }
    return bands, [it.id for it in items], meta


def build_composite(
    bbox: BBox,
    grid: Grid,
    window: tuple[str, str],
    *,
    orbit_state: str | None = None,
    use_sar: bool = True,
) -> Composite:
    """Optical + SAR composite for one window. Either side may be empty."""
    start, end = window
    if date.fromisoformat(start) >= date.fromisoformat(end):
        raise ValueError(f"Window start {start} must be before end {end}.")
    opt, s2_ids, opt_meta = optical_composite(bbox, grid, start, end)
    sar, s1_ids, sar_meta = (
        sar_composite(bbox, grid, start, end, orbit_state=orbit_state)
        if use_sar
        else ({}, [], {})
    )
    return Composite(
        window=window,
        grid=grid,
        bands={**opt, **sar},
        scenes={"sentinel-2-l2a": s2_ids, "sentinel-1-rtc": s1_ids},
        meta={**opt_meta, **sar_meta},
    )
