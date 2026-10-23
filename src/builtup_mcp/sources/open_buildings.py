"""Google Open Buildings 2.5D Temporal reader (validation reference source).

Reads the annual ``building_presence`` layer from the public, anonymous GCS
bucket (no Earth Engine auth) and resamples it onto the analysis grid. Used to
validate Sentinel-derived built-up gains: Open Buildings provides yearly layers
(2016-2023) at ~4m effective resolution over the Global South. License: CC-BY-4.0 / ODbL.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

import numpy as np

from .raster_io import Grid, read_warped

_BUCKET = "open-buildings-temporal-data"
_MANIFEST = (
    f"https://storage.googleapis.com/{_BUCKET}/v1/manifests/"
    "{token}_EPSG_{epsg}_{year}_06_30.json"
)
_PRESENCE_BAND = 3  # band order: fractional_count, height, presence

_manifest_index: list[str] | None = None


def _list_manifests() -> list[str]:
    """List (and cache) all manifest object names in the bucket."""
    global _manifest_index
    if _manifest_index is not None:
        return _manifest_index
    names: list[str] = []
    page = None
    while True:
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{_BUCKET}/o"
            "?prefix=v1/manifests/&maxResults=1000"
        )
        if page:
            url += f"&pageToken={page}"
        with urllib.request.urlopen(url, timeout=60) as fh:
            resp = json.load(fh)
        names += [i["name"] for i in resp.get("items", [])]
        page = resp.get("nextPageToken")
        if not page:
            break
    _manifest_index = names
    return names


def _candidate_tokens(epsg: int, year: int) -> list[str]:
    """Tokens whose manifest matches this UTM zone and year (usually 1-3)."""
    pat = re.compile(rf"v1/manifests/(.+?)_EPSG_{epsg}_{year}_06_30\.json$")
    return [m.group(1) for n in _list_manifests() if (m := pat.search(n))]


def _https(uri: str) -> str:
    return uri.replace("gs://", "https://storage.googleapis.com/")


def _tile_bounds(aff: dict, dims: dict) -> tuple[float, float, float, float]:
    sx, sy = aff["scaleX"], aff["scaleY"]
    tx, ty = aff["translateX"], aff["translateY"]
    w, h = dims["width"], dims["height"]
    xs = (tx, tx + w * sx)
    ys = (ty, ty + h * sy)
    return (min(xs), min(ys), max(xs), max(ys))


def _overlap_area(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _best_tile(manifest: dict, grid: Grid):
    """Return (uriPrefix, source) for the tile with most overlap with the grid."""
    prefix = manifest["uriPrefix"]
    best, best_area = None, 0.0
    for src in manifest["tilesets"][0]["sources"]:
        area = _overlap_area(
            grid.bounds, _tile_bounds(src["affineTransform"], src["dimensions"])
        )
        if area > best_area:
            best, best_area = src, area
    return prefix, best, best_area


def _overlapping_tiles(grid: Grid, year: int) -> list[tuple[str, dict, float]]:
    """All (uriPrefix, source, overlap_area) tiles intersecting the grid.

    Open Buildings tiles are stored in the UTM zone of their S2 cell, which near
    a zone boundary is often the neighbouring zone of the analysis grid (Kigali,
    at 30.1 E, is mostly in EPSG:32735 tiles). The grid's zone and both
    neighbours are searched, and overlap is computed in each tile's own CRS.
    """
    from rasterio.warp import transform_bounds

    zone = grid.epsg % 100
    base = grid.epsg - zone
    epsgs = [base + z for z in (zone - 1, zone, zone + 1) if 1 <= z <= 60]
    out: list[tuple[str, dict, float]] = []
    for epsg in epsgs:
        bounds = transform_bounds(f"EPSG:{grid.epsg}", f"EPSG:{epsg}", *grid.bounds)
        for token in _candidate_tokens(epsg, year):
            url = _MANIFEST.format(token=token, epsg=epsg, year=year)
            with urllib.request.urlopen(url, timeout=120) as fh:
                manifest = json.load(fh)
            prefix = manifest["uriPrefix"]
            for src in manifest["tilesets"][0]["sources"]:
                area = _overlap_area(bounds, _tile_bounds(src["affineTransform"], src["dimensions"]))
                if area > 0:
                    out.append((prefix, src, area))
    if not out and not any(_candidate_tokens(e, year) for e in epsgs):
        raise ValueError(
            f"No Open Buildings manifest near EPSG:{grid.epsg} for {year} "
            "(AOI likely outside Open Buildings coverage)."
        )
    return out


def _read_presence_tile(prefix: str, src: dict, grid: Grid) -> np.ndarray:
    """Windowed read of one tile's presence band, with nodata cleaned to NaN.

    Open Buildings encodes missing data as a sentinel (e.g. -99); presence is
    defined on [0, 1], so anything outside that range is set to NaN.
    """
    href = _https(prefix + src["uris"][0])
    arr = read_warped(href, grid, "bilinear", band=_PRESENCE_BAND).astype("float32")
    arr[(arr < 0.0) | (arr > 1.0)] = np.nan
    return arr


def fetch_building_presence(grid: Grid, year: int, *, mosaic: bool = True) -> np.ndarray:
    """Read Open Buildings ``building_presence`` (0..1) for ``year`` onto ``grid``.

    Auto-discovers the Sentinel-2 cell token(s) for the grid's UTM zone and does a
    windowed read. With ``mosaic=True`` (default) all overlapping tiles are merged
    (NaN-aware), which both spans tile seams and skips per-tile nodata; with
    ``mosaic=False`` only the single best-overlap tile is read. EE manifest
    convention: ``uriPrefix`` concatenates DIRECTLY with each uri (no separator).
    """
    tiles = _overlapping_tiles(grid, year)
    if not tiles:
        raise ValueError(f"No Open Buildings tile overlaps the AOI for {year}.")

    if not mosaic:
        prefix, src, _ = max(tiles, key=lambda t: t[2])
        return _read_presence_tile(prefix, src, grid)

    out = np.full(grid.shape, np.nan, dtype="float32")
    for prefix, src, _ in tiles:
        out = np.fmax(out, _read_presence_tile(prefix, src, grid))
    return out
