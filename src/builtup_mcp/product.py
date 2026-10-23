"""Write a city change product as COGs, GeoParquet and a STAC item (v0.2).

Layout under ``PRODUCTS_DIR`` (a self-contained STAC catalog)::

    catalog.json
    builtup-change/collection.json
    builtup-change/<item-id>/<item-id>.json      STAC item
    builtup-change/<item-id>/builtup_probability.tif   COG, one band per window
    builtup-change/<item-id>/gain_epoch.tif            COG, window index of first built-up
    builtup-change/<item-id>/gain_polygons.parquet     GeoParquet (+ .geojson)
    builtup-change/<item-id>/h3_stats.parquet          per-hexagon statistics (+ .geojson)
    builtup-change/<item-id>/footprints_age.parquet    footprints with first-detected window
    builtup-change/<item-id>/summary.json              provenance, accuracy, static comparison

Any STAC client (QGIS, pystac, rstac, a STAC browser, DDH) can read it, and
the folder can be synced to object storage unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pystac
import rasterio
from rasterio.features import shapes
from shapely.geometry import mapping, shape

from . import config
from .processing.align import grid_transform
from .sources.raster_io import Grid

COLLECTION_ID = "builtup-change"


def write_cog(path: Path, arrays: list[np.ndarray], grid: Grid, *, nodata, dtype: str, descriptions=None):
    profile = {
        "driver": "COG",
        "width": grid.width,
        "height": grid.height,
        "count": len(arrays),
        "dtype": dtype,
        "crs": f"EPSG:{grid.epsg}",
        "transform": grid_transform(grid),
        "nodata": nodata,
        "compress": "DEFLATE",
        "overview_resampling": "nearest",
    }
    with rasterio.open(path, "w", **profile) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr.astype(dtype), i)
            if descriptions:
                dst.set_band_description(i, descriptions[i - 1])


def gain_polygons(
    gain_epoch: np.ndarray, confidence: np.ndarray, grid: Grid, labels: list[str], last_k: int | None = None
) -> gpd.GeoDataFrame:
    """Polygonize gain pixels by epoch, with area and mean confidence."""
    mask = (gain_epoch > 0) & (gain_epoch != 255)
    rows = []
    for geom, val in shapes(gain_epoch.astype("uint8"), mask=mask, transform=grid_transform(grid)):
        rows.append({"gain_epoch": int(val), "geometry": shape(geom)})
    if not rows:
        return gpd.GeoDataFrame(
            {"gain_epoch": [], "gain_window": [], "area_ha": [], "confidence": []},
            geometry=[], crs=f"EPSG:{grid.epsg}",
        ).to_crs(4326)
    gdf = gpd.GeoDataFrame(rows, crs=f"EPSG:{grid.epsg}")
    gdf["gain_window"] = [labels[k] for k in gdf["gain_epoch"]]
    gdf["area_ha"] = (gdf.area / 1e4).round(3)
    # Mean confidence per polygon via rasterized ids.
    from rasterio.features import rasterize

    ids = rasterize(
        [(g, i + 1) for i, g in enumerate(gdf.geometry)],
        out_shape=grid.shape, transform=grid_transform(grid), dtype="int32",
    )
    sums = np.bincount(ids.ravel(), weights=confidence.ravel(), minlength=len(gdf) + 1)
    cnts = np.bincount(ids.ravel(), minlength=len(gdf) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gdf["confidence"] = np.round(sums[1:] / np.maximum(cnts[1:], 1), 3)
    last_k = len(labels) - 1 if last_k is None else last_k
    gdf["provisional"] = gdf["gain_epoch"] > last_k
    return gdf.to_crs(4326)


def _pixel_lonlat(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    left, _b, _r, top = grid.bounds
    xs = left + (np.arange(grid.width) + 0.5) * grid.resolution_m
    ys = top - (np.arange(grid.height) + 0.5) * grid.resolution_m
    X, Y = np.meshgrid(xs, ys)
    tr = Transformer.from_crs(f"EPSG:{grid.epsg}", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(X, Y)
    return lon, lat


def h3_stats(
    grid: Grid,
    aoi: np.ndarray,
    built_by_window: np.ndarray,  # (T, H, W) bool, latest state per window
    gain_epoch: np.ndarray,
    labels: list[str],
    resolution: int = config.H3_RESOLUTION,
    last_k: int | None = None,
) -> gpd.GeoDataFrame:
    """Built-up hectares per window and gain per epoch, per H3 cell."""
    import h3
    import pandas as pd
    from shapely.geometry import Polygon

    lon, lat = _pixel_lonlat(grid)
    sel = aoi
    cells = np.array(
        [h3.latlng_to_cell(la, lo, resolution) for la, lo in zip(lat[sel], lon[sel])]
    )
    px_ha = grid.resolution_m**2 / 1e4
    df = pd.DataFrame({"h3": cells})
    last_k = len(labels) - 1 if last_k is None else last_k
    first, last = labels[0], labels[last_k]
    df["built_first_ha"] = built_by_window[0][sel] * px_ha
    df["built_latest_ha"] = built_by_window[last_k][sel] * px_ha
    ge = gain_epoch[sel]
    for k in range(1, len(labels)):
        df[f"gain_ha_e{k}"] = (ge == k) * px_ha
    df["gain_ha_total"] = ((ge > 0) & (ge <= last_k)) * px_ha
    df["gain_ha_provisional"] = ((ge > last_k) & (ge != 255)) * px_ha
    df["area_ha"] = px_ha
    agg = df.groupby("h3").sum().reset_index()
    agg["growth_pct"] = np.where(
        agg["built_first_ha"] > 0,
        100 * (agg["built_latest_ha"] - agg["built_first_ha"]) / agg["built_first_ha"],
        np.nan,
    ).round(1)
    agg = agg.round(3)
    agg.attrs["first_window"] = first
    agg.attrs["latest_window"] = last
    geoms = [Polygon([(lo, la) for la, lo in h3.cell_to_boundary(c)]) for c in agg["h3"]]
    return gpd.GeoDataFrame(agg, geometry=geoms, crs=4326)


def footprints_age(
    footprints: gpd.GeoDataFrame, grid: Grid, gain_epoch: np.ndarray, built_first: np.ndarray, labels: list[str]
) -> gpd.GeoDataFrame:
    """Attach the window in which each footprint's pixels first read as built-up.

    ``builtup_since`` is the first window label, ``before <first window>`` for
    built-up present from the start, or empty where the signal is inconclusive.
    """
    if footprints is None or footprints.empty:
        return footprints
    fp = footprints.to_crs(grid.epsg).copy()
    c = fp.geometry.representative_point()
    t = grid_transform(grid)
    cols = np.floor((c.x - t.c) / t.a).astype(int).to_numpy()
    rows = np.floor((c.y - t.f) / t.e).astype(int).to_numpy()
    ok = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    since = np.array([""] * len(fp), dtype=object)
    epoch = np.full(len(fp), -1)
    r, cc = rows[ok], cols[ok]
    ge = gain_epoch[r, cc]
    bf = built_first[r, cc]
    sub = np.array([""] * ok.sum(), dtype=object)
    sub[bf] = f"before {labels[0]}"
    is_gain = (ge > 0) & (ge != 255)
    sub[is_gain] = [labels[k] for k in ge[is_gain]]
    since[ok] = sub
    epoch[np.flatnonzero(ok)[is_gain]] = ge[is_gain]
    fp["builtup_since"] = since
    fp["gain_epoch"] = epoch
    return fp.to_crs(4326)


def _catalog(root: Path) -> tuple[pystac.Catalog, pystac.Collection]:
    root.mkdir(parents=True, exist_ok=True)
    cat_path = root / "catalog.json"
    if cat_path.exists():
        cat = pystac.Catalog.from_file(str(cat_path))
        coll = cat.get_child(COLLECTION_ID)
        if coll is not None:
            return cat, coll
    else:
        cat = pystac.Catalog(
            id="builtup-mcp-products",
            description="On-demand built-up change products generated by builtup-mcp.",
            title="builtup-mcp products",
        )
    coll = pystac.Collection(
        id=COLLECTION_ID,
        description=(
            "Built-up change products from Sentinel-2 and Sentinel-1 seasonal composites, "
            "classified per request with open building footprints as weak labels. "
            "Each item is one city request."
        ),
        extent=pystac.Extent(
            pystac.SpatialExtent([[-180, -90, 180, 90]]),
            pystac.TemporalExtent([[datetime(2015, 6, 23, tzinfo=timezone.utc), None]]),
        ),
        license="CC-BY-4.0",
        title="Built-up change (on demand)",
        providers=[
            pystac.Provider("Copernicus / ESA", roles=[pystac.ProviderRole.PRODUCER], url="https://dataspace.copernicus.eu"),
            pystac.Provider("Microsoft Planetary Computer", roles=[pystac.ProviderRole.HOST], url="https://planetarycomputer.microsoft.com"),
            pystac.Provider("Overture Maps Foundation (labels, ODbL)", roles=[pystac.ProviderRole.LICENSOR], url="https://overturemaps.org"),
            pystac.Provider("builtup-mcp", roles=[pystac.ProviderRole.PROCESSOR], url="https://github.com/cgiovando/builtup-mcp"),
        ],
    )
    cat.add_child(coll)
    return cat, coll


def publish_item(
    root: Path,
    item_id: str,
    geometry: dict,
    bbox: list[float],
    windows: list[tuple[str, str]],
    properties: dict,
    assets: dict[str, tuple[Path, str, list[str], str]],
) -> Path:
    """Add (or replace) an item in the local catalog; returns the item path."""
    cat, coll = _catalog(root)
    if coll.get_item(item_id) is not None:
        coll.remove_item(item_id)
    start = datetime.fromisoformat(windows[0][0]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(windows[-1][1]).replace(tzinfo=timezone.utc)
    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=None,
        start_datetime=start,
        end_datetime=end,
        properties=properties,
    )
    for key, (path, media, roles, title) in assets.items():
        item.add_asset(key, pystac.Asset(href=str(path), media_type=media, roles=roles, title=title))
    coll.add_item(item)
    # Collection extent = union of items.
    coll.update_extent_from_items()
    cat.normalize_hrefs(str(root))
    cat.make_all_asset_hrefs_relative()
    cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return Path(item.get_self_href())


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    return obj


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")


def aoi_mask(geometry: dict, grid: Grid) -> np.ndarray:
    from rasterio.features import geometry_mask
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{grid.epsg}", always_xy=True).transform
    g = shp_transform(tr, shape(geometry))
    return ~geometry_mask([mapping(g)], out_shape=grid.shape, transform=grid_transform(grid))


def corner_coords(grid: Grid) -> list[list[float]]:
    """WGS84 corners (TL, TR, BR, BL) of the grid, for web image overlays."""
    from pyproj import Transformer

    left, bottom, right, top = grid.bounds
    tr = Transformer.from_crs(f"EPSG:{grid.epsg}", "EPSG:4326", always_xy=True)
    return [list(tr.transform(x, y)) for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]


def write_quicklooks(out: Path, composites, aoi: np.ndarray) -> list[str]:
    """True-colour PNG per window, one shared stretch so years are comparable."""
    from PIL import Image

    rgbs = []
    for c in composites:
        if not c.has_optical:
            rgbs.append(None)
            continue
        rgbs.append(np.stack([c.bands["red"], c.bands["green"], c.bands["blue"]], axis=-1))
    stack = np.concatenate([r[aoi] for r in rgbs if r is not None]) if any(r is not None for r in rgbs) else None
    if stack is None:
        return []
    lo, hi = np.nanpercentile(stack, 2, axis=0), np.nanpercentile(stack, 98, axis=0)
    names = []
    for k, rgb in enumerate(rgbs):
        name = f"quicklook_{k:02d}.png"
        if rgb is None:
            names.append("")
            continue
        x = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8
        img = np.nan_to_num(x * 255).astype("uint8")
        alpha = np.where(np.all(np.isfinite(rgb), axis=-1), 255, 0).astype("uint8")
        Image.fromarray(np.dstack([img, alpha]), "RGBA").save(out / name, optimize=True)
        names.append(name)
    return names


def write_mask_png(path: Path, mask: np.ndarray, rgba: tuple[int, int, int, int]) -> None:
    from PIL import Image

    img = np.zeros(mask.shape + (4,), dtype="uint8")
    img[mask] = rgba
    Image.fromarray(img, "RGBA").save(path, optimize=True)
