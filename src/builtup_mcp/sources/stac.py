"""Planetary Computer STAC access helpers (Phase 3)."""

from __future__ import annotations

from datetime import date, timedelta

import planetary_computer
import pystac
import pystac_client
from shapely.geometry import box as shp_box
from shapely.geometry import shape

from .. import config
from ..geo import BBox


def open_catalog() -> pystac_client.Client:
    """Open the Planetary Computer STAC catalog with automatic asset signing."""
    return pystac_client.Client.open(
        config.PC_STAC_URL, modifier=planetary_computer.sign_inplace
    )


def search_items(
    collection: str,
    bbox: BBox,
    target_date: str,
    *,
    window_days: int = config.SEARCH_WINDOW_DAYS,
    query: dict | None = None,
) -> list[pystac.Item]:
    """Search a collection for items intersecting ``bbox`` up to ``target_date``.

    Looks back ``window_days`` from the target date (and a couple of days forward
    to tolerate revisit timing). Implements the "go back in time for a usable
    scene" fallback by widening the temporal window rather than failing.
    """
    end = date.fromisoformat(target_date) + timedelta(days=2)
    start = date.fromisoformat(target_date) - timedelta(days=window_days)
    catalog = open_catalog()
    search = catalog.search(
        collections=[collection],
        bbox=bbox.as_tuple(),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query=query or {},
    )
    return list(search.items())


def _epsg_from_props(props: dict) -> int | None:
    """Extract an EPSG code from a STAC proj-extension property mapping.

    Handles both the legacy ``proj:epsg`` (int) and the current ``proj:code``
    (string like ``"EPSG:32737"``).
    """
    if props.get("proj:epsg") is not None:
        return int(props["proj:epsg"])
    code = props.get("proj:code")
    if isinstance(code, str) and code.upper().startswith("EPSG:"):
        return int(code.split(":", 1)[1])
    return None


def aoi_coverage(item: pystac.Item, bbox: BBox) -> float:
    """Fraction of the AOI bbox covered by the item's footprint (0..1).

    Scenes whose swath edge clips the AOI return < 1.0; used to avoid partial-
    coverage acquisitions that would leave nodata in the analysis window.
    """
    if item.geometry is None:
        return 0.0
    aoi = shp_box(*bbox.as_tuple())
    if aoi.area == 0:
        return 0.0
    return shape(item.geometry).intersection(aoi).area / aoi.area


def covering_items(
    items: list[pystac.Item], bbox: BBox, *, min_coverage: float = 0.999
) -> list[pystac.Item]:
    """Keep only items that (almost) fully cover the AOI; fall back to all."""
    covering = [it for it in items if aoi_coverage(it, bbox) >= min_coverage]
    return covering or items


def item_epsg(item: pystac.Item) -> int:
    """Return the item's native projected EPSG code.

    Checks item-level properties first, then per-asset proj fields (the proj
    extension increasingly lives on assets), since all bands of a Sentinel item
    share one UTM zone.
    """
    epsg = _epsg_from_props(item.properties)
    if epsg is not None:
        return epsg
    for asset in item.assets.values():
        epsg = _epsg_from_props(asset.extra_fields)
        if epsg is not None:
            return epsg
    raise ValueError(f"Item {item.id} exposes no proj:epsg/proj:code.")
