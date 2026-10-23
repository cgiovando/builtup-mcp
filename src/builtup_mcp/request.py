"""City change requests (v0.2).

A request names a place, an AOI and a period, and asks for a built-up change
product. It is the unit a requester (or an agent acting for one) submits, and
it is echoed verbatim into the product's provenance so any run can be repeated.

Example (JSON)::

    {
      "request_id": "malakal-builtup-change",
      "city": "Malakal", "country": "South Sudan", "iso3": "SSD",
      "aoi": {"bbox": [31.643, 9.492, 31.695, 9.578]},
      "period": {"start_year": 2019, "end_year": 2026, "season_months": [12, 1, 2, 3]},
      "latest_window_days": 90,
      "options": {"detect_loss": false}
    }

``season_months`` fixes the calendar months compared each year (here the dry
season), so phenology and soil moisture do not masquerade as change. A month
list that wraps the year end (12, 1, 2, 3) is anchored on the later year.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from shapely.geometry import box as shp_box
from shapely.geometry import mapping, shape

from . import config
from .geo import BBox, BBoxValidationError, bbox_area_km2, parse_bbox


@dataclass
class ChangeRequest:
    request_id: str
    city: str
    aoi_geometry: dict  # GeoJSON geometry, WGS84
    windows: list[tuple[str, str]]
    country: str = ""
    iso3: str = ""
    window_labels: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def bbox(self) -> BBox:
        return parse_bbox(list(shape(self.aoi_geometry).bounds))

    @property
    def area_km2(self) -> float:
        return bbox_area_km2(self.bbox)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["windows"] = [list(w) for w in self.windows]
        return d


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _contiguous_run(months: list[int]) -> list[int]:
    """Order calendar months as one contiguous run, wrapping the year end.

    [12, 1, 2, 3] and [1, 2, 3, 12] both give [12, 1, 2, 3]. Non-contiguous
    lists (for example [1, 3]) are rejected rather than silently widened.
    """
    ms = sorted({int(m) for m in months})
    if not ms or any(m < 1 or m > 12 for m in ms):
        raise BBoxValidationError(f"season_months must be calendar months 1-12; got {months}.")
    if len(ms) == 12:
        return ms
    for start in ms:
        run = [((start - 1 + i) % 12) + 1 for i in range(len(ms))]
        if sorted(run) == ms:
            return run
    raise BBoxValidationError(f"season_months must be contiguous (wrapping allowed); got {months}.")


def seasonal_windows(
    start_year: int, end_year: int, months: list[int], today: date | None = None
) -> tuple[list[tuple[str, str]], list[str]]:
    """One window per year covering ``months`` (wrapping the year end if needed).

    A window that would end after ``today`` is clipped and labelled partial.
    """
    today = today or date.today()
    run = _contiguous_run(months)
    first_m, last_m = run[0], run[-1]
    wraps = first_m > last_m
    windows, labels = [], []
    for year in range(start_year, end_year + 1):
        start = date(year - 1 if wraps else year, first_m, 1)
        nominal_end = date(year + (last_m == 12), (last_m % 12) + 1, 1) - timedelta(days=1)
        end = min(nominal_end, today)
        if start >= today or (end - start).days + 1 < 28:
            continue
        windows.append((start.isoformat(), end.isoformat()))
        labels.append(f"{year} season" + (f" (partial to {end.isoformat()})" if end < nominal_end else ""))
    return windows, labels


def build_request(spec: dict, today: date | None = None) -> ChangeRequest:
    """Validate a request spec and expand its period into composite windows."""
    today = today or date.today()
    city = spec.get("city") or "AOI"
    aoi = spec.get("aoi") or {}
    if "bbox" in aoi:
        b = parse_bbox(aoi["bbox"])
        geom = mapping(shp_box(*b.as_tuple()))
    elif "geometry" in aoi:
        geom = aoi["geometry"]
    elif "path" in aoi:
        import geopandas as gpd

        gdf = gpd.read_file(aoi["path"]).to_crs(4326)
        geom = mapping(gdf.union_all())
    else:
        raise BBoxValidationError("Request needs aoi.bbox, aoi.geometry or aoi.path.")

    req = ChangeRequest(
        request_id=spec.get("request_id") or f"{_slug(city)}-builtup-change",
        city=city,
        country=spec.get("country", ""),
        iso3=spec.get("iso3", ""),
        aoi_geometry=json.loads(json.dumps(geom)),
        windows=[],
        options=spec.get("options", {}),
        raw=spec,
    )
    if req.area_km2 > config.CITY_MAX_AREA_KM2:
        raise BBoxValidationError(
            f"AOI bbox is {req.area_km2:.1f} sq km; the city-request cap is "
            f"{config.CITY_MAX_AREA_KM2} sq km (BUILTUP_MCP_CITY_MAX_KM2)."
        )

    if "windows" in spec:
        windows = [tuple(w) for w in spec["windows"]]
        labels = spec.get("window_labels") or [f"{a}..{b}" for a, b in windows]
    else:
        p = spec.get("period", {})
        windows, labels = seasonal_windows(
            int(p.get("start_year", today.year - 5)),
            int(p.get("end_year", today.year)),
            p.get("season_months", [1, 2, 3, 4]),
            today,
        )
    latest = spec.get("latest_window_days")
    if latest:
        start = (today - timedelta(days=int(latest))).isoformat()
        if not windows or start > windows[-1][1]:
            windows.append((start, today.isoformat()))
            labels.append(f"latest ({today.isoformat()})")
    if len(windows) < 2:
        raise BBoxValidationError("A change request needs at least two windows.")
    req.windows = windows
    req.window_labels = labels
    return req


def load_request(path: str | Path) -> ChangeRequest:
    return build_request(json.loads(Path(path).read_text(encoding="utf-8")))
