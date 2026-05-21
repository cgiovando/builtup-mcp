"""Geospatial helpers: BBox validation and area estimation (WGS84).

Phase 1 only depends on the standard library here so the server can be
imported and validated without the heavy geospatial stack installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import config

# Mean Earth radius based scale factors (km per degree).
_KM_PER_DEG_LAT: float = 110.574  # meridional, ~constant
_KM_PER_DEG_LON_EQ: float = 111.320  # at the equator; scaled by cos(lat)


class BBoxValidationError(ValueError):
    """Raised when an input BBox fails validation."""


@dataclass(frozen=True)
class BBox:
    """An axis-aligned WGS84 bounding box.

    Coordinates are ordered (min_lon, min_lat, max_lon, max_lat), matching the
    GeoJSON / STAC convention.
    """

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.min_lon + self.max_lon) / 2.0,
            (self.min_lat + self.max_lat) / 2.0,
        )


def parse_bbox(bbox: list[float] | tuple[float, ...]) -> BBox:
    """Coerce a 4-element sequence into a :class:`BBox`.

    Raises:
        BBoxValidationError: if the sequence is not exactly four numbers.
    """
    if len(bbox) != 4:
        raise BBoxValidationError(
            f"bbox must have exactly 4 values (min_lon, min_lat, max_lon, max_lat), "
            f"got {len(bbox)}."
        )
    try:
        values = [float(v) for v in bbox]
    except (TypeError, ValueError) as exc:
        raise BBoxValidationError(f"bbox values must be numbers: {exc}") from exc
    return BBox(*values)


def bbox_area_km2(bbox: BBox) -> float:
    """Approximate the area of a small WGS84 BBox in square kilometers.

    Uses an equirectangular approximation (longitude degrees scaled by the
    cosine of the mean latitude). This is accurate to well within a percent for
    the sub-kilometer boxes this prototype handles.
    """
    mean_lat_rad = math.radians((bbox.min_lat + bbox.max_lat) / 2.0)
    width_km = (bbox.max_lon - bbox.min_lon) * _KM_PER_DEG_LON_EQ * math.cos(mean_lat_rad)
    height_km = (bbox.max_lat - bbox.min_lat) * _KM_PER_DEG_LAT
    return width_km * height_km


def validate_bbox(bbox: list[float] | tuple[float, ...]) -> BBox:
    """Validate a BBox for WGS84 bounds, ordering, and maximum area.

    Returns:
        A validated :class:`BBox`.

    Raises:
        BBoxValidationError: on any validation failure, with a descriptive message.
    """
    box = parse_bbox(bbox)

    # WGS84 coordinate ranges.
    if not (config.MIN_LON <= box.min_lon <= config.MAX_LON) or not (
        config.MIN_LON <= box.max_lon <= config.MAX_LON
    ):
        raise BBoxValidationError(
            f"Longitudes must be within [{config.MIN_LON}, {config.MAX_LON}] (WGS84); "
            f"got min_lon={box.min_lon}, max_lon={box.max_lon}."
        )
    if not (config.MIN_LAT <= box.min_lat <= config.MAX_LAT) or not (
        config.MIN_LAT <= box.max_lat <= config.MAX_LAT
    ):
        raise BBoxValidationError(
            f"Latitudes must be within [{config.MIN_LAT}, {config.MAX_LAT}] (WGS84); "
            f"got min_lat={box.min_lat}, max_lat={box.max_lat}."
        )

    # Ordering: min must be strictly less than max (non-degenerate box).
    if box.min_lon >= box.max_lon:
        raise BBoxValidationError(
            f"min_lon ({box.min_lon}) must be less than max_lon ({box.max_lon})."
        )
    if box.min_lat >= box.max_lat:
        raise BBoxValidationError(
            f"min_lat ({box.min_lat}) must be less than max_lat ({box.max_lat})."
        )

    # Maximum area guard.
    area = bbox_area_km2(box)
    if area > config.MAX_BBOX_AREA_KM2:
        raise BBoxValidationError(
            f"BBox area is {area:.4f} sq km, which exceeds the prototype maximum of "
            f"{config.MAX_BBOX_AREA_KM2} sq km. Please request a smaller area."
        )

    return box


def utm_epsg(bbox: BBox) -> int:
    """Return the EPSG code of the UTM zone containing the BBox center.

    326xx for the northern hemisphere, 327xx for the southern.
    """
    lon, lat = bbox.center
    if not (-80.0 <= lat <= 84.0):
        raise BBoxValidationError(
            f"UTM is undefined outside [-80, 84] latitude; got center lat {lat}."
        )
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)  # guard the lon == 180 edge (would give 61)
    return (32600 if lat >= 0 else 32700) + zone


def validate_date(date: str) -> str:
    """Validate that ``date`` is an ISO ``YYYY-MM-DD`` string.

    Returns the normalized date string.

    Raises:
        BBoxValidationError: if the date is malformed.
    """
    from datetime import date as _date

    try:
        parsed = _date.fromisoformat(date)
    except (TypeError, ValueError) as exc:
        raise BBoxValidationError(
            f"date must be an ISO 'YYYY-MM-DD' string; got {date!r} ({exc})."
        ) from exc
    return parsed.isoformat()


def validate_change_dates(
    target_date: str, baseline_date: str | None = None
) -> tuple[str, str | None]:
    """Validate the target and optional baseline acquisition dates.

    Implements the option-C baseline model: a ``baseline_date`` is optional. When
    supplied it defines a pre/post pair and must fall strictly before
    ``target_date``; when omitted the caller compares against a static reference
    baseline dataset instead.

    Returns:
        ``(target, baseline_or_none)`` as normalized ISO date strings.

    Raises:
        BBoxValidationError: on a malformed date or a baseline not before target.
    """
    target = validate_date(target_date)
    if baseline_date is None:
        return target, None
    baseline = validate_date(baseline_date)
    if baseline >= target:
        raise BBoxValidationError(
            f"baseline_date ({baseline}) must be strictly before target_date ({target})."
        )
    return target, baseline
