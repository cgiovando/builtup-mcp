"""Tests for BBox / date validation and area estimation (Phase 1)."""

from __future__ import annotations

import math

import pytest

from builtup_mcp import config
from builtup_mcp.geo import (
    BBoxValidationError,
    bbox_area_km2,
    validate_bbox,
    validate_change_dates,
    validate_date,
)

# A ~0.5 x 0.5 km box near San Francisco (well under the 1 sq km cap).
SMALL_BBOX = [-122.4200, 37.7740, -122.4143, 37.7785]


def test_area_estimate_reasonable():
    area = bbox_area_km2(validate_bbox(SMALL_BBOX))
    assert 0.2 < area < 0.4  # roughly 0.5km x 0.5km


def test_valid_bbox_passes():
    box = validate_bbox(SMALL_BBOX)
    assert box.as_tuple() == tuple(SMALL_BBOX)
    assert box.center == pytest.approx(
        ((SMALL_BBOX[0] + SMALL_BBOX[2]) / 2, (SMALL_BBOX[1] + SMALL_BBOX[3]) / 2)
    )


def test_rejects_wrong_length():
    with pytest.raises(BBoxValidationError, match="exactly 4"):
        validate_bbox([-122.42, 37.77, -122.41])


def test_rejects_out_of_range_lon():
    with pytest.raises(BBoxValidationError, match="Longitudes"):
        validate_bbox([-181.0, 37.77, -122.41, 37.78])


def test_rejects_out_of_range_lat():
    with pytest.raises(BBoxValidationError, match="Latitudes"):
        validate_bbox([-122.42, -91.0, -122.41, 37.78])


def test_rejects_reversed_lon():
    with pytest.raises(BBoxValidationError, match="min_lon"):
        validate_bbox([-122.41, 37.77, -122.42, 37.78])


def test_rejects_reversed_lat():
    with pytest.raises(BBoxValidationError, match="min_lat"):
        validate_bbox([-122.42, 37.78, -122.41, 37.77])


def test_rejects_too_large_area():
    # ~0.1 degree square near the equator is ~123 sq km, far over the cap.
    with pytest.raises(BBoxValidationError, match="exceeds the prototype maximum"):
        validate_bbox([0.0, 0.0, 0.1, 0.1])


def test_demo_aoi_within_cap():
    # The default demo AOI must validate under the configured area cap.
    box = validate_bbox(list(config.DEMO_BBOX))
    assert bbox_area_km2(box) <= config.MAX_BBOX_AREA_KM2


def test_validate_date_ok():
    assert validate_date("2024-06-15") == "2024-06-15"


def test_validate_date_bad():
    with pytest.raises(BBoxValidationError, match="ISO"):
        validate_date("06/15/2024")


def test_change_dates_single():
    target, baseline = validate_change_dates("2024-06-15")
    assert target == "2024-06-15"
    assert baseline is None


def test_change_dates_pair():
    target, baseline = validate_change_dates("2024-06-15", "2023-06-15")
    assert (target, baseline) == ("2024-06-15", "2023-06-15")


def test_change_dates_baseline_must_precede_target():
    with pytest.raises(BBoxValidationError, match="strictly before"):
        validate_change_dates("2023-06-15", "2024-06-15")


def test_change_dates_equal_rejected():
    with pytest.raises(BBoxValidationError, match="strictly before"):
        validate_change_dates("2024-06-15", "2024-06-15")
