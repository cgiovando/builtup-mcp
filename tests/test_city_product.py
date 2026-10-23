"""Offline tests for the v0.2 city-request path (no network)."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from builtup_mcp.geo import BBoxValidationError
from builtup_mcp.processing.timeseries import NODATA, epoch_change, states
from builtup_mcp.request import build_request, seasonal_windows


def test_seasonal_windows_wrap_year_end():
    windows, labels = seasonal_windows(2020, 2022, [12, 1, 2, 3], today=date(2026, 1, 1))
    assert windows[0] == ("2019-12-01", "2020-03-31")
    assert windows[-1] == ("2021-12-01", "2022-03-31")
    assert labels == ["2020 season", "2021 season", "2022 season"]


def test_seasonal_windows_truncate_at_today():
    windows, _ = seasonal_windows(2025, 2026, [6, 7, 8], today=date(2026, 7, 15))
    assert windows[-1] == ("2026-06-01", "2026-07-15")


def test_build_request_adds_latest_window():
    req = build_request(
        {
            "city": "Test",
            "aoi": {"bbox": [31.64, 9.49, 31.66, 9.51]},
            "period": {"start_year": 2024, "end_year": 2025, "season_months": [1, 2, 3]},
            "latest_window_days": 60,
        },
        today=date(2026, 10, 1),
    )
    assert len(req.windows) == 3
    assert req.window_labels[-1].startswith("latest")
    assert req.windows[-1] == ("2026-08-02", "2026-10-01")
    json.dumps(req.to_dict())  # provenance must serialize


def test_build_request_rejects_huge_aoi():
    with pytest.raises(BBoxValidationError):
        build_request({"aoi": {"bbox": [30.0, 0.0, 31.0, 1.0]}, "period": {"start_year": 2020, "end_year": 2021}})


def _series(values):
    """(T, 1, 1) -> tile to a 6x6 block so cluster filters keep it."""
    arr = np.array(values, dtype="float32")[:, None, None]
    return np.tile(arr, (1, 6, 6))


def test_states_dead_band():
    s = states(np.array([0.1, 0.5, 0.9, np.nan], dtype="float32"))
    assert s.tolist() == [0, -1, 1, -1]


def test_gain_dated_to_first_built_after_last_not_built():
    probs = _series([0.1, 0.2, 0.1, 0.9, 0.95, 0.9])
    ec = epoch_change(probs, min_neighbors=0, min_cluster_px=1)
    assert np.all(ec.gain_epoch == 3)
    assert not ec.provisional.any()
    assert ec.confidence.min() > 0.5


def test_flicker_then_built_uses_last_transition():
    probs = _series([0.1, 0.9, 0.1, 0.5, 0.9, 0.9])
    ec = epoch_change(probs, min_neighbors=0, min_cluster_px=1)
    assert np.all(ec.gain_epoch == 4)


def test_established_builtup_with_one_bad_window_is_not_gain():
    probs = _series([0.9, 0.9, 0.9, 0.1, 0.9, 0.9])
    ec = epoch_change(probs, min_neighbors=0, min_cluster_px=1)
    assert np.all(ec.gain_epoch == 0)
    assert ec.built_first.all()


def test_always_built_is_not_gain():
    probs = _series([0.9, 0.95, 0.9, 0.9])
    ec = epoch_change(probs, min_neighbors=0, min_cluster_px=1)
    assert np.all(ec.gain_epoch == 0)
    assert ec.built_first.all()


def test_gain_in_last_window_is_provisional():
    probs = _series([0.1, 0.1, 0.9])
    ec = epoch_change(probs, min_neighbors=0, min_cluster_px=1)
    assert np.all(ec.gain_epoch == 2)
    assert ec.provisional.all()


def test_unknown_windows_do_not_break_dating():
    probs = _series([0.1, np.nan, 0.5, 0.9, np.nan])
    ec = epoch_change(probs, min_neighbors=0, min_cluster_px=1)
    assert np.all(ec.gain_epoch == 3)


def test_loss_only_when_requested():
    probs = _series([0.9, 0.9, 0.1, 0.1])
    off = epoch_change(probs, min_neighbors=0, min_cluster_px=1)
    on = epoch_change(probs, detect_loss=True, min_neighbors=0, min_cluster_px=1)
    assert np.all(off.loss_epoch == 0)
    assert np.all(on.loss_epoch == 2)


def test_aoi_mask_sets_nodata():
    probs = _series([0.1, 0.9, 0.9])
    aoi = np.ones((6, 6), dtype=bool)
    aoi[0, 0] = False
    ec = epoch_change(probs, aoi, min_neighbors=0, min_cluster_px=1)
    assert ec.gain_epoch[0, 0] == NODATA
    assert ec.gain_epoch[3, 3] == 1


def test_linear_gain_removed_but_blob_kept():
    probs = np.full((3, 30, 30), 0.1, dtype="float32")
    probs[1:, 2, :] = 0.9  # a one-pixel-wide "road"
    probs[1:, 10:16, 10:16] = 0.9  # a 6x6 block
    ec = epoch_change(probs, min_neighbors=0, min_cluster_px=1)
    assert (ec.gain_epoch[2] == 0).all()
    assert (ec.gain_epoch[10:16, 10:16] == 1).all()


def test_isolated_gain_pixel_removed_by_cluster_filter():
    probs = np.full((3, 20, 20), 0.1, dtype="float32")
    probs[1:, 5, 5] = 0.9
    ec = epoch_change(probs)
    assert (ec.gain_epoch > 0).sum() == 0


def test_weak_labels_and_fraction():
    import geopandas as gpd
    from shapely.geometry import box

    from builtup_mcp.processing.classify import building_fraction, weak_labels
    from builtup_mcp.sources.raster_io import Grid

    grid = Grid(32636, (0.0, 0.0, 200.0, 200.0), 20, 20, 10.0)
    fp = gpd.GeoDataFrame(geometry=[box(0, 180, 20, 200)], crs=32636)
    frac = building_fraction(fp, grid)
    assert frac[0, 0] == pytest.approx(1.0)
    assert frac[10, 10] == 0
    labels = weak_labels(frac, 10.0)
    assert labels[0, 0] == 1
    assert labels[19, 19] == 0  # far from any footprint
    assert labels[0, 3] == -1  # within the negative buffer


def test_classifier_learns_separable_signal():
    from builtup_mcp.processing.classify import predict_probability, train_models
    from builtup_mcp.processing.features import OPTICAL_FEATURES
    from builtup_mcp.sources.composite import Composite
    from builtup_mcp.sources.raster_io import Grid

    rng = np.random.default_rng(1)
    h = w = 120
    grid = Grid(32636, (0.0, 0.0, 1200.0, 1200.0), w, h, 10.0)
    built = np.zeros((h, w), dtype=bool)
    built[:, :60] = True
    def comp(window, built_mask):
        base = lambda lo, hi: np.where(built_mask, hi, lo) + rng.normal(0, 80, (h, w))
        bands = {
            "blue": base(500, 1100), "green": base(700, 1300), "red": base(600, 1600),
            "nir": base(3200, 2300), "swir1": base(1800, 3000), "swir2": base(1000, 2600),
        }
        return Composite(window, grid, {k: v.astype("float32") for k, v in bands.items()})
    c0 = comp(("2020-01-01", "2020-03-31"), built)
    labels = np.where(built, 1, 0).astype("int8")
    models = train_models([c0], labels, "2020-02-01")
    assert "optical" in models.models and "fused" not in models.models
    assert models.metrics["optical"]["f1"] > 0.9
    later = built.copy()
    later[:, 60:80] = True
    p, src = predict_probability(models, comp(("2024-01-01", "2024-03-31"), later))
    assert p[:, 65:75].mean() > 0.8 and p[:, 100:].mean() < 0.2
    assert set(np.unique(src)) == {2}
    assert len(OPTICAL_FEATURES) == 8


def test_season_months_sorted_wrap_is_reordered():
    a, _ = seasonal_windows(2021, 2021, [1, 2, 3, 12], today=date(2026, 1, 1))
    b, _ = seasonal_windows(2021, 2021, [12, 1, 2, 3], today=date(2026, 1, 1))
    assert a == b == [("2020-12-01", "2021-03-31")]


def test_single_month_season_kept():
    windows, _ = seasonal_windows(2020, 2021, [6], today=date(2026, 1, 1))
    assert windows == [("2020-06-01", "2020-06-30"), ("2021-06-01", "2021-06-30")]


def test_non_contiguous_months_rejected():
    with pytest.raises(BBoxValidationError):
        seasonal_windows(2020, 2021, [1, 3], today=date(2026, 1, 1))


def test_current_season_clipped_and_labelled_partial():
    windows, labels = seasonal_windows(2025, 2026, [12, 1, 2, 3], today=date(2026, 1, 20))
    assert windows[-1] == ("2025-12-01", "2026-01-20")
    assert "partial" in labels[-1] and "partial" not in labels[0]
