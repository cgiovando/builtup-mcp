"""Unit tests for Phase 4 processing: indices, change classification, vectorize."""

from __future__ import annotations

import numpy as np

from builtup_mcp.processing import change as ch
from builtup_mcp.processing.indices import ndbi
from builtup_mcp.processing.vectorize import change_to_points
from builtup_mcp.sources.raster_io import Grid

# A tiny grid in UTM 37S so we can vectorize without network.
GRID = Grid(epsg=32737, bounds=(800000.0, 9870000.0, 800030.0, 9870030.0),
            width=3, height=3, resolution_m=10.0)


def test_ndbi_range_and_nan():
    nir = np.array([[0.2, 0.0]], dtype="float32")
    swir = np.array([[0.4, 0.0]], dtype="float32")
    out = ndbi(nir, swir)
    assert np.isclose(out[0, 0], (0.4 - 0.2) / (0.4 + 0.2))
    assert np.isnan(out[0, 1])  # 0/0 -> NaN


def test_pair_detects_gain_and_loss():
    valid = np.ones((1, 3), dtype=bool)
    # pixel0: both indices rise sharply -> gain; pixel1: flat -> none; pixel2: both fall -> loss.
    ndbi_b = np.array([[0.0, 0.0, 0.3]], dtype="float32")
    ndbi_t = np.array([[0.3, 0.0, 0.0]], dtype="float32")
    vv_b = np.array([[-15.0, -10.0, -5.0]], dtype="float32")
    vv_t = np.array([[-5.0, -10.0, -15.0]], dtype="float32")
    res = ch.detect_change_pair(
        ndbi_b, ndbi_t, vv_b, vv_t, valid,
        min_neighbors=0, detect_loss=True, min_gain_cluster_px=1,
    )
    assert res.change_code.tolist() == [[ch.GAIN, ch.NONE, ch.LOSS]]
    assert res.counts() == {"gain": 1, "loss": 1, "valid_pixels": 3}
    assert 0.0 <= res.confidence.max() <= 1.0


def test_pair_respects_validity_mask():
    valid = np.array([[False]], dtype=bool)
    res = ch.detect_change_pair(
        np.array([[0.0]], "float32"), np.array([[0.5]], "float32"),
        np.array([[-15.0]], "float32"), np.array([[-2.0]], "float32"), valid,
        min_neighbors=0, detect_loss=True, min_gain_cluster_px=1,
    )
    assert res.change_code.tolist() == [[ch.NONE]]


def test_reference_gain_outside_overture_loss_inside():
    valid = np.ones((1, 2), dtype=bool)
    ndbi_t = np.array([[0.4, -0.4]], dtype="float32")   # strong / weak built-up signal
    vv_t = np.array([[-2.0, -18.0]], dtype="float32")
    overture = np.array([[False, True]], dtype=bool)    # px0 unmapped, px1 mapped
    res = ch.detect_change_reference(
        ndbi_t, vv_t, overture, valid,
        min_neighbors=0, detect_loss=True, min_gain_cluster_px=1,
    )
    assert res.change_code[0, 0] == ch.GAIN   # built-up where nothing mapped
    assert res.change_code[0, 1] == ch.LOSS   # mapped but no signal now


def test_keep_large_clusters_drops_small():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:5, 0:5] = True  # 25-pixel block
    mask[9, 9] = True       # isolated pixel
    out = ch.keep_large_clusters(mask, 25)
    assert out[0:5, 0:5].all()
    assert not out[9, 9]


def test_loss_disabled_and_small_gain_dropped_by_default():
    valid = np.ones((6, 6), dtype=bool)
    # A clear, large loss region; gain region is only 4 px (< default 25).
    ndbi_b = np.full((6, 6), 0.3, "float32")
    ndbi_t = np.full((6, 6), -0.3, "float32")
    vv_b = np.full((6, 6), -2.0, "float32")
    vv_t = np.full((6, 6), -18.0, "float32")
    res = ch.detect_change_pair(ndbi_b, ndbi_t, vv_b, vv_t, valid)  # config defaults
    assert res.counts()["loss"] == 0  # loss disabled by default
    assert "cluster_id" in res.signals


def test_vectorize_produces_points_with_attrs():
    code = np.array([[ch.GAIN, ch.NONE, ch.LOSS]], dtype="uint8")
    # GRID is 3x3 but our arrays are 1x3; pad to grid shape for the test.
    code = np.vstack([code, np.zeros((2, 3), "uint8")])
    conf = np.zeros((3, 3), "float32")
    conf[0, 0] = 0.8
    conf[0, 2] = 0.6
    res = ch.ChangeResult(code, conf, {"ndbi_delta": np.zeros((3, 3), "float32")},
                          np.ones((3, 3), bool))
    gdf = change_to_points(res, GRID)
    assert len(gdf) == 2
    assert set(gdf["change_type"]) == {"gain", "loss"}
    assert gdf.crs.to_epsg() == 4326
    assert "ndbi_delta" in gdf.columns
