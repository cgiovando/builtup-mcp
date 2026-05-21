"""Unit tests for Open Buildings tile geometry helpers (Phase 4 validation)."""

from __future__ import annotations

from builtup_mcp.sources.open_buildings import _overlap_area, _tile_bounds


def test_tile_bounds_from_affine():
    aff = {"scaleX": 0.5, "translateX": 234532.0, "scaleY": -0.5, "translateY": 8055632.0}
    dims = {"width": 25000, "height": 25000}
    left, bottom, right, top = _tile_bounds(aff, dims)
    assert left == 234532.0
    assert right == 234532.0 + 25000 * 0.5
    assert top == 8055632.0
    assert bottom == 8055632.0 - 25000 * 0.5


def test_overlap_area():
    a = (0.0, 0.0, 10.0, 10.0)
    assert _overlap_area(a, (5.0, 5.0, 15.0, 15.0)) == 25.0  # 5x5 overlap
    assert _overlap_area(a, (20.0, 20.0, 30.0, 30.0)) == 0.0  # disjoint
