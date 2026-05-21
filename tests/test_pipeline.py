"""Live end-to-end pipeline test (Phase 4). Marked ``network``, opt-in.

Run with: ``uv run pytest -m network -k pipeline``.
"""

from __future__ import annotations

import json

import pytest

from builtup_mcp import config
from builtup_mcp.pipeline import run_change_detection


@pytest.mark.network
def test_pipeline_pair_mode_writes_geojson(tmp_path):
    run = run_change_detection(
        list(config.DEMO_BBOX), "2024-06-15", "2021-06-15", output_dir=tmp_path
    )
    assert run.mode == "pre/post pair"
    assert run.geojson_path.exists()
    # A plausible (not runaway) fraction of the AOI flags as change.
    total = run.counts["gain"] + run.counts["loss"]
    assert 0 < total < 0.30 * run.counts["valid_pixels"]
    # Output is a valid FeatureCollection with the expected attributes.
    fc = json.loads(run.geojson_path.read_text())
    assert fc["type"] == "FeatureCollection"
    props = fc["features"][0]["properties"]
    assert "change_type" in props and "confidence" in props
