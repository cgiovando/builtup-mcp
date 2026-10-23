"""End-to-end built-up change-detection pipeline (Phase 4).

Orchestrates: validate -> windowed S1+S2 reads (target, and baseline date if
given) -> align to the S2 grid -> change detection -> GeoJSON. In reference mode
the Overture baseline is also fetched and its data-freshness reported, so the
"change" is time-qualified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .geo import BBox, bbox_area_km2, validate_bbox, validate_change_dates
from .processing import change as change_mod
from .processing.align import reproject_array, reproject_stack
from .processing.indices import ndbi
from .processing.rasterize import rasterize_buildings
from .processing.vectorize import change_to_points
from .sources import sentinel1, sentinel2
from .sources.overture import baseline_age, fetch_buildings, resolve_release


@dataclass
class ChangeRun:
    """Result of a pipeline run."""

    bbox: BBox
    mode: str
    target_date: str
    baseline_date: str | None
    counts: dict
    geojson_path: Path
    baseline_desc: str
    n_overture: int | None = None
    overture_age: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        c = self.counts
        parts = [
            f"Built-up GAIN over bbox {self.bbox.as_tuple()} "
            f"(~{bbox_area_km2(self.bbox):.2f} sq km). Mode: {self.mode}.",
            f"Gain pixels: {c['gain']} (of {c['valid_pixels']} valid pixels @10m).",
            f"Baseline: {self.baseline_desc}",
            f"GeoJSON: {self.geojson_path}.",
        ]
        if self.notes:
            parts.append("Notes: " + " ".join(self.notes))
        return " ".join(parts)


def _valid_from_scl(stack) -> np.ndarray:
    mask = stack.meta.get("valid_mask")
    return np.ones(stack.shape, dtype=bool) if mask is None else mask


def _years_between(start: str, end: str) -> float:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days / 365.25


def run_change_detection(
    bbox: list[float],
    target_date: str,
    baseline_date: str | None = None,
    *,
    output_dir: Path | None = None,
) -> ChangeRun:
    """Run the full change-detection pipeline and write a GeoJSON result."""
    box = validate_bbox(bbox)
    target, baseline = validate_change_dates(target_date, baseline_date)
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    # Target-date EO on the canonical S2 grid.
    s2_t = sentinel2.fetch_optical(box, target)
    grid = s2_t.grid
    s1_t = reproject_stack(sentinel1.fetch_sar(box, target), grid)
    ndbi_t = ndbi(s2_t["nir"], s2_t["swir"])
    valid_t = _valid_from_scl(s2_t)

    n_overture: int | None = None
    overture_age: dict = {}

    if baseline is not None:
        # Pre/post pair: baseline age is exactly known (the dates are chosen).
        mode = "pre/post pair"
        s2_b = sentinel2.fetch_optical(box, baseline)
        s1_b = reproject_stack(sentinel1.fetch_sar(box, baseline), grid)
        ndbi_b = ndbi(s2_b["nir"], s2_b["swir"])
        if s2_b.grid != grid:
            ndbi_b = reproject_array(ndbi_b, s2_b.grid, grid)
            valid_b = reproject_array(
                _valid_from_scl(s2_b).astype("float32"), s2_b.grid, grid
            ) > 0.5
        else:
            valid_b = _valid_from_scl(s2_b)
        result = change_mod.detect_change_pair(
            ndbi_b, ndbi_t, s1_b["vv_db"], s1_t["vv_db"], valid_t & valid_b
        )
        baseline_desc = (
            f"Sentinel {baseline} -> {target} ({_years_between(baseline, target):.1f} yr, "
            "exact). Gain = built-up change over this period."
        )
        notes.append(
            "SAR signal is backscatter delta (RTC has no phase/coherence); "
            "treat as a complement to optical."
        )
    else:
        # Reference: Overture baseline, with unknown age -> report data freshness.
        mode = "single date vs Overture reference"
        buildings = fetch_buildings(box)
        n_overture = int(len(buildings))
        overture_age = baseline_age(buildings, target)
        overture_mask = rasterize_buildings(buildings, grid)
        result = change_mod.detect_change_reference(
            ndbi_t, s1_t["vv_db"], overture_mask, valid_t
        )
        if overture_age.get("n_dated"):
            baseline_desc = (
                f"Overture {resolve_release()} ({n_overture} buildings; "
                f"{overture_age['n_dated']} dated). Median source update "
                f"{overture_age['median_update']} (~{overture_age['median_age_years']} yr "
                f"before target; range {overture_age['oldest_update']}.."
                f"{overture_age['newest_update']}). Gain ~ change over up to "
                f"~{overture_age['max_age_years']} yr."
            )
        else:
            baseline_desc = (
                f"Overture {resolve_release()} ({n_overture} buildings; no "
                "source timestamps). Baseline age unknown - interpret gain with caution."
            )
        notes.append(
            "Reference-mode gain depends on Overture freshness above; "
            "loss is disabled (not yet reliable)."
        )
        # Temporal-leakage guard: a baseline newer than the target makes "new
        # built-up since baseline" ill-defined.
        newest = overture_age.get("newest_update")
        if newest and newest > target:
            notes.append(
                f"WARNING: Overture baseline contains updates after the target date "
                f"({newest} > {target}); reference-mode 'gain' may be unreliable. "
                "Prefer pre/post pair mode."
            )

    gdf = change_to_points(result, grid)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    geojson_path = output_dir / f"builtup_change_{stamp}.geojson"
    if gdf.empty:
        geojson_path.write_text(
            '{"type": "FeatureCollection", "features": []}', encoding="utf-8"
        )
        notes.append("No gain clusters passed the thresholds.")
    else:
        gdf.to_file(geojson_path, driver="GeoJSON")

    return ChangeRun(
        bbox=box,
        mode=mode,
        target_date=target,
        baseline_date=baseline,
        counts=result.counts(),
        geojson_path=geojson_path,
        baseline_desc=baseline_desc,
        n_overture=n_overture,
        overture_age=overture_age,
        notes=notes,
    )
