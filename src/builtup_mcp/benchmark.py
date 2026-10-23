"""Benchmark a city product against independent published layers (v0.2).

Two references, each compared only over the period it actually covers:

- **WSF Tracker** (DLR/MindEarth, six-monthly, to Jan 2026): built-up extent at a
  matching date, and new built-up between the first window and Jan 2026.
- **Google Open Buildings 2.5D Temporal** (annual, to 2023): new built-up between
  the first window's year and 2023 (presence rising from < 0.3 to >= 0.5).

Both are themselves model outputs, so agreement is evidence of consistency, not
ground truth. A one-pixel tolerance absorbs co-registration offsets at 10 m.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from scipy.ndimage import binary_dilation

from . import config


def _agree(ours: np.ndarray, ref: np.ndarray, valid: np.ndarray, tol: int = 1) -> dict:
    ours, ref = ours & valid, ref & valid
    ref_t = binary_dilation(ref, iterations=tol) if tol else ref
    ours_t = binary_dilation(ours, iterations=tol) if tol else ours
    n_o, n_r = int(ours.sum()), int(ref.sum())
    px_ha = 0.01  # 10 m pixels
    out = {"ours_ha": round(n_o * px_ha, 1), "reference_ha": round(n_r * px_ha, 1), "tolerance_px": tol}
    if n_o:
        out["precision"] = round(float((ours & ref_t).sum() / n_o), 3)
    if n_r:
        out["recall"] = round(float((ref & ours_t).sum() / n_r), 3)
    if n_o and n_r:
        p, r = out["precision"], out["recall"]
        out["f1"] = round(2 * p * r / (p + r), 3) if p + r else 0.0
    return out


def _window_index_at(windows: list[tuple[str, str]], when: str) -> int | None:
    """Last window ending on or before ``when``."""
    idx = [k for k, (_a, b) in enumerate(windows) if b <= when]
    return idx[-1] if idx else None


def vs_wsf_tracker(grid, bbox, aoi, windows, built, gain_epoch) -> dict:
    from .sources.wsf_tracker import EPOCHS, fetch_wsf_tracker

    wsf = fetch_wsf_tracker(grid, bbox)
    out: dict = {"layer": "WSF Tracker (DLR/MindEarth)", "latest_epoch": EPOCHS[20]}
    # Our last window ending by Mar 2026 (Dec-Mar composites centre on January).
    k = _window_index_at(windows, "2026-03-31")
    if k is None:
        return out
    # Compare at matching dates: WSF epochs up to the end of our window k.
    end_k = windows[k][1]
    j = max((i for i, d in EPOCHS.items() if d <= end_k), default=0)
    wsf_built = (wsf >= 1) & (wsf <= j)
    out["extent"] = {"our_window": windows[k], "wsf_epoch": EPOCHS.get(j), **_agree(built[k], wsf_built, aoi)}
    # New built-up after our first window, up to the same matched date.
    first_end = windows[0][1]
    first_idx = max((i for i, d in EPOCHS.items() if d <= first_end), default=0)
    wsf_new = (wsf > first_idx) & (wsf <= j)
    ours_new = (gain_epoch >= 1) & (gain_epoch <= k)
    out["new_builtup"] = {
        "period": f"{first_end} .. {EPOCHS.get(j)}",
        **_agree(ours_new, wsf_new, aoi),
    }
    out["wsf_new_ha_by_epoch"] = {
        EPOCHS[i]: round(float(((wsf == i) & aoi).sum()) * 0.01, 1) for i in range(first_idx + 1, j + 1)
    }
    out["_wsf"] = wsf
    return out


def vs_open_buildings(grid, aoi, windows, gain_epoch) -> dict:
    from .sources.open_buildings import fetch_building_presence

    y0 = date.fromisoformat(windows[0][1]).year
    y1 = config.OPEN_BUILDINGS_LATEST_YEAR
    if y0 >= y1:
        return {}
    p0 = fetch_building_presence(grid, y0)
    p1 = fetch_building_presence(grid, y1)
    valid = aoi & np.isfinite(p0) & np.isfinite(p1)
    ob_new = (p1 >= 0.5) & (p0 < 0.3)
    k = _window_index_at(windows, f"{y1}-12-31")
    ours_new = (gain_epoch >= 1) & (gain_epoch <= (k or 0))
    return {
        "layer": "Google Open Buildings 2.5D Temporal",
        "period": f"{y0} .. {y1}",
        "new_builtup": _agree(ours_new, ob_new, valid),
        "base_rate_new": round(float(ob_new[valid].mean()), 4) if valid.any() else None,
    }
