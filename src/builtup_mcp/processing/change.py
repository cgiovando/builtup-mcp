"""Built-up change detection (Phase 4, placeholder fusion logic).

Two modes matching the option-C baseline model:

- **pair**: built-up state at baseline vs target, where "built-up" requires BOTH
  optical (NDBI) and SAR (VV dB) to agree (min-fusion). Gain = not built-up
  before, built-up after; loss = the reverse. This is the disaster pre/post path.
- **reference**: built-up state at the target date diffed against the rasterized
  Overture baseline. Gain = built-up where no building is mapped (new / unmapped);
  loss = mapped building with no built-up signal now (potential demolition).

Both produce a per-pixel ``change_code`` (0 none, 1 gain, 2 loss) and a 0..1
``confidence``, then drop spatially isolated pixels. Requiring sensor agreement
plus a state transition (not just an index delta) sharply reduces the noise of
naive differencing. Thresholds live in config and stand in for a learned model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .. import config

_CONNECTIVITY_8 = np.ones((3, 3), dtype=int)

NONE, GAIN, LOSS = 0, 1, 2


@dataclass
class ChangeResult:
    """Per-pixel change classification plus the contributing signal layers."""

    change_code: np.ndarray  # uint8: 0 none, 1 gain, 2 loss
    confidence: np.ndarray  # float32 0..1
    signals: dict[str, np.ndarray]  # named evidence layers for output attributes
    valid: np.ndarray  # bool: pixels with usable data

    def counts(self) -> dict[str, int]:
        return {
            "gain": int((self.change_code == GAIN).sum()),
            "loss": int((self.change_code == LOSS).sum()),
            "valid_pixels": int(self.valid.sum()),
        }


def _unit(x: np.ndarray, scale: float) -> np.ndarray:
    """Clip ``x / scale`` into 0..1 (NaN -> 0)."""
    return np.clip(np.nan_to_num(x, nan=0.0) / scale, 0.0, 1.0)


def builtup_score(ndbi: np.ndarray, vv_db: np.ndarray) -> np.ndarray:
    """Fused 0..1 built-up score requiring BOTH optical and SAR evidence.

    Uses the minimum of the two normalized signals, so a pixel only scores high
    where NDBI *and* SAR backscatter both indicate built-up - this is the main
    lever against single-sensor noise.
    """
    opt = _unit(ndbi - config.NDBI_BUILTUP_THRESHOLD, config.NDBI_DELTA_SCALE)
    sar = _unit(vv_db - config.SAR_VV_DB_BUILTUP_THRESHOLD, config.SAR_VV_DELTA_SCALE_DB)
    return np.minimum(opt, sar)


def _count_neighbors(mask: np.ndarray) -> np.ndarray:
    """Count True 8-neighbors for every cell (numpy-only, no scipy)."""
    p = np.pad(mask.astype("uint8"), 1)
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )


def _despeckle(mask: np.ndarray, min_neighbors: int) -> np.ndarray:
    """Drop changed pixels with fewer than ``min_neighbors`` like neighbors."""
    if min_neighbors <= 0:
        return mask
    return mask & (_count_neighbors(mask) >= min_neighbors)


def label_clusters(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label 8-connected components; return (labels, per-pixel cluster size)."""
    labels, _n = ndimage.label(mask, structure=_CONNECTIVITY_8)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    return labels.astype("int32"), sizes[labels].astype("int32")


def keep_large_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Keep only connected components with at least ``min_size`` pixels."""
    if min_size <= 1:
        return mask
    _labels, sizes = label_clusters(mask)
    return mask & (sizes >= min_size)


def detect_change_pair(
    ndbi_baseline: np.ndarray,
    ndbi_target: np.ndarray,
    vv_db_baseline: np.ndarray,
    vv_db_target: np.ndarray,
    valid: np.ndarray,
    *,
    min_neighbors: int = config.MIN_CHANGE_NEIGHBORS,
    detect_loss: bool = config.DETECT_LOSS,
    min_gain_cluster_px: int = config.MIN_GAIN_CLUSTER_PIXELS,
) -> ChangeResult:
    """Pre/post change as a built-up state transition (both sensors agree)."""
    score_b = builtup_score(ndbi_baseline, vv_db_baseline)
    score_t = builtup_score(ndbi_target, vv_db_target)
    thr = config.BUILTUP_SCORE_THRESHOLD
    mask_b = score_b >= thr
    mask_t = score_t >= thr

    gain = valid & mask_t & ~mask_b
    loss = valid & mask_b & ~mask_t
    confidence = np.where(gain, score_t, np.where(loss, score_b, 0.0)).astype("float32")

    return _finalize(
        gain,
        loss,
        confidence,
        valid,
        signals={
            "ndbi_delta": (ndbi_target - ndbi_baseline).astype("float32"),
            "vv_db_delta": (vv_db_target - vv_db_baseline).astype("float32"),
        },
        min_neighbors=min_neighbors,
        detect_loss=detect_loss,
        min_gain_cluster_px=min_gain_cluster_px,
    )


def detect_change_reference(
    ndbi_target: np.ndarray,
    vv_db_target: np.ndarray,
    overture_mask: np.ndarray,
    valid: np.ndarray,
    *,
    min_neighbors: int = config.MIN_CHANGE_NEIGHBORS,
    detect_loss: bool = config.DETECT_LOSS,
    min_gain_cluster_px: int = config.MIN_GAIN_CLUSTER_PIXELS,
) -> ChangeResult:
    """Single-date built-up vs the Overture baseline prior."""
    score_t = builtup_score(ndbi_target, vv_db_target)
    mask_t = score_t >= config.BUILTUP_SCORE_THRESHOLD

    gain = valid & mask_t & ~overture_mask  # built-up where nothing mapped
    loss = valid & overture_mask & ~mask_t  # mapped, but no built-up signal now
    confidence = np.where(gain, score_t, np.where(loss, 1.0 - score_t, 0.0)).astype(
        "float32"
    )

    return _finalize(
        gain,
        loss,
        confidence,
        valid,
        signals={
            "ndbi_target": ndbi_target.astype("float32"),
            "vv_db_target": vv_db_target.astype("float32"),
            "in_overture": overture_mask.astype("uint8"),
        },
        min_neighbors=min_neighbors,
        detect_loss=detect_loss,
        min_gain_cluster_px=min_gain_cluster_px,
    )


def _finalize(
    gain: np.ndarray,
    loss: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    signals: dict[str, np.ndarray],
    min_neighbors: int,
    detect_loss: bool,
    min_gain_cluster_px: int,
) -> ChangeResult:
    """Apply spatial cleanup + large-cluster gain filter and assemble the result."""
    gain = _despeckle(gain, min_neighbors)
    gain = keep_large_clusters(gain, min_gain_cluster_px)
    loss = _despeckle(loss, min_neighbors) if detect_loss else np.zeros_like(loss)

    code = np.full(gain.shape, NONE, dtype="uint8")
    code[gain] = GAIN
    code[loss] = LOSS
    confidence = np.where(code == NONE, 0.0, confidence).astype("float32")

    # Tag each gain pixel with its cluster id and size for downstream grouping.
    labels, sizes = label_clusters(gain)
    signals = {
        **signals,
        "cluster_id": np.where(gain, labels, 0).astype("int32"),
        "cluster_size_px": np.where(gain, sizes, 0).astype("int32"),
    }
    return ChangeResult(code, confidence, signals, valid)
