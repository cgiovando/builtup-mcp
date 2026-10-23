"""Epoch-of-change from a time series of built-up probabilities (v0.2).

Each window gets a state per pixel from its probability, with a dead band so
borderline pixels do not flicker:

    built      p >= PROB_BUILT
    not built  p <= PROB_NOT_BUILT
    unknown    otherwise, or no data (cloud, SAR gap)

Gain is then dated as the first "built" window after the LAST "not built"
window. That rule needs no tuning and is persistent by construction: after the
gain epoch the pixel is never again observed as not built. A gain seen only in
the final window cannot yet be confirmed and is flagged provisional. Loss (the
mirror rule) is computed only on request, for crisis cases, and is experimental.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from scipy import ndimage

from .change import _CONNECTIVITY_8, _despeckle, keep_large_clusters

NODATA = 255


@dataclass
class EpochChange:
    gain_epoch: np.ndarray  # uint8: 0 no gain, k = first built window index (1..T-1), 255 nodata
    loss_epoch: np.ndarray  # uint8, same coding; all 0 when loss is off
    built_first: np.ndarray  # bool: built in the first observed window and never lost
    built_latest: np.ndarray  # bool: latest observed state is built
    confidence: np.ndarray  # float32 0..1: mean p after gain minus mean p before
    provisional: np.ndarray  # bool: gain only seen in the final window
    observed: np.ndarray  # uint8: number of windows with a definite state


def states(probs: np.ndarray) -> np.ndarray:
    """(T, H, W) float probabilities -> int8 states (1 built, 0 not, -1 unknown)."""
    s = np.full(probs.shape, -1, dtype="int8")
    s[probs >= config.PROB_BUILT] = 1
    s[probs <= config.PROB_NOT_BUILT] = 0
    return s


def _last_index(mask: np.ndarray) -> np.ndarray:
    """Index of the last True along axis 0, or -1."""
    t = mask.shape[0]
    rev = mask[::-1].argmax(axis=0)
    return np.where(mask.any(axis=0), t - 1 - rev, -1)


def _first_index_after(mask: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Index of the first True along axis 0 strictly after ``after``, or -1."""
    t = mask.shape[0]
    idx = np.arange(t)[:, None, None]
    ok = mask & (idx > after[None])
    first = ok.argmax(axis=0)
    return np.where(ok.any(axis=0), first, -1)


def keep_cored_clusters(mask: np.ndarray) -> np.ndarray:
    """Keep clusters at least 3 px (30 m) thick somewhere.

    Drops one- and two-pixel-wide linear features, mostly roads being graded or
    resurfaced, which read as "new built-up" at 10 m but are not settlement.
    """
    core = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool))
    labels, _n = ndimage.label(mask, structure=_CONNECTIVITY_8)
    keep = np.unique(labels[core])
    return mask & np.isin(labels, keep[keep > 0])


def epoch_change(
    probs: np.ndarray,
    valid_aoi: np.ndarray | None = None,
    *,
    detect_loss: bool = False,
    min_neighbors: int = 2,
    min_cluster_px: int = config.MIN_GAIN_CLUSTER_PIXELS,
    require_core: bool = True,
) -> EpochChange:
    s = states(probs)
    t = s.shape[0]
    is1, is0 = s == 1, s == 0
    observed = (s >= 0).sum(axis=0).astype("uint8")

    last0 = _last_index(is0)
    gain_k = _first_index_after(is1, last0)
    gain = (last0 >= 0) & (gain_k > 0)

    # A pixel that was mostly built before its "gain" is established built-up
    # with a bad window (cloud residue, flood, sensor switch), not new built-up.
    idx = np.arange(t)[:, None, None]
    before = idx < gain_k[None]
    n_before = ((s >= 0) & before).sum(axis=0)
    ones_before = (is1 & before).sum(axis=0)
    flicker = gain & (ones_before * 2 >= n_before) & (n_before > 0)
    gain &= ~flicker

    last1 = _last_index(is1)
    first_def = np.argmax(s >= 0, axis=0)
    first_state = np.take_along_axis(s, first_def[None], axis=0)[0]
    n_def = np.maximum(observed.astype(int), 1)
    built_first = (first_state == 1) & (is0.sum(axis=0) * 3 <= n_def) & ~gain
    built_first |= flicker
    built_latest = last1 > last0

    loss = np.zeros(gain.shape, dtype=bool)
    loss_k = np.full(gain.shape, -1)
    if detect_loss:
        loss_k = _first_index_after(is0, last1)
        loss = (last1 >= 0) & (loss_k > 0)

    if valid_aoi is not None:
        gain &= valid_aoi
        loss &= valid_aoi

    gain = keep_large_clusters(_despeckle(gain, min_neighbors), min_cluster_px)
    if require_core:
        gain = keep_cored_clusters(gain)
    if detect_loss:
        loss = keep_large_clusters(_despeckle(loss, min_neighbors), min_cluster_px)

    # Confidence: how cleanly the probability steps up at the gain epoch.
    idx = np.arange(t)[:, None, None]
    after = idx >= gain_k[None]
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p_after = np.nanmean(np.where(after, probs, np.nan), axis=0)
            p_before = np.nanmean(np.where(~after, probs, np.nan), axis=0)
    conf = np.clip(np.nan_to_num(p_after - p_before), 0, 1).astype("float32")
    conf = np.where(gain, conf, 0).astype("float32")

    gain_epoch = np.where(gain, gain_k, 0).astype("uint8")
    loss_epoch = np.where(loss, loss_k, 0).astype("uint8")
    if valid_aoi is not None:
        gain_epoch[~valid_aoi] = NODATA
        loss_epoch[~valid_aoi] = NODATA
    return EpochChange(
        gain_epoch=gain_epoch,
        loss_epoch=loss_epoch,
        built_first=built_first,
        built_latest=built_latest,
        confidence=conf,
        provisional=gain & (gain_k == t - 1),
        observed=observed,
    )
