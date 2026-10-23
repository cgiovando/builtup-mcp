"""Per-request built-up classifier trained on open building footprints (v0.2).

The v0.1 detector used fixed NDBI and VV thresholds, which do not transfer
between cities (bare soil in Malakal is not bare soil in Lagos). Instead, each
request trains its own small model, using the footprints that already exist for
the AOI as weak labels:

- positives: 10 m pixels with at least ``LABEL_POS_FRACTION`` building cover;
- negatives: pixels farther than ``LABEL_NEG_BUFFER_M`` from any footprint.

Footprint maps miss buildings (that is the point of the product), so labels are
noisy; gradient boosting with balanced classes tolerates that, and the model is
only asked "what does built-up look like here", not "when was it built". Change
timing comes from applying the one model to every window (``timeseries``).

Three models are fitted where data allow: fused (optical + SAR), optical-only and
SAR-only. Each pixel in each window is scored by the richest model its data
support, so cloud gaps and Sentinel-1 coverage gaps degrade gracefully instead
of failing. Accuracy is estimated on spatially held-out blocks, and reported
with every product.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .. import config
from ..sources.composite import Composite
from ..sources.raster_io import Grid
from .align import grid_transform
from .features import OPTICAL_FEATURES, SAR_FEATURES, feature_stack

MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "fused": OPTICAL_FEATURES + SAR_FEATURES,
    "optical": OPTICAL_FEATURES,
    "sar": SAR_FEATURES,
}


def building_fraction(gdf: gpd.GeoDataFrame, grid: Grid, supersample: int = 4) -> np.ndarray:
    """Fraction (0..1) of each grid cell covered by footprints."""
    if gdf is None or gdf.empty:
        return np.zeros(grid.shape, dtype="float32")
    geoms = [g for g in gdf.to_crs(epsg=grid.epsg).geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(grid.shape, dtype="float32")
    fine = Grid(
        grid.epsg,
        grid.bounds,
        grid.width * supersample,
        grid.height * supersample,
        grid.resolution_m / supersample,
    )
    burned = rasterize(
        [(g, 1) for g in geoms],
        out_shape=fine.shape,
        transform=grid_transform(fine),
        fill=0,
        dtype="uint8",
    )
    h, w = grid.shape
    return (
        burned.reshape(h, supersample, w, supersample).mean(axis=(1, 3)).astype("float32")
    )


def weak_labels(frac: np.ndarray, resolution_m: float) -> np.ndarray:
    """int8 labels: 1 built-up, 0 not built-up, -1 ignore."""
    labels = np.full(frac.shape, -1, dtype="int8")
    labels[frac >= config.LABEL_POS_FRACTION] = 1
    dist_m = ndimage.distance_transform_edt(frac <= 0) * resolution_m
    labels[dist_m > config.LABEL_NEG_BUFFER_M] = 0
    return labels


def _blocks(grid: Grid) -> np.ndarray:
    """Spatial block id per pixel, for held-out evaluation."""
    side = max(1, int(round(config.HOLDOUT_BLOCK_M / grid.resolution_m)))
    rows = np.arange(grid.height)[:, None] // side
    cols = np.arange(grid.width)[None, :] // side
    return (rows * 10007 + cols).astype("int64")


@dataclass
class TrainedModels:
    models: dict[str, HistGradientBoostingClassifier] = field(default_factory=dict)
    metrics: dict[str, dict] = field(default_factory=dict)
    train_window: dict[str, tuple[str, str]] = field(default_factory=dict)
    n_labels: dict[str, int] = field(default_factory=dict)


def _sample(idx: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return idx if idx.size <= n else rng.choice(idx, n, replace=False)


def _fit(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    return clf.fit(X, y)


def train_models(
    composites: list[Composite],
    labels: np.ndarray,
    label_date: str | None,
    *,
    seed: int = 0,
) -> TrainedModels:
    """Fit fused / optical / SAR models on samples pooled across windows.

    Pooling teaches each model what built-up looks like across years, sensors
    (Sentinel-1A vs 1C, Sentinel-2A/B/C) and wet or dry years, so a pixel does not
    change class just because the radiometry of one window differs. Negatives
    (far from any footprint) are taken from every window. Positives are taken only
    from windows that end no earlier than ``POSITIVE_LOOKBACK_YEARS`` before the
    footprints' median update date, since older windows predate some buildings.
    """
    from datetime import date, timedelta

    rng = np.random.default_rng(seed)
    blocks = _blocks(composites[0].grid).ravel()
    out = TrainedModels()
    ld = date.fromisoformat(label_date) if label_date else None
    pos_from = (
        (ld - timedelta(days=int(365.25 * config.POSITIVE_LOOKBACK_YEARS))).isoformat() if ld else None
    )
    # Mirror rule for negatives: after the footprints were mapped, "far from any
    # footprint" no longer means "not built" (that is the change being detected).
    neg_until = (ld + timedelta(days=int(365.25 * config.NEGATIVE_LOOKAHEAD_YEARS))).isoformat() if ld else None
    flat = labels.ravel()

    for kind, names in MODEL_FEATURES.items():
        cands = [c for c in composites if feature_stack(c, names) is not None]
        if not cands:
            continue
        per_window = max(config.MIN_LABELS_PER_CLASS, config.LABEL_MAX_PER_CLASS // len(cands))
        Xs, ys, bs = [], [], []
        for comp in cands:
            X_all = feature_stack(comp, names).reshape(-1, len(names))
            ok = np.all(np.isfinite(X_all), axis=-1) & (flat >= 0)
            neg = np.flatnonzero(ok & (flat == 0))
            pos = np.flatnonzero(ok & (flat == 1))
            if pos_from and comp.window[1] < pos_from:
                pos = pos[:0]
            if neg_until and comp.window[0] > neg_until:
                neg = neg[:0]
            idx = np.concatenate([_sample(pos, per_window, rng), _sample(neg, per_window, rng)])
            Xs.append(X_all[idx])
            ys.append(flat[idx])
            bs.append(blocks[idx])
        X, y, b = np.concatenate(Xs), np.concatenate(ys), np.concatenate(bs)
        if (y == 1).sum() < config.MIN_LABELS_PER_CLASS or (y == 0).sum() < config.MIN_LABELS_PER_CLASS:
            continue
        comp = cands[0]
        idx = np.arange(y.size)

        # Spatial hold-out: ~20% of blocks never seen in training.
        ub = np.unique(b)
        test_blocks = rng.choice(ub, max(1, ub.size // 5), replace=False)
        test = np.isin(b, test_blocks)
        metrics: dict = {}
        if test.any() and (~test).any() and len(np.unique(y[test])) == 2:
            clf = _fit(X[~test], y[~test])
            p = clf.predict_proba(X[test])[:, 1]
            # Scored the way the product classifies: built at PROB_BUILT; pixels in
            # the dead band count as "not built" here and are reported separately.
            yhat = p >= config.PROB_BUILT
            metrics = {
                "holdout_pixels": int(test.sum()),
                "threshold": config.PROB_BUILT,
                "dead_band_fraction": round(float(((p > config.PROB_NOT_BUILT) & (p < config.PROB_BUILT)).mean()), 3),
                "precision": round(float(precision_score(y[test], yhat)), 3),
                "recall": round(float(recall_score(y[test], yhat)), 3),
                "f1": round(float(f1_score(y[test], yhat)), 3),
                "roc_auc": round(float(roc_auc_score(y[test], p)), 3),
            }
        out.models[kind] = _fit(X, y)
        out.metrics[kind] = metrics
        out.train_window[kind] = (cands[0].window[0], cands[-1].window[1])
        out.n_labels[kind] = int(idx.size)
    return out


def predict_probability(
    models: TrainedModels, comp: Composite, kinds: tuple[str, ...] = ("fused", "optical", "sar")
) -> tuple[np.ndarray, np.ndarray]:
    """Built-up probability per pixel, and which model scored it.

    Returns ``(prob, source)`` where ``source`` is 0 none, 1 fused, 2 optical,
    3 SAR-only.
    """
    shape = comp.grid.shape
    prob = np.full(shape, np.nan, dtype="float32")
    source = np.zeros(shape, dtype="uint8")
    for code, kind in ((1, "fused"), (2, "optical"), (3, "sar")):
        if kind not in models.models or kind not in kinds:
            continue
        names = MODEL_FEATURES[kind]
        X = feature_stack(comp, names)
        if X is None:
            continue
        ok = np.all(np.isfinite(X), axis=-1) & np.isnan(prob)
        if not ok.any():
            continue
        prob[ok] = models.models[kind].predict_proba(X[ok])[:, 1]
        source[ok] = code
    return prob, source
