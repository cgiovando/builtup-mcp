"""Per-pixel features from a composite (v0.2).

Optical indices separate vegetation, water and bright surfaces; SAR statistics
separate rough, stable built structures from bare soil, which NDBI alone
confuses with built-up. Local standard deviation adds texture, which is high in
built-up fabric and low on fields or bare ground.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..sources.composite import Composite

OPTICAL_FEATURES = (
    "ndbi",
    "ndvi",
    "mndwi",
    "bsi",
    "brightness",
    "swir_ratio",
    "nir_tex",
    "ndvi_tex",
)
SAR_FEATURES = ("vv_med", "vh_med", "vv_minus_vh", "vv_std", "vh_std", "vv_tex")


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out.astype("float32")


def local_std(arr: np.ndarray, size: int = 5) -> np.ndarray:
    """NaN-aware moving-window standard deviation (texture)."""
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, np.nan, dtype="float32")
    filled = np.where(finite, arr, 0.0).astype("float64")
    w = ndimage.uniform_filter(finite.astype("float64"), size)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = ndimage.uniform_filter(filled, size) / w
        m2 = ndimage.uniform_filter(filled**2, size) / w
    out = np.sqrt(np.clip(m2 - m**2, 0.0, None)).astype("float32")
    out[~finite] = np.nan
    return out


def optical_features(c: Composite) -> dict[str, np.ndarray]:
    if not c.has_optical:
        return {}
    b = c.bands
    ndvi = _nd(b["nir"], b["red"])
    with np.errstate(divide="ignore", invalid="ignore"):
        swir_ratio = (b["swir2"] / b["swir1"]).astype("float32")
    return {
        "ndbi": _nd(b["swir1"], b["nir"]),
        "ndvi": ndvi,
        "mndwi": _nd(b["green"], b["swir1"]),
        "bsi": _nd(b["swir1"] + b["red"], b["nir"] + b["blue"]),
        "brightness": (
            (b["blue"] + b["green"] + b["red"] + b["nir"]) / 40000.0
        ).astype("float32"),
        "swir_ratio": swir_ratio,
        "nir_tex": local_std(b["nir"] / 10000.0),
        "ndvi_tex": local_std(ndvi),
    }


def sar_features(c: Composite) -> dict[str, np.ndarray]:
    if not c.has_sar:
        return {}
    b = c.bands
    return {
        "vv_med": b["vv_med"],
        "vh_med": b["vh_med"],
        "vv_minus_vh": (b["vv_med"] - b["vh_med"]).astype("float32"),
        "vv_std": b["vv_std"],
        "vh_std": b["vh_std"],
        "vv_tex": local_std(b["vv_med"]),
    }


def feature_stack(c: Composite, names: tuple[str, ...]) -> np.ndarray | None:
    """(H, W, F) array of the named features, or None if any is unavailable."""
    feats = {**optical_features(c), **sar_features(c)}
    if not all(n in feats for n in names):
        return None
    return np.stack([feats[n] for n in names], axis=-1)
