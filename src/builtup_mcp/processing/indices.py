"""Spectral indices (Phase 4)."""

from __future__ import annotations

import numpy as np


def ndbi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index = (SWIR - NIR) / (SWIR + NIR).

    Returns float32 with NaN where the denominator is zero. Positive values tend
    to indicate built-up / bare surfaces; this is a placeholder for a future
    learned built-up model.
    """
    nir = nir.astype("float32")
    swir = swir.astype("float32")
    denom = swir + nir
    out = np.full(nir.shape, np.nan, dtype="float32")
    nz = denom != 0
    out[nz] = (swir[nz] - nir[nz]) / denom[nz]
    return out
