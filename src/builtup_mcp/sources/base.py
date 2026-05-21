"""Shared container for fetched raster bands (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from affine import Affine

from .raster_io import Grid


@dataclass
class RasterStack:
    """A set of co-registered bands on a common grid, plus provenance.

    The schema is intentionally open (``meta``) so future sources/attributes can
    attach without breaking downstream consumers.
    """

    bands: dict[str, np.ndarray]
    grid: Grid
    item_id: str
    datetime: str
    collection: str
    meta: dict = field(default_factory=dict)

    @property
    def transform(self) -> Affine:
        left, _bottom, _right, top = self.grid.bounds
        return Affine(self.grid.resolution_m, 0, left, 0, -self.grid.resolution_m, top)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def __getitem__(self, name: str) -> np.ndarray:
        return self.bands[name]
