"""Project-wide configuration constants for builtup-mcp."""

from __future__ import annotations

import os
from pathlib import Path

# --- BBox constraints -------------------------------------------------------

# The prototype caps the BBox to keep local processing fast and cheap. The AOI
# itself is a free parameter; this is only an upper bound. Override via env var.
MAX_BBOX_AREA_KM2: float = float(os.environ.get("BUILTUP_MCP_MAX_BBOX_KM2", "5.0"))

# WGS84 (EPSG:4326) coordinate bounds.
MIN_LON, MAX_LON = -180.0, 180.0
MIN_LAT, MAX_LAT = -90.0, 90.0

# --- Output -----------------------------------------------------------------

# Directory where result GeoJSON files are written. Overridable via env var.
OUTPUT_DIR: Path = Path(os.environ.get("BUILTUP_MCP_OUTPUT_DIR", "/tmp/builtup-mcp"))

# --- Coordinate reference systems -------------------------------------------

WGS84_EPSG: int = 4326

# --- Overture Maps (Phase 2) ------------------------------------------------

# Overture release. "latest" is resolved at run time from Overture's STAC catalog
# (https://stac.overturemaps.org), and the resolved id is written into every
# product's provenance. Overture keeps only the last few releases online, so a
# hard-coded id goes stale within months; pin one via env var for reproducibility.
OVERTURE_RELEASE: str = os.environ.get("BUILTUP_MCP_OVERTURE_RELEASE", "latest")
OVERTURE_S3_REGION: str = "us-west-2"
OVERTURE_STAC_URL: str = "https://stac.overturemaps.org"

# Fallback floor-to-meters factor when a building has num_floors but no height.
FLOOR_HEIGHT_M: float = 3.0

# --- Cloud-native EO / STAC (Phase 3) ---------------------------------------

# Microsoft Planetary Computer STAC API (anonymous; assets signed on read).
PC_STAC_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"

S2_COLLECTION: str = "sentinel-2-l2a"
S1_COLLECTION: str = "sentinel-1-rtc"

# Planetary Computer asset keys.
S2_ASSETS: dict[str, str] = {"nir": "B08", "swir": "B11", "scl": "SCL"}
S1_ASSETS: dict[str, str] = {"vv": "vv", "vh": "vh"}

# Common analysis grid resolution (meters), in the scene's native UTM.
TARGET_RESOLUTION_M: float = 10.0

# Temporal search: look back this many days from a requested date for a usable scene.
SEARCH_WINDOW_DAYS: int = 45

# Scene-level cloud prefilter (eo:cloud_cover, percent) and AOI-level acceptance.
MAX_SCENE_CLOUD_PCT: float = 60.0
MAX_AOI_CLOUD_FRACTION: float = 0.35

# Sentinel-2 SCL classes treated as cloud / shadow / invalid for masking.
# 0 no-data, 1 saturated, 3 cloud shadow, 8 cloud med, 9 cloud high, 10 thin cirrus.
SCL_CLOUD_CLASSES: tuple[int, ...] = (0, 1, 3, 8, 9, 10)

# --- Composites + city requests (v0.2) --------------------------------------

# City requests cover whole urban extents, so they get their own, larger cap.
CITY_MAX_AREA_KM2: float = float(os.environ.get("BUILTUP_MCP_CITY_MAX_KM2", "150.0"))

# Scenes per window. More S2 scenes = fewer cloud gaps; more S1 = less speckle.
COMPOSITE_MAX_S2_SCENES: int = 16
COMPOSITE_MAX_S1_SCENES: int = 12
COMPOSITE_MIN_S1_SCENES: int = 2
COMPOSITE_MIN_S1_FOR_MODEL: int = 4  # fewer acquisitions -> window scored optical-only

# A Sentinel-2 scene joins a composite only if this fraction of the AOI is clear.
MIN_SCENE_CLEAR_FRACTION: float = 0.15

# Parallel windowed COG reads.
READ_WORKERS: int = int(os.environ.get("BUILTUP_MCP_READ_WORKERS", "12"))

# Default composite window length when only dates are given (days before the date).
DEFAULT_WINDOW_DAYS: int = 120

# Per-request classifier (weak supervision from open building footprints).
LABEL_POS_FRACTION: float = 0.25  # >= this building cover in a 10 m pixel = built-up
LABEL_NEG_BUFFER_M: float = 40.0  # negatives must be this far from any footprint
LABEL_MAX_PER_CLASS: int = 60000  # per class, split across pooled windows
POSITIVE_LOOKBACK_YEARS: float = 3.0
# Negatives after the label date are noisy (post-mapping growth is labelled "not
# built"), but dropping or thinning them is worse: the model then never sees
# post-mapping radiometry for bare ground. On Malakal (labels median 2022) a
# 1-year cut-off with a 250 m late buffer gave 284 ha of new built-up against
# 148 ha with all windows, and lower agreement with WSF Tracker (F1 0.16 vs
# 0.21). Default: negatives from every window. Lower this to test the trade-off.
NEGATIVE_LOOKAHEAD_YEARS: float = 100.0
LATE_NEGATIVE_BUFFER_M: float = 250.0  # applies only within NEGATIVE_LOOKAHEAD_YEARS
MIN_LABELS_PER_CLASS: int = 300
# Time-series state thresholds on built-up probability (hysteresis).
PROB_BUILT: float = 0.6
PROB_NOT_BUILT: float = 0.35
# Spatial blocks (m) for the held-out accuracy estimate.
HOLDOUT_BLOCK_M: float = 500.0

# H3 resolution for neighbourhood statistics (res 8 ~ 0.74 sq km cells).
H3_RESOLUTION: int = 8

# --- Change detection (Phase 4, placeholder spectral/backscatter heuristics) -
# These thresholds stand in for a future learned model. NDBI = (SWIR-NIR)/(SWIR+NIR);
# positive NDBI tends to indicate built-up / bare surfaces. SAR VV in dB rises with
# built-up double-bounce. All are deliberately conservative and tunable.

# Single-date "is this pixel built-up now?" detection (reference mode).
NDBI_BUILTUP_THRESHOLD: float = 0.0
SAR_VV_DB_BUILTUP_THRESHOLD: float = -10.0

# Two-date change magnitudes (pre/post pair mode) and the scales used to grade
# evidence into a 0..1 confidence.
NDBI_DELTA_SCALE: float = 0.20  # an NDBI delta of this size -> full optical evidence
SAR_VV_DELTA_SCALE_DB: float = 4.0  # a VV dB delta of this size -> full SAR evidence

# A pixel counts as "built-up" on a date when its fused (optical AND SAR) score
# reaches this level. Change = a transition of this binary state between dates.
BUILTUP_SCORE_THRESHOLD: float = 0.5

# Spatial cleanup: a changed pixel must have at least this many like-changed
# neighbors (of 8) to survive, removing salt-and-pepper false positives.
MIN_CHANGE_NEIGHBORS: int = 3

# Built-up LOSS is not yet reliable (sparse small Overture footprints vs 10m
# pixels + positional offset). Disabled by default; only GAIN is reported for now.
DETECT_LOSS: bool = False

# Report only GAIN clusters at least this many connected pixels (8-connectivity).
# 10 px @10m ~= 0.10 ha. Lowered from 25 once precision was validated (7-9x enrichment
# vs Open Buildings), to recover recall on smaller new developments.
MIN_GAIN_CLUSTER_PIXELS: int = 10

# --- Demo AOI ---------------------------------------------------------------

# Validation AOI: ~4 sq km over the Ruiru / Tatu City peri-urban corridor,
# Kiambu County, Kenya (NE of Nairobi) - a known zone of rapid recent urban
# expansion. The AOI is just a default; any bbox under MAX_BBOX_AREA_KM2 works.
DEMO_AOI_NAME: str = "Ruiru / Tatu City corridor, Kiambu, Kenya"
DEMO_BBOX: tuple[float, float, float, float] = (36.9510, -1.1580, 36.9700, -1.1400)

# Named test AOIs (all in Google Open Buildings coverage) for multi-site checks.
# Each is a documented zone of rapid recent expansion; UTM zone is derived per AOI.
TEST_AOIS: dict[str, tuple[float, float, float, float]] = {
    "ruiru_kenya": (36.9510, -1.1580, 36.9700, -1.1400),
    "new_cairo_egypt": (31.6800, 29.9850, 31.7050, 30.0030),
    "dhaka_purbachal_bd": (90.4900, 23.8050, 90.5120, 23.8230),
}

# Larger city regions to SCAN (with the evidence scanner) for the sub-AOI with the
# most built-up change; coordinates need only contain the city, not a precise spot.
SCAN_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "nairobi_ne": (36.8800, -1.2000, 36.9900, -1.1000),
    "greater_dhaka": (90.3500, 23.7000, 90.5200, 23.8800),
    "lagos_lekki": (3.5000, 6.4300, 3.6200, 6.4900),
    "bengaluru_east": (77.6800, 12.9300, 77.7800, 13.0100),
    "kampala_ug": (32.5800, 0.2800, 32.6800, 0.3800),
}

# --- Products + catalog (v0.2) ----------------------------------------------

# Latest annual layer published in Google Open Buildings 2.5D Temporal.
OPEN_BUILDINGS_LATEST_YEAR: int = 2023

# Root of the local STAC catalog that city-request products are written into.
PRODUCTS_DIR: Path = Path(
    os.environ.get("BUILTUP_MCP_PRODUCTS_DIR", str(Path.home() / ".cache" / "builtup-mcp" / "products"))
)
# Composite cache (re-runs and demos skip the network).
CACHE_DIR: Path = Path(
    os.environ.get("BUILTUP_MCP_CACHE_DIR", str(Path.home() / ".cache" / "builtup-mcp" / "composites"))
)
PRODUCT_VERSION: str = "0.2.0"
