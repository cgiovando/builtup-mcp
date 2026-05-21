# Methodology

`builtup-mcp` produces an on-demand **built-up change signal** (where built-up area has increased)
over small bounding boxes, at higher temporal cadence than published epoch products such as GHSL or
the World Settlement Footprint. It is designed for new-construction detection and, in future, disaster
damage assessment. This document describes the data, the processing pipeline, the change-detection
logic, and how the prototype was validated.

> Status: research prototype. The current detector is a transparent spectral + radar heuristic that
> stands in for a future learned model. See [Limitations](#limitations) and [Roadmap](#roadmap).

## Design principles

- **Cloud-native and lean.** Everything is read on demand from public cloud sources with windowed
  reads (only the pixels overlapping the AOI are fetched). Nothing is bulk-downloaded.
- **Swappable adapters.** Data sources (vector, optical, SAR, reference) and the change detector sit
  behind stable interfaces, so individual components can be replaced (e.g. Sentinel-1 -> NISAR, or the
  heuristic -> a learned model) without changing the pipeline or output schema.
- **Honest, time-qualified outputs.** A "change" statement is meaningless without a known baseline
  date, so the baseline age is always reported (see [Baseline modes](#baseline-modes)).

## Data sources

| Role | Source | Access | Resolution | License |
| --- | --- | --- | --- | --- |
| Optical | Sentinel-2 L2A (`sentinel-2-l2a`) | Microsoft Planetary Computer STAC | 10-20 m | open (Copernicus) |
| SAR | Sentinel-1 RTC (`sentinel-1-rtc`) | Microsoft Planetary Computer STAC | ~10 m | open (Copernicus) |
| Vector baseline | Overture Maps buildings | Public S3 GeoParquet (DuckDB) | n/a | ODbL |
| Validation reference | Google Open Buildings 2.5D Temporal | Public GCS COGs | ~4 m | CC-BY-4.0 / ODbL |

All reads are anonymous; no credentials are required.

## Pipeline

1. **Input validation.** A WGS84 bbox (`min_lon, min_lat, max_lon, max_lat`) and date(s). The bbox is
   capped (default 5 sq km) to keep iteration fast; coordinates and ordering are validated, and the
   area is estimated with an equirectangular approximation.
2. **Vector baseline (optional).** Overture building footprints intersecting the bbox are fetched with
   DuckDB, filtering on Overture's `bbox` struct for predicate pushdown. Per-feature
   `sources[].update_time` is read to characterise the baseline's data freshness.
3. **Cloud-native EO reads.** For each date, the least-cloudy Sentinel-2 scene over the AOI is selected
   (by reading the SCL scene-classification band, not just scene metadata), with a temporal fallback
   that widens the search window. Sentinel-1 is selected by closest acquisition that fully covers the
   AOI. NIR (B08), SWIR (B11), SCL, and SAR VV/VH are read in a single windowed call onto a common 10 m
   grid in the scene's native UTM zone.
4. **Alignment.** All rasters are aligned to the Sentinel-2 grid (Sentinel-1 is reprojected onto it).
   Overture footprints are reprojected into the same grid and rasterised when used as a baseline.
5. **Change detection.** See below.
6. **Output.** Changed pixels are vectorised to point centroids and written as GeoJSON (WGS84) with
   attributes (`change_type`, `confidence`, contributing signal values, cluster id/size), plus a concise
   text summary.

## Change detection

The current detector is a deliberately simple, transparent placeholder for a future learned model.

- **Built-up score.** Per date, a 0-1 score is computed as the **minimum** of a normalised NDBI signal
  (optical: `(SWIR - NIR) / (SWIR + NIR)`) and a normalised Sentinel-1 VV backscatter (dB) signal.
  Using the minimum means a pixel only scores high where optical **and** radar agree, which is the main
  lever against single-sensor noise.
- **State transition.** A pixel is "built-up" on a date when its score crosses a threshold. **Gain** is
  a transition from not-built-up to built-up between the two reference points; this is far less noisy
  than thresholding a raw index difference.
- **Spatial cleanup.** Isolated pixels are removed (minimum-neighbour filter), and only connected
  **gain clusters** above a minimum size are reported (8-connectivity). This focuses on coherent new
  developments and suppresses speckle.
- **Loss** (built-up disappearing) is implemented but **disabled by default**, because at 10 m against
  sparse/old vector baselines it is not yet reliable.

These thresholds live in `config.py` and are intended to be tuned or replaced.

## Baseline modes

The "before" reference is configurable (option C):

- **Pre/post pair** (`baseline_date` supplied): two Sentinel acquisitions. The baseline age is *exactly
  known* (the chosen dates), e.g. "change over 2018-2023". This is the recommended, unambiguous mode.
- **Single date vs reference** (`baseline_date` omitted): the target-date built-up signal is compared
  against the rasterised Overture baseline. Because Overture footprints come from mixed sources (OSM,
  ML) with varying age, the pipeline reports the **data-freshness distribution** from
  `sources[].update_time` (median, oldest, newest), so the change can be time-qualified rather than
  presented as if the baseline were current.

## Validation

Detected gains are validated against **Google Open Buildings 2.5D Temporal**, which provides annual
`building_presence` layers (2016-2023). This is an independent, model-derived reference; the detector
uses only Sentinel data, so the comparison is genuinely out-of-sample.

- **Evidence-driven AOI selection.** Rather than guessing locations, a scanner reads Open Buildings
  over a city region and selects the ~4 sq km window with the most newly-built pixels (presence rising
  from below 0.3 to at/above 0.5 between two years). The detector is then run on that confirmed-change
  AOI.
- **Metrics.** With a 1-pixel tolerance (to absorb the 10 m vs ~4 m georeferencing offset):
  - *Precision* = fraction of detected gain pixels that coincide with Open Buildings "new" pixels.
  - *Recall* = fraction of Open Buildings "new" pixels that are detected.
  - *Enrichment* = precision divided by the AOI base rate (how much better than chance).
- **Figures** overlay three classes on a faint ESRI World Imagery basemap: agreement (yellow),
  detected-only (red), and reference-only / undetected (cyan).

### Results (auto-selected hotspots, 2018 -> 2023)

| City | Gain px | Precision (OB-new) | Recall | Context |
| --- | --- | --- | --- | --- |
| Lagos (Lekki) | 4,364 | 81.5% | ~44% | large contiguous development |
| Greater Dhaka | 880 | 73.1% | ~18% | dense active expansion |
| Bengaluru (east) | 1,092 | 38.8% | ~13% | already-dense infill |
| Nairobi (NE) | 558 | 28.7% | ~15% | sparse expanding edge |

**Reading the results.** Precision is strong where genuine, substantial change occurs (73-82%), and
recall scales with how large and contiguous the change is (high in Lagos, low for dispersed growth).
Recall is structurally limited (~13-44%) by the conservative "both sensors must agree" design - which
is exactly what buys the high precision. Closing that gap without losing precision is the motivation
for a learned model.

## Limitations

- **Heuristic, not learned.** NDBI + backscatter thresholds are a stand-in; they conflate built-up with
  bare soil and active construction, and they cannot achieve both high precision and high recall.
- **Resolution.** Output is a 10 m pixel-level signal, not building footprints. Genuine information
  content from Sentinel-2 saturates around 3-4 m even with super-resolution, so finer grids would be
  interpolation, not measurement.
- **SAR is backscatter, not coherence.** The RTC product has no interferometric phase, so the SAR
  signal is a backscatter change, not coherence loss (the gold standard for damage mapping).
- **Loss detection** is not yet trustworthy and is disabled by default.
- **Coverage.** Google Open Buildings (validation) covers the Global South only; Overture height and
  freshness vary by region.

## Roadmap

- Replace the heuristic with a learned built-up/change model (e.g. an Earth-observation foundation
  backbone) behind the same detector interface, to lift recall while preserving precision.
- Add an SLC SAR source for true coherence-based change/damage detection.
- Add a NISAR `SARSource` adapter once calibrated, cloud-native data is available.
- Optional richer attributes (height, footprint geometry) as reliable sources emerge.
