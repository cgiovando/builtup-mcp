# Methodology

`builtup-mcp` produces on-demand **built-up change products** for a city or any area of interest: where built-up area appeared (and optionally disappeared), dated to the composite window in which it was first observed, at 10 m, up to the most recent satellite pass. It is designed for requests that annual or static global layers cannot answer: fast-growing towns, displacement settlements, and areas where the newest open layer predates the change of interest.

> Status: research prototype, version 0.2. Version 0.1 (single scenes, fixed NDBI and backscatter thresholds) is kept as the `detect_builtup_change` tool and as a baseline. See [Limitations](#limitations).

## Design principles

- **A request is the unit.** A request names an AOI, a period and the calendar months to compare; the product is built for that request and records it verbatim, so any run can be repeated.
- **Cloud-native, nothing bulk-downloaded.** Imagery is read with windowed COG reads from STAC catalogs; footprints are read from the Overture GeoParquet files that intersect the AOI only.
- **Accuracy travels with the product.** Every product carries its own held-out accuracy and a comparison against independent published layers over the period each covers.
- **Standard delivery.** Products are STAC items with COG, GeoParquet and JSON assets in a self-contained catalog, readable by QGIS, Python, R or any STAC client.

## Data sources

| Role | Source | Access | Resolution | License |
| --- | --- | --- | --- | --- |
| Optical | Sentinel-2 L2A (`sentinel-2-l2a`) | Microsoft Planetary Computer STAC | 10-20 m | open (Copernicus) |
| SAR | Sentinel-1 RTC (`sentinel-1-rtc`) | Microsoft Planetary Computer STAC | 10 m | open (Copernicus) |
| Labels and footprint baseline | Overture Maps buildings (latest release, resolved via the Overture STAC catalog) | Public GeoParquet (DuckDB) | vector | ODbL |
| Comparison | WSF Tracker (DLR / MindEarth), six-monthly to Jan 2026 | Public GeoZarr, Source Cooperative | 10 m | see dataset DOI 10.5281/zenodo.20424537 |
| Comparison | Google Open Buildings 2.5D Temporal, annual to 2023 | Public GCS | ~4 m effective | CC-BY-4.0 / ODbL |
| Comparison | Esri / Impact Observatory land cover (to 2023), ESA WorldCover (2020-2021) | Planetary Computer STAC | 10 m | CC-BY-4.0 |

All reads are anonymous; no credentials are required.

## Reference footprints: one source, no conflation step

The project does not combine footprint datasets itself. The only footprints it reads are the Overture Maps buildings theme, latest release, for the AOI. Overture already conflates OpenStreetMap with machine-learned footprints (Microsoft, Google Open Buildings and others) and publishes the result with a per-feature source and update date, so the merge, de-duplication and source priority are Overture's, not this project's.

Those footprints are used twice: as training labels (rasterised to building-cover fraction, step 4 below) and as the footprint baseline that each product dates by season (`footprints_age`). Google Open Buildings Temporal and WSF Tracker are only benchmarks; they are never merged into the labels, so the comparison against them stays independent.

A separate merge step was not needed for labels: a pixel counts as built-up at 25% building cover and as not built only beyond 40 m from any footprint, which tolerates the offsets and partial duplicates that conflation resolves. A conflated set with more complete coverage (for example one that adds a height-bearing dataset to fill gaps) can still help as a footprint baseline, and is a planned input.

## Pipeline

1. **Request and windows.** A request gives a bbox or polygon, a year range and `season_months`. One composite window is built per year over the same calendar months (for example December-March, the dry season in South Sudan), so seasonal vegetation and soil moisture are not mistaken for change. An optional rolling window covers the last N days to give the latest picture.
2. **Composites.** For each window the clearest Sentinel-2 scenes (up to 16 per MGRS tile, so AOIs across tile edges are covered on both sides) are warped onto one 10 m UTM grid; cloud, shadow, cirrus and snow pixels are masked with the scene classification band, and the per-band median is taken. Scenes from processing baseline 04.00 onward (from 25 Jan 2022) have the +1000 reflectance offset removed, otherwise indices drift across 2022. Sentinel-1 acquisitions from one orbit direction are reduced to the median and standard deviation of VV and VH backscatter in dB. Built structures are bright and temporally stable in C-band, which separates them from bare soil, the main confusion of optical built-up indices.
3. **Features.** NDBI, NDVI, MNDWI, a bare-soil index, brightness, the SWIR2/SWIR1 ratio and local texture (5x5 standard deviation of NIR and NDVI) from Sentinel-2; VV and VH medians, their difference, their temporal standard deviation and VV texture from Sentinel-1.
4. **Per-request classifier.** The AOI's own Overture footprints are rasterised to building-cover fraction per 10 m pixel. Pixels with at least 25% cover are positive labels; pixels more than 40 m from any footprint are negative; everything in between is ignored. Samples are pooled across windows so the model learns built-up across years, sensors (Sentinel-1A vs 1C, Sentinel-2A/B/C) and wet or dry years: positives come only from windows ending no more than three years before the footprints' median update date, and negatives from every window. Negatives after the mapping date are noisy (growth since then is labelled "not built"), but dropping them is worse: the model then never sees post-mapping radiometry (new sensors, wet years) for bare ground. On Malakal a one-year cut-off produced almost twice the new built-up of the pooled rule and lower agreement with WSF Tracker, so the pooled rule is the default (`NEGATIVE_LOOKAHEAD_YEARS` keeps the alternative testable). A gradient-boosted classifier is fitted three ways where the data allow: fused (optical and SAR), optical-only and SAR-only. Every pixel of every window is scored by the richest model its data support, so cloud gaps and Sentinel-1 gaps degrade the product instead of stopping it. Windows with fewer than four Sentinel-1 acquisitions are treated as SAR gaps.
5. **Held-out accuracy.** Before the final fit, 20% of 500 m spatial blocks are withheld and the model is scored on them at the product's own built-up threshold (p >= 0.6), with the share of pixels falling in the dead band reported alongside. Because the labels come from incomplete footprint maps, this measures agreement with mapped buildings, not ground truth.
6. **Epoch of change.** Each window's probability becomes a state with a dead band: built (p >= 0.6), not built (p <= 0.35) or unknown. A pixel's gain is dated to the first built window after its last not-built window. This rule is persistent by construction (after the gain the pixel is never again observed as not built) and needs no tuning. Isolated pixels, clusters under 10 pixels (0.1 ha) and linear features under 30 m wide (mostly graded roads) are removed. The rolling latest window (and a current season clipped to today) is kept out of change dating, so it cannot erase gains confirmed in earlier seasons; it only adds provisional detections, because it is usually off-season or thinly observed. Loss (the mirror rule) is available on request and is experimental.
7. **Products.** Built-up probability per window (multi-band COG), gain epoch (COG), new built-up polygons with first-seen window, area and confidence (GeoParquet and GeoJSON), H3 resolution-8 neighbourhood statistics, the footprints with the window in which each first read as built-up, the static-layer comparison, the benchmark, and a `summary.json` holding provenance (scene ids per window, Overture release, model and thresholds). A STAC item ties them together.

## Model choice

The classifier is supervised, but nobody draws labels: they come automatically from the AOI's own open footprints, so a new city needs no annotation.

| Option considered | Outcome |
| --- | --- |
| Fixed index thresholds (NDBI and VV backscatter), the v0.1 method | Replaced. No training, but thresholds tuned on one city do not transfer, and requiring both sensors capped recall at 13-44% (see [Version 0.1 validation](#version-01-validation-kept-for-reference)). |
| Off-the-shelf global built-up classifiers (Dynamic World built class, EMC-BUILT, WSF) | Used as comparisons, not as the engine. They are fixed products or models: optical only (Dynamic World), biennial (EMC-BUILT) or published on the producer's schedule (WSF Tracker), and none reports accuracy for the requested city. |
| Deep segmentation or fine-tuned EO foundation models (Prithvi, DOFA via TerraTorch) | Not used yet. They need a GPU and fine-tuning per region; the gain over hand-made features at 10 m has not been shown for this task. |
| Foundation-model embeddings as features (OlmoEarth) | Next experiment ([#7](https://github.com/cgiovando/builtup-mcp/issues/7)): replace the hand-made features in step 3, keep the per-request classifier and its accuracy reporting. |
| **Gradient-boosted trees trained per request** (scikit-learn `HistGradientBoostingClassifier`) | **Used.** Trains in seconds on a laptop CPU, is fitted in fused, optical-only and radar-only variants so data gaps degrade the product instead of stopping it, and gives a held-out accuracy for the place requested. |

WSF products use the same family of model: WSF 2019 is a Random Forest on Sentinel-1 and Sentinel-2 temporal features, trained once for the globe (Marconcini et al.). The WSF Tracker data are public on Source Cooperative; its processing code is not published. The difference here is not the algorithm but where it is trained (on each request's own footprints), when it runs (on request, up to the latest pass) and what ships with it (accuracy and benchmarks for that area).

## Benchmarks run with every product

- **WSF Tracker.** Built-up extent at the last composite window ending by March 2026, against WSF epochs up to that window's end date, and new built-up between the first window and the same matched date, pixel by pixel with a one-pixel tolerance.
- **Google Open Buildings 2.5D Temporal.** New built-up between the first window's year and 2023 (presence rising from below 0.3 to at least 0.5).

Both references are model outputs. Agreement is evidence of consistency; disagreement shows where a reference sample (for example very-high-resolution imagery) is needed before figures are used for decisions.

## Results (version 0.2, October 2026)

Dry-season composites, 2019 to 2026. Agreement is pixel F1 with a one-pixel tolerance on the same 10 m grid.

| | Malakal | Renk | Juba | Kigali |
| --- | --- | --- | --- | --- |
| Area, season | 66 sq km, Dec-Mar | 77 sq km, Dec-Mar | 66 sq km, Dec-Mar | 69 sq km, Jun-Aug |
| Built-up, first to last season (ha) | 760 to 908 | 847 to 1,115 | 4,103 to 4,724 | 4,289 to 4,549 |
| New built-up, confirmed / provisional (ha) | 148 / 105 | 268 / 178 | 620 / 105 | 260 / 0 |
| Held-out F1 vs footprint labels (fused model) | 0.91 | 0.96 | 0.96 | 0.97 |
| WSF Tracker: extent agreement | 0.76 | 0.88 | 0.97 | 0.96 |
| WSF Tracker: new built-up, ours / WSF (ha) and agreement | 148 / 96, 0.21 | 268 / 53, 0.36 | 620 / 169, 0.42 | 210 / 108, 0.49 |
| Open Buildings Temporal: new built-up 2019-2023, ours / OB (ha) and agreement | 66 / 7, 0.07 | 86 / 5, 0.04 | 373 / 175, 0.20 | 140 / 82, 0.24 |
| Seasons without usable Sentinel-1 | 2022-2025 | 2022-2025 | none | none |

Both products find the towns in the same place and show similar growth trends, but they rarely flag the same pixels as new. In Malakal, image checks found errors on both sides: WSF Tracker caught a new shelter block by the river that this product had already scored as built-up in 2019 (bright bare ground), and this product caught dispersed infill that WSF Tracker missed. Settling which is right needs an independent reference sample.

Open Buildings Temporal sees very little new building in the South Sudan towns (5-7 ha), because it measures roof presence at about 4 m and small tukuls and shelters rarely reach its 0.5 presence threshold; agreement there says more about the reference than about this product.

Static layers over the same areas (newest epoch, ha built): Malakal 254 (ESA WorldCover 2021), 45 (Open Buildings 2023), 905 (Esri / Impact Observatory 2023), 712 (WSF Tracker Jan 2026). They answer different questions (roof presence, land-cover class, settlement extent) at different dates.

## Reporting growth in buildings and in areas to check

The product measures built-up surface in hectares, but two of its outputs translate it into the units that building-inventory work reports:

- **Building counts.** `footprints_age` gives each footprint the season in which its pixel first read as built-up, so growth can be counted in buildings against a citywide footprint baseline. Over the Malakal request area, 618 of the 18,519 Overture footprints sit in pixels first read as built-up in a dry season from 2020 to 2026, and 178 more only in the provisional 2026 rainy-season window.
- **Areas the footprint baseline misses.** New built-up with no footprint under it is reported as `n_unmapped_gain_ha`: 234 ha of the 243 ha confirmed new built-up in the Malakal request area. That is either growth the footprint maps have not caught up with or a false detection (bright bare ground is the known one), and either way it is where imagery should be checked first. Intersecting the gain polygons with the footprints gives its extent, so it can be expressed as square kilometres of imagery to task.

## Cost per request

All inputs are open and read anonymously, so the cost of a request is compute time. Measured run times (`runtime_s` in each product's `summary.json`) on a 10-core laptop with 16 GB of memory, for areas of 66 to 106 sq km and eight dry seasons from 2019 to 2026:

| Run | Time |
| --- | --- |
| First run, composites built from scratch (Renk, Malakal request area, Kigali) | 8.5 to 12 minutes |
| Re-run with cached composites (Malakal, Juba) | 1 to 2 minutes |

On a general-purpose cloud VM of similar size (8 vCPU, 32 GB, about $0.40 an hour on demand), a first run costs under $0.10 in compute; a city request costs well under $1 even with several re-runs. The costs that matter are elsewhere: a very-high-resolution reference sample to validate the figures, and analyst time to review them.

## Version 0.1 validation (kept for reference)

The v0.1 heuristic (single scenes, NDBI and VV thresholds, both sensors required) was validated against Google Open Buildings 2.5D Temporal on auto-selected 4 sq km hotspots, 2018 to 2023, with a one-pixel tolerance:

| City | Gain px | Precision (OB-new) | Recall |
| --- | --- | --- | --- |
| Lagos (Lekki) | 4,364 | 81.5% | ~44% |
| Greater Dhaka | 880 | 73.1% | ~18% |
| Bengaluru (east) | 1,092 | 38.8% | ~13% |
| Nairobi (NE) | 558 | 28.7% | ~15% |

Recall was structurally limited by the fixed both-sensors rule, which motivated the v0.2 classifier. Those runs also predate the processing-baseline offset fix, which biased post-2022 NDBI downward.

## Limitations

- **Labels are footprints, not ground truth.** Where footprint maps are sparse or old, the classifier learns from what is mapped. Held-out scores measure consistency with those maps, and the labels inherit Overture's coverage and its choice between overlapping sources.
- **No independent accuracy yet.** Every benchmark so far is against another model's output. Agreement on extent is high (0.73 to 0.97 F1), agreement on which pixels are new is low (0.15 to 0.49), and only a very-high-resolution reference sample can say which product is right where they differ ([#1](https://github.com/cgiovando/builtup-mcp/issues/1)).
- **Season of first detection, not building age.** A gain is dated to the first composite window that reads it as built-up; structures that predate 2019 have no age, and a building replaced in place is not a gain.
- **Bright bare ground.** Graded plots, dry riverbanks and some wetlands can read as built-up ([#2](https://github.com/cgiovando/builtup-mcp/issues/2)).
- **10 m, not buildings.** The signal is built-up surface at 10 m, suited to neighbourhood and block scale. Individual small structures (for example tukuls or shelters) are detected as clusters, not counted.
- **Season.** Comparisons are only as clean as the season choice. Off-season windows (wet season, flooding) change surface reflectance and backscatter; gains first seen there are provisional.
- **Sentinel-1 gaps.** After Sentinel-1B failed in December 2021, parts of Africa had few or no acquisitions until Sentinel-1C (2025). Those windows are scored optical-only and flagged.
- **Backscatter, not coherence.** RTC has no phase. Coherence-based damage mapping needs SLC data.
- **References end early.** Open Buildings Temporal ends in 2023 and WSF Tracker in January 2026, so 2026 change has no independent reference yet.
- **Infrastructure dependency.** Planetary Computer is a free service without an SLA. The STAC reader is generic, so another catalog with the same collections can be substituted.

## Roadmap

- Reference samples from very-high-resolution imagery for per-request validation.
- NISAR L-band as a third sensor once validated products are released.
- Coherence from Sentinel-1 SLC for damage mapping.
- A fallback STAC catalog (Copernicus Data Space or Earth Search) behind the same reader.
