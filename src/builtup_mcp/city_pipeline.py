"""City request -> built-up change product (v0.2).

    request -> windows -> S2/S1 composites (cached) -> weak labels from footprints
            -> per-request classifier (held-out accuracy) -> probability per window
            -> epoch of change -> COGs + GeoParquet + H3 stats + footprint age
            -> static-layer comparison -> STAC item in the local catalog

Every number a requester sees (areas, accuracy, data gaps, static-layer ages)
is written into ``summary.json`` and the STAC item properties.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config, product
from .processing.classify import building_fraction, predict_probability, train_models, weak_labels
from .processing.timeseries import epoch_change
from .request import ChangeRequest
from .sources.composite import Composite, build_composite, dominant_orbit_state
from .sources.raster_io import aoi_grid


@dataclass
class CityProduct:
    request: ChangeRequest
    item_path: Path
    out_dir: Path
    summary: dict = field(default_factory=dict)

    def text(self) -> str:
        s = self.summary
        lines = [
            f"{self.request.city}: built-up change product {s['item_id']} "
            f"({s['aoi_km2']} sq km, {len(self.request.windows)} windows "
            f"{self.request.windows[0][0]} .. {self.request.windows[-1][1]}).",
            f"Built-up in first window ({s['window_labels'][0]}): {s['built_first_ha']} ha; "
            f"{s['latest_confirmed_window']}: {s['built_latest_ha']} ha; "
            f"new built-up detected: {s['gain_ha_total']} ha "
            f"(plus {s['provisional_gain_ha']} ha provisional, seen only in the latest window).",
            "New built-up by window: "
            + ", ".join(f"{k} {v} ha" for k, v in s["gain_ha_by_window"].items())
            + ".",
        ]
        acc = s.get("accuracy", {})
        best = next((k for k, m in acc.items() if m), None)
        if best:
            m = acc[best]
            if m:
                lines.append(
                    f"Held-out agreement with footprint labels ({best} model): "
                    f"F1 {m.get('f1')}, precision {m.get('precision')}, recall {m.get('recall')}."
                )
        gaps = [w["label"] for w in s["windows"] if not w["sentinel-1"]]
        if gaps:
            lines.append("Sentinel-1 unavailable (optical-only) for: " + ", ".join(gaps) + ".")
        for lay in s.get("static_layers", []):
            if "latest_year" in lay:
                lines.append(
                    f"{lay['layer']}: latest epoch {lay['latest_year']}, {lay['built_ha']} ha built."
                )
        if s.get("footprint_baseline"):
            fb = s["footprint_baseline"]
            lines.append(
                f"Footprint baseline: {fb['n_footprints']} footprints, median source update "
                f"{fb.get('median_update')}; {fb.get('n_unmapped_gain_ha', 0)} ha of new built-up "
                "has no footprint yet."
            )
        lines.append(f"STAC item: {self.item_path}")
        return " ".join(lines)


def _cache_key(req: ChangeRequest, window, grid, orbit) -> str:
    raw = json.dumps(
        [
            req.bbox.as_tuple(), window, grid.epsg, grid.bounds, orbit, 3,
            config.COMPOSITE_MAX_S2_SCENES, config.COMPOSITE_MAX_S1_SCENES,
            config.MIN_SCENE_CLEAR_FRACTION, config.SCL_CLOUD_CLASSES,
        ],
        default=str,
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _load_or_build(req: ChangeRequest, grid, window, orbit, use_sar: bool, log) -> Composite:
    import os
    import zipfile

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CACHE_DIR / f"{_cache_key(req, window, grid, orbit)}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                bands = {k: z[k] for k in z.files if not k.startswith("__")}
                meta = json.loads(str(z["__meta"]))
            log(f"  {window}: cached")
            return Composite(window=tuple(window), grid=grid, bands=bands, scenes=meta["scenes"], meta=meta["meta"])
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            log(f"  {window}: cache unreadable, rebuilding")
    t0 = time.time()
    c = build_composite(req.bbox, grid, tuple(window), orbit_state=orbit, use_sar=use_sar)
    failed = c.meta.get("s2_failed", 0) + c.meta.get("s1_failed", 0)
    tried = c.meta.get("s2_candidates", 0) + c.meta.get("s1_candidates", 0)
    # A scene or two that never reads (a corrupt asset) should not block caching
    # forever; a throttled burst (many failures) should.
    if failed > max(1, 0.1 * tried):
        # Do not freeze a transient outage into the cache; the next run retries.
        log(f"  {window}: {failed} scene reads failed; composite not cached")
    else:
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **c.bands, __meta=np.array(json.dumps({"scenes": c.scenes, "meta": c.meta})))
        os.replace(tmp, path)
    log(f"  {window}: S2 {c.meta.get('s2_used', 0)} scenes, S1 {c.meta.get('s1_used', 0)} ({time.time() - t0:.0f}s)")
    return c


def run_city_request(
    req: ChangeRequest,
    *,
    products_dir: Path | None = None,
    footprints=None,
    log=None,
) -> CityProduct:
    import sys

    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    t_start = time.time()
    root = products_dir or config.PRODUCTS_DIR
    grid = aoi_grid(req.bbox, config.TARGET_RESOLUTION_M)
    aoi = product.aoi_mask(req.aoi_geometry, grid)
    use_sar = req.options.get("use_sar", True)
    orbit = dominant_orbit_state(req.bbox, req.windows[0][0], req.windows[-1][1]) if use_sar else None
    log(f"{req.city}: grid {grid.width}x{grid.height} EPSG:{grid.epsg}, {len(req.windows)} windows, S1 orbit {orbit}")

    comps = [_load_or_build(req, grid, w, orbit, use_sar, log) for w in req.windows]

    # Footprints: Overture by default, or a requester-supplied file (e.g. a
    # best-available-footprints product) for both labels and the age join.
    from .sources.overture import baseline_age, fetch_buildings, resolve_release

    fp_source = req.options.get("footprints", "overture")
    if footprints is None:
        if fp_source == "overture":
            footprints = fetch_buildings(req.bbox)
            fp_desc = f"Overture {resolve_release()} buildings"
        else:
            import geopandas as gpd

            footprints = gpd.read_file(fp_source).to_crs(4326)
            fp_desc = f"requester footprints ({Path(fp_source).name})"
    else:
        fp_desc = "supplied footprints"
    if "source_update_time" not in footprints and "footprint_date" in footprints:
        footprints = footprints.assign(source_update_time=footprints["footprint_date"].astype(str))
    age = baseline_age(footprints, req.windows[-1][1]) if len(footprints) else {"n_dated": 0}
    label_date = age.get("median_update")

    frac = building_fraction(footprints, grid)
    labels = weak_labels(frac, grid.resolution_m)
    labels[~aoi] = -1
    from .processing.classify import far_from_footprints

    far = far_from_footprints(frac, grid.resolution_m, config.LATE_NEGATIVE_BUFFER_M) & aoi
    models = train_models(comps, labels, label_date, far_mask=far)
    if not models.models:
        raise RuntimeError("Too few footprint labels in the AOI to train a model.")
    log(f"  models: {list(models.models)}; held-out: {models.metrics}")

    probs, sources = zip(*(predict_probability(models, c) for c in comps))
    probs = np.stack(probs)
    probs[:, ~aoi] = np.nan
    # The rolling "latest" window (and a current season clipped to today) is
    # the least comparable: usually off-season or thinly observed. It is kept out
    # of change dating entirely, so it cannot erase gains confirmed in earlier
    # seasons; it can only add provisional detections, flagged as such.
    last_label = req.window_labels[-1]
    has_latest = last_label.startswith("latest") or "partial" in last_label
    last_k = len(req.windows) - 2 if has_latest else len(req.windows) - 1
    ec = epoch_change(probs[: last_k + 1], aoi, detect_loss=req.options.get("detect_loss", False))
    gain_epoch = ec.gain_epoch.copy()
    confidence = ec.confidence.copy()
    gain_ok = (gain_epoch > 0) & (gain_epoch != 255)
    built_conf = ec.built_first | gain_ok
    if has_latest:
        from .processing.change import _despeckle, keep_large_clusters
        from .processing.timeseries import keep_cored_clusters

        p_last = probs[-1]
        prov = (p_last >= config.PROB_BUILT) & ~built_conf & aoi
        prov = keep_cored_clusters(keep_large_clusters(_despeckle(prov, 2), config.MIN_GAIN_CLUSTER_PIXELS))
        gain_epoch[prov] = len(req.windows) - 1
        with np.errstate(invalid="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                before = np.nanmean(probs[: last_k + 1], axis=0)
        confidence[prov] = np.clip(np.nan_to_num(p_last - before), 0, 1)[prov]
        gain_ok = (gain_epoch > 0) & (gain_epoch != 255)

    # Built-up extent per window, monotone by construction (the WSF Tracker
    # convention): built from the first window and never observed as not built,
    # plus every gain dated to that window or earlier. Year-to-year noise in the
    # raw probabilities therefore cannot make a city shrink and regrow.
    built = np.stack(
        [(ec.built_first | (gain_ok & (gain_epoch <= t))) & aoi for t in range(probs.shape[0])]
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    item_id = f"{req.request_id}-{stamp}"
    out = root / product.COLLECTION_ID / item_id
    out.mkdir(parents=True, exist_ok=True)
    labels_w = req.window_labels
    px_ha = grid.resolution_m**2 / 1e4

    prob_u8 = [np.where(np.isfinite(p), np.round(p * 100), 255).astype("uint8") for p in probs]
    product.write_cog(out / "builtup_probability.tif", prob_u8, grid, nodata=255, dtype="uint8",
                      descriptions=[f"{l} ({a}..{b})" for l, (a, b) in zip(labels_w, req.windows)])
    product.write_cog(out / "gain_epoch.tif", [gain_epoch], grid, nodata=255, dtype="uint8",
                      descriptions=["window index of first built-up (0 = no gain)"])
    if req.options.get("detect_loss"):
        product.write_cog(out / "loss_epoch.tif", [ec.loss_epoch], grid, nodata=255, dtype="uint8",
                          descriptions=["window index of first not-built-up after built-up (experimental)"])

    polys = product.gain_polygons(gain_epoch, confidence, grid, labels_w, last_k)
    polys.to_parquet(out / "gain_polygons.parquet")
    polys.to_file(out / "gain_polygons.geojson", driver="GeoJSON")
    hexes = product.h3_stats(grid, aoi, built, gain_epoch, labels_w, last_k=last_k)
    hexes.to_parquet(out / "h3_stats.parquet")
    hexes.to_file(out / "h3_stats.geojson", driver="GeoJSON")

    fp_age = product.footprints_age(footprints, grid, gain_epoch, ec.built_first, labels_w)
    unmapped_gain = (gain_epoch > 0) & (gain_epoch <= last_k) & (frac <= 0)
    if fp_age is not None and len(fp_age):
        keep = [c for c in ("id", "integrated_id", "source_update_time", "builtup_since", "gain_epoch", "geometry") if c in fp_age]
        fp_age = fp_age[keep]
        fp_age["source_update_time"] = fp_age["source_update_time"].astype(str) if "source_update_time" in fp_age else None
        fp_age.to_parquet(out / "footprints_age.parquet")

    from .static_layers import static_snapshot

    statics = static_snapshot(req.bbox, grid, aoi, out_dir=out) if req.options.get("static_layers", True) else []
    quicklooks = product.write_quicklooks(out, comps, aoi)

    from . import benchmark

    bench: dict = {}
    if req.options.get("benchmark", True):
        try:
            wsf = benchmark.vs_wsf_tracker(grid, req.bbox, aoi, req.windows[: last_k + 1], built, gain_epoch)
            arr = wsf.pop("_wsf", None)
            if arr is not None:
                product.write_mask_png(out / "static_wsf-tracker.png", (arr >= 1) & aoi, (0, 191, 255, 200))
                statics.append({
                    "layer": "WSF Tracker (DLR/MindEarth), built by Jan 2026",
                    "source": "s3://us-west-2.opendata.source.coop/mindearth/wsf",
                    "latest_year": 2026, "latest_epoch": "2026-01-01",
                    "built_ha": round(float(((arr >= 1) & aoi).sum()) * px_ha, 1),
                    "license": "see DOI 10.5281/zenodo.20424537",
                })
            bench["wsf_tracker"] = wsf
        except Exception as exc:
            bench["wsf_tracker"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            bench["open_buildings"] = benchmark.vs_open_buildings(grid, aoi, req.windows[: last_k + 1], gain_epoch)
        except Exception as exc:
            bench["open_buildings"] = {"error": f"{type(exc).__name__}: {exc}"}
        log(f"  benchmark: {bench}")
    built_pngs = []
    for k in range(len(req.windows)):
        name = f"built_{k:02d}.png"
        product.write_mask_png(out / name, built[k], (255, 140, 0, 210))
        built_pngs.append(name)

    gain_by = {
        labels_w[k]: round(float((gain_epoch == k).sum()) * px_ha, 1)
        for k in range(1, len(labels_w))
    }
    win_info = [
        {
            "label": l,
            "start": a,
            "end": b,
            "sentinel-2": len(c.scenes.get("sentinel-2-l2a", [])),
            "sentinel-1": len(c.scenes.get("sentinel-1-rtc", [])) if c.has_sar else 0,
            "scored_by": {
                k: round(float(((s == code) & aoi).sum() / aoi.sum()), 3)
                for code, k in ((1, "fused"), (2, "optical"), (3, "sar"))
            },
            "built_ha": round(float(bt.sum()) * px_ha, 1),
            "scene_ids": c.scenes,
        }
        for l, (a, b), c, s, bt in zip(labels_w, req.windows, comps, sources, built)
    ]
    summary = {
        "item_id": item_id,
        "product_version": config.PRODUCT_VERSION,
        "generated_utc": stamp,
        "request": req.to_dict(),
        "city": req.city,
        "country": req.country,
        "aoi_km2": round(float(aoi.sum()) * px_ha / 100, 2),
        "grid": {"epsg": grid.epsg, "resolution_m": grid.resolution_m, "width": grid.width, "height": grid.height},
        "window_labels": labels_w,
        "windows": win_info,
        "built_first_ha": round(float(built[0].sum()) * px_ha, 1),
        "latest_confirmed_window": labels_w[last_k],
        "built_latest_ha": round(float(built[last_k].sum()) * px_ha, 1),
        "gain_ha_total": round(float(((gain_epoch > 0) & (gain_epoch <= last_k)).sum()) * px_ha, 1),
        "built_latest_window_ha": round(float(built[-1].sum()) * px_ha, 1),
        "gain_ha_by_window": gain_by,
        "provisional_gain_ha": round(
            float(((gain_epoch > last_k) & (gain_epoch != 255)).sum()) * px_ha, 1
        ),
        "accuracy": models.metrics,
        "accuracy_note": (
            "Agreement with footprint-derived labels on spatially held-out 500 m blocks, pooled "
            "over all windows. Labels are incomplete footprint maps, so this measures consistency "
            "with mapped buildings, not ground truth."
        ),
        "model": {
            "type": "HistGradientBoostingClassifier",
            "train_window": models.train_window,
            "n_labels": models.n_labels,
            "label_source": fp_desc,
            "thresholds": {"built": config.PROB_BUILT, "not_built": config.PROB_NOT_BUILT},
        },
        "footprint_baseline": {
            "source": fp_desc,
            "n_footprints": int(len(footprints)),
            **{k: v for k, v in age.items() if k != "n_total"},
            "n_unmapped_gain_ha": round(float(unmapped_gain.sum()) * px_ha, 1),
        },
        "static_layers": statics,
        "benchmark": bench,
        "quicklooks": quicklooks,
        "built_masks": built_pngs,
        "corners_wgs84": product.corner_coords(grid),
        "runtime_s": round(time.time() - t_start, 1),
    }
    product.write_json(out / "summary.json", summary)

    assets = {
        "builtup_probability": (out / "builtup_probability.tif", pystac_media("cog"), ["data"], "Built-up probability per window (0-100)"),
        "gain_epoch": (out / "gain_epoch.tif", pystac_media("cog"), ["data"], "Window of first built-up"),
        "gain_polygons": (out / "gain_polygons.parquet", pystac_media("parquet"), ["data"], "New built-up polygons"),
        "gain_polygons_geojson": (out / "gain_polygons.geojson", pystac_media("geojson"), ["data"], "New built-up polygons (GeoJSON)"),
        "h3_stats": (out / "h3_stats.parquet", pystac_media("parquet"), ["data"], "Per-hexagon statistics (H3 res 8)"),
        "summary": (out / "summary.json", pystac_media("json"), ["metadata"], "Provenance, accuracy and static-layer comparison"),
    }
    for k, q in enumerate(quicklooks):
        if q:
            assets[f"quicklook_{k:02d}"] = (out / q, "image/png", ["overview"], f"True colour composite, {labels_w[k]}")
    if (out / "footprints_age.parquet").exists():
        assets["footprints_age"] = (out / "footprints_age.parquet", pystac_media("parquet"), ["data"], "Footprints with first-detected window")
    if (out / "loss_epoch.tif").exists():
        assets["loss_epoch"] = (out / "loss_epoch.tif", pystac_media("cog"), ["data"], "Loss epoch (experimental)")
    props = {
        "title": f"{req.city} built-up change {req.windows[0][0][:4]}-{req.windows[-1][1][:4]}",
        "city": req.city,
        "country": req.country,
        "proj:code": f"EPSG:{grid.epsg}",
        "gsd": grid.resolution_m,
        "builtup:windows": [{"label": l, "start": a, "end": b} for l, (a, b) in zip(labels_w, req.windows)],
        "builtup:gain_ha_total": summary["gain_ha_total"],
        "builtup:accuracy": models.metrics,
        "builtup:label_source": fp_desc,
        "processing:software": {"builtup-mcp": config.PRODUCT_VERSION},
    }
    item_path = product.publish_item(
        root, item_id, req.aoi_geometry, list(req.bbox.as_tuple()), req.windows, props, assets
    )
    log(f"  done in {summary['runtime_s']}s -> {item_path}")
    return CityProduct(request=req, item_path=item_path, out_dir=out, summary=summary)


def pystac_media(kind: str) -> str:
    import pystac

    return {
        "cog": pystac.MediaType.COG,
        "parquet": "application/vnd.apache.parquet",
        "geojson": pystac.MediaType.GEOJSON,
        "json": pystac.MediaType.JSON,
    }[kind]
