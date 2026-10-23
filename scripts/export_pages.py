"""Publish the product explorer and a light copy of the catalog to docs/ (GitHub Pages).

    uv run python scripts/export_pages.py

Copies ``app/index.html`` to ``docs/index.html`` and, for the newest product of
each request, the files the explorer reads into ``docs/products/``: the STAC
item and summary, gain-epoch COG, new built-up polygons and H3 statistics
(GeoJSON, coordinates rounded to ~1 m), per-season masks, and quicklooks
re-encoded as WebP. The large assets (probability COG, GeoParquet, footprint
ages) stay local; the published item lists only what is published.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from builtup_mcp import config

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SRC = config.PRODUCTS_DIR / "builtup-change"
DST = DOCS / "products"
KEEP_ASSETS = {"gain_epoch", "gain_polygons_geojson", "summary"}


def _round_geojson(src: Path, dst: Path, digits: int = 5) -> None:
    fc = json.loads(src.read_text())

    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c, list) and c and isinstance(c[0], list) else [round(v, digits) for v in c]

    for f in fc.get("features", []):
        g = f.get("geometry")
        if g:
            g["coordinates"] = rnd(g["coordinates"])
    dst.write_text(json.dumps(fc, separators=(",", ":")))


def newest_items() -> list[Path]:
    """Newest product per request (run ids sort by time)."""
    newest: dict[str, Path] = {}
    for d in sorted(p for p in SRC.iterdir() if (p / "summary.json").exists()):
        req = json.loads((d / "summary.json").read_text())["request"]
        newest[req["request_id"]] = d
    return list(newest.values())


def public_request(s: dict) -> dict:
    """Drop local file paths from the echoed request; the AOI polygon itself is
    already in ``aoi_geometry`` and the STAC item geometry."""
    raw = s.get("request", {}).get("raw", {})
    aoi = raw.get("aoi") or {}
    if "path" in aoi:
        raw["aoi"] = {"geometry": "see request.aoi_geometry"}
    opts = raw.get("options") or {}
    if isinstance(opts.get("footprints"), str) and opts["footprints"] != "overture":
        opts["footprints"] = "requester file"
    if "footprint_baseline" in s and "requester footprints" in str(s["footprint_baseline"].get("source", "")):
        s["footprint_baseline"]["source"] = "requester footprints"
        s.get("model", {})["label_source"] = "requester footprints"
    return s


def main() -> None:
    shutil.copy(ROOT / "app" / "index.html", DOCS / "index.html")
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "builtup-change").mkdir(parents=True)
    links = []
    for d in newest_items():
        out = DST / "builtup-change" / d.name
        out.mkdir()
        s = public_request(json.loads((d / "summary.json").read_text()))
        webp = []
        for name in s.get("quicklooks", []):
            if not name:
                webp.append("")
                continue
            w = name.replace(".png", ".webp")
            Image.open(d / name).save(out / w, "WEBP", quality=78, method=6)
            webp.append(w)
        s["quicklooks"] = webp
        for name in s.get("built_masks", []) + [p.name for p in d.glob("static_*.png")]:
            shutil.copy(d / name, out / name)
        shutil.copy(d / "gain_epoch.tif", out / "gain_epoch.tif")
        _round_geojson(d / "gain_polygons.geojson", out / "gain_polygons.geojson")
        _round_geojson(d / "h3_stats.geojson", out / "h3_stats.geojson")
        (out / "summary.json").write_text(json.dumps(s, indent=1))

        item = json.loads((d / f"{d.name}.json").read_text())
        item["properties"].pop("builtup:label_source", None) if "requester" in str(item["properties"].get("builtup:label_source", "")) else None
        item["assets"] = {k: v for k, v in item["assets"].items() if k in KEEP_ASSETS}
        item["links"] = [
            {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
            {"rel": "collection", "href": "../collection.json", "type": "application/json"},
            {"rel": "parent", "href": "../collection.json", "type": "application/json"},
        ]
        (out / f"{d.name}.json").write_text(json.dumps(item, indent=1))
        links.append({"rel": "item", "href": f"./{d.name}/{d.name}.json", "type": "application/geo+json"})
        print(f"{s['city']}: {sum(f.stat().st_size for f in out.iterdir()) / 1e6:.1f} MB")

    coll = json.loads((SRC / "collection.json").read_text())
    coll["links"] = [
        {"rel": "root", "href": "../catalog.json", "type": "application/json"},
        {"rel": "parent", "href": "../catalog.json", "type": "application/json"},
    ] + links
    (DST / "builtup-change" / "collection.json").write_text(json.dumps(coll, indent=1))
    cat = json.loads((config.PRODUCTS_DIR / "catalog.json").read_text())
    cat["links"] = [
        {"rel": "root", "href": "./catalog.json", "type": "application/json"},
        {"rel": "child", "href": "./builtup-change/collection.json", "type": "application/json"},
    ]
    (DST / "catalog.json").write_text(json.dumps(cat, indent=1))
    print("published to", DST)


if __name__ == "__main__":
    main()
