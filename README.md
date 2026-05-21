# builtup-mcp

[![Explore the interactive map](docs/assets/banner.jpg)](https://cgiovando.github.io/builtup-mcp/)

*Detected built-up gain (yellow = confirmed by Google Open Buildings, red = detected only, cyan = reference only) over an evidence-selected hotspot in Lagos, 2018-2023. [Explore all four cities interactively.](https://cgiovando.github.io/builtup-mcp/)*

An on-demand [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that
detects **built-up change** (where built-up area has increased or decreased) over small,
user-defined bounding boxes, at higher temporal cadence than published epoch products
(GHSL, WSF). It targets new-construction detection and disaster-driven destruction mapping,
fusing SAR (Sentinel-1, all-weather) with optical (Sentinel-2) and using Overture buildings
as a baseline prior. Cloud-native data (STAC, COGs, GeoParquet) keeps local processing lean.

A single tool, `detect_builtup_change`, takes a `bbox`, a `target_date`, and an optional
`baseline_date`, and returns a local GeoJSON file plus a concise text summary. Height is not a
required output; the schema is extensible so future sources (e.g. NISAR) can add attributes.

## Status

Prototype, built in phases:

1. **Phase 1 - MCP server skeleton + input validation.** Done.
2. **Phase 2 - Overture vector baseline (DuckDB on S3 GeoParquet).** Done.
3. **Phase 3 - Cloud-native EO fetching: Sentinel-1 + Sentinel-2 (STAC, windowed reads).** Done.
4. **Phase 4 - Change detection (fused optical + SAR) and GeoJSON output.** Done.
5. **Phase 5 (roadmap) - real models / sources (EO foundation model, SLC coherence, NISAR adapter).** Planned.

The Phase 4 detector uses an NDBI + SAR-backscatter heuristic as a placeholder for a future learned
model; thresholds in `config.py` are tunable. Output is a GeoJSON of 10m pixel centroids tagged
`change_type` (gain/loss), `confidence`, and the contributing signal values.

See **[METHODOLOGY.md](METHODOLOGY.md)** for the full approach, change-detection logic, and validation
(including a four-city comparison against Google Open Buildings 2.5D).

## Interactive map

An interactive MapLibre explorer of the four validation AOIs (Lagos, Dhaka, Bengaluru, Nairobi) lives in
[`docs/`](docs/) and is published via GitHub Pages:

**https://cgiovando.github.io/builtup-mcp/**

Toggle layers (agreement / detected-only / reference-only), switch basemaps (ESRI imagery, ESRI
imagery with labels, OpenStreetMap), and jump between cities. Regenerate the data layers with:

```bash
uv run python scripts/export_web_layers.py        # writes docs/data/*.geojson
```

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync                                  # Phase 1 (just the MCP SDK)
uv sync --extra vector --extra raster    # later phases (geospatial stack)
```

## Usage

Run the server over stdio:

```bash
uv run builtup-mcp
```

Register it with an MCP client (e.g. Claude Desktop) by pointing the client at the
`builtup-mcp` command. The exposed tool:

| Tool | Inputs | Output |
| --- | --- | --- |
| `detect_builtup_change` | `bbox` = `[min_lon, min_lat, max_lon, max_lat]` (WGS84, < 5 sq km), `target_date` = `YYYY-MM-DD`, optional `baseline_date` = `YYYY-MM-DD` | Text summary + (Phase 4) a GeoJSON file of the built-up change signal |

If `baseline_date` is given, the tool compares that "before" date against `target_date`
(pre/post pair). If omitted, it compares the target date against a static reference baseline.

### Constraints

- Inputs must be WGS84 (EPSG:4326).
- BBox area is capped at **5 sq km** (`BUILTUP_MCP_MAX_BBOX_KM2`) to keep local iteration fast.

## Development

```bash
uv run pytest -q
```

## AI-assisted development

> This project was developed with significant assistance from AI coding tools.

- **[Claude Code](https://claude.ai/claude-code)** (Anthropic) - code generation, architecture, debugging, and documentation
- All functionality has been tested and verified to work as intended
- Features and infrastructure choices have been reviewed and approved by the maintainer

This disclosure follows emerging best practices for transparency in AI-assisted software development.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

Data and references retain their own licenses: Overture Maps (ODbL), Copernicus Sentinel-1/2 (open),
and Google Open Buildings 2.5D Temporal (CC-BY-4.0 / ODbL).
