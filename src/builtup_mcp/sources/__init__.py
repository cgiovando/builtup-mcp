"""Pluggable data-source adapters for builtup-mcp.

Each adapter fetches one input layer behind a stable interface so the pipeline
can swap implementations (e.g. Sentinel-1 -> NISAR) without downstream changes.
"""
