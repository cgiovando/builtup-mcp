"""Tests that the MCP server exposes the tool with a correct schema (Phase 1)."""

from __future__ import annotations

import asyncio

from builtup_mcp.server import detect_builtup_change, mcp

SMALL_BBOX = [-122.4200, 37.7740, -122.4143, 37.7785]


def test_tool_is_registered():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert "detect_builtup_change" in names


def test_tool_schema_inputs():
    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == "detect_builtup_change")
    props = tool.inputSchema["properties"]
    assert props["bbox"]["type"] == "array"
    assert props["target_date"]["type"] == "string"
    # baseline_date is optional, so only bbox + target_date are required.
    assert set(tool.inputSchema["required"]) == {"bbox", "target_date"}
    assert "baseline_date" in props


def test_tool_has_description():
    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == "detect_builtup_change")
    assert tool.description and "change" in tool.description.lower()


def test_rejects_oversized_bbox():
    out = detect_builtup_change([0.0, 0.0, 0.1, 0.1], "2024-06-15")
    assert "validation failed" in out.lower()
    assert "exceeds" in out.lower()


def test_rejects_bad_date():
    out = detect_builtup_change(SMALL_BBOX, "not-a-date")
    assert "validation failed" in out.lower()


def test_rejects_baseline_after_target():
    out = detect_builtup_change(SMALL_BBOX, "2023-06-15", baseline_date="2024-06-15")
    assert "validation failed" in out.lower()
    assert "strictly before" in out.lower()
