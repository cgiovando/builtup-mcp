"""builtup-mcp: an MCP server for on-demand built-up change detection."""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> None:
    """Console-script entrypoint: launch the MCP server over stdio."""
    from .server import run

    run()
