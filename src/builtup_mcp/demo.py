"""Serve the demo app and the local STAC product catalog over HTTP.

    uv run builtup-demo            # http://localhost:8765
    uv run builtup-demo --port 9000

``/`` serves ``app/`` and ``/products/`` serves ``PRODUCTS_DIR``, so the app reads
the catalog the same way any STAC client would. Works offline once products and
basemap tiles are cached, except for the Esri basemap itself.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

from . import config

APP_DIR = Path(__file__).resolve().parents[2] / "app"


class _Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        from urllib.parse import unquote

        clean = unquote(path.split("?", 1)[0].split("#", 1)[0])
        if clean.startswith("/products/"):
            root = config.PRODUCTS_DIR.resolve()
            target = (root / clean[len("/products/"):]).resolve()
            # Never serve anything outside the catalog folder.
            return str(target) if target.is_relative_to(root) else str(root / "__forbidden__")
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args) -> None:  # keep the terminal quiet during a demo
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--city", default="")
    args = ap.parse_args()
    handler = functools.partial(_Handler, directory=str(APP_DIR))
    url = f"http://localhost:{args.port}/" + (f"?city={args.city}" if args.city else "")
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"builtup-mcp demo: {url}  (products: {config.PRODUCTS_DIR})")
        if not args.no_browser:
            webbrowser.open(url)
        httpd.serve_forever()
