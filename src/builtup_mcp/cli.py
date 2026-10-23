"""Run a city request from the command line.

    uv run builtup-request requests/malakal.json
    uv run builtup-request requests/malakal.json --plan
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a builtup-mcp city change request.")
    ap.add_argument("request", help="Request JSON file")
    ap.add_argument("--plan", action="store_true", help="Only list windows and scene counts")
    args = ap.parse_args()
    from .server import plan_city_request

    if args.plan:
        print(plan_city_request(request_path=args.request))
        return
    from .city_pipeline import run_city_request
    from .request import load_request

    product = run_city_request(load_request(args.request), log=lambda m: print(m, file=sys.stderr, flush=True))
    print(product.text())
