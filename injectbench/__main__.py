"""Start the injectbench dashboard: ``injectbench`` or ``python -m injectbench``."""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="injectbench", description="Start the injectbench dashboard.")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    p.add_argument("--port", type=int, default=8765, help="port, 0 for any free one (default: %(default)s)")
    p.add_argument("--out", default="results", help="where runs are saved (default: %(default)s)")
    p.add_argument("--techniques-file", help="JSON file with extra techniques")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    return p


def main(argv: list[str] | None = None) -> int:
    from .dashboard.server import serve

    args = build_parser().parse_args(argv)
    serve(args.host, args.port, args.out, args.techniques_file, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
