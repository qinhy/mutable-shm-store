from __future__ import annotations

import argparse
import sys

from .server import MStoreServer, default_endpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mutable shared-memory object server")
    parser.add_argument(
        "--endpoint",
        default=default_endpoint(),
        help="unix:///path/to.sock on Linux or tcp://127.0.0.1:PORT (default: %(default)s)",
    )
    parser.add_argument("--debug", action="store_true", help="include tracebacks in internal errors")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = MStoreServer(args.endpoint, debug=args.debug)
    print(f"mstore server listening on {args.endpoint}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    except Exception as exc:
        print(f"mstore server failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
