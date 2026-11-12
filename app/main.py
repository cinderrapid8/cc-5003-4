"""Run the museum audio-guide loudness service."""
from __future__ import annotations

import argparse
import os

from .server import build_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Loudness production service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="task storage root",
    )
    args = parser.parse_args()
    httpd = build_server(os.path.abspath(args.data), args.host, args.port)
    print(f"loudness service on http://{args.host}:{args.port} (data: {os.path.abspath(args.data)})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
