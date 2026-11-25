"""Allow running `python -m invocation_protocol`."""

from __future__ import annotations

import sys

from invocation_protocol.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
