"""Tokenizer training entrypoint (delegates to ``jaxchat.tokenizer``)."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jaxchat.tokenizer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
