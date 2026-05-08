"""Tokenizer eval entrypoint (delegates to ``training.eval_tokenizer``)."""

from __future__ import annotations

import os
import runpy
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    runpy.run_module("training.eval_tokenizer", run_name="__main__")
