#!/usr/bin/env python3
"""Evaluate submission against ground-truth answers according to Rulebook Section 9."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure 'src' is in sys.path when running script directly
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from zaic_2026_4.evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main())
