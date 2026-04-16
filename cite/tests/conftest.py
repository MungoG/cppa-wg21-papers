"""Shared path setup for cite tests. Auto-loaded by pytest."""
import sys
from pathlib import Path

_tests_dir = str(Path(__file__).resolve().parent)
_cite_dir = str(Path(__file__).resolve().parent.parent)

if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
if _cite_dir not in sys.path:
    sys.path.insert(0, _cite_dir)
