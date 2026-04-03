"""Shared pytest configuration for the local test suite."""

import os
import sys


SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
