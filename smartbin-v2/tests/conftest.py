"""
Pytest configuration and environment fixtures for SmartBin AI v2 tests.
Ensures repository root and smartbin-v2 packages are on sys.path.
"""

import sys
from pathlib import Path

# Add repository root and smartbin-v2 to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
smartbin_v2_dir = root_dir / "smartbin-v2"

for p in [str(root_dir), str(smartbin_v2_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)
