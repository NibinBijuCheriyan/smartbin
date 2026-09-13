"""
SmartBin v2 Package Bridge
Allows 'import smartbin_v2' to seamlessly route to 'smartbin-v2' submodules.
"""
import sys
from pathlib import Path

_base = Path(__file__).resolve().parent.parent / "smartbin-v2"
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))

__path__ = [str(_base)]
