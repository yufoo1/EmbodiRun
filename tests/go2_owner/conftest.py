"""Expose only the optional owner package to its dedicated regression suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations/go2_owner/src"))
