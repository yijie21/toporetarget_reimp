"""Canonical locations of the vendored robot assets.

Everything the package needs to run is inside the repository, so a fresh clone
works with no downloads.  Dataset roots (ContactPose / GRAB) are *not* vendored
and are passed explicitly on the command line.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets" / "wuji_right"

WUJI_URDF = ASSETS / "right.urdf"
WUJI_MJCF = ASSETS / "mjcf" / "right.xml"
WUJI_MESHES = ASSETS / "meshes"

CONFIGS = REPO_ROOT / "configs"
