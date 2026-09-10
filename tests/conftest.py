import pytest

from toporetarget import WujiHand
from toporetarget.paths import WUJI_URDF


@pytest.fixture(scope="session")
def hand():
    """The vendored Wuji right hand — proves a fresh clone is self-contained."""
    assert WUJI_URDF.exists(), "vendored URDF missing; the repo is not self-contained"
    return WujiHand(str(WUJI_URDF))
