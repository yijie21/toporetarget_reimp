"""DexPilot baseline, via the upstream `dex_retargeting` package.

DexPilot solves joint angles from thumb-to-finger vectors in the wrist frame; it
does not place the wrist.  `place_base_by_kabsch` supplies the base pose (see
`_common.py` for why that choice is the fair one).
"""
from __future__ import annotations

import contextlib
import os
import time

import numpy as np
import yaml

from toporetarget.paths import ASSETS, CONFIGS
from toporetarget.solve import Solution

from . import register
from ._common import place_base_by_kabsch

_CACHE = {}


@contextlib.contextmanager
def _chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _build(cfg_path, scaling=None):
    """Build (and cache) a SeqRetargeting from our config file."""
    from dex_retargeting.retargeting_config import RetargetingConfig

    key = (str(cfg_path), scaling)
    if key in _CACHE:
        r = _CACHE[key]
        r.reset()
        return r

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["retargeting"]
    cfg["urdf_path"] = str(ASSETS / cfg["urdf_path"].replace("ASSETS/", ""))
    if scaling is not None:
        cfg["scaling_factor"] = float(scaling)
    # yourdfpy resolves the URDF's relative mesh paths against the *process* cwd,
    # which floods stderr with "Unable to resolve filename".  DexPilot only needs
    # kinematics, so build from the asset directory and move back.
    with _chdir(ASSETS):
        r = RetargetingConfig.from_dict(cfg).build()
    _CACHE[key] = r
    return r


@register("dexpilot")
def solve(hand, frame, *, config=None, scaling=None, **kw) -> Solution:
    cfg_path = config or (CONFIGS / "baselines" / "dexpilot_wuji.yml")
    r = _build(cfg_path, scaling)

    t0 = time.time()
    human = np.asarray(frame.human_kpts, np.float32)
    idx = r.optimizer.target_link_human_indices          # (2, N): origin, task
    ref = human[idx[1], :] - human[idx[0], :]
    qpos = r.retarget(ref)                               # full robot dof order

    # dex_retargeting returns joints in its own (pinocchio) order; map by name.
    names = list(r.optimizer.robot.dof_joint_names)
    lut = {n: i for i, n in enumerate(names)}
    q = np.array([qpos[lut[n]] for n in hand.joint_names], np.float64)
    q = np.clip(q, hand.lower.numpy(), hand.upper.numpy())
    d6, t = place_base_by_kabsch(hand, q, human)
    return Solution(q=q, d6=np.asarray(d6, np.float64), t=np.asarray(t, np.float64),
                    seconds=time.time() - t0, method="dexpilot",
                    extra=dict(scaling=r.optimizer.scaling if hasattr(
                        r.optimizer, "scaling") else scaling))
