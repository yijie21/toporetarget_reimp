"""One calling convention for every retargeting method.

A solver takes a `WujiHand` and a `Frame` and returns a `Solution`, so the
evaluation harness can score our method and the baselines with identical code
and identical metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Solution:
    q: np.ndarray                    # (n_joints,) joint angles
    d6: np.ndarray                   # (6,) base rotation, continuous 6-D
    t: np.ndarray                    # (3,) base translation
    seconds: float = 0.0
    method: str = ""
    extra: dict = field(default_factory=dict)


def solve_ours(hand, frame, *, weights=None, n_iters=500, lr=0.02,
               pen_per_link=40, kappa=80.0, method="ours", prev: "Solution" = None,
               **kw) -> Solution:
    """This repository's implementation of the paper's Stage 1-4 optimisation.

    `prev` chains a sequence: the previous frame's solution both warm-starts the
    optimisation and becomes the reference E_reg smooths towards.
    """
    from .optimize import retarget
    t0 = time.time()
    chain = dict(q_init=(prev.q, prev.d6, prev.t), ref=(prev.q, prev.d6, prev.t)) \
        if prev is not None else {}
    out = retarget(hand, frame, weights=weights, n_iters=n_iters, lr=lr,
                   kappa=kappa, pen_per_link=pen_per_link,
                   log_every=max(n_iters // 10, 1), **chain, **kw)
    f = out["final"]
    return Solution(q=np.asarray(f["q"], np.float64),
                    d6=np.asarray(f["d6"], np.float64),
                    t=np.asarray(f["t"], np.float64),
                    seconds=time.time() - t0, method=method,
                    extra=dict(weights=out["weights"], n_edges=len(out["edges"])))
