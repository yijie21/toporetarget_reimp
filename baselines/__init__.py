"""Baseline retargeting methods, run through their **own upstream libraries**.

The comparison is only worth something if the baselines are not our
reimplementations of someone else's method, so `dexpilot` calls Yuzhe Qin's
`dex_retargeting` and `mink` calls the MuJoCo IK library of that name.  What we
supply is a per-robot configuration file, and those live in `configs/baselines/`
in full view.
"""
from typing import Callable

_REGISTRY = {}

# solver name -> module providing it (a module cannot be called `mink`: it would
# collide with the upstream package it imports)
_MODULES = {"dexpilot": "dexpilot", "mink": "mink_ik"}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_solver(name: str) -> Callable:
    if name not in _REGISTRY:
        # lazy import so that using one baseline does not require the other's deps
        __import__(f"baselines.{_MODULES.get(name, name)}")
    if name not in _REGISTRY:
        raise SystemExit(f"unknown baseline {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]
