#!/usr/bin/env python3
"""Fit every method's hyper-parameters on the held-out split.

Protocol, identical for our method and for each baseline:

* **Split**: a different ContactPose participant from the benchmark
  (`configs/tuning_split.yaml`), so no evaluation grasp is ever seen.
* **Objective**: mean over the split of ``E_prec_mm + D_pen_max_mm``.  Both
  terms are in millimetres and equally weighted, so the objective has no free
  parameter of its own to tune.
* **Budget**: the same number of randomly sampled configurations per method
  (seeded, so the search is reproducible).  Our method spends its budget on a
  3-D space and each baseline on its 1-D scaling factor, which favours the
  baselines -- deliberately, that is the safe direction to err in.

    python scripts/tune.py --grasps ... --models ... --method ours --budget 40
"""
import os

# Each worker must stay single-threaded: with N workers each defaulting to one
# OpenMP thread per core, the pool spawns hundreds of threads and dies with
# BrokenProcessPool.  These must be set BEFORE numpy/torch/igl are imported,
# because the threading libraries read them at load time -- setting them inside
# the worker initialiser is too late.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from toporetarget.paths import CONFIGS

_STATE = {}


def _worker_init(grasps, models, split):
    import torch
    torch.set_num_threads(1)
    from toporetarget import WujiHand
    from toporetarget.contactpose import load_contactpose_frame
    from toporetarget.metrics import ExactSDF
    from toporetarget.paths import WUJI_URDF

    _STATE["hand"] = WujiHand(str(WUJI_URDF))
    _STATE["frames"] = []
    for obj in split["objects"]:
        f = load_contactpose_frame(grasps, models, obj,
                                   participant=split["participant"],
                                   intent=split["intent"])
        _STATE["frames"].append((f, ExactSDF(f.obj)))


def _score(args):
    """Mean of (E_prec + D_pen_max) over the split for one configuration."""
    method, params, solver_cfg = args
    from toporetarget.metrics import evaluate
    from toporetarget.solve import solve_ours

    hand = _STATE["hand"]
    totals = []
    for frame, sdf in _STATE["frames"]:
        if method == "ours":
            sol = solve_ours(hand, frame, weights=params, **solver_cfg)
        else:
            from baselines import get_solver
            sol = get_solver(method)(hand, frame, **params)
        m = evaluate(hand, frame, sol.q, sol.d6, sol.t, sdf=sdf)
        totals.append(m["E_prec_mm"] + m["D_pen_max_mm"])
    return dict(params=params, objective=float(np.mean(totals)),
                per_grasp=[round(v, 3) for v in totals])


def sample(method, budget, seed):
    """Randomly sampled configurations, log-uniform over each range."""
    rng = np.random.default_rng(seed)

    def logu(lo, hi, n):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), n))

    if method == "ours":
        # IM is the scale anchor (only ratios matter) and lim enforces a hard
        # constraint rather than trading off, so both are fixed.
        bone, reg, pen = logu(0.5, 50, budget), logu(0.02, 4, budget), logu(2, 400, budget)
        return [dict(IM=1.0, bone=float(b), reg=float(r), pen=float(p), lim=200.0)
                for b, r, p in zip(bone, reg, pen)]
    return [dict(scaling=float(s)) for s in logu(0.8, 2.5, budget)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--method", required=True, choices=["ours", "dexpilot", "mink"])
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--pen-per-link", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6,
                    help="each worker holds torch plus the split's meshes (~2 GB); "
                         "raise only if the machine has the memory")
    ap.add_argument("--split", default=str(CONFIGS / "tuning_split.yaml"))
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--freeze", action="store_true",
                    help="write the winning parameters into the method's config, "
                         "so reproducing needs no hand-editing")
    args = ap.parse_args()

    split = yaml.safe_load(open(args.split))
    configs = sample(args.method, args.budget, args.seed)
    solver_cfg = (dict(n_iters=args.iters, pen_per_link=args.pen_per_link)
                  if args.method == "ours" else {})
    jobs = [(args.method, c, solver_cfg) for c in configs]

    # "spawn", not the Linux default "fork": torch, libigl and mujoco all carry
    # native thread pools, and forking a process that has already loaded them is
    # unsafe -- it showed up as BrokenProcessPool once every worker was busy.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(args.grasps, args.models, split)) as ex:
        results = []
        for i, r in enumerate(ex.map(_score, jobs), 1):
            results.append(r)
            print(f"[{i:3d}/{len(jobs)}] objective {r['objective']:8.3f}   {r['params']}",
                  flush=True)

    results.sort(key=lambda r: r["objective"])
    best = results[0]
    out = Path(args.out_dir) / f"tuning_{args.method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(method=args.method, split=split, budget=args.budget,
                   seed=args.seed, objective="mean(E_prec_mm + D_pen_max_mm)",
                   results=results, best=best), open(out, "w"), indent=1)
    print(f"\nbest objective {best['objective']:.3f} with {best['params']}")
    print(f"full sweep -> {out}")
    if args.freeze:
        freeze(args.method, best)
    else:
        print("re-run with --freeze to write these into the method's config")


def freeze(method, best):
    """Write the winning parameters into the config the evaluation reads."""
    import re
    from datetime import date

    if method == "ours":
        w = {k: round(float(v), 4) for k, v in best["params"].items()}
        (CONFIGS / "weights.yaml").write_text(f"""# Loss weights for the TopoRetarget objective.
#
# The paper does not publish its weights.  These were FITTED, not hand-picked:
# scripts/tune.py sampled configurations on configs/tuning_split.yaml (a
# different ContactPose participant from the benchmark) and minimised
# mean(E_prec_mm + D_pen_max_mm).  Best objective on that split: {best['objective']:.3f} mm
# (frozen {date.today().isoformat()}).
# Reproduce with:  python scripts/tune.py --method ours --budget 40 --seed 0 --freeze
#
# One single set is used for every object in the main table, as the paper claims
# for its own parameters.  Positional terms are in mm^2 so the weights stay O(1).
# IM is the scale anchor (only ratios matter) and lim enforces a hard constraint,
# so both were held fixed during the search.
weights:
  IM: {w['IM']}          # E_IM   interaction-mesh Laplacian coordinates  (Stage 3)
  bone: {w['bone']}       # E_bone relative bone-direction                 (Stage 1)
  reg: {w['reg']}        # E_reg  smoothness / base prior                 (Stage 4)
  pen: {w['pen']}      # E_pen  penetration hinge on link surfaces      (Stage 4)
  lim: {w['lim']}      # E_lim  joint-limit barrier
""")
        print(f"froze weights into {CONFIGS / 'weights.yaml'}")
        return

    path = CONFIGS / "baselines" / f"{method}_wuji.yml"
    key = "scaling_factor" if method == "dexpilot" else "scaling"
    text = path.read_text()
    new, n = re.subn(rf"^(\s*){key}: [0-9.eE+-]+",
                     rf"\g<1>{key}: {best['params']['scaling']:.4f}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"could not find a single `{key}:` line in {path}")
    path.write_text(new)
    print(f"froze {key} = {best['params']['scaling']:.4f} into {path}")


if __name__ == "__main__":
    main()
