#!/usr/bin/env python3
"""Score every method on the frozen ContactPose evaluation set.

Writes a JSON record of every per-grasp measurement (so tables are never typed
by hand) and prints the aggregate.  Methods are named `ours`, `ours:no_IM`,
`dexpilot`, `mink`, ...; an `ours:<ablation>` name zeroes that loss weight.

    python scripts/eval_contactpose.py --grasps ... --models ... \
        --methods ours,dexpilot,mink --out runs/contactpose.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from toporetarget import WujiHand
from toporetarget.contactpose import load_contactpose_frame
from toporetarget.metrics import ExactSDF, evaluate
from toporetarget.paths import CONFIGS, WUJI_URDF
from toporetarget.solve import solve_ours

ABLATIONS = {"no_IM": dict(IM=0.0), "no_pen": dict(pen=0.0),
             "no_bone": dict(bone=0.0), "no_reg": dict(reg=0.0)}


def load_weights(path):
    with open(path) as f:
        return yaml.safe_load(f)["weights"]


def make_solver(name, base_weights, cfg):
    """`name` -> callable(hand, frame) -> Solution."""
    if name == "ours":
        return lambda h, f: solve_ours(h, f, weights=base_weights, method=name, **cfg)
    if name.startswith("ours:"):
        ab = name.split(":", 1)[1]
        if ab not in ABLATIONS:
            raise SystemExit(f"unknown ablation {ab!r}; choose from {list(ABLATIONS)}")
        w = dict(base_weights); w.update(ABLATIONS[ab])
        return lambda h, f: solve_ours(h, f, weights=w, method=name, **cfg)
    from baselines import get_solver
    return get_solver(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--methods", default="ours")
    ap.add_argument("--eval-set", default=str(CONFIGS / "eval_set.yaml"))
    ap.add_argument("--weights", default=str(CONFIGS / "weights.yaml"))
    ap.add_argument("--objects", default=None, help="comma list, overrides the eval set")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--pen-per-link", type=int, default=40)
    ap.add_argument("--n-obj", type=int, default=80)
    ap.add_argument("--tau-mm", type=float, default=10.0)
    ap.add_argument("--eval-per-link", type=int, default=200)
    ap.add_argument("--out", default="runs/contactpose.json")
    args = ap.parse_args()

    with open(args.eval_set) as f:
        eval_set = yaml.safe_load(f)
    objects = (args.objects.split(",") if args.objects else eval_set["objects"])
    excluded = set(eval_set["paper_comparable"]["excluded"])
    base_weights = load_weights(args.weights)
    solver_cfg = dict(n_iters=args.iters, pen_per_link=args.pen_per_link)
    methods = args.methods.split(",")

    hand = WujiHand(str(WUJI_URDF))
    rows, t_start = [], time.time()
    for oi, obj in enumerate(objects, 1):
        frame = load_contactpose_frame(args.grasps, args.models, obj,
                                       participant=eval_set["participant"],
                                       intent=eval_set["intent"], n_obj=args.n_obj)
        sdf = ExactSDF(frame.obj)          # built once, shared by every method
        for m in methods:
            sol = make_solver(m, base_weights, solver_cfg)(hand, frame)
            met = evaluate(hand, frame, sol.q, sol.d6, sol.t, sdf=sdf,
                           tau_mm=args.tau_mm, n_per_link=args.eval_per_link)
            rows.append(dict(object=obj, method=m, seconds=round(sol.seconds, 2),
                             paper_comparable=obj not in excluded,
                             bimanual=frame.meta["bimanual"], **met))
            print(f"[{oi:2d}/{len(objects)}] {obj:14s} {m:12s} "
                  f"E_prec {met['E_prec_mm']:7.2f} mm   D_pen_max {met['D_pen_max_mm']:6.2f} mm "
                  f"  |C|={met['n_contacts']:2d}  ({sol.seconds:.1f}s)", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for m in methods:
        for subset, keep in (("all", lambda r: True),
                             ("paper_comparable", lambda r: r["paper_comparable"])):
            sel = [r for r in rows if r["method"] == m and keep(r)]
            if not sel:
                continue
            summary[f"{m}|{subset}"] = dict(
                n=len(sel),
                E_prec_mm=round(statistics.fmean(r["E_prec_mm"] for r in sel), 3),
                E_prec_median_mm=round(statistics.median(r["E_prec_mm"] for r in sel), 3),
                D_pen_max_mm=round(max(r["D_pen_max_mm"] for r in sel), 3),
                D_pen_max_mean_mm=round(statistics.fmean(r["D_pen_max_mm"] for r in sel), 3),
                D_pen_mean_mm=round(statistics.fmean(r["D_pen_mean_mm"] for r in sel), 3),
                seconds_mean=round(statistics.fmean(r["seconds"] for r in sel), 2))
    with open(out, "w") as f:
        json.dump(dict(config=vars(args), eval_set=eval_set["paper_comparable"],
                       weights=base_weights, rows=rows, summary=summary,
                       total_seconds=round(time.time() - t_start, 1)), f, indent=1)

    print(f"\n{'method|subset':28s} {'n':>3s} {'E_prec':>8s} {'med':>8s} "
          f"{'Dpen_max':>9s} {'Dpen_mean_of_max':>17s}")
    for k, v in summary.items():
        print(f"{k:28s} {v['n']:3d} {v['E_prec_mm']:8.2f} {v['E_prec_median_mm']:8.2f} "
              f"{v['D_pen_max_mm']:9.2f} {v['D_pen_max_mean_mm']:17.2f}")
    print(f"\nwrote {out}  ({time.time()-t_start:.0f}s total)")


if __name__ == "__main__":
    main()
