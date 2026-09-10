#!/usr/bin/env python3
"""Retarget one frame and print its metrics — the smallest useful entry point.

    python scripts/run_retarget.py                              # synthetic cylinder
    python scripts/run_retarget.py --source synthetic:sphere
    python scripts/run_retarget.py --source contactpose:mug --grasps ... --models ...
    python scripts/run_retarget.py --source contactpose:mug ... --method mink

For the viewer payload use scripts/export_viewer.py; for the whole benchmark use
scripts/eval_contactpose.py.
"""
import argparse

import numpy as np
import yaml

from toporetarget import WujiHand
from toporetarget.data import make_synthetic_grasp
from toporetarget.metrics import evaluate
from toporetarget.paths import CONFIGS, WUJI_URDF
from toporetarget.solve import solve_ours

ABLATIONS = {"no_IM": dict(IM=0.0), "no_pen": dict(pen=0.0),
             "no_bone": dict(bone=0.0), "no_reg": dict(reg=0.0)}


def build_frame(args):
    kind, _, rest = args.source.partition(":")
    if kind == "synthetic":
        return make_synthetic_grasp(rest or "cylinder", n_obj=args.n_obj, seed=args.seed)
    if kind == "contactpose":
        from toporetarget.contactpose import load_contactpose_frame
        cfg = yaml.safe_load(open(CONFIGS / "eval_set.yaml"))
        if not (args.grasps and args.models):
            raise SystemExit("--grasps and --models are required for contactpose")
        return load_contactpose_frame(args.grasps, args.models, rest,
                                      participant=cfg["participant"],
                                      intent=cfg["intent"], n_obj=args.n_obj)
    raise SystemExit(f"unknown --source {args.source!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic:cylinder")
    ap.add_argument("--method", default="ours",
                    help="ours | dexpilot | mink (baselines need the extra deps)")
    ap.add_argument("--ablations", action="store_true",
                    help="also run each single-term ablation")
    ap.add_argument("--grasps"); ap.add_argument("--models")
    ap.add_argument("--weights", default=str(CONFIGS / "weights.yaml"))
    ap.add_argument("--n_obj", type=int, default=80)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    weights = yaml.safe_load(open(args.weights))["weights"]
    hand = WujiHand(str(WUJI_URDF))
    frame = build_frame(args)
    print(f"[data] {frame.meta}")

    runs = {args.method: {}}
    if args.ablations:
        if args.method != "ours":
            raise SystemExit("--ablations only applies to --method ours")
        runs.update({f"ours:{k}": v for k, v in ABLATIONS.items()})

    print(f"\n{'variant':14s} {'E_prec':>8s} {'E_orig':>8s} {'D_pen_max':>10s} "
          f"{'D_pen_mean':>11s} {'|C|':>4s} {'secs':>6s}")
    for name, override in runs.items():
        if name.startswith("ours"):
            w = dict(weights); w.update(override)
            sol = solve_ours(hand, frame, weights=w, n_iters=args.iters, method=name)
        else:
            from baselines import get_solver
            sol = get_solver(name)(hand, frame)
        m = evaluate(hand, frame, sol.q, sol.d6, sol.t)
        print(f"{name:14s} {m['E_prec_mm']:8.2f} {m['E_prec_origin_mm']:8.2f} "
              f"{m['D_pen_max_mm']:10.2f} {m['D_pen_mean_mm']:11.2f} "
              f"{m['n_contacts']:4d} {sol.seconds:6.1f}")
    if not m["watertight"]:
        print("\nnote: this object's mesh is not watertight; signed distance uses the "
              "winding number, which is defined for open meshes.")


if __name__ == "__main__":
    main()
