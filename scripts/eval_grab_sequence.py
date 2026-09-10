#!/usr/bin/env python3
"""Retarget a GRAB sequence frame by frame and measure temporal smoothness.

ContactPose grasps are static, so the main table cannot say whether `E_reg` does
anything.  This is the experiment that can: solve a real 120 Hz sequence with
each frame warm-started from the previous one, with `E_reg` on and off, and
report jitter alongside the usual contact and penetration numbers.

Jitter is the mean magnitude of the second difference (the discrete
acceleration) of the solved trajectory -- the quantity a downstream RL tracking
controller would have to follow.  The human demonstration's own jitter is
reported as the floor to beat.

    python scripts/eval_grab_sequence.py --seq .../cylindersmall_lift.npz \
        --mano .../MANO_RIGHT.pkl --subject-meshes .../subject_meshes \
        --object-meshes .../contactdb_meshes --out runs/grab_sequence.json
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import yaml

from toporetarget import WujiHand
from toporetarget.grab import GrabSequence
from toporetarget.metrics import ExactSDF, evaluate
from toporetarget.paths import CONFIGS, WUJI_URDF
from toporetarget.solve import solve_ours

METHODS = {"ours": {}, "ours:no_reg": dict(reg=0.0)}


def second_difference(series: np.ndarray) -> float:
    """Mean |x[t+1] - 2x[t] + x[t-1]| over a (T, ...) trajectory."""
    if len(series) < 3:
        return float("nan")
    d2 = series[2:] - 2 * series[1:-1] + series[:-2]
    return float(np.abs(d2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--mano", required=True, help="path to MANO_RIGHT.pkl")
    ap.add_argument("--subject-meshes", required=True)
    ap.add_argument("--object-meshes", required=True)
    ap.add_argument("--weights", default=str(CONFIGS / "weights.yaml"))
    ap.add_argument("--stride", type=int, default=8, help="120 Hz is denser than needed")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--iters", type=int, default=300,
                    help="fewer than a cold solve: every frame is warm-started")
    ap.add_argument("--tip-mm", type=float, default=20.0)
    ap.add_argument("--out", default="runs/grab_sequence.json")
    args = ap.parse_args()

    base_weights = yaml.safe_load(open(args.weights))["weights"]
    seq = GrabSequence(args.seq, args.mano, args.subject_meshes, args.object_meshes)
    window = seq.grasp_window(max_tip_mm=args.tip_mm, stride=args.stride)
    if len(window) < 3:
        raise SystemExit(f"{seq.path.stem}: only {len(window)} frames in contact; "
                         f"raise --tip-mm or lower --stride")
    window = window[:args.max_frames]
    print(f"{seq.path.stem}: {seq.obj_name}, {len(seq)} frames @ {seq.framerate} Hz "
          f"-> {len(window)} in-contact frames (stride {args.stride})")

    hand = WujiHand(str(WUJI_URDF))
    frames = [seq.frame(i) for i in window]
    sdf = ExactSDF(frames[0].obj)          # object is static in its own frame

    human = np.stack([f.human_kpts for f in frames]).astype(np.float64)
    results = {}
    for name, override in METHODS.items():
        w = dict(base_weights); w.update(override)
        prev, rows, qs, kps = None, [], [], []
        for f, idx in zip(frames, window):
            sol = solve_ours(hand, f, weights=w, n_iters=args.iters, prev=prev)
            met = evaluate(hand, f, sol.q, sol.d6, sol.t, sdf=sdf)
            rows.append(dict(frame=int(idx), **met))
            qs.append(sol.q)
            kps.append(hand.keypoints(*[__import__("torch").as_tensor(v,
                       dtype=__import__("torch").float32)[None]
                       for v in (sol.q, sol.d6, sol.t)])[0].detach().numpy())
            prev = sol
            print(f"  [{name:11s}] frame {idx:5d}  E_prec {met['E_prec_mm']:6.2f} "
                  f"D_pen_max {met['D_pen_max_mm']:5.2f}", flush=True)
        qs, kps = np.stack(qs), np.stack(kps)
        results[name] = dict(
            n_frames=len(rows), weights=w, per_frame=rows,
            E_prec_mm=round(statistics.fmean(r["E_prec_mm"] for r in rows), 3),
            D_pen_max_over_t_mm=round(max(r["D_pen_max_mm"] for r in rows), 3),
            D_pen_max_mean_mm=round(statistics.fmean(r["D_pen_max_mm"] for r in rows), 3),
            joint_jitter_rad=round(second_difference(qs), 6),
            keypoint_jitter_mm=round(second_difference(kps) * 1000, 4))

    human_jitter = round(second_difference(human) * 1000, 4)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(sequence=seq.path.stem, object=seq.obj_name,
                   framerate=seq.framerate, stride=args.stride,
                   frames=[int(i) for i in window],
                   human_keypoint_jitter_mm=human_jitter,
                   results=results), open(out, "w"), indent=1)

    print(f"\nhuman demonstration keypoint jitter: {human_jitter:.4f} mm/frame^2 (the floor)")
    print(f"{'method':12s} {'E_prec':>8s} {'max_t Dpen':>11s} "
          f"{'joint jitter':>13s} {'kpt jitter':>11s}")
    for k, v in results.items():
        print(f"{k:12s} {v['E_prec_mm']:8.2f} {v['D_pen_max_over_t_mm']:11.2f} "
              f"{v['joint_jitter_rad']:13.5f} {v['keypoint_jitter_mm']:11.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
