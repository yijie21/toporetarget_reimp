#!/usr/bin/env python3
"""Export a retargeting run for the three.js viewer.

The 26 Wuji link meshes are written **once** to `viewer/robot_meshes.js` and
shared by every payload; they used to be duplicated into each dataset file,
which is most of why the viewer data was 62 MB.

Only synthetic payloads are committed.  ContactPose and GRAB payloads embed
dataset-derived geometry that we are not permitted to redistribute, so they are
git-ignored and regenerated locally by this script.

    python scripts/export_viewer.py --source synthetic:cylinder
    python scripts/export_viewer.py --source contactpose:mug --grasps ... --models ...
    python scripts/export_viewer.py --source grab:cylindersmall_lift:1034 --mano ... \
        --subject-meshes ... --object-meshes ...
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml

from toporetarget import WujiHand
from toporetarget.data import make_synthetic_grasp
from toporetarget.metrics import ExactSDF, evaluate
from toporetarget.optimize import retarget
from toporetarget.paths import CONFIGS, REPO_ROOT, WUJI_MESHES, WUJI_URDF

VIEWER = REPO_ROOT / "viewer"
ABLATIONS = {"full": {}, "no_IM": dict(IM=0.0), "no_pen": dict(pen=0.0),
             "no_bone": dict(bone=0.0), "no_reg": dict(reg=0.0)}


def rounded(x, n=5):
    if isinstance(x, float):
        return round(x, n)
    if isinstance(x, (list, tuple)):
        return [rounded(v, n) for v in x]
    return x


def write_js(path, var, payload):
    with open(path, "w") as f:
        f.write(f"window.{var} = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";")
    return os.path.getsize(path) / 1e6


def export_robot_meshes(target_faces=500):
    """Decimated link meshes, written once and shared by every payload."""
    meshes = {}
    for p in sorted(WUJI_MESHES.glob("*.STL")):
        m = trimesh.load(p, process=False)
        if len(m.faces) > target_faces:
            try:
                m = m.simplify_quadric_decimation(face_count=target_faces)
            except Exception:
                pass
        meshes[p.stem] = dict(vertices=rounded(m.vertices.astype(float).tolist(), 5),
                              faces=m.faces.astype(int).tolist())
    mb = write_js(VIEWER / "robot_meshes.js", "TR_ROBOT_MESHES", meshes)
    print(f"[robot] {len(meshes)} link meshes -> viewer/robot_meshes.js ({mb:.1f} MB)")


def pretty_label(frame, key):
    """A short, human name for the viewer's dataset switch ("mug", not "grab mug drink 1")."""
    meta = frame.meta
    name = meta.get("object") or meta.get("kind") or key
    return str(name).replace("_", " ")


def build_frame(args):
    kind, _, rest = args.source.partition(":")
    if kind == "synthetic":
        return make_synthetic_grasp(rest or "cylinder", n_obj=args.n_obj,
                                    seed=args.seed), f"syn_{rest or 'cylinder'}"
    if kind == "contactpose":
        from toporetarget.contactpose import load_contactpose_frame
        cfg = yaml.safe_load(open(CONFIGS / "eval_set.yaml"))
        return load_contactpose_frame(args.grasps, args.models, rest,
                                      participant=cfg["participant"],
                                      intent=cfg["intent"],
                                      n_obj=args.n_obj), f"contactpose_{rest}"
    if kind == "grab":
        from toporetarget.grab import GrabSequence
        seq_name, _, frame_idx = rest.partition(":")
        matches = glob.glob(os.path.join(args.grab_root, "**", f"{seq_name}.npz"),
                            recursive=True)
        if not matches:
            raise SystemExit(f"no sequence {seq_name}.npz under {args.grab_root}")
        seq = GrabSequence(matches[0], args.mano, args.subject_meshes,
                           args.object_meshes)
        if frame_idx in ("", "best"):
            # the tightest grasp in the sequence: smallest mean fingertip gap
            window = seq.grasp_window(max_tip_mm=25.0, stride=10)
            if not window:
                raise SystemExit(f"{seq_name}: no in-contact frame found")
            idx = min(window, key=lambda i: seq.tip_distances_mm(i).mean())
            print(f"[frame] picked {idx} of {len(seq)} "
                  f"(mean fingertip gap {seq.tip_distances_mm(idx).mean():.1f} mm)")
        else:
            idx = int(frame_idx)
        return seq.frame(idx, n_obj=args.n_obj), f"grab_{seq_name}"
    raise SystemExit(f"unknown --source {args.source!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic:cylinder")
    ap.add_argument("--grasps"); ap.add_argument("--models")
    ap.add_argument("--grab-root"); ap.add_argument("--mano")
    ap.add_argument("--subject-meshes"); ap.add_argument("--object-meshes")
    ap.add_argument("--weights", default=str(CONFIGS / "weights.yaml"))
    ap.add_argument("--n_obj", type=int, default=80)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--skip-robot-meshes", action="store_true")
    args = ap.parse_args()

    VIEWER.mkdir(exist_ok=True)
    if not args.skip_robot_meshes:
        export_robot_meshes()

    base_weights = yaml.safe_load(open(args.weights))["weights"]
    hand = WujiHand(str(WUJI_URDF))
    frame, key = build_frame(args)
    label = args.label or pretty_label(frame, key)
    print(f"[data] {frame.meta}")

    sdf = ExactSDF(frame.obj)
    overt, ofaces = frame.obj.trimesh()
    runs, edges, kinds = {}, None, None
    for name, override in ABLATIONS.items():
        w = dict(base_weights); w.update(override)
        out = retarget(hand, frame, weights=w, n_iters=args.iters, log_every=20)
        edges, kinds = out["edges"], out["edge_kinds"]
        trace = []
        for rec in out["trace"]:
            q, d6, t = (torch.tensor(rec[k]) for k in ("q", "d6", "t"))
            poses = hand.all_link_poses(q[None], d6[None], t[None])
            trace.append(dict(
                iter=rec["iter"], losses=rounded(rec["losses"]),
                kpts=rounded(rec["kpts"]),
                link_T={k: rounded(v.detach().cpu().numpy().reshape(-1).tolist(), 5)
                        for k, v in poses.items()}))
        f = out["final"]
        met = evaluate(hand, frame, np.array(f["q"]), np.array(f["d6"]),
                       np.array(f["t"]), sdf=sdf)
        for rec, tr in zip(out["trace"], trace):
            tr["contact_mm"] = rounded(rec["contact_mm"], 2)
            tr["max_pen_mm"] = rounded(rec["max_pen_mm"], 2)
        runs[name] = dict(weights=w, trace=trace,
                          final={k: rounded(v, 3) for k, v in met.items()
                                 if isinstance(v, (int, float))})
        print(f"[run] {name:8s} E_prec {met['E_prec_mm']:6.2f} mm   "
              f"D_pen_max {met['D_pen_max_mm']:5.2f} mm")

    payload = dict(
        meta=dict(source=frame.meta, label=label, urdf="wuji-hand-right",
                  n_joints=hand.n_joints, ablations=list(ABLATIONS)),
        object=dict(vertices=rounded(np.asarray(overt, float).tolist(), 5),
                    faces=np.asarray(ofaces, int).tolist()),
        human_kpts=rounded(frame.human_kpts.astype(float).tolist()),
        human_mesh=(dict(vertices=rounded(frame.hand_verts.astype(float).tolist(), 5),
                         faces=frame.hand_faces.astype(int).tolist())
                    if frame.hand_verts is not None else None),
        obj_pts=rounded(frame.obj_pts.astype(float).tolist()),
        edges=[[int(i), int(j), k] for (i, j), k in zip(edges, kinds)],
        bones=[[int(a), int(b)] for a, b in hand.bones],
        kp_frames=hand.kp_frames, runs=runs)

    var = f"TR_{key.upper()}"
    mb = write_js(VIEWER / f"data_{key}.js", var, payload)
    print(f"[out] viewer/data_{key}.js ({mb:.1f} MB, robot meshes not duplicated)")
    update_manifest()


def update_manifest():
    """List the payloads present so index.html needs no hand-edited script tags."""
    entries = []
    for p in sorted(VIEWER.glob("data_*.js")):
        key = p.stem[len("data_"):]
        entries.append(dict(key=key, var=f"TR_{key.upper()}", file=p.name,
                            label=key.replace("_", " ")))
    write_js(VIEWER / "manifest.js", "TR_MANIFEST", entries)
    print(f"[manifest] {len(entries)} payload(s): {[e['key'] for e in entries]}")


if __name__ == "__main__":
    main()
