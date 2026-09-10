#!/usr/bin/env python3
"""Run TopoRetarget on a frame, with a full ablation set, and export everything
the three.js viewer needs into viewer/data.js (as `window.TR_DATA`).

Usage:
    python run_retarget.py                       # synthetic cylinder (default)
    python run_retarget.py --object sphere
    python run_retarget.py --grab SEQ.npz --meshes GRAB/tools/object_meshes/contact_meshes
"""
import argparse, glob, json, os
import numpy as np
import trimesh

from toporetarget import WujiHand, make_synthetic_grasp
from toporetarget.data import load_grab_frame, load_grab_preprocessed
from toporetarget.optimize import retarget

from toporetarget.paths import REPO_ROOT, WUJI_URDF, WUJI_MESHES

URDF = str(WUJI_URDF)
MESHDIR = str(WUJI_MESHES)

ABLATIONS = {
    "full":    {},
    "no_IM":   dict(IM=0.0),
    "no_pen":  dict(pen=0.0),
    "no_bone": dict(bone=0.0),
    "no_reg":  dict(reg=0.0),
}


def r(x, n=5):  # round nested lists for compact JSON
    if isinstance(x, float):
        return round(x, n)
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    return x


def decimate_link_meshes(target_faces=500):
    meshes = {}
    for f in sorted(glob.glob(os.path.join(MESHDIR, "*.STL"))):
        name = os.path.splitext(os.path.basename(f))[0]
        m = trimesh.load(f, process=False)
        if len(m.faces) > target_faces:
            try:
                m = m.simplify_quadric_decimation(face_count=target_faces)
            except Exception:
                pass
        meshes[name] = dict(vertices=r(m.vertices.astype(float).tolist(), 5),
                            faces=m.faces.astype(int).tolist())
    return meshes


def build_frame(args):
    if args.grab_pre:
        return load_grab_preprocessed(args.grab_pre, args.contactdb, n_obj=args.n_obj)
    if args.grab:
        return load_grab_frame(args.grab, args.meshes, frame_idx=args.frame, n_obj=args.n_obj)
    return make_synthetic_grasp(args.object, n_obj=args.n_obj, seed=args.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="cylinder", choices=["cylinder", "sphere"])
    ap.add_argument("--grab", default=None, help="path to a GRAB sequence .npz")
    ap.add_argument("--meshes", default=None, help="GRAB contact_meshes dir")
    ap.add_argument("--grab-pre", dest="grab_pre", default=None,
                    help="preprocessed GRAB frame .npz from grab_extract.py")
    ap.add_argument("--contactdb", default=None, help="dir with contactdb *.stl meshes")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--n_obj", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "viewer" / "data_syn_cyl.js"))
    ap.add_argument("--var", default="TR_DATA", help="JS global to assign in the output")
    args = ap.parse_args()

    hand = WujiHand(URDF)
    frame = build_frame(args)
    print(f"[data] source={frame.meta} human_kpts={frame.human_kpts.shape} "
          f"obj_pts={frame.obj_pts.shape}")

    overt, ofaces = frame.obj.trimesh()
    link_meshes = decimate_link_meshes()
    print(f"[mesh] decimated {len(link_meshes)} robot link meshes")

    # run all ablations; capture per-iter robot link transforms for full-mesh render
    runs = {}
    edges = kinds = None
    for name, w in ABLATIONS.items():
        out = retarget(hand, frame, weights=w, n_iters=args.iters, lr=0.02, log_every=20)
        edges, kinds = out["edges"], out["edge_kinds"]
        trace = []
        import torch
        for rec in out["trace"]:
            q = torch.tensor(rec["q"]); d6 = torch.tensor(rec["d6"]); t = torch.tensor(rec["t"])
            poses = hand.all_link_poses(q[None], d6[None], t[None])
            link_T = {k: r(v.detach().cpu().numpy().reshape(-1).tolist(), 5)
                      for k, v in poses.items()}
            trace.append(dict(iter=rec["iter"], losses=r(rec["losses"]),
                              contact_mm=r(rec["contact_mm"], 2),
                              max_pen_mm=r(rec["max_pen_mm"], 2),
                              kpts=r(rec["kpts"]), link_T=link_T))
        runs[name] = dict(weights=out["weights"], trace=trace,
                          final=dict(contact_mm=r(out["final"]["contact_mm"], 2),
                                     max_pen_mm=r(out["final"]["max_pen_mm"], 2)))
        print(f"[run] {name:8s} contact={out['final']['contact_mm']:.2f}mm "
              f"max_pen={out['final']['max_pen_mm']:.2f}mm  ({len(trace)} frames)")

    data = dict(
        meta=dict(source=frame.meta, urdf="wuji-hand-right", n_joints=hand.n_joints,
                  ablations=list(ABLATIONS.keys())),
        object=dict(vertices=r(overt.astype(float).tolist(), 5),
                    faces=ofaces.astype(int).tolist()),
        human_kpts=r(frame.human_kpts.astype(float).tolist()),
        human_mesh=(dict(vertices=r(frame.hand_verts.astype(float).tolist(), 5),
                         faces=frame.hand_faces.astype(int).tolist())
                    if frame.hand_verts is not None else None),
        obj_pts=r(frame.obj_pts.astype(float).tolist()),
        edges=[[int(i), int(j), k] for (i, j), k in zip(edges, kinds)],
        bones=[[int(a), int(b)] for a, b in hand.bones],
        kp_frames=hand.kp_frames,
        robot_meshes=link_meshes,
        runs=runs,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"window.{args.var} = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";")
    mb = os.path.getsize(args.out) / 1e6
    print(f"[out] wrote {args.out}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
