#!/usr/bin/env python3
"""Search GRAB s1 for a clean grasp frame per object and dump a preprocessed npz
(21 keypoints + MANO mesh + object pose) for each.  RUN IN conda 'manoconv' env.

    conda run -n manoconv python grab_showcase.py
"""
import glob, os
import numpy as np, torch, trimesh, smplx
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "models")
SEQDIR = os.path.join(HERE, "..", "grab_raw", "s1_data", "s1")
MESHDIR = os.path.join(HERE, "..", "grab_raw", "contactdb_meshes")
OUT = os.path.join(HERE, "grab_frames")

TIP = dict(thumb=744, index=320, middle=443, ring=554, pinky=671)
MJ = dict(thumb=[13, 14, 15], index=[1, 2, 3], middle=[4, 5, 6], ring=[10, 11, 12], pinky=[7, 8, 9])
ORD = ["thumb", "index", "middle", "ring", "pinky"]
NAME = {"cubesmall": "cube_small", "cubemedium": "cube_medium", "cubelarge": "cube_large",
        "cylindersmall": "cylinder_small", "cylindermedium": "cylinder_medium",
        "spheresmall": "sphere_small", "spheremedium": "sphere_medium", "spherelarge": "sphere_large",
        "waterbottle": "water_bottle", "wineglass": "wine_glass", "phone": "cell_phone",
        "stanfordbunny": "stanford_bunny", "rubberduck": "rubber_duck", "duck": "rubber_duck",
        "pyramidmedium": "pyramid_medium", "torusmedium": "torus_medium"}

# (showcase key, GRAB object token)
SHOWCASES = [
    ("grab_wineglass", "wineglass"),
    ("grab_hammer",    "hammer"),
    ("grab_flute",     "flute"),
]


def aaR(aa):
    th = np.linalg.norm(aa) + 1e-12; k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def kpts_from(J, V):
    kp = np.zeros((21, 3), np.float32); kp[0] = J[0]
    for fs, n in enumerate(ORD):
        b = 1 + fs * 4
        kp[b:b + 3] = J[MJ[n]]; kp[b + 3] = V[TIP[n]]
    return kp


def best_frame_for(obj_token):
    seqs = sorted(glob.glob(os.path.join(SEQDIR, f"{obj_token}_*.npz")))
    mn = NAME.get(obj_token, obj_token)
    stl = os.path.join(MESHDIR, mn + ".stl")
    if not seqs or not os.path.exists(stl):
        return None
    v = np.asarray(trimesh.load(stl, process=False).vertices, float)
    if (v.max(0) - v.min(0)).max() > 5:
        v = v / 1000.0
    best = None
    for seq in seqs:
        d = np.load(seq, allow_pickle=True)
        rp = d["rhand"].item()["params"]; op = d["object"].item()["params"]
        T = len(rp["transl"]); fr = list(range(0, T, max(1, T // 40))); B = len(fr)
        mano = smplx.create(MODEL, model_type="mano", is_rhand=True, use_pca=False,
                            flat_hand_mean=True, batch_size=B)
        out = mano(betas=torch.zeros(B, 10),
                   global_orient=torch.tensor(rp["global_orient"][fr], dtype=torch.float32),
                   hand_pose=torch.tensor(rp["fullpose"][fr], dtype=torch.float32),
                   transl=torch.tensor(rp["transl"][fr], dtype=torch.float32), return_verts=True)
        J = out.joints.detach().numpy(); V = out.vertices.detach().numpy()
        for ii, f in enumerate(fr):
            tips = np.stack([V[ii, TIP[n]] for n in ORD])
            R = aaR(op["global_orient"][f]); vw = v @ R.T + op["transl"][f]
            dmean = float(cKDTree(vw).query(tips)[0].mean())
            if best is None or dmean < best[0]:
                best = (dmean, seq, f, J[ii].copy(), V[ii].copy(),
                        op["global_orient"][f].copy(), op["transl"][f].copy(), mn,
                        np.asarray(mano.faces, np.int32))
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    done = []
    for key, tok in SHOWCASES:
        b = best_frame_for(tok)
        if b is None:
            print(f"[skip] {key} ({tok}): no seq/mesh"); continue
        dmean, seq, f, J, V, oo, ot, mn, faces = b
        kp = kpts_from(J, V)
        out = os.path.join(OUT, key + ".npz")
        np.savez(out, human_kpts=kp, hand_verts=V.astype(np.float32), hand_faces=faces,
                 obj_mesh_name=mn, obj_name=tok, obj_transl=ot.astype(np.float32),
                 obj_orient=oo.astype(np.float32), seq=os.path.basename(seq), frame=int(f))
        print(f"[ok] {key:16s} {os.path.basename(seq):26s} f={f:4d} meanTip={dmean*1000:5.1f}mm -> {out}")
        done.append(key)
    print("DONE:", done)


if __name__ == "__main__":
    main()
