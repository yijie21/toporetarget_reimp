#!/usr/bin/env python3
"""Preprocess one GRAB frame -> 21 MANO-order right-hand keypoints + object pose.

RUN IN THE conda 'manoconv' ENV (needs smplx + chumpy + torch):
    conda run -n manoconv python grab_extract.py \
        --seq  ../grab_raw/s1_data/s1/mug_drink_1.npz \
        --frame 777 --out grab_frames/mug_drink_1_f777.npz

Outputs a tiny npz the main (py3.13) pipeline reads via data.load_grab_preprocessed().
"""
import argparse, os
import numpy as np
import torch
import smplx

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "models")  # contains mano/MANO_RIGHT.pkl

# smplx MANO joint order (16): 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky,
# 10-12 ring, 13-15 thumb.  MANO right-hand fingertip vertex indices:
TIP_VERTS = dict(thumb=744, index=320, middle=443, ring=554, pinky=671)
# remap to OUR order: [wrist, thumb(mcp,pip,dip), index(...), middle, ring, pinky]
MANO_JOINT = dict(thumb=[13, 14, 15], index=[1, 2, 3], middle=[4, 5, 6],
                  ring=[10, 11, 12], pinky=[7, 8, 9])
ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# GRAB object-name -> contactdb stl name
NAME_MAP = {
    "cubesmall": "cube_small", "cubelarge": "cube_large", "cubemedium": "cube_medium",
    "cylindersmall": "cylinder_small", "cylinderlarge": "cylinder_large",
    "cylindermedium": "cylinder_medium", "spheresmall": "sphere_small",
    "spherelarge": "sphere_large", "spheremedium": "sphere_medium",
    "pyramidsmall": "pyramid_small", "pyramidlarge": "pyramid_large",
    "pyramidmedium": "pyramid_medium", "torussmall": "torus_small",
    "toruslarge": "torus_large", "torusmedium": "torus_medium",
    "waterbottle": "water_bottle", "wineglass": "wine_glass", "phone": "cell_phone",
    "doorknob": "door_knob", "lightbulb": "light_bulb", "pschips": "ps_controller",
    "alarmclock": "alarm_clock", "stanfordbunny": "stanford_bunny",
    "flashlight": "flashlight", "rubberduck": "rubber_duck", "duck": "rubber_duck",
    "teapot": "utah_teapot", "watch": "wristwatch", "stamp": "stapler",
}


def aa_to_R(aa):
    th = np.linalg.norm(aa) + 1e-12
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frame", type=int, default=None, help="default: peak object contact")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.seq, allow_pickle=True)
    obj_name = str(d["obj_name"])
    rp = d["rhand"].item()["params"]
    op = d["object"].item()["params"]
    contact_obj = d["contact"].item()["object"]
    fi = args.frame if args.frame is not None else int(np.argmax((contact_obj > 0).sum(1)))

    # ---- MANO forward for the single frame -------------------------------
    mano = smplx.create(MODEL_PATH, model_type="mano", is_rhand=True,
                        use_pca=False, flat_hand_mean=True, batch_size=1)
    out = mano(global_orient=torch.tensor(rp["global_orient"][fi][None], dtype=torch.float32),
               hand_pose=torch.tensor(rp["fullpose"][fi][None], dtype=torch.float32),
               transl=torch.tensor(rp["transl"][fi][None], dtype=torch.float32),
               return_verts=True)
    J = out.joints[0].detach().numpy()      # (16,3)
    V = out.vertices[0].detach().numpy()    # (778,3)

    kpts = np.zeros((21, 3), np.float32)
    kpts[0] = J[0]
    for fslot, name in enumerate(ORDER):
        base = 1 + fslot * 4
        j = MANO_JOINT[name]
        kpts[base + 0] = J[j[0]]
        kpts[base + 1] = J[j[1]]
        kpts[base + 2] = J[j[2]]
        kpts[base + 3] = V[TIP_VERTS[name]]

    # ---- object pose -----------------------------------------------------
    obj_transl = op["transl"][fi].astype(np.float32)
    obj_orient = op["global_orient"][fi].astype(np.float32)
    mesh_name = NAME_MAP.get(obj_name, obj_name)

    hand_faces = np.asarray(mano.faces, np.int32)   # MANO surface (1538,3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, human_kpts=kpts, hand_verts=V.astype(np.float32), hand_faces=hand_faces,
             obj_mesh_name=mesh_name, obj_name=obj_name,
             obj_transl=obj_transl, obj_orient=obj_orient, seq=os.path.basename(args.seq),
             frame=fi)
    # quick contact sanity: distance of fingertips to posed object centroid
    print(f"[grab_extract] {os.path.basename(args.seq)} frame={fi} obj={obj_name}->{mesh_name}")
    print(f"  wrist={kpts[0].round(3)}  obj_transl={obj_transl.round(3)}")
    print(f"  hand-obj centroid dist={np.linalg.norm(kpts[0]-obj_transl):.3f} m  -> wrote {args.out}")


if __name__ == "__main__":
    main()
