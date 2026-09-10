"""Data interface.

A `Frame` is everything the optimiser needs for one timestep:
  - human_kpts : (21,3) MANO-order hand keypoints (the demonstration)
  - obj        : an object with .sdf / .sample_surface / .trimesh
  - obj_pts    : (No,3) sampled object-surface points (the interaction anchors)
  - obj_nrm    : (No,3) their outward normals

Two sources are provided:
  * make_synthetic_grasp(...)  -- zero downloads, fully controllable
  * load_grab_frame(...)       -- reads a GRAB sequence the user has downloaded
"""
import os
from dataclasses import dataclass
import numpy as np

from .geometry import Sphere, Cylinder, MeshObject

MANO_GROUPS = ["thumb", "index", "middle", "ring", "pinky"]


@dataclass
class Frame:
    human_kpts: np.ndarray   # (21,3)
    obj: object
    obj_pts: np.ndarray      # (No,3)
    obj_nrm: np.ndarray      # (No,3)
    meta: dict
    hand_verts: np.ndarray = None   # (Vh,3) MANO surface, world (optional)
    hand_faces: np.ndarray = None   # (Fh,3)


# --------------------------------------------------------------------------- #
# Synthetic MANO-topology grasp around a primitive.                            #
# --------------------------------------------------------------------------- #
def make_synthetic_grasp(kind="cylinder", n_obj=50, seed=0, hand_scale=0.16):
    """Build a plausible power/precision grasp whose fingertips contact the object.

    The hand is intentionally a *different size* from the Wuji hand to exercise the
    embodiment gap.  All 21 keypoints are returned in MANO order.
    """
    rng = np.random.default_rng(seed)
    if kind == "cylinder":
        # tall-ish can: height prevents fingers from cheating over the top rim,
        # so the retargeted grasp must press the lateral wall like the human.
        obj = Cylinder(center=[0.0, 0.0, 0.0], radius=0.042, hz=0.11)
        R = obj.radius
    elif kind == "sphere":
        obj = Sphere(center=[0.0, 0.0, 0.0], radius=0.05)
        R = obj.radius
    else:
        raise ValueError(kind)

    # Wrist sits on the -x side, fingers wrap toward +/-x around the object.
    wrist = np.array([-(R + 0.085), 0.0, -0.01], np.float32)

    # The four fingers (index..pinky): contact angles around the cylinder near the
    # front (angle pi = -x side), spread in height(z) and wrap angle.
    # name : (wrap_angle_rad, height_z, y_knuckle).  Thumb opposes from +y side.
    finger_specs = {
        "thumb":  (np.pi * 0.5,  0.015,  0.045),
        "index":  (np.pi - 0.55, 0.045,  0.028),
        "middle": (np.pi - 0.20, 0.050,  0.010),
        "ring":   (np.pi + 0.15, 0.040, -0.010),
        "pinky":  (np.pi + 0.50, 0.028, -0.030),
    }
    kpts = np.zeros((21, 3), np.float32)
    kpts[0] = wrist

    def surface_point(ang, z):
        if kind == "cylinder":
            return obj.center + np.array([R * np.cos(ang), R * np.sin(ang), z], np.float32)
        d = np.array([np.cos(ang), np.sin(ang) * 0.6, z / max(R, 1e-6)])
        d = d / np.linalg.norm(d)
        return (obj.center + R * d).astype(np.float32)

    def place_finger(slot, mcp, tip, bulge):
        outward = mcp - obj.center
        outward = outward / (np.linalg.norm(outward) + 1e-8)
        pip = 0.45 * mcp + 0.55 * tip + bulge * outward
        dip = 0.18 * mcp + 0.82 * tip + 0.5 * bulge * outward
        base = 1 + slot * 4
        kpts[base:base + 4] = np.stack([mcp, pip, dip, tip])

    for name, (ang, z, yk) in finger_specs.items():
        slot = MANO_GROUPS.index(name)
        tip = surface_point(ang, z)
        mcp = wrist + np.array([0.035, yk, z * 0.5], np.float32)
        place_finger(slot, mcp.astype(np.float32), tip, bulge=0.011)

    # rescale the whole hand about the wrist to `hand_scale` reach (embodiment gap)
    reach = np.linalg.norm(kpts[8] - kpts[0]) + 1e-8
    kpts = wrist + (kpts - wrist) * (hand_scale / reach)
    # re-snap ALL five tips onto the surface so every finger truly contacts
    for name, (ang, z, _) in finger_specs.items():
        slot = MANO_GROUPS.index(name)
        kpts[1 + slot * 4 + 3] = surface_point(ang, z)
    kpts += rng.normal(scale=0.0008, size=kpts.shape).astype(np.float32)

    obj_pts, obj_nrm = obj.sample_surface(n_obj, seed=seed)
    return Frame(human_kpts=kpts, obj=obj, obj_pts=obj_pts, obj_nrm=obj_nrm,
                 meta={"source": "synthetic", "kind": kind})


# --------------------------------------------------------------------------- #
# GRAB loader.                                                                 #
# --------------------------------------------------------------------------- #
def load_grab_frame(seq_npz, contact_meshes_dir, frame_idx=0, n_obj=50):
    """Load one frame from a GRAB sequence the user has downloaded.

    GRAB layout (after you register & download from https://grab.is.tue.mpg.de):
        GRAB/
          grab/<subject>/<seq>.npz        # per-frame params (object + body/hands)
          tools/object_meshes/contact_meshes/<obj>.ply

    The .npz stores SMPL-X / MANO params and the object 6-DoF pose.  Getting the 21
    hand keypoints requires running the body/hand model (SMPL-X or MANO) -- both are
    license-gated.  This loader expects you to have produced, per frame, either:
      (a) 'rhand_joints' : (T,21,3) world-frame right-hand keypoints, or
      (b) MANO params we can forward through `smplx` (if installed + models present).
    It always reads the object mesh + 6-DoF pose directly (ungated).
    """
    import os, numpy as np, trimesh
    data = np.load(seq_npz, allow_pickle=True)

    # ---- object: mesh + per-frame 6-DoF pose ---------------------------------
    obj_name = str(data["obj_name"]) if "obj_name" in data else \
        str(data["object"].item().get("name"))
    mesh_path = os.path.join(contact_meshes_dir, f"{obj_name}.ply")
    om = trimesh.load(mesh_path, process=False)
    verts0 = np.asarray(om.vertices, np.float32)
    faces = np.asarray(om.faces, np.int32)

    obj_params = data["object"].item() if "object" in data else data
    transl = np.asarray(obj_params["params"]["transl"])[frame_idx]      # (3,)
    glob = np.asarray(obj_params["params"]["global_orient"])[frame_idx]  # (3,) axis-angle
    Rm = _aa_to_R(glob)
    verts = (verts0 @ Rm.T) + transl
    obj = MeshObject(verts.astype(np.float32), faces)

    # ---- right-hand 21 keypoints --------------------------------------------
    if "rhand_joints" in data:
        kpts = np.asarray(data["rhand_joints"])[frame_idx][:21].astype(np.float32)
    else:
        kpts = _grab_hand_via_smplx(data, frame_idx)

    obj_pts, obj_nrm = obj.sample_surface(n_obj, seed=frame_idx)
    return Frame(human_kpts=kpts, obj=obj, obj_pts=obj_pts, obj_nrm=obj_nrm,
                 meta={"source": "grab", "obj_name": obj_name, "frame": frame_idx})


def load_grab_preprocessed(npz_path, contactdb_dir, n_obj=80):
    """Load a frame produced by grab_extract.py (21 keypoints + object pose) and
    pose the contactdb object mesh into world coordinates."""
    import trimesh
    z = np.load(npz_path, allow_pickle=True)
    kpts = z["human_kpts"].astype(np.float32)
    mesh_name = str(z["obj_mesh_name"])
    stl = os.path.join(contactdb_dir, mesh_name + ".stl")
    om = trimesh.load(stl, process=False)
    verts0 = np.asarray(om.vertices, np.float64)
    faces = np.asarray(om.faces, np.int32)
    # contactdb STLs are in millimetres; GRAB poses are in metres
    if (verts0.max(0) - verts0.min(0)).max() > 5.0:
        verts0 = verts0 / 1000.0
    R = _aa_to_R(np.asarray(z["obj_orient"], np.float64))
    verts = (verts0 @ R.T) + np.asarray(z["obj_transl"], np.float64)
    # recentre the scene at the object centroid (GRAB lives at standing height)
    c = verts.mean(0)
    verts = verts - c
    kpts = (kpts - c).astype(np.float32)
    obj = MeshObject(verts.astype(np.float32), faces)
    obj_pts, obj_nrm = obj.sample_surface(n_obj, seed=int(z["frame"]))
    hv = (np.asarray(z["hand_verts"], np.float64) - c).astype(np.float32) if "hand_verts" in z else None
    hf = np.asarray(z["hand_faces"], np.int32) if "hand_faces" in z else None
    return Frame(human_kpts=kpts, obj=obj, obj_pts=obj_pts, obj_nrm=obj_nrm,
                 hand_verts=hv, hand_faces=hf,
                 meta={"source": "grab", "seq": str(z["seq"]), "obj": str(z["obj_name"]),
                       "frame": int(z["frame"])})


def _aa_to_R(aa):
    th = np.linalg.norm(aa) + 1e-12
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _grab_hand_via_smplx(data, frame_idx):
    raise NotImplementedError(
        "No precomputed 'rhand_joints' found in the GRAB npz. Either preprocess the "
        "sequence with the GRAB tools to dump right-hand joints, or install `smplx` "
        "with the gated SMPL-X/MANO models and extend _grab_hand_via_smplx().")
