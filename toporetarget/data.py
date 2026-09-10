"""Data interface.

A `Frame` is everything the optimiser needs for one timestep:
  - human_kpts : (21,3) MANO-order hand keypoints (the demonstration)
  - obj        : an object with .sdf / .sample_surface / .trimesh
  - obj_pts    : (No,3) sampled object-surface points (the interaction anchors)
  - obj_nrm    : (No,3) their outward normals

Sources build `Frame`s and are interchangeable downstream:
  * `make_synthetic_grasp(...)` here -- zero downloads, fully controllable
  * `toporetarget.contactpose` -- the benchmark
  * `toporetarget.grab`        -- real motion, for the sequence experiment
"""
from dataclasses import dataclass

import numpy as np

from .geometry import Cylinder, Sphere

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
