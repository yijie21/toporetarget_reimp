"""ContactPose loader — the dataset the paper's main table is built on.

ContactPose ships **3-D hand joints directly** in `annotations.json`, in the
object's own coordinate frame and in metres, so this path needs neither MANO nor
`chumpy` nor a second Python environment.  (The `oTw` entry in each frame is the
turntable pose of the capture rig and is only needed to project into images.)

Layout expected (after unzipping the official archives):

    <grasps_root>/<participant>_<intent>/<object>/annotations.json
    <models_dir>/<object>.ply                      # millimetres

Verified against participant 28: the 21 joints are in the OpenPose/MANO order
`[wrist, thumb x4, index x4, middle x4, ring x4, pinky x4]` — per-bone lengths
agree to <= 11.5 % CV across all 24 of that participant's grasps — and the mesh
needs only a millimetre-to-metre rescale to sit in the same frame (nearest joint
to surface: 1.4 mm).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import Frame
from .geometry import MeshObject

RIGHT, LEFT = 1, 0


def grasp_dir(grasps_root, object_name: str, participant: str = "full28",
              intent: str = "use") -> Path:
    return Path(grasps_root) / f"{participant}_{intent}" / object_name


def list_objects(grasps_root, participant: str = "full28", intent: str = "use"):
    """Objects for which this participant/intent actually has annotations."""
    d = Path(grasps_root) / f"{participant}_{intent}"
    return sorted(p.name for p in d.iterdir()
                  if (p / "annotations.json").exists())


def load_annotations(grasps_root, object_name, participant="full28", intent="use"):
    path = grasp_dir(grasps_root, object_name, participant, intent) / "annotations.json"
    with open(path) as f:
        return json.load(f)


def load_contactpose_frame(grasps_root, models_dir, object_name: str, *,
                           participant: str = "full28", intent: str = "use",
                           hand: int = RIGHT, n_obj: int = 80,
                           seed: int = 0) -> Frame:
    """One ContactPose grasp as a `Frame`, identical downstream to every other source."""
    import trimesh

    ann = load_annotations(grasps_root, object_name, participant, intent)
    h = ann["hands"][hand]
    if not h.get("valid", False):
        raise ValueError(f"{participant}_{intent}/{object_name}: "
                         f"{'right' if hand == RIGHT else 'left'} hand is not annotated")
    kpts = np.asarray(h["joints"], np.float32)          # (21,3), object frame, metres
    if kpts.shape != (21, 3):
        raise ValueError(f"expected 21 joints, got {kpts.shape}")

    mesh_path = Path(models_dir) / f"{object_name}.ply"
    m = trimesh.load(mesh_path, process=False)
    verts = np.asarray(m.vertices, np.float64) / 1000.0     # mm -> m
    faces = np.asarray(m.faces, np.int32)

    obj = MeshObject(verts.astype(np.float32), faces)
    obj_pts, obj_nrm = obj.sample_surface(n_obj, seed=seed)
    return Frame(
        human_kpts=kpts, obj=obj, obj_pts=obj_pts, obj_nrm=obj_nrm,
        meta=dict(source="contactpose", participant=participant, intent=intent,
                  object=object_name, hand="right" if hand == RIGHT else "left",
                  bimanual=bool(ann["hands"][LEFT].get("valid", False)
                                and ann["hands"][RIGHT].get("valid", False))),
    )
