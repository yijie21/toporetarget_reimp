"""GRAB sequence loader — the only source here with real hand *motion*.

ContactPose grasps are static, so nothing in the main table can say whether
`E_reg` does anything.  GRAB is a 120 Hz motion capture dataset, so it is where
the smoothness term is actually tested.

Every frame is expressed in the **object's** frame, which (a) matches how
ContactPose stores its grasps, so both datasets feed the identical `Frame`
interface, (b) states the task as "preserve the hand-object relationship"
rather than "track the object through the world", and (c) makes the object
static across the sequence, so its signed-distance structures are built once.

GRAB and MANO are licence-gated and are never redistributed here; the caller
supplies the paths to their own downloads.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import Frame
from .geometry import MeshObject
from .mano import ManoRightHand, rodrigues

TIP_IDS = [4, 8, 12, 16, 20]


def _normalise(name: str) -> str:
    """`cylinder_small` and `cylindersmall` name the same ContactDB object."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@lru_cache(maxsize=8)
def _mesh_index(mesh_dir: str) -> dict:
    out = {}
    for p in Path(mesh_dir).iterdir():
        if p.suffix.lower() in (".ply", ".stl", ".obj"):
            out[_normalise(p.stem)] = p
    return out


class GrabSequence:
    """One GRAB `.npz`, posed through MANO, in the object's frame."""

    def __init__(self, seq_npz, mano_model, subject_meshes_dir, object_meshes_dir):
        self.path = Path(seq_npz)
        z = np.load(self.path, allow_pickle=True)
        self.obj_name = str(z["obj_name"])
        self.n_frames = int(z["n_frames"])
        self.framerate = float(z["framerate"])
        self._rh = z["rhand"].item()["params"]
        self._obj = z["object"].item()["params"]

        vtemp_name = Path(str(z["rhand"].item()["vtemp"])).name
        vtemp_path = _find(subject_meshes_dir, vtemp_name)
        import trimesh
        v_template = np.asarray(trimesh.load(vtemp_path, process=False).vertices,
                                np.float64)
        self.hand = ManoRightHand(mano_model, v_template=v_template)

        mesh_path = _mesh_index(str(object_meshes_dir)).get(_normalise(self.obj_name))
        if mesh_path is None:
            raise FileNotFoundError(
                f"no mesh for object {self.obj_name!r} in {object_meshes_dir}")
        om = trimesh.load(mesh_path, process=False)
        verts = np.asarray(om.vertices, np.float64)
        if (verts.max(0) - verts.min(0)).max() > 5.0:      # ContactDB STLs are in mm
            verts = verts / 1000.0
        self.obj_verts, self.obj_faces = verts, np.asarray(om.faces, np.int32)
        self._object = MeshObject(self.obj_verts.astype(np.float32), self.obj_faces)

    def __len__(self):
        return self.n_frames

    def hand_in_object_frame(self, idx: int):
        """(verts (778,3), keypoints (21,3)) with the object at the origin."""
        verts, kpts = self.hand(self._rh["fullpose"][idx],
                                self._rh["global_orient"][idx],
                                self._rh["transl"][idx])
        R = rodrigues(self._obj["global_orient"][idx])[0]
        t = np.asarray(self._obj["transl"][idx], np.float64)
        return (verts - t) @ R, (kpts - t) @ R          # R^T applied on the right

    def frame(self, idx: int, n_obj: int = 80, seed: int | None = None) -> Frame:
        verts, kpts = self.hand_in_object_frame(idx)
        obj_pts, obj_nrm = self._object.sample_surface(n_obj, seed=seed if seed
                                                       is not None else 0)
        return Frame(human_kpts=kpts.astype(np.float32), obj=self._object,
                     obj_pts=obj_pts, obj_nrm=obj_nrm,
                     hand_verts=verts.astype(np.float32),
                     hand_faces=self.hand.faces.astype(np.int32),
                     meta=dict(source="grab", seq=self.path.stem,
                               object=self.obj_name, frame=int(idx),
                               framerate=self.framerate))

    def tip_distances_mm(self, idx: int) -> np.ndarray:
        """Fingertip-to-object-surface distances, used to find the grasp window."""
        from scipy.spatial import cKDTree
        _, kpts = self.hand_in_object_frame(idx)
        tree = _kdtree(self)
        return tree.query(kpts[TIP_IDS])[0] * 1000.0

    def grasp_window(self, max_tip_mm: float = 20.0, min_tips: int = 3,
                     stride: int = 1):
        """Frames where at least `min_tips` fingertips are near the object.

        A GRAB sequence is mostly reach and retreat; the retargeting problem only
        exists while the hand is actually holding the object.
        """
        keep = [i for i in range(0, self.n_frames, stride)
                if int((self.tip_distances_mm(i) <= max_tip_mm).sum()) >= min_tips]
        return keep


@lru_cache(maxsize=8)
def _kdtree_for(key):
    raise RuntimeError("internal")


def _kdtree(seq):
    from scipy.spatial import cKDTree
    if getattr(seq, "_tree", None) is None:
        seq._tree = cKDTree(seq.obj_verts)
    return seq._tree


def _find(root, name):
    hits = list(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]
