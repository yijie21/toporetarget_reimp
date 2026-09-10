"""A minimal, dependency-light MANO right-hand forward model.

Why this exists: GRAB stores hand *parameters*, not joints, so turning a GRAB
sequence into keypoints normally means `smplx` + `chumpy`, and `chumpy` needs an
old NumPy and therefore a second Python environment.  The MANO pickle is almost
entirely plain arrays -- only `shapedirs` is a chumpy object, and we do not need
it, because GRAB ships each subject's own hand template (`vtemp`), which is
strictly better than posing the generic template at betas = 0.

So the whole dependency chain collapses to: read the pickle with a stub class
standing in for chumpy, then do standard linear blend skinning in NumPy.

The MANO model itself is licence-gated and is never redistributed here; the
caller supplies the path to their own copy.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# Fingertips are not MANO joints; these are the conventional tip vertices
# (the same indices `smplx` uses for the MANO hand).
TIP_VERTEX_IDS = dict(thumb=744, index=320, middle=443, ring=554, pinky=671)

# MANO's 16 joints are ordered [wrist, index x3, middle x3, pinky x3, ring x3,
# thumb x3]; we need [wrist, thumb, index, middle, ring, pinky] with tips.
_MANO_JOINT_ORDER = dict(thumb=(13, 14, 15), index=(1, 2, 3), middle=(4, 5, 6),
                         ring=(10, 11, 12), pinky=(7, 8, 9))
_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


class _Ch:
    """Stands in for `chumpy.Ch` while unpickling so chumpy need not be installed."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Ch
        return super().find_class(module, name)


def load_mano(model_path) -> dict:
    """Read `MANO_RIGHT.pkl` into plain NumPy arrays, with no chumpy involved."""
    with open(model_path, "rb") as f:
        raw = _Unpickler(f, encoding="latin1").load()
    jr = raw["J_regressor"]
    return dict(
        v_template=np.asarray(raw["v_template"], np.float64),
        J_regressor=np.asarray(jr.toarray() if hasattr(jr, "toarray") else jr, np.float64),
        posedirs=np.asarray(raw["posedirs"], np.float64),
        weights=np.asarray(raw["weights"], np.float64),
        kintree_table=np.asarray(raw["kintree_table"], np.int64),
        faces=np.asarray(raw["f"], np.int64),
        hands_mean=np.asarray(raw["hands_mean"], np.float64),
    )


def rodrigues(aa: np.ndarray) -> np.ndarray:
    """(...,3) axis-angle -> (...,3,3) rotation matrices."""
    aa = np.asarray(aa, np.float64).reshape(-1, 3)
    theta = np.linalg.norm(aa, axis=1, keepdims=True)
    k = np.divide(aa, theta, out=np.zeros_like(aa), where=theta > 1e-12)
    K = np.zeros((len(aa), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    th = theta[:, :, None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class ManoRightHand:
    """Posed MANO right hand: parameters in, 778 vertices and 21 keypoints out."""

    def __init__(self, model_path, v_template=None):
        self.m = load_mano(model_path)
        if v_template is not None:                      # a subject's own template
            v_template = np.asarray(v_template, np.float64)
            if v_template.shape != self.m["v_template"].shape:
                raise ValueError(f"hand template must be {self.m['v_template'].shape}, "
                                 f"got {v_template.shape}")
            self.m["v_template"] = v_template
        parents = self.m["kintree_table"][0].copy()
        parents[0] = -1
        self.parents = parents
        self.faces = self.m["faces"]

    def __call__(self, fullpose, global_orient, transl=None, flat_hand_mean=True):
        """fullpose (45,) axis-angle, global_orient (3,), transl (3,).

        `flat_hand_mean=True` matches how GRAB stores its parameters: the pose is
        used as-is rather than as an offset from MANO's mean hand pose.
        """
        m = self.m
        pose = np.asarray(fullpose, np.float64).reshape(45)
        if not flat_hand_mean:
            pose = pose + m["hands_mean"]
        full = np.concatenate([np.asarray(global_orient, np.float64).reshape(3), pose])
        R = rodrigues(full.reshape(16, 3))                       # (16,3,3)

        v_shaped = m["v_template"]
        J = m["J_regressor"] @ v_shaped                          # (16,3)
        pose_feature = (R[1:] - np.eye(3)).reshape(135)
        offset = (m["posedirs"].reshape(-1, 135) @ pose_feature).reshape(-1, 3)
        v_posed = v_shaped + offset

        # rigid transform chain
        A = np.zeros((16, 4, 4))
        A[0] = _rt(R[0], J[0])
        for i in range(1, 16):
            A[i] = A[self.parents[i]] @ _rt(R[i], J[i] - J[self.parents[i]])
        joints = A[:, :3, 3].copy()
        # subtract the rest pose so the template is not translated twice
        A_rel = A.copy()
        A_rel[:, :3, 3] -= np.einsum("nij,nj->ni", A[:, :3, :3], J)

        T = np.einsum("vn,nij->vij", m["weights"], A_rel)        # (778,4,4)
        verts = np.einsum("vij,vj->vi", T[:, :3, :3], v_posed) + T[:, :3, 3]

        if transl is not None:
            t = np.asarray(transl, np.float64).reshape(3)
            verts, joints = verts + t, joints + t
        return verts, self._keypoints21(verts, joints)

    def _keypoints21(self, verts, joints):
        """[wrist, thumb x4, index x4, middle x4, ring x4, pinky x4] in MANO order."""
        out = [joints[0]]
        for f in _FINGERS:
            out += [joints[j] for j in _MANO_JOINT_ORDER[f]]
            out.append(verts[TIP_VERTEX_IDS[f]])
        return np.asarray(out, np.float64)


def _rt(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T
