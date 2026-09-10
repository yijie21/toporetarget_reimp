"""Evaluation metrics, implemented to match the definitions in the TopoRetarget
paper (arXiv:2606.16272, Eq. 10 and Eq. 12).

Two deliberate differences from the optimiser live here:

* the optimiser uses a differentiable *surrogate* signed distance for meshes;
  evaluation uses an **exact** signed distance, because it needs no gradients.
* the optimiser penalises a sparse set of keypoints and bone midpoints;
  evaluation samples the **actual robot link surfaces**, as Eq. 12 requires.

Everything here takes a solved configuration `(q, d6, t)` and a `Frame`, so the
same code evaluates our method and every baseline.
"""
from __future__ import annotations

import numpy as np
import torch
import trimesh

MM = 1000.0


# --------------------------------------------------------------------------- #
# Exact signed distance (evaluation only)                                      #
# --------------------------------------------------------------------------- #
class ExactSDF:
    """Exact signed distance to an object, with our sign convention: >0 outside.

    `trimesh.proximity.signed_distance` returns *positive inside*, so the sign is
    flipped here.  Sign correctness relies on the mesh being watertight; the flag
    is exposed so evaluation reports can state it rather than hide it.
    """

    def __init__(self, obj):
        # analytic primitives expose closed-form queries -> use them (zero
        # discretisation error); arbitrary meshes fall back to trimesh.
        self.analytic = hasattr(obj, "exact_signed_distance")
        self._obj = obj
        if self.analytic:
            self.watertight = True
            return
        verts, faces = obj.trimesh()
        self.mesh = trimesh.Trimesh(np.asarray(verts, np.float64),
                                    np.asarray(faces, np.int64), process=False)
        self.watertight = bool(self.mesh.is_watertight)
        self._pq = trimesh.proximity.ProximityQuery(self.mesh)

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) -> (N,) signed distance, positive outside the object."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        if self.analytic:
            return np.asarray(self._obj.exact_signed_distance(pts), np.float64)
        return -np.asarray(self._pq.signed_distance(pts), np.float64)

    def nearest_surface(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) -> (N,3) closest point on the object surface."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        if self.analytic:
            return np.asarray(self._obj.nearest_surface(pts), np.float64)
        closest, _, _ = self._pq.on_surface(pts)
        return np.asarray(closest, np.float64)


# --------------------------------------------------------------------------- #
# Robot hand surface sampling (Eq. 12 needs surface points, not keypoints)      #
# --------------------------------------------------------------------------- #
def robot_surface_points(hand, q, d6, t, n_per_link: int = 200, seed: int = 0):
    """World-frame samples of the whole robot hand surface -> (N,3) float64.

    Delegates to the hand's own (differentiable, cached) sampler so evaluation
    and optimisation are guaranteed to describe the same geometry.
    """
    with torch.no_grad():
        pts = hand.surface_points(_t(q)[None], _t(d6)[None], _t(t)[None],
                                  n_per_link=n_per_link, seed=seed)[0]
    return pts.cpu().numpy().astype(np.float64)


def _t(x):
    return x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, np.float32))


# --------------------------------------------------------------------------- #
# Eq. 12 — maximum penetration depth                                           #
# --------------------------------------------------------------------------- #
def max_penetration_mm(sdf: ExactSDF, surface_pts: np.ndarray) -> float:
    """D_pen^max = max_x [-d_M(x)]_+ over sampled robot-hand surface points, in mm."""
    phi = sdf.signed_distance(surface_pts)
    return float(np.clip(-phi, 0.0, None).max() * MM)


def mean_penetration_mm(sdf: ExactSDF, surface_pts: np.ndarray) -> float:
    """Mean depth over the penetrating points only (0 if nothing penetrates)."""
    depth = np.clip(-sdf.signed_distance(surface_pts), 0.0, None)
    pen = depth[depth > 0]
    return float(pen.mean() * MM) if pen.size else 0.0


# --------------------------------------------------------------------------- #
# Eq. 10 — contact precision error                                             #
# --------------------------------------------------------------------------- #
def contact_set(sdf: ExactSDF, human_kpts: np.ndarray, tau_mm: float = 10.0):
    """C = hand keypoints that are in contact with the object in the SOURCE demo.

    Contact is defined on the human demonstration only, so the same set is scored
    for every method — a method cannot improve its score by contacting elsewhere.
    """
    d = np.abs(sdf.signed_distance(human_kpts)) * MM
    idx = np.where(d <= tau_mm)[0]
    if idx.size == 0:                      # degenerate demo: fall back to the 5 tips
        idx = np.array([4, 8, 12, 16, 20])
    return idx


def contact_precision_error_mm(sdf: ExactSDF, robot_kpts: np.ndarray,
                               human_kpts: np.ndarray, contact_idx: np.ndarray,
                               mode: str = "surface") -> float:
    """E_prec (Eq. 10), in mm.

    `o_c` is under-specified in the paper.  Two readings are implemented:

    * ``surface`` (default, what we report): ``o_c`` is the closest point on the
      object surface to that hand point, computed separately for the robot and
      for the human.  ``h_c - o_c`` is then the *contact offset vector*, and the
      metric asks whether the robot reproduces the human's contact offsets.  It
      is invariant to sliding along the surface, which is what makes it a
      *contact* metric rather than a keypoint metric.
    * ``origin``: ``o_c`` is the object origin.  Because the object pose is
      identical in source and retarget, this collapses to the plain keypoint
      error ``||h_c^r - h_c^s||``.  Reported alongside for transparency.
    """
    hr = np.asarray(robot_kpts, np.float64)[contact_idx]
    hs = np.asarray(human_kpts, np.float64)[contact_idx]
    if mode == "surface":
        off_r = hr - sdf.nearest_surface(hr)
        off_s = hs - sdf.nearest_surface(hs)
    elif mode == "origin":
        off_r, off_s = hr, hs
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return float(np.linalg.norm(off_r - off_s, axis=-1).mean() * MM)


# --------------------------------------------------------------------------- #
# One-call evaluation, shared by our method and every baseline                  #
# --------------------------------------------------------------------------- #
def evaluate(hand, frame, q, d6, t, *, tau_mm: float = 10.0,
             n_per_link: int = 200, seed: int = 0) -> dict:
    """Score one solved configuration against one frame."""
    sdf = ExactSDF(frame.obj)
    kp = hand.keypoints(_t(q)[None], _t(d6)[None], _t(t)[None])[0]
    kp = kp.detach().cpu().numpy().astype(np.float64)
    surf = robot_surface_points(hand, q, d6, t, n_per_link=n_per_link, seed=seed)
    cidx = contact_set(sdf, frame.human_kpts, tau_mm)
    return dict(
        E_prec_mm=contact_precision_error_mm(sdf, kp, frame.human_kpts, cidx, "surface"),
        E_prec_origin_mm=contact_precision_error_mm(sdf, kp, frame.human_kpts, cidx, "origin"),
        D_pen_max_mm=max_penetration_mm(sdf, surf),
        D_pen_mean_mm=mean_penetration_mm(sdf, surf),
        n_contacts=int(len(cidx)),
        n_surface_pts=int(len(surf)),
        watertight=sdf.watertight,
    )
