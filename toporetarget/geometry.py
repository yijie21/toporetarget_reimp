"""Geometry helpers: rotation parameterisation, Procrustes init, and object SDFs.

All SDFs follow the convention: phi(p) > 0 outside, < 0 inside, == 0 on the
surface.  They are differentiable w.r.t. the query points so the penetration
loss can push robot keypoints out of the object.
"""
import numpy as np
import torch

EPS = 1e-8
# Epsilon added *inside* a sqrt must be far smaller than EPS: sqrt(1e-8) is
# 1e-4 m = 0.1 mm, which biased the cylinder SDF by that much everywhere inside
# the object.  sqrt(1e-18) is 1e-9 m, negligible against millimetre metrics,
# while still keeping the gradient finite at the degenerate point.
SQRT_EPS = 1e-18


# --------------------------------------------------------------------------- #
# Rotation parameterisation: continuous 6-D representation (Zhou et al. 2019). #
# --------------------------------------------------------------------------- #
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(...,6) -> (...,3,3) rotation matrix via Gram-Schmidt on two 3-vectors."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + EPS)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / (a2.norm(dim=-1, keepdim=True) + EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns are the basis vectors


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """3x3 -> 6-D (first two columns), the inverse used for initialisation."""
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def kabsch(A: np.ndarray, B: np.ndarray):
    """Best-fit rigid transform mapping A -> B (no scale).  Returns (R, t)."""
    muA, muB = A.mean(0), B.mean(0)
    H = (A - muA).T @ (B - muB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = muB - R @ muA
    return R.astype(np.float32), t.astype(np.float32)


# --------------------------------------------------------------------------- #
# Object primitives                                                           #
# --------------------------------------------------------------------------- #
class Sphere:
    kind = "sphere"

    def __init__(self, center, radius):
        self.center = np.asarray(center, np.float32)
        self.radius = float(radius)

    def sdf(self, p):  # p: (...,3) torch
        c = torch.as_tensor(self.center, dtype=p.dtype, device=p.device)
        return (p - c).norm(dim=-1) - self.radius

    def sample_surface(self, n, seed=0):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + EPS
        pts = self.center + self.radius * v
        return pts.astype(np.float32), v.astype(np.float32)

    # ---- exact (non-differentiable) queries used by evaluation --------------- #
    def exact_signed_distance(self, pts):
        v = np.asarray(pts, np.float64) - self.center
        return np.linalg.norm(v, axis=-1) - self.radius

    def nearest_surface(self, pts):
        v = np.asarray(pts, np.float64) - self.center
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        n = np.where(n < EPS, 1.0, n)
        return self.center + self.radius * v / n

    def trimesh(self):
        import trimesh
        m = trimesh.creation.icosphere(subdivisions=3, radius=self.radius)
        m.apply_translation(self.center)
        return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)


class Cylinder:
    """Axis-aligned along +z, centred at `center`, with `radius` and half-height `hz`."""
    kind = "cylinder"

    def __init__(self, center, radius, hz):
        self.center = np.asarray(center, np.float32)
        self.radius = float(radius)
        self.hz = float(hz)

    def sdf(self, p):
        c = torch.as_tensor(self.center, dtype=p.dtype, device=p.device)
        q = p - c
        d_rad = torch.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 + SQRT_EPS) - self.radius
        d_z = torch.abs(q[..., 2]) - self.hz
        # 2-D rounded-box SDF in (radial, axial) coordinates
        ax = torch.clamp(d_rad, min=0.0)
        az = torch.clamp(d_z, min=0.0)
        outside = torch.sqrt(ax ** 2 + az ** 2 + SQRT_EPS)
        inside = torch.clamp(torch.maximum(d_rad, d_z), max=0.0)
        return outside + inside

    def sample_surface(self, n, seed=0):
        rng = np.random.default_rng(seed)
        pts, nrm = [], []
        for _ in range(n):
            if rng.random() < 0.8:  # lateral wall
                th = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(-self.hz, self.hz)
                d = np.array([np.cos(th), np.sin(th), 0.0])
                pts.append(self.center + np.array([self.radius * np.cos(th),
                                                   self.radius * np.sin(th), z]))
                nrm.append(d)
            else:  # caps
                th = rng.uniform(0, 2 * np.pi)
                r = self.radius * np.sqrt(rng.random())
                z = self.hz if rng.random() < 0.5 else -self.hz
                pts.append(self.center + np.array([r * np.cos(th), r * np.sin(th), z]))
                nrm.append(np.array([0, 0, np.sign(z)]))
        return np.asarray(pts, np.float32), np.asarray(nrm, np.float32)

    # ---- exact (non-differentiable) queries used by evaluation --------------- #
    def exact_signed_distance(self, pts):
        q = np.asarray(pts, np.float64) - self.center
        d_rad = np.hypot(q[..., 0], q[..., 1]) - self.radius
        d_z = np.abs(q[..., 2]) - self.hz
        outside = np.hypot(np.clip(d_rad, 0, None), np.clip(d_z, 0, None))
        inside = np.clip(np.maximum(d_rad, d_z), None, 0)
        return outside + inside

    def nearest_surface(self, pts):
        q = np.asarray(pts, np.float64) - self.center
        rho = np.hypot(q[..., 0], q[..., 1])
        safe = np.where(rho < EPS, 1.0, rho)
        ux, uy = q[..., 0] / safe, q[..., 1] / safe
        # candidate on the lateral wall, and on the nearer cap
        wall = np.stack([self.radius * ux, self.radius * uy,
                         np.clip(q[..., 2], -self.hz, self.hz)], -1)
        r_cap = np.minimum(rho, self.radius)
        cap = np.stack([r_cap * ux, r_cap * uy,
                        np.where(q[..., 2] >= 0, self.hz, -self.hz)], -1)
        pick_wall = (np.linalg.norm(q - wall, axis=-1)
                     <= np.linalg.norm(q - cap, axis=-1))[..., None]
        return self.center + np.where(pick_wall, wall, cap)

    def trimesh(self):
        import trimesh
        m = trimesh.creation.cylinder(radius=self.radius, height=2 * self.hz, sections=48)
        m.apply_translation(self.center)
        return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)


class MeshObject:
    """Arbitrary triangle mesh (e.g. a GRAB object).  Differentiable penetration via
    nearest-surface-point + normal (recomputed, detached, each query)."""
    kind = "mesh"

    def __init__(self, vertices, faces):
        import trimesh
        self.mesh = trimesh.Trimesh(np.asarray(vertices, np.float64),
                                    np.asarray(faces, np.int64), process=False)
        self._verts = np.asarray(vertices, np.float32)
        self._faces = np.asarray(faces, np.int32)

    # Dense surface point cloud + normals, built once, used as a fast nearest-
    # neighbour surrogate.  trimesh's exact query needs `contains()` (ray casting)
    # which is far too slow to call on ~1k hand-surface points every iteration.
    _CLOUD_N = 40000

    def _cloud(self):
        if getattr(self, "_kdt", None) is None:
            import trimesh
            from scipy.spatial import cKDTree
            pts, fid = trimesh.sample.sample_surface(self.mesh, self._CLOUD_N, seed=0)
            self._cloud_pts = np.asarray(pts, np.float64)
            self._cloud_nrm = np.asarray(self.mesh.face_normals[fid], np.float64)
            self._kdt = cKDTree(self._cloud_pts)
        return self._kdt, self._cloud_pts, self._cloud_nrm

    def sdf(self, p):
        """Differentiable signed-distance *surrogate*, >0 outside.

        phi(p) = dot(p - nearest_surface_pt, outward_normal), with the nearest
        point and its normal detached.  Exact on the surface, correct in sign and
        direction nearby, and linear in `p` so gradients flow.  Evaluation never
        uses this -- see `metrics.ExactSDF`.
        """
        kdt, cloud, nrm = self._cloud()
        pd = p.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
        # workers=-1 spawns a thread pool per call; for the ~1k points queried
        # each iteration the spawn cost dwarfs the query itself.
        _, idx = kdt.query(pd, k=1, workers=1)
        closest_t = torch.as_tensor(cloud[idx], dtype=p.dtype, device=p.device).view_as(p)
        n_t = torch.as_tensor(nrm[idx], dtype=p.dtype, device=p.device).view_as(p)
        return ((p - closest_t) * n_t).sum(-1)

    def sample_surface(self, n, seed=0):
        import trimesh
        pts, fid = trimesh.sample.sample_surface(self.mesh, n, seed=seed)
        nrm = self.mesh.face_normals[fid]
        return pts.astype(np.float32), nrm.astype(np.float32)

    def trimesh(self):
        return self._verts, self._faces
