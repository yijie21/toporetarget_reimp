"""Interaction-mesh construction (paper Stage 2 + the weights of Stage 3).

Build a Delaunay tetrahedralisation over the *source* vertices V^s = [hand;object],
derive the shared edge set, distance-aware normalised weights from the source
geometry, and assemble the weighted-Laplacian operator  L = I - W  so that the
Laplacian coordinates are simply  Delta(V) = L @ V.
"""
import numpy as np
import torch
from scipy.spatial import Delaunay


def build_edges(V_source):
    """Delaunay tetrahedralisation -> undirected edge set (list of (i,j))."""
    V = np.asarray(V_source, np.float64)
    try:
        tri = Delaunay(V, qhull_options="QJ")  # QJ = joggle to survive degeneracies
        edges = set()
        for tet in tri.simplices:
            for a in range(4):
                for b in range(a + 1, 4):
                    i, j = int(tet[a]), int(tet[b])
                    edges.add((min(i, j), max(i, j)))
        return sorted(edges)
    except Exception:
        # fallback: kNN graph if Delaunay fails (e.g. too few/coplanar points)
        from scipy.spatial import cKDTree
        tree = cKDTree(V)
        k = min(8, len(V))
        edges = set()
        for i in range(len(V)):
            _, idx = tree.query(V[i], k=k)
            for j in idx[1:]:
                edges.add((min(i, int(j)), max(i, int(j))))
        return sorted(edges)


def laplacian_operator(V_source, edges, kappa, device="cpu", dtype=torch.float32):
    """Row-normalised weighted-Laplacian L = I - W  (Nv x Nv) from SOURCE geometry.

    w_ij = exp(-kappa * ||v_i - v_j||) normalised so each row sums to 1.
    Returns (L, W, edges) with L,W dense torch tensors (Nv small ~ 71).
    """
    V = np.asarray(V_source, np.float64)
    Nv = len(V)
    W = np.zeros((Nv, Nv), np.float64)
    for i, j in edges:
        d = np.linalg.norm(V[i] - V[j])
        w = np.exp(-kappa * d)
        W[i, j] += w
        W[j, i] += w
    rowsum = W.sum(1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    W = W / rowsum
    L = np.eye(Nv) - W
    Lt = torch.tensor(L, dtype=dtype, device=device)
    Wt = torch.tensor(W, dtype=dtype, device=device)
    return Lt, Wt


def laplacian_coords(L, V):
    """Delta(V) = L @ V.   V:(...,Nv,3)  L:(Nv,Nv)  ->  (...,Nv,3)."""
    return L @ V


def edge_kind(edges, n_hand):
    """Label each edge as 'hh' (hand-hand), 'oo' (obj-obj) or 'ho' (cross)."""
    kinds = []
    for i, j in edges:
        hi, hj = i < n_hand, j < n_hand
        kinds.append("hh" if hi and hj else "oo" if (not hi and not hj) else "ho")
    return kinds
