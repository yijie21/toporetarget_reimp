"""Interaction-mesh construction and the weighted Laplacian operator."""
import numpy as np
import torch

from toporetarget.interaction import (build_edges, edge_kind, laplacian_coords,
                                      laplacian_operator)


def cube_points():
    """Eight cube corners plus the centre: a non-degenerate tetrahedralisation."""
    c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    return np.vstack([c, [[0.5, 0.5, 0.5]]])


def test_edges_are_undirected_unique_and_sorted():
    edges = build_edges(cube_points())
    assert edges == sorted(set(edges))
    assert all(i < j for i, j in edges)
    assert all(0 <= i < 9 and 0 <= j < 9 for i, j in edges)


def test_every_vertex_is_connected():
    edges = build_edges(cube_points())
    seen = {v for e in edges for v in e}
    assert seen == set(range(9))


def test_degenerate_input_falls_back_to_a_knn_graph():
    """Coplanar points break Delaunay; the fallback must still produce edges."""
    flat = np.array([[x, y, 0.0] for x in range(4) for y in range(4)], float)
    edges = build_edges(flat)
    assert len(edges) > 0
    assert {v for e in edges for v in e} == set(range(16))


def test_laplacian_rows_sum_to_zero():
    V = cube_points()
    L, W = laplacian_operator(V, build_edges(V), kappa=10.0)
    assert torch.allclose(L.sum(1), torch.zeros(len(V)), atol=1e-6)
    assert torch.allclose(W.sum(1), torch.ones(len(V)), atol=1e-6)


def test_laplacian_weights_are_symmetric_before_normalisation_and_positive():
    V = cube_points()
    _, W = laplacian_operator(V, build_edges(V), kappa=10.0)
    assert torch.all(W >= 0)
    assert torch.all(torch.diagonal(W) == 0)


def test_laplacian_coordinates_are_translation_invariant():
    """Delta(V + c) == Delta(V): the interaction term cannot see a global shift."""
    V = cube_points()
    L, _ = laplacian_operator(V, build_edges(V), kappa=10.0)
    Vt = torch.tensor(V, dtype=torch.float32)
    shifted = Vt + torch.tensor([3.0, -1.0, 2.0])
    assert torch.allclose(laplacian_coords(L, Vt), laplacian_coords(L, shifted), atol=1e-5)


def test_larger_kappa_concentrates_weight_on_nearer_neighbours():
    V = cube_points()
    edges = build_edges(V)
    _, W_soft = laplacian_operator(V, edges, kappa=1.0)
    _, W_sharp = laplacian_operator(V, edges, kappa=50.0)
    centre = 8                                  # the centre vertex, nearest to all
    assert W_sharp[0, centre] > W_soft[0, centre]


def test_edge_kind_labels_hand_object_and_cross_edges():
    edges = [(0, 1), (0, 5), (5, 6)]
    assert edge_kind(edges, n_hand=3) == ["hh", "ho", "oo"]
