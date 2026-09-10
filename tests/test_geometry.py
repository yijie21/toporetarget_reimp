"""Rotation, Procrustes and analytic-SDF properties with known answers."""
import numpy as np
import pytest
import torch

from toporetarget.geometry import (Cylinder, Sphere, kabsch, matrix_to_rot6d,
                                   rot6d_to_matrix)


def random_rotation(rng):
    """A uniformly random proper rotation (QR of a Gaussian, sign-corrected)."""
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))          # make the QR factorisation unique
    if np.linalg.det(Q) < 0:                      # force det = +1
        Q[:, 0] = -Q[:, 0]
    return Q


def test_rot6d_roundtrip_recovers_the_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = random_rotation(rng)
        back = rot6d_to_matrix(torch.tensor(matrix_to_rot6d(R))[None])[0].numpy()
        assert np.allclose(back, R, atol=1e-5)


def test_rot6d_output_is_always_a_rotation_even_for_junk_input():
    rng = np.random.default_rng(1)
    d6 = torch.tensor(rng.normal(size=(8, 6)), dtype=torch.float32)
    R = rot6d_to_matrix(d6)
    eye = torch.eye(3).expand(8, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(8), atol=1e-5)


def test_kabsch_recovers_a_known_rigid_transform():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(21, 3))
    R_true, t_true = random_rotation(rng), rng.normal(size=3)
    B = A @ R_true.T + t_true
    R, t = kabsch(A, B)
    assert np.allclose(R, R_true, atol=1e-5)
    assert np.allclose(t, t_true, atol=1e-5)
    assert np.allclose(A @ R.T + t, B, atol=1e-5)


def test_kabsch_never_returns_a_reflection():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(10, 3))
    B = A * np.array([1.0, 1.0, -1.0])          # mirrored: no rigid fit exists
    R, _ = kabsch(A, B)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("obj", [Sphere([0.01, 0, 0], 0.05),
                                 Cylinder([0, 0.02, 0], 0.042, 0.11)])
def test_analytic_sdf_is_zero_on_the_surface(obj):
    pts, _ = obj.sample_surface(500, seed=0)
    assert np.abs(obj.exact_signed_distance(pts)).max() < 1e-5


@pytest.mark.parametrize("obj", [Sphere([0.01, 0, 0], 0.05),
                                 Cylinder([0, 0.02, 0], 0.042, 0.11)])
def test_analytic_nearest_point_lies_on_the_surface_at_the_reported_distance(obj):
    rng = np.random.default_rng(4)
    p = np.asarray(obj.center) + rng.normal(scale=0.08, size=(400, 3))
    near = obj.nearest_surface(p)
    assert np.abs(obj.exact_signed_distance(near)).max() < 1e-5
    d = np.linalg.norm(p - near, axis=-1)
    assert np.allclose(d, np.abs(obj.exact_signed_distance(p)), atol=1e-6)


def test_analytic_and_differentiable_sdf_agree():
    """The torch surrogate used in the loss must match the analytic evaluator."""
    for obj in (Sphere([0, 0, 0], 0.05), Cylinder([0, 0, 0], 0.042, 0.11)):
        rng = np.random.default_rng(5)
        p = rng.normal(scale=0.08, size=(300, 3))
        torch_sdf = obj.sdf(torch.tensor(p, dtype=torch.float64)).numpy()
        assert np.allclose(torch_sdf, obj.exact_signed_distance(p), atol=1e-6)


def test_sphere_sdf_sign_convention_is_positive_outside():
    obj = Sphere([0, 0, 0], 0.05)
    assert obj.exact_signed_distance(np.zeros((1, 3)))[0] == pytest.approx(-0.05)
    assert obj.exact_signed_distance(np.array([[1.0, 0, 0]]))[0] == pytest.approx(0.95)
