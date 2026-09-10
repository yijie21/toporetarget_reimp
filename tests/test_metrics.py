"""The two metrics the paper's main table is built on."""
import numpy as np
import pytest
import torch

from toporetarget.data import make_synthetic_grasp
from toporetarget.geometry import Sphere
from toporetarget.metrics import (ExactSDF, contact_precision_error_mm,
                                  contact_set, evaluate, max_penetration_mm,
                                  robot_surface_points)


def test_contact_precision_is_zero_when_the_robot_matches_the_human():
    frame = make_synthetic_grasp("sphere", n_obj=20, seed=0)
    sdf = ExactSDF(frame.obj)
    idx = contact_set(sdf, frame.human_kpts)
    e = contact_precision_error_mm(sdf, frame.human_kpts, frame.human_kpts, idx)
    assert e == pytest.approx(0.0, abs=1e-6)


def test_contact_precision_ignores_sliding_along_the_surface():
    """Both readings differ here, which is exactly why we report both."""
    obj = Sphere([0, 0, 0], 0.05)
    sdf = ExactSDF(obj)
    human = np.zeros((21, 3)); human[:, 0] = 0.05          # all on the +x pole
    robot = np.zeros((21, 3)); robot[:, 1] = 0.05          # slid to the +y pole
    idx = np.arange(21)
    assert contact_precision_error_mm(sdf, robot, human, idx, "surface") \
        == pytest.approx(0.0, abs=1e-3)
    assert contact_precision_error_mm(sdf, robot, human, idx, "origin") \
        == pytest.approx(50 * np.sqrt(2), abs=1e-2)


def test_contact_precision_reports_a_known_offset():
    obj = Sphere([0, 0, 0], 0.05)
    sdf = ExactSDF(obj)
    human = np.tile([0.05, 0, 0], (21, 1))                 # touching
    robot = np.tile([0.058, 0, 0], (21, 1))                # 8 mm off the surface
    assert contact_precision_error_mm(sdf, robot, human, np.arange(21)) \
        == pytest.approx(8.0, abs=1e-3)


def test_contact_set_selects_only_joints_within_the_threshold():
    obj = Sphere([0, 0, 0], 0.05)
    sdf = ExactSDF(obj)
    kpts = np.tile([0.20, 0, 0], (21, 1))                  # all far away
    kpts[3] = [0.052, 0, 0]                                # 2 mm off: in contact
    kpts[7] = [0.056, 0, 0]                                # 6 mm off: in contact
    assert set(contact_set(sdf, kpts, tau_mm=10.0)) == {3, 7}


def test_contact_set_falls_back_to_the_fingertips_when_nothing_touches():
    sdf = ExactSDF(Sphere([0, 0, 0], 0.05))
    far = np.tile([1.0, 0, 0], (21, 1))
    assert set(contact_set(sdf, far, tau_mm=1.0)) == {4, 8, 12, 16, 20}


def test_max_penetration_measures_a_known_depth():
    obj = Sphere([0, 0, 0], 0.05)
    sdf = ExactSDF(obj)
    pts = np.array([[0.05, 0, 0],          # on the surface  -> 0 mm
                    [0.047, 0, 0],         # 3 mm inside
                    [0.20, 0, 0]])         # far outside     -> 0 mm
    assert max_penetration_mm(sdf, pts) == pytest.approx(3.0, abs=1e-3)


def test_max_penetration_is_zero_when_nothing_is_inside():
    sdf = ExactSDF(Sphere([0, 0, 0], 0.05))
    assert max_penetration_mm(sdf, np.array([[0.2, 0, 0], [0, 0.3, 0]])) == 0.0


def test_evaluation_samples_the_link_surfaces_not_just_keypoints(hand):
    """Regression guard: penetration must be measured on hand geometry."""
    q, d6, t = hand.q_mid.numpy(), np.array([1.0, 0, 0, 0, 1, 0]), np.zeros(3)
    pts = robot_surface_points(hand, q, d6, t, n_per_link=50)
    assert len(pts) == 26 * 50
    frame = make_synthetic_grasp("cylinder", n_obj=20, seed=0)
    out = evaluate(hand, frame, q, d6, t, n_per_link=50)
    assert out["n_surface_pts"] == 26 * 50
    assert out["n_surface_pts"] > 21 + 20         # more than keypoints + bone midpoints
