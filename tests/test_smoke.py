"""End-to-end guards that run with no dataset: synthetic data only, CPU only.

These are what CI can actually run — ContactPose and GRAB are licence-gated and
never enter the repository, so the full benchmark is a local `reproduce.sh`.
"""
import numpy as np
import pytest

from toporetarget.data import make_synthetic_grasp
from toporetarget.metrics import evaluate
from toporetarget.solve import solve_ours


ITERS = 250          # enough to converge; these run on every CI push


@pytest.fixture(scope="module")
def frame():
    return make_synthetic_grasp("cylinder", n_obj=50, seed=0)


@pytest.fixture(scope="module")
def solved(hand, frame):
    """The full method, solved once and reused by every test below."""
    sol = solve_ours(hand, frame, n_iters=ITERS)
    return frame, sol, evaluate(hand, frame, sol.q, sol.d6, sol.t)


def test_the_synthetic_grasp_converges_to_a_plausible_solution(hand, solved):
    frame, sol, m = solved
    assert m["E_prec_mm"] < 25.0, "contact precision regressed badly"
    assert m["D_pen_max_mm"] < 15.0, "the hand is sinking into the object"
    assert np.all(np.isfinite(sol.q)) and np.all(np.isfinite(sol.t))


def test_the_solution_respects_the_joint_limits(hand, solved):
    _, sol, _ = solved
    assert np.all(sol.q >= hand.lower.numpy() - 1e-6)
    assert np.all(sol.q <= hand.upper.numpy() + 1e-6)


def test_dropping_the_interaction_term_destroys_contact(hand, frame, solved):
    """E_IM is what pulls the hand onto the object; without it nothing does."""
    _, _, m_full = solved
    no_im = solve_ours(hand, frame, weights=dict(IM=0.0), n_iters=ITERS)
    e_none = evaluate(hand, frame, no_im.q, no_im.d6, no_im.t)["E_prec_mm"]
    assert e_none > 3 * m_full["E_prec_mm"]


def test_dropping_the_penetration_term_lets_the_hand_sink_in(hand, frame, solved):
    _, _, m_full = solved
    no_pen = solve_ours(hand, frame, weights=dict(pen=0.0), n_iters=ITERS)
    d_none = evaluate(hand, frame, no_pen.q, no_pen.d6, no_pen.t)["D_pen_max_mm"]
    assert d_none > 2 * m_full["D_pen_max_mm"]


def test_the_run_is_deterministic(hand):
    seq_frame = make_synthetic_grasp("sphere", n_obj=30, seed=1)
    a = solve_ours(hand, seq_frame, n_iters=120)
    b = solve_ours(hand, seq_frame, n_iters=120)
    assert np.allclose(a.q, b.q, atol=1e-6)
    assert np.allclose(a.t, b.t, atol=1e-6)
