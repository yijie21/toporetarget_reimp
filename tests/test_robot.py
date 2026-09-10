"""Forward kinematics, joint limits and hand-surface sampling."""
import numpy as np
import torch

from toporetarget.geometry import rot6d_to_matrix


def ident_pose(n=1):
    return (torch.tensor([[1.0, 0, 0, 0, 1, 0]]).repeat(n, 1), torch.zeros(n, 3))


def test_keypoint_correspondence_covers_the_21_mano_joints(hand):
    assert len(hand.kp_frames) == 21
    assert len(set(hand.kp_frames)) == 21
    assert hand.kp_frames[0] == "right_palm_link"          # wrist
    assert len(hand.bones) == 20
    # every bone connects joints that exist, and no bone is a self-loop
    for a, b in hand.bones:
        assert 0 <= a < 21 and 0 <= b < 21 and a != b


def test_joint_limits_are_well_formed(hand):
    assert hand.lower.shape == hand.upper.shape == (hand.n_joints,)
    assert torch.all(hand.lower < hand.upper)
    assert torch.all((hand.q_mid >= hand.lower) & (hand.q_mid <= hand.upper))


def test_forward_kinematics_is_rigidly_equivariant(hand):
    """Moving the base must move the keypoints by exactly that rigid transform."""
    q = hand.q_mid[None]
    d6, t = ident_pose()
    base = hand.keypoints(q, d6, t)[0]

    d6b = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])       # 90 deg about z
    tb = torch.tensor([[0.1, -0.2, 0.3]])
    moved = hand.keypoints(q, d6b, tb)[0]

    R = rot6d_to_matrix(d6b)[0]
    assert torch.allclose(moved, base @ R.T + tb[0], atol=1e-5)


def test_changing_a_joint_moves_only_its_own_finger(hand):
    q0 = hand.q_mid.clone()
    d6, t = ident_pose()
    kp0 = hand.keypoints(q0[None], d6, t)[0]
    q1 = q0.clone()
    j = hand.joint_names.index("right_finger2_joint2")
    q1[j] += 0.3
    kp1 = hand.keypoints(q1[None], d6, t)[0]
    moved = (kp1 - kp0).norm(dim=-1) > 1e-6
    index_finger = {5, 6, 7, 8}                    # MANO slots for the index finger
    assert set(torch.nonzero(moved).flatten().tolist()) <= index_finger
    assert moved.any()


def test_surface_sampling_covers_every_link_and_is_deterministic(hand):
    n = 16
    a = hand.surface_points(hand.q_mid[None], *ident_pose(), n_per_link=n)
    b = hand.surface_points(hand.q_mid[None], *ident_pose(), n_per_link=n)
    assert a.shape == (1, 26 * n, 3)               # 26 link meshes
    assert torch.equal(a, b)                        # cached, seeded


def test_surface_points_are_differentiable_wrt_joints(hand):
    q = hand.q_mid.clone().requires_grad_(True)
    hand.surface_points(q[None], *ident_pose(), n_per_link=4).square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


def test_surface_points_agree_with_keypoints_under_the_same_base(hand):
    """Both paths must place geometry in the same world frame."""
    q, (d6, t) = hand.q_mid[None], ident_pose()
    d6 = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])
    t = torch.tensor([[0.05, 0.0, -0.02]])
    mats = hand.fk_local(q)
    kp_shared = hand.keypoints(q, d6, t, mats=mats)
    kp_alone = hand.keypoints(q, d6, t)
    assert torch.allclose(kp_shared, kp_alone, atol=1e-6)
    sp = hand.surface_points(q, d6, t, n_per_link=8, mats=mats)
    # the palm samples must lie within a hand's reach of the wrist keypoint
    assert (sp[0] - kp_alone[0, 0]).norm(dim=-1).max() < 0.5


def test_procrustes_init_lands_near_the_human_hand(hand):
    from toporetarget.data import make_synthetic_grasp
    frame = make_synthetic_grasp("cylinder", n_obj=20, seed=0)
    q0, d6_0, t0 = hand.procrustes_init(frame.human_kpts)
    kp = hand.keypoints(torch.tensor(q0)[None], torch.tensor(d6_0)[None],
                        torch.tensor(t0)[None])[0].detach().numpy()
    assert np.linalg.norm(kp.mean(0) - frame.human_kpts.mean(0)) < 1e-5
