"""Dataset-dependent checks, skipped unless the data is present.

These encode the facts the loaders rely on, which were established empirically
rather than read out of documentation: ContactPose's joints are in the object
frame in the MANO joint order, and MANO posed from GRAB parameters produces an
anatomically sensible hand that actually touches the object.  CI cannot run them
(the datasets are licence-gated), so point the environment variables at your own
copies to enable them locally.
"""
import os

import numpy as np
import pytest

CP_GRASPS = os.environ.get("CONTACTPOSE_GRASPS")
CP_MODELS = os.environ.get("CONTACTPOSE_MODELS")
GRAB_SEQ = os.environ.get("GRAB_SEQ")
MANO_MODEL = os.environ.get("MANO_MODEL")
GRAB_SUBJECT_MESHES = os.environ.get("GRAB_SUBJECT_MESHES")
GRAB_OBJECT_MESHES = os.environ.get("GRAB_OBJECT_MESHES")

needs_contactpose = pytest.mark.skipif(
    not (CP_GRASPS and CP_MODELS),
    reason="set CONTACTPOSE_GRASPS and CONTACTPOSE_MODELS to run")
needs_grab = pytest.mark.skipif(
    not (GRAB_SEQ and MANO_MODEL and GRAB_SUBJECT_MESHES and GRAB_OBJECT_MESHES),
    reason="set GRAB_SEQ, MANO_MODEL, GRAB_SUBJECT_MESHES, GRAB_OBJECT_MESHES to run")

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def bone_lengths_mm(kpts):
    """(5, 4) per-finger segment lengths, wrist -> MCP -> PIP -> DIP -> tip."""
    out = []
    for f in range(5):
        chain = [0] + [1 + f * 4 + k for k in range(4)]
        out.append([np.linalg.norm(kpts[b] - kpts[a]) * 1000
                    for a, b in zip(chain[:-1], chain[1:])])
    return np.asarray(out)


@needs_contactpose
def test_contactpose_joints_are_already_in_the_object_frame():
    """The loader applies no transform; that is only valid if this holds."""
    from scipy.spatial import cKDTree

    from toporetarget.contactpose import load_contactpose_frame
    frame = load_contactpose_frame(CP_GRASPS, CP_MODELS, "mug")
    verts, _ = frame.obj.trimesh()
    d = cKDTree(np.asarray(verts, np.float64)).query(frame.human_kpts)[0] * 1000
    assert d.min() < 15.0, "no joint is near the object: the frames do not agree"
    assert d.max() < 400.0, "a joint is implausibly far away"


@needs_contactpose
def test_contactpose_joint_order_is_consistent_across_every_grasp():
    """One person's bone lengths are fixed; a wrong order would scramble them."""
    from toporetarget.contactpose import list_objects, load_contactpose_frame
    objects = list_objects(CP_GRASPS)
    lengths = np.stack([bone_lengths_mm(load_contactpose_frame(
        CP_GRASPS, CP_MODELS, o).human_kpts) for o in objects])
    cv = lengths.std(0) / lengths.mean(0)
    assert cv.max() < 0.20, f"bone lengths vary too much across grasps: {cv.max():.2f}"

    proximal = lengths.mean(0)[:, 0]        # wrist -> MCP, per finger
    idx, mid, ring, pinky = (proximal[FINGERS.index(f)]
                             for f in ("index", "middle", "ring", "pinky"))
    assert idx > pinky and ring > pinky, "proximal lengths are not anatomically ordered"


@needs_grab
def test_mano_forward_reproduces_a_real_grasp():
    """Loaded without chumpy; the result must still be a hand holding the object."""
    from scipy.spatial import cKDTree

    from toporetarget.grab import GrabSequence
    seq = GrabSequence(GRAB_SEQ, MANO_MODEL, GRAB_SUBJECT_MESHES, GRAB_OBJECT_MESHES)
    window = seq.grasp_window(stride=40)
    assert window, "no in-contact frame found in the sequence"

    verts, kpts = seq.hand_in_object_frame(window[len(window) // 2])
    assert verts.shape == (778, 3) and kpts.shape == (21, 3)

    lengths = bone_lengths_mm(kpts)
    assert (lengths > 5).all() and (lengths < 130).all(), "implausible bone lengths"

    tree = cKDTree(seq.obj_verts)
    assert tree.query(verts)[0].min() * 1000 < 10.0, "the hand is not touching"


@needs_grab
def test_the_object_is_static_once_expressed_in_its_own_frame():
    from toporetarget.grab import GrabSequence
    seq = GrabSequence(GRAB_SEQ, MANO_MODEL, GRAB_SUBJECT_MESHES, GRAB_OBJECT_MESHES)
    a = seq.frame(0)
    b = seq.frame(min(500, len(seq) - 1))
    assert a.obj is b.obj                      # so the SDF is built once
    assert not np.allclose(a.human_kpts, b.human_kpts), "the hand should have moved"
