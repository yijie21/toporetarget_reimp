"""Shared helpers for baselines that solve joint angles but not the base pose."""
import numpy as np

from toporetarget.geometry import kabsch, matrix_to_rot6d


def place_base_by_kabsch(hand, q, human_kpts):
    """Best-fit the robot's 21 keypoints (at joints `q`) onto the human's.

    DexPilot and Mink both retarget *within* the wrist frame and leave the wrist
    pose to the caller.  Our method optimises the base jointly, so handing the
    baselines the human wrist pose verbatim would understate them; a Kabsch fit
    over all 21 keypoints is the strongest simple placement and is what we use.
    """
    import torch
    qt = torch.as_tensor(np.asarray(q, np.float32))
    ident = torch.tensor([[1.0, 0, 0, 0, 1, 0]])
    local = hand.keypoints(qt[None], ident, torch.zeros(1, 3))[0]
    local = local.detach().cpu().numpy().astype(np.float32)
    R, t = kabsch(local, np.asarray(human_kpts, np.float32))
    return matrix_to_rot6d(R), t
