"""Wuji-hand kinematics wrapper.

Exposes a differentiable map  (base rotation, base translation, 20 joints)
-> 21 robot keypoints in MANO order, plus all link transforms for rendering.

MANO 21-keypoint order (MediaPipe convention) used throughout:
    0  wrist
    1-4    thumb  (MCP, PIP, DIP, TIP)
    5-8    index
    9-12   middle
    13-16  ring
    17-20  pinky
"""
import os
import numpy as np
import torch
import pytorch_kinematics as pk

from .geometry import rot6d_to_matrix, kabsch

# Which robot link frame represents each MANO keypoint.  finger1 = thumb.
# We use link2/link3/link4/tip as the four per-finger joints (link1 is a short
# coupling near the palm).
_FINGER_OF_MANO = {  # mano finger -> wuji finger index
    "thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5,
}
_MANO_GROUPS = ["thumb", "index", "middle", "ring", "pinky"]


def _keypoint_frames():
    frames = ["right_palm_link"]  # 0: wrist
    for g in _MANO_GROUPS:
        f = _FINGER_OF_MANO[g]
        frames += [f"right_finger{f}_link2",   # MCP
                   f"right_finger{f}_link3",   # PIP
                   f"right_finger{f}_link4",   # DIP
                   f"right_finger{f}_tip_link"]  # TIP
    return frames  # length 21


# Bones along each finger: (wrist|MCP|PIP|DIP)->next, as index pairs into the 21.
def _bones():
    bones = []
    for fi in range(5):
        base = 1 + fi * 4
        chain = [0, base, base + 1, base + 2, base + 3]  # wrist + 4 joints
        for a, b in zip(chain[:-1], chain[1:]):
            bones.append((a, b))
    return bones  # 5 fingers x 4 = 20 bones


# Adjacent bone pairs along each finger (for the relative bone-direction loss).
def _adjacent_bone_pairs():
    pairs = []
    for fi in range(5):
        b0 = fi * 4  # first bone index of this finger in _bones()
        for k in range(3):  # (b0,b1),(b1,b2),(b2,b3)
            pairs.append((b0 + k, b0 + k + 1))
    return pairs


class WujiHand:
    def __init__(self, urdf_path, device="cpu", dtype=torch.float32):
        self.device, self.dtype = device, dtype
        with open(urdf_path, "rb") as f:
            self.chain = pk.build_chain_from_urdf(f.read())
        self.chain = self.chain.to(dtype=dtype, device=device)
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_joints = len(self.joint_names)              # 20
        self.kp_frames = _keypoint_frames()                # 21 frame names
        self.bones = _bones()
        self.adjacent_bone_pairs = _adjacent_bone_pairs()

        # joint limits, in chain joint order
        lo, hi = [], []
        limmap = self._limits_from_urdf(urdf_path)
        for jn in self.joint_names:
            l, h = limmap[jn]
            lo.append(l); hi.append(h)
        self.lower = torch.tensor(lo, dtype=dtype, device=device)
        self.upper = torch.tensor(hi, dtype=dtype, device=device)
        self.q_mid = 0.5 * (self.lower + self.upper)

        # all link names (for rendering) and a default mid-open pose
        self.link_names = list(self.chain.get_link_names())

    @staticmethod
    def _limits_from_urdf(path):
        import re
        x = open(path).read()
        out = {}
        for m in re.finditer(
            r'<joint\s+name="([^"]+)"\s+type="revolute".*?lower="([^"]+)"\s+upper="([^"]+)"',
            x, re.S):
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        return out

    # ----- forward kinematics ------------------------------------------------ #
    def fk_local(self, q):
        """q: (B,20) -> dict frame_name -> (B,4,4) in the palm/root frame."""
        ret = self.chain.forward_kinematics(q)
        return {k: v.get_matrix() for k, v in ret.items()}

    def keypoints(self, q, d6, t):
        """q:(B,20) joints, d6:(B,6) base rotation, t:(B,3) base translation.
        Returns world-frame keypoints (B,21,3)."""
        mats = self.fk_local(q)
        local = torch.stack([mats[f][:, :3, 3] for f in self.kp_frames], dim=1)  # (B,21,3)
        R = rot6d_to_matrix(d6)                       # (B,3,3)
        world = torch.einsum("bij,bkj->bki", R, local) + t[:, None, :]
        return world

    def all_link_poses(self, q, d6, t):
        """Returns {link_name: (4,4) world matrix} for B=1 (rendering)."""
        mats = self.fk_local(q)
        R = rot6d_to_matrix(d6)[0]
        base = torch.eye(4, dtype=self.dtype, device=self.device)
        base[:3, :3] = R; base[:3, 3] = t[0]
        return {k: (base @ v[0]) for k, v in mats.items()}

    # ----- initialisation ---------------------------------------------------- #
    def procrustes_init(self, human_kpts, q0=None):
        """Place the base by best-fit aligning zero/mid-pose robot keypoints to the
        human keypoints.  Returns (q0, d6_0, t0) as numpy arrays."""
        if q0 is None:
            q0 = self.q_mid.clone()
        with torch.no_grad():
            local = self.keypoints(q0[None],
                                   torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=self.dtype),
                                   torch.zeros(1, 3, dtype=self.dtype))[0].cpu().numpy()
        R, t = kabsch(local, np.asarray(human_kpts, np.float32))
        from .geometry import matrix_to_rot6d
        return q0.cpu().numpy(), matrix_to_rot6d(R), t
