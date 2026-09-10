"""Mink (MuJoCo differential IK) baseline, via the upstream `mink` package.

The vendored MJCF fixes the palm to the world.  A `<freejoint>` is injected so
Mink solves the base pose jointly with the joint angles -- the fixed-base
alternative would force us to place the wrist for it, which is a weaker and less
comparable baseline than the one our own method is measured against.
"""
from __future__ import annotations

import re
import time

import numpy as np
import yaml

from toporetarget.geometry import matrix_to_rot6d
from toporetarget.paths import CONFIGS, WUJI_MJCF
from toporetarget.solve import Solution

from . import register

_MODEL_CACHE = {}


def _fixed_tip_offsets():
    """Fingertip frames folded away by the URDF->MJCF conversion.

    MuJoCo merges a link attached by a fixed joint into its parent body, so the
    five `*_tip_link` frames our keypoint correspondence uses do not exist as
    bodies in the MJCF.  Recover each one as (parent body, translation) by
    walking the URDF's fixed joints.  All five happen to be pure translations
    with zero rpy; a non-zero rpy would need a site quaternion too and raises.
    """
    import xml.etree.ElementTree as ET
    from toporetarget.paths import WUJI_URDF

    root = ET.parse(WUJI_URDF).getroot()
    out = {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        if not child.endswith("tip_link") or j.get("type") != "fixed":
            continue
        origin = j.find("origin")
        rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
        if any(abs(v) > 1e-9 for v in rpy):
            raise NotImplementedError(
                f"{child}: fixed joint has non-zero rpy {rpy}; the injected site "
                "would need a quaternion")
        out[child] = (j.find("parent").get("link"),
                      [float(v) for v in origin.get("xyz").split()])
    return out


def _floating_model():
    """MJCF with a free base and sites standing in for the merged tip frames."""
    import mujoco
    if "m" not in _MODEL_CACHE:
        xml = WUJI_MJCF.read_text()
        xml, n = re.subn(r'(<body name="right_palm_link">)',
                         r'\1<freejoint name="base_free"/>', xml, count=1)
        if n != 1:
            raise RuntimeError("could not inject <freejoint> into the Wuji MJCF")
        sites = {}
        for tip, (parent, xyz) in _fixed_tip_offsets().items():
            site = f'<site name="{tip}" pos="{xyz[0]} {xyz[1]} {xyz[2]}" size="0.001"/>'
            xml, n = re.subn(rf'(<body name="{parent}"[^>]*>)', r"\1" + site, xml, count=1)
            if n != 1:
                raise RuntimeError(f"could not attach a site for {tip} to {parent}")
            sites[tip] = parent
        _MODEL_CACHE["m"] = mujoco.MjModel.from_xml_string(
            xml, {p.name: p.read_bytes() for p in WUJI_MJCF.parent.parent
                  .joinpath("meshes").iterdir()})
        _MODEL_CACHE["sites"] = set(sites)
    return _MODEL_CACHE["m"]


def _frame_type(name):
    return "site" if name in _MODEL_CACHE.get("sites", ()) else "body"


@register("mink")
def solve(hand, frame, *, config=None, scaling=None, **kw) -> Solution:
    import mink
    import mujoco

    cfg = yaml.safe_load(open(config or (CONFIGS / "baselines" / "mink_wuji.yml")))["mink"]
    if scaling is not None:
        cfg["scaling"] = float(scaling)

    model = _floating_model()
    t0 = time.time()

    human = np.asarray(frame.human_kpts, np.float64)
    targets = human[0] + (human - human[0]) * cfg["scaling"]

    configuration = mink.Configuration(model)
    tasks = [mink.PostureTask(model, cost=cfg["posture_cost"])]
    tasks[0].set_target(configuration.q.copy())
    frame_tasks = []
    for kp_frame, target in zip(hand.kp_frames, targets):
        task = mink.FrameTask(kp_frame, _frame_type(kp_frame),
                              position_cost=cfg["position_cost"],
                              orientation_cost=0.0,
                              lm_damping=cfg["lm_damping"])
        pose = mink.SE3.from_rotation_and_translation(
            mink.SO3.identity(), np.asarray(target, np.float64))
        task.set_target(pose)
        frame_tasks.append(task)
    tasks += frame_tasks
    limits = [mink.ConfigurationLimit(model)]

    for _ in range(int(cfg["iters"])):
        vel = mink.solve_ik(configuration, tasks, cfg["dt"], cfg["solver"],
                            damping=cfg["damping"], limits=limits)
        configuration.integrate_inplace(vel, cfg["dt"])

    # split the solved qpos into (base pose, joint angles) in our conventions
    q_full = configuration.q.copy()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    adr = model.jnt_qposadr[jid]
    t = q_full[adr:adr + 3].astype(np.float64)
    quat = q_full[adr + 3:adr + 7]                       # mujoco order: w x y z
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    d6 = matrix_to_rot6d(R.reshape(3, 3))

    lut = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_qposadr[j]
           for j in range(model.njnt)}
    q = np.array([q_full[lut[n]] for n in hand.joint_names], np.float64)
    q = np.clip(q, hand.lower.numpy(), hand.upper.numpy())
    return Solution(q=q, d6=np.asarray(d6, np.float64), t=t,
                    seconds=time.time() - t0, method="mink",
                    extra=dict(scaling=cfg["scaling"]))
