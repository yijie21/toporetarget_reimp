<div align="center">

# TopoRetarget — reimplementation

**Teach a robot hand by copying the *interaction*, not the pose.**

An unofficial, from-scratch implementation of
[*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*](https://arxiv.org/abs/2606.16272),
with an interactive 3-D viewer for watching the optimisation actually happen.

[![tests](https://github.com/yijie21/toporetarget_reimp/actions/workflows/ci.yml/badge.svg)](https://github.com/yijie21/toporetarget_reimp/actions/workflows/ci.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](pyproject.toml)
[![live demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://yijie21.github.io/toporetarget_reimp/viewer/)

### ▶ **[Open the interactive viewer](https://yijie21.github.io/toporetarget_reimp/viewer/)**

<img src="docs/media/hero.gif" width="880" alt="Human hand on the left, Wuji robot hand on the right, converging onto the same grasp">

<sub>A real GRAB grasp, retargeted. Left: the human demonstration. Right: the Wuji
Hand, solved from its initial pose to reproduce the same hand-object
relationship. In the viewer you drag to orbit — both panes share one camera.</sub>

</div>

---

## The problem, in one picture

A human hand and a robot hand have different shapes, sizes and joints. Copy the
finger angles across and the fingers pass through the object or lose contact
entirely — and a policy trained on broken references learns broken behaviour.

TopoRetarget instead builds a small **interaction graph** over the hand keypoints
*and* points on the object, and asks the robot to preserve the *local relative
geometry* of that graph — who touches what, from which direction, at what
distance — while staying out of the object.

<div align="center">
<img src="docs/media/ablation.png" width="880" alt="The same grasp solved with the full method, without the interaction term, and without the penetration term">
<br><sub>The same grasp, three ways. Drop <code>E_IM</code> and the hand never
commits to the object; drop <code>E_pen</code> and it grips <em>better</em> —
by sinking 6 mm into the mug. Every term is switchable in the viewer, live.</sub>
</div>

## Try it

Nothing to download: the Wuji right hand (URDF, 26 meshes, MJCF) is vendored, and
the demo runs on synthetic grasps.

```bash
git clone https://github.com/yijie21/toporetarget_reimp && cd toporetarget_reimp
pip install -e ".[dev]"
pytest -q                                              # 43 tests, CPU only

python scripts/export_viewer.py --source synthetic:cylinder
open viewer/index.html                                 # orbit · scrub · toggle losses
```

Real grasps need ContactPose or GRAB, which are licence-gated and not
redistributed here — see [THIRD_PARTY.md](THIRD_PARTY.md). Once you have them:

```bash
python scripts/export_viewer.py --source contactpose:mug \
    --grasps /path/to/grasps --models /path/to/ply_files_mm
```

## What the viewer gives you

| | |
|---|---|
| **Synchronised panes** | one shared camera renders the human and the robot side by side, so poses can be compared directly rather than by memory |
| **Scrub the optimisation** | play the solver from initialisation to convergence, or drag to any iteration |
| **Ablation switch** | full / no `E_IM` / no `E_pen` / no `E_bone` / no `E_reg`, with the loss curves and live contact and penetration readouts |
| **Interaction mesh overlay** | the Delaunay edges, colour-coded hand-hand / object-object / **cross** — the cross edges are the ones carrying the interaction |

<div align="center">
<img src="docs/media/interaction-mesh.png" width="880" alt="The interaction mesh overlaid on both hands, with loss curves and live metrics">
<br><sub>The interaction mesh switched on: the graph whose local relative
geometry the optimisation is trying to preserve.</sub>
</div>

## How it works

Four stages, each mapped to where it lives:

| Paper | Here |
|---|---|
| ① relative bone directions `E_bone`, Procrustes base placement | `optimize.py`, `robot.py` |
| ② interaction mesh — Delaunay over hand keypoints + object anchors | `interaction.py` |
| ③ distance-aware weights, Laplacian coordinates `E_IM` | `interaction.py`, `optimize.py` |
| ④ regularisation `E_reg`, penetration `E_pen`, joint limits `E_lim` | `optimize.py` |

```
L = w_IM·E_IM + w_bone·E_bone + w_reg·E_reg + w_pen·E_pen + w_lim·E_lim
```

Base orientation uses the continuous 6-D rotation representation. Penetration is
measured on **sampled link surfaces**, not on keypoints — penalising only
keypoints lets the finger geometry sink into the object while the score looks
clean (0.80 mm reported where the link surfaces were 9.00 mm inside).

Two explainers walk through the maths with animations:
**[the method](https://yijie21.github.io/toporetarget_reimp/toporetarget_explained.html)** ·
**[Stage 3 in detail](https://yijie21.github.io/toporetarget_reimp/stage3_laplacian_explained.html)**

## Honest scope

- **Unofficial.** Not affiliated with the authors, and at the time of writing
  their project page links no code, so nothing here was checked against a
  reference implementation.
- **Retargeting only.** The downstream RL tracking controller — and therefore the
  pen-spinning and sim-to-real results — is not implemented.
- **The paper does not publish its loss weights.** Ours were fitted on a
  *different* ContactPose participant from any we evaluate on, then frozen; the
  same single set is used everywhere. `scripts/tune.py` reproduces the search.
- **A full benchmark harness is included** — Eq. 10 and Eq. 12 metrics, DexPilot
  and Mink baselines through their own upstream libraries, a frozen evaluation
  set — but the complete table has not been run for this release. The numbers
  this repository does publish are generated by `scripts/make_results.py`; none
  are typed by hand.

## Layout

```
assets/wuji_right/   vendored Wuji right hand (MIT): URDF, 26 STL meshes, MJCF
toporetarget/        geometry robot interaction optimize metrics solve
                     contactpose grab mano  ← chumpy-free MANO forward model
baselines/           DexPilot and Mink, via dex_retargeting and mink
configs/             frozen evaluation set, tuning split, weights
scripts/             export_viewer · run_retarget · tune · eval_* · reproduce.sh
viewer/              three.js viewer; link meshes shared across payloads
docs/                explainers and the GitHub Pages entry point
```

One detail worth calling out: `toporetarget/mano.py` reads the MANO model without
`chumpy`, so the GRAB path needs no second Python environment — the whole
repository runs in one.

## Licence

Code is MIT. Vendored Wuji assets are MIT
(`assets/wuji_right/PROVENANCE.md`). **No dataset content is redistributed**;
ContactPose, GRAB, ContactDB and MANO each carry their own terms, listed in
[THIRD_PARTY.md](THIRD_PARTY.md).

## Citing

Cite the paper, not this repository:

```bibtex
@article{wu2026toporetarget,
  title   = {TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation},
  author  = {Wu, Jielin and Yao, Shenzhe and He, Guanqi and Liu, Xiaohan and
             Zeng, Zhaoqing and Jiang, Xiangrui and Yang, Han and Zhang, Wentao
             and Zhao, Hang},
  journal = {arXiv preprint arXiv:2606.16272},
  year    = {2026}
}
```
