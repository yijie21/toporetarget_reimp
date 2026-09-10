# toporetarget-reimpl

**Unofficial reimplementation** of *TopoRetarget: Interaction-Preserving
Retargeting for Dexterous Manipulation*
([arXiv:2606.16272](https://arxiv.org/abs/2606.16272)), evaluated on
**ContactPose** with the **Wuji Hand**.

> Not affiliated with, endorsed by, or reviewed by the paper's authors. At the
> time of writing the authors' [project page](https://toporetarget2026.github.io/TopoRetarget/)
> links no code, so nothing here could be checked against a reference
> implementation. Every number below was produced by this repository; where our
> setup necessarily departs from the paper, it is listed in
> [Known deviations](#known-deviations-from-the-paper) rather than smoothed over.

Given a human hand-object interaction, it solves for the robot base pose and
joint angles that reproduce the **same interaction** — preserving the relative
hand-object geometry while keeping the hand out of the object — and ships an
interactive viewer for inspecting the optimisation and the ablations.

## Results

<!-- RESULTS:BEGIN -->
Run `scripts/reproduce.sh` to generate this table.
<!-- RESULTS:END -->

## Quick start

Nothing needs downloading to run the code: the Wuji right hand (URDF, 26 meshes,
MJCF) is vendored under `assets/`, and the demo works on synthetic grasps.

```bash
git clone https://github.com/<you>/toporetarget-reimpl && cd toporetarget-reimpl
pip install -e ".[dev]"          # add ".[baselines]" for DexPilot and Mink
pytest -q                        # 39 tests, CPU only, no datasets needed

python scripts/export_viewer.py --source synthetic:cylinder
open viewer/index.html           # orbit, scrub the optimisation, toggle losses
```

To run on real data you need ContactPose (and, for the sequence experiment, GRAB
and MANO). See [THIRD_PARTY.md](THIRD_PARTY.md) for where to get them and under
what terms, then:

```bash
export CONTACTPOSE_GRASPS=/path/to/grasps CONTACTPOSE_MODELS=/path/to/ply_files_mm
./scripts/reproduce.sh           # eval set -> tuning -> main table -> RESULTS.md
```

## What is and is not implemented

| | |
|---|---|
| ✅ Stage 1 — relative bone-direction initialisation (`E_bone`) + Procrustes base placement | `optimize.py`, `robot.py` |
| ✅ Stage 2 — interaction mesh (Delaunay over hand + object anchors) | `interaction.py` |
| ✅ Stage 3 — distance-aware weights, Laplacian coordinates (`E_IM`) | `interaction.py`, `optimize.py` |
| ✅ Stage 4 — regularisation (`E_reg`), penetration (`E_pen`), joint limits (`E_lim`) | `optimize.py` |
| ✅ Eq. 10 / Eq. 12 metrics, exact signed distance at evaluation | `metrics.py` |
| ✅ DexPilot and Mink baselines, through their upstream libraries | `baselines/` |
| ❌ The downstream **RL tracking controller** (PPO), and therefore the pen-spinning and sim-to-real results | — |
| ❌ OmniRetarget and GeoRT baselines | — |

The objective, with positional terms in mm² so the weights stay O(1):

```
L = w_IM·E_IM + w_bone·E_bone + w_reg·E_reg + w_pen·E_pen + w_lim·E_lim
```

Base orientation uses the continuous 6-D rotation representation; joint limits
are enforced by a barrier *and* by projected clamping each step.

## How the evaluation is set up

Two choices do most of the work in making the numbers trustworthy:

**Hyper-parameters are fitted on a different person.** The paper does not publish
its weights, so ours are fitted — `scripts/tune.py` samples 40 configurations on
`configs/tuning_split.yaml`, which is a **different ContactPose participant**
from the benchmark, and minimises `mean(E_prec + D_pen_max)`. Each baseline gets
the same split, the same objective and the same budget for its own scaling
factor. Ours spends that budget on three weights and each baseline on one
scalar, which favours the baselines. The winners are frozen in `configs/` and
one single set is used for every grasp.

**Penetration is measured on the hand's geometry, not on its keypoints.** Eq. 12
is defined over sampled robot-hand surface points, so both the loss and the
metric sample the 26 link meshes (1040 points in the loss, 5200 in the metric).
An earlier version of this code penalised only the 21 keypoints and 20 bone
midpoints; on the synthetic cylinder that scored 0.80 mm while the actual link
surfaces were 9.00 mm inside the object.

## Known deviations from the paper

Things we had to decide because the paper does not say, or could not match:

1. **Loss weights are not published.** Fitted as described above. This is the
   most likely source of any gap between our numbers and theirs.
2. **`E_prec`'s `o_c` is under-specified.** We read it as the nearest point on
   the object surface, so `h_c − o_c` is a *contact offset* and the metric is
   invariant to sliding along the surface. Under the alternative reading (object
   origin) the term cancels and Eq. 10 collapses to plain keypoint error; that
   variant is reported too, as `E_prec_origin_mm`.
3. **The contact set `C` is ours.** "In-contact hand links" needs a threshold; we
   use hand keypoints within **10 mm** of the surface *in the human demonstration*,
   so every method is scored on the same set and cannot improve by contacting
   elsewhere.
4. **The paper's 25-of-28 grasp list is not published.** We report all 24 grasps
   of one participant, and mark a paper-comparable subset by removing the three
   lowest-solidity objects — the count is the paper's, the objective selection
   rule is ours.
5. **Optimisation uses a differentiable signed-distance surrogate**, evaluation
   uses exact signed distance (libigl fast winding number). The two therefore
   disagree slightly; only the exact one is ever reported.
6. **Keypoint correspondence** (which Wuji link stands for which MANO joint) is a
   modelling choice in `robot.py::_keypoint_frames()`.
7. **Baseline configurations are ours.** `configs/baselines/` is public; a better
   configuration would change the comparison.
8. **One participant, one hand.** No cross-embodiment claim is made or tested.

## Layout

```
assets/wuji_right/   vendored Wuji right hand (MIT) — URDF, 26 STL meshes, MJCF
configs/             frozen eval set, tuning split, weights, baseline configs
toporetarget/        geometry robot interaction optimize data metrics solve
                     contactpose grab mano   (mano.py: chumpy-free MANO forward)
baselines/           dexpilot.py, mink_ik.py — thin adapters over upstream libraries
scripts/             make_eval_set  make_tuning_split  tune  eval_contactpose
                     eval_grab_sequence  export_viewer  make_results  reproduce.sh
tests/               39 tests, CPU only, no dataset required
viewer/              three.js viewer; robot meshes shared across payloads
docs/                explainers + the GitHub Pages entry point
```

`toporetarget/mano.py` is worth a note: GRAB stores hand *parameters*, so turning
it into keypoints normally needs `smplx` + `chumpy`, and `chumpy` needs an old
NumPy and hence a second environment. The MANO pickle is almost all plain arrays,
and GRAB ships each subject's own hand template, so the whole chain collapses to
a stub class during unpickling plus standard linear blend skinning. **This repo
runs in one environment.**

## Licence

Our code is MIT. The vendored Wuji assets are MIT (see
`assets/wuji_right/PROVENANCE.md`). **No dataset content is redistributed** —
ContactPose, GRAB, ContactDB and MANO all carry their own terms, listed in
[THIRD_PARTY.md](THIRD_PARTY.md). Viewer payloads derived from those datasets are
git-ignored and regenerated locally.

## Citing

Cite the original paper, not this repository:

```bibtex
@article{wu2026toporetarget,
  title  = {TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation},
  author = {Wu, Jielin and Yao, Shenzhe and He, Guanqi and Liu, Xiaohan and
            Zeng, Zhaoqing and Jiang, Xiangrui and Yang, Han and Zhang, Wentao
            and Zhao, Hang},
  journal = {arXiv preprint arXiv:2606.16272},
  year   = {2026}
}
```
