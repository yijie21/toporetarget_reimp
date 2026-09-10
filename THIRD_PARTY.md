# Third-party assets, datasets and licences

This repository's own code is MIT (see `LICENSE`). Everything below belongs to
someone else and carries its own terms. **No dataset content is redistributed
here** — the repository ships only robot assets that permit redistribution, plus
demo payloads generated from synthetic data we create ourselves.

## Vendored in this repository

| Asset | Licence | Notes |
|---|---|---|
| `assets/wuji_right/` — Wuji Hand URDF, 26 STL meshes, MJCF | **MIT** | From [wuji-technology/wuji-description](https://github.com/wuji-technology/wuji-description) `2026.8.19`. See `assets/wuji_right/PROVENANCE.md`. |
| `viewer/data_syn_*.js` | MIT (ours) | Generated from `make_synthetic_grasp()`; contains no dataset-derived geometry. |

## NOT vendored — you must obtain these yourself

| Dataset / model | Where | Terms |
|---|---|---|
| **ContactPose** | <https://contactpose.cc.gatech.edu/> | Annotations and non-model data are MIT; the 3-D object models carry individual licences documented in the model directories. Used for the main table. |
| **GRAB** (subject `s1`) | <https://grab.is.tue.mpg.de/> | Registration required, research-only, **redistribution prohibited**. Used only for the sequence experiment. |
| **ContactDB object meshes** | distributed with GRAB | Same terms as GRAB. |
| **MANO** | <https://mano.is.tue.mpg.de/> | Registration required, research-only, **redistribution prohibited**. Needed only to turn GRAB's MANO parameters into hand keypoints. |

Because of the terms above, viewer payloads derived from ContactPose or GRAB are
**git-ignored** and must be regenerated locally (`scripts/export_viewer.py`).
The figures in `README.md` that show real grasps are screenshots, not data.
