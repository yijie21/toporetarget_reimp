# Vendored Wuji Hand assets

Source: <https://github.com/wuji-technology/wuji-description>, calendar version
`2026.8.19`, licensed **MIT** (see `LICENSE` in this directory, copied verbatim
from upstream).

Only the right-hand `body` variant is vendored — the upstream repository is
~400 MB, of which this subset is ~2.7 MB:

| Here | Upstream |
|---|---|
| `right.urdf` | `hand/body/urdf/right.urdf` |
| `meshes/*.STL` (26) | `hand/body/meshes/right/*.STL` |
| `mjcf/right.xml` | `hand/body/mjcf/right.xml` |

**Only modification:** mesh search paths were rewritten from
`../meshes/right/` to `../meshes/` to match this flattened layout
(`filename=` attributes in the URDF, `meshdir=` in the MJCF). Geometry, joint
definitions and limits are untouched.

The URDF declares 25 joints, of which 20 are revolute and actuated; the MJCF is
used only by the Mink IK baseline.
