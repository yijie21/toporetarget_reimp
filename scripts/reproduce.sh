#!/usr/bin/env bash
# Reproduce every number in RESULTS.md from scratch.
#
# The datasets are licence-gated and are not in this repository, so point these
# at your own copies (see THIRD_PARTY.md for where to get them).  Everything
# else -- robot model, configs, eval set -- is vendored.
#
#   CONTACTPOSE_GRASPS=... CONTACTPOSE_MODELS=... ./scripts/reproduce.sh
#
# Optional, for the sequence experiment only:
#   GRAB_ROOT=... MANO_MODEL=... GRAB_SUBJECT_MESHES=... GRAB_OBJECT_MESHES=...
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CONTACTPOSE_GRASPS:?set it to the dir holding <participant>_<intent>/}"
: "${CONTACTPOSE_MODELS:?set it to the dir holding <object>.ply (millimetres)}"

ITERS="${ITERS:-500}"
mkdir -p runs

echo "=== 1/4  freeze the evaluation set and the tuning split ==="
python scripts/make_eval_set.py     --grasps "$CONTACTPOSE_GRASPS" --models "$CONTACTPOSE_MODELS"
python scripts/make_tuning_split.py --grasps "$CONTACTPOSE_GRASPS"

echo "=== 2/4  fit every method's hyper-parameters on the held-out split ==="
# Skip with SKIP_TUNING=1 to use the frozen values already in configs/.
if [ "${SKIP_TUNING:-0}" != "1" ]; then
  for m in ours dexpilot mink; do
    python scripts/tune.py --grasps "$CONTACTPOSE_GRASPS" --models "$CONTACTPOSE_MODELS" \
      --method "$m" --budget "${BUDGET:-40}" --seed 0 --workers "${WORKERS:-6}"
  done
  echo "NOTE: tuning writes runs/tuning_*.json; the winning values are already"
  echo "      frozen in configs/. Re-freeze them by hand if the search moved."
fi

echo "=== 3/4  main table: ours vs baselines vs ablations on ContactPose ==="
python scripts/eval_contactpose.py --grasps "$CONTACTPOSE_GRASPS" --models "$CONTACTPOSE_MODELS" \
  --methods ours,dexpilot,mink,ours:no_IM,ours:no_pen,ours:no_bone,ours:no_reg \
  --iters "$ITERS" --out runs/contactpose.json

echo "=== 4/4  sequence experiment (GRAB): does E_reg buy smoothness? ==="
if [ -n "${GRAB_ROOT:-}" ] && [ -n "${MANO_MODEL:-}" ]; then
  SEQ="${GRAB_SEQ:-$(find "$GRAB_ROOT" -name 'cylindersmall_lift.npz' | head -1)}"
  python scripts/eval_grab_sequence.py --seq "$SEQ" --mano "$MANO_MODEL" \
    --subject-meshes "$GRAB_SUBJECT_MESHES" --object-meshes "$GRAB_OBJECT_MESHES" \
    --out runs/grab_sequence.json
else
  echo "  skipped: set GRAB_ROOT, MANO_MODEL, GRAB_SUBJECT_MESHES, GRAB_OBJECT_MESHES"
fi

echo "=== rendering RESULTS.md ==="
python scripts/make_results.py --out RESULTS.md
echo "done — see RESULTS.md"
