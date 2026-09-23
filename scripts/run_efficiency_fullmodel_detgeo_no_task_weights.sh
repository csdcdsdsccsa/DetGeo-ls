#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
OUTPUT_CSV="${OUTPUT_CSV:-results/efficiency_fullmodel_vs_detgeo_no_task_weights.csv}"
OUTPUT_MD="${OUTPUT_MD:-results/efficiency_fullmodel_vs_detgeo_no_task_weights.md}"
OUTPUT_LOG="${OUTPUT_LOG:-logs/efficiency_fullmodel_vs_detgeo_no_task_weights.log}"

# DetGeo's constructor always loads this original YOLOv3 initialization.
[[ -f saved_models/yolov3.weights ]] || { echo 'Missing DetGeo initialization: saved_models/yolov3.weights' >&2; exit 1; }
"$PYTHON_BIN" -c 'import fvcore' >/dev/null 2>&1 || { echo "fvcore is not installed in $PYTHON_BIN" >&2; exit 1; }

mkdir -p results logs
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" tools/profile_efficiency.py \
  --device cuda:0 --no_task_checkpoint \
  --warmup "${WARMUP:-100}" --iterations "${ITERATIONS:-500}" --repeats "${REPEATS:-5}" \
  --output_csv "$OUTPUT_CSV" --output_md "$OUTPUT_MD" \
  --comparison_md results/efficiency_checkpoint_effect_comparison.md \
  2>&1 | tee "$OUTPUT_LOG"
