#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
DRONE_CKPT="saved_models/fullmodel_unshared_swin_t_drone_naturalrng_seed2024_model_best.pth.tar"
SVI_CKPT="saved_models/fullmodel_unshared_swin_t_svi_naturalrng_seed2024_model_best.pth.tar"
OUTPUT_CSV="${OUTPUT_CSV:-results/efficiency_unshared_fullmodel.csv}"
OUTPUT_MD="${OUTPUT_MD:-results/efficiency_unshared_fullmodel.md}"
OUTPUT_LOG="${OUTPUT_LOG:-logs/efficiency_unshared_fullmodel.log}"

[[ -f "$DRONE_CKPT" ]] || { echo "Missing checkpoint: $DRONE_CKPT" >&2; exit 1; }
[[ -f "$SVI_CKPT" ]] || { echo "Missing checkpoint: $SVI_CKPT" >&2; exit 1; }
[[ -f results/efficiency_fullmodel_vs_detgeo.csv ]] || { echo 'Missing Shared efficiency CSV' >&2; exit 1; }
"$PYTHON_BIN" -c 'import fvcore' >/dev/null 2>&1 || { echo "fvcore is not installed in $PYTHON_BIN" >&2; exit 1; }

mkdir -p results logs
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" tools/profile_efficiency_unshared_fullmodel.py \
  --device cuda:0 --warmup "${WARMUP:-100}" --iterations "${ITERATIONS:-500}" --repeats "${REPEATS:-5}" \
  --output_csv "$OUTPUT_CSV" --output_md "$OUTPUT_MD" \
  --comparison_md results/efficiency_shared_vs_unshared_fullmodel.md \
  2>&1 | tee "$OUTPUT_LOG"
