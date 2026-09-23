#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
CHECKPOINTS=(
  "saved_models/scratch25_worker24_square_seed13_model_best.pth.tar"
  "saved_models/scratch25_worker24_square_svi_naturalrng_seed13_model_best.pth.tar"
  "saved_models/trogeo_ms_e4_amhcsfi_res_bi_hisym_crgpe_core25_outer50_naturalrng_swin_t_drone_seed2024_model_best.pth.tar"
  "saved_models/trogeo_ms_e4_amhcsfi_res_bi_hisym_crgpe_corey25_corex50_outery50_outerx100_naturalrng_swin_t_svi_seed2024_model_best.pth.tar"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
done
[[ -f saved_models/yolov3.weights ]] || { echo "Missing DetGeo initialization: saved_models/yolov3.weights" >&2; exit 1; }
"$PYTHON_BIN" -c "import fvcore" >/dev/null 2>&1 || {
  echo "fvcore is not installed in $PYTHON_BIN" >&2
  exit 1
}

mkdir -p results logs
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" tools/profile_efficiency.py \
  --device cuda:0 --warmup "${WARMUP:-100}" --iterations "${ITERATIONS:-500}" --repeats "${REPEATS:-5}" \
  --output_csv results/efficiency_fullmodel_vs_detgeo.csv \
  --output_md results/efficiency_fullmodel_vs_detgeo.md \
  2>&1 | tee logs/efficiency_fullmodel_vs_detgeo.log
