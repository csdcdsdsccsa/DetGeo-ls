#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
CKPT="saved_models/trogeo_ms_e4_amhcsfi_res_bi_hisym_crgpe_core25_outer50_naturalrng_swin_t_drone_seed2024_model_best.pth.tar"
for split in val test; do
  PYTHONPATH=. "$PYTHON_BIN" tools/eval_fullmodel_sps.py --gpu "${GPU:-0}" --data_root data --seg_root data-Seg --data_name CVOGL_DroneAerial --split "$split" --checkpoint "$CKPT" --sam_checkpoint "${SAM_CKPT:-checkpoints/sam_vit_h_4b8939.pth}" --output_dir results/fullmodel_sps
done
