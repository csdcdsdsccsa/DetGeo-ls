#!/usr/bin/env bash
# CSFI-removal controls: fine guidance -> bidirectional guidance.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
VARIANTS=(h2_ind_fg_nocsfi h2_ind_bi_nocsfi)
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --standard_rng
  --seed 2024 --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2
  --coarse_sigma 1.5 --fine_sigma 3.0)

PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_cross_scale_hier.py --batch_size 7
for VARIANT in "${VARIANTS[@]}"; do
  NAME="trogeo_ms_e4_${VARIANT}_standardrng_swin_t_drone_seed2024"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
    --trogeo_ms_det_variant "$VARIANT" --savename "$NAME"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_ms_det_variant "$VARIANT" \
    --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
  rm -f "saved_models/${NAME}_checkpoint.pth.tar"
done
