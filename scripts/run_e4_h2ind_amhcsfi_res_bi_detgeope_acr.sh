#!/usr/bin/env bash
# Frozen Bi-Res + DetGeo-PE + Agreement-Conditioned Residual refinement.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
BASE_CHECKPOINT="saved_models/trogeo_ms_e4_amhcsfi_res_bi_detgeope_naturalrng_swin_t_drone_seed2024_model_best.pth.tar"
NAME="trogeo_ms_e4_amhcsfi_res_bi_detgeope_acr_naturalrng_swin_t_drone_seed2024"

COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi
  --trogeo_position_mode detgeo --trogeo_click_map_mode distance --dadpe_mode none --amr_pe_mode none
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
  --acr_head --acr_iou_threshold 0.5 --acr_lr 1e-3 --acr_giou_weight 1.0
  --acr_reg_weight 0.5 --acr_alpha_weight 0.01 --acr_weight_decay 1e-4
)

PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_acr_head.py
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "$BASE_CHECKPOINT" --max_epoch 25 --savename "$NAME"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
