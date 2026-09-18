#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
NAME="hisym_crgpe_bires_thresholdreg_af_naturalrng_swin_t_drone_seed2024"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian --gaussian_sigma 25 --dadpe_mode none --amr_pe_mode none --bbox_threshold_reg --bbox_threshold_reg_weight 0.2 --bbox_threshold_temperature 0.05 --bbox_threshold_weight25 0.5 --bbox_threshold_weight50 1.0 --agreement_fusion_decode --h3_iou_threshold 0.5 --standard_rng --seed 2024 --beta 1.0 --print_freq 50)
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 --savename "$NAME"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
