#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
NAME="trogeo_ms_e4_amhcsfi_res_bi_hisymgpe_gaussian_sigma25_naturalrng_swin_t_drone_seed2024"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi --trogeo_position_mode hisym_pe --trogeo_click_map_mode gaussian --gaussian_sigma 25 --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024 --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0)
PYTHONPATH=. "$PYTHON_BIN" tools/test_frontend_position_encoders.py
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 --savename "$NAME"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
