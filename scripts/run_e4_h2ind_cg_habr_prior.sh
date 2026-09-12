#!/usr/bin/env bash
# A2 coarse guidance plus HABR-Prior, replacing A3's post-matching CSFI.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
NAME="trogeo_ms_e4_h2_ind_cg_habr_prior_standardrng_swin_t_drone_seed2024"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --standard_rng
  --seed 2024 --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2 --coarse_sigma 1.5)
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_cross_scale_hier.py --batch_size 7
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
  --trogeo_ms_det_variant h2_ind_cg_habr_prior --savename "$NAME"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_ms_det_variant h2_ind_cg_habr_prior \
  --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
