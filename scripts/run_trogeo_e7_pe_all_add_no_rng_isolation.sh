#!/usr/bin/env bash
# E7 three-scale PE-All -> K/V, with no PE-initialization RNG restoration.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
NAME="trogeo_ms_e7_h2_ind_3scale_pe_all_add_noinitrng_swin_t_drone_seed2024"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --standard_rng
  --seed 2024 --beta 1.0 --print_freq 50)

PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_query_pe_3scale.py --batch_size 7
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
  --trogeo_ms_det_variant h2_ind_3scale_pe_all_add --savename "$NAME"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" \
  --trogeo_ms_det_variant h2_ind_3scale_pe_all_add \
  --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
