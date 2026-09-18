#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
VARIANT="h2_ind_detgeo2s"
NAME="trogeo_ms_e4_h2_ind_detgeo2s_naturalrng_swin_t_drone_seed2024"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --trogeo_position_mode current --trogeo_click_map_mode distance
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024
  --beta 1.0 --print_freq 50)

PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_detgeo2s.py
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
  --trogeo_ms_det_variant "$VARIANT" --savename "$NAME"
if [[ -f "saved_models/${NAME}_model_best.pth.tar" ]]; then
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_ms_det_variant "$VARIANT" \
    --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
fi
