#!/usr/bin/env bash
# Strict E4 controls: original DetGeo augmentation, then original DetGeo PE.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --standard_rng
  --seed 2024 --beta 1.0 --print_freq 50 --trogeo_ms_det_variant h2_ind)

PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_e4_detgeo_controls.py --batch_size 7

run_one() {
  local name="$1"
  shift
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
    --savename "$name" "$@"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" \
    --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test" "$@"
  rm -f "saved_models/${name}_checkpoint.pth.tar"
}

run_one trogeo_ms_e4_h2_ind_detgeo_aug_swin_t_drone_seed2024 --trogeo_aug_mode detgeo
run_one trogeo_ms_e4_h2_ind_detgeo_pe_swin_t_drone_seed2024 --trogeo_position_mode detgeo
