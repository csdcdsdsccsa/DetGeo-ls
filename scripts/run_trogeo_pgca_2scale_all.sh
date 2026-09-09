#!/usr/bin/env bash
# Strict E4 H2-Ind two-scale PGCA C -> A -> B: validation-best then one test.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --standard_rng
  --seed 2024 --beta 1.0 --print_freq 50)

PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_pgca_2scale.py --batch_size 7

for SPEC in \
  "h2_ind_pgca_c_direct:trogeo_ms_e4_pgca_c_direct_bias_swin_t_drone_seed2024" \
  "h2_ind_pgca_a_conv:trogeo_ms_e4_pgca_a_conv_bias_swin_t_drone_seed2024" \
  "h2_ind_pgca_b_dynamic:trogeo_ms_e4_pgca_b_dynamic_lambda_swin_t_drone_seed2024"; do
  VARIANT="${SPEC%%:*}"
  NAME="${SPEC#*:}"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
    --trogeo_ms_det_variant "$VARIANT" --savename "$NAME"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_ms_det_variant "$VARIANT" \
    --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
  rm -f "saved_models/${NAME}_checkpoint.pth.tar"
done
