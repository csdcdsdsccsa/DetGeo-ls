#!/usr/bin/env bash
# Strict E4 H2-Ind backbone comparison: tiled ImageNet ViT-T then ViT-S.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --standard_rng --seed 2024
  --beta 1.0 --print_freq 50 --trogeo_ms_det_variant h2_ind)

for BACKBONE in vit_t vit_s; do
  NAME="trogeo_ms_e4_h2_ind_${BACKBONE}_tiled256_drone_seed2024"
  PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_e4_vit_backbones.py --backbone "$BACKBONE" --batch_size 1
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_backbone "$BACKBONE" \
    --max_epoch 25 --lr 1e-4 --savename "$NAME"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_backbone "$BACKBONE" \
    --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
  rm -f "saved_models/${NAME}_checkpoint.pth.tar"
done
