#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
VARIANT="h2_ind_bi_nocsfi"
NAME="trogeo_ms_e4_h2_ind_bi_nocsfi_detgeope_distance_naturalrng_swin_t_svi_seed2024"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_SVI --trogeo_backbone swin_t
  --trogeo_ms_det_variant "$VARIANT"
  --trogeo_position_mode detgeo --trogeo_click_map_mode distance
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024
  --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

if [[ -e "saved_models/${NAME}_model_best.pth.tar" ]]; then
  echo "Refusing to overwrite existing checkpoint: ${NAME}" >&2
  exit 1
fi

PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_click_maps.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_cross_scale_hier.py --batch_size 7
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 --savename "$NAME"
test -f "saved_models/${NAME}_model_best.pth.tar"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" \
  --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
