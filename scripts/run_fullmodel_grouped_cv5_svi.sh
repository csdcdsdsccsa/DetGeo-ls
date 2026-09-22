#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"

for FOLD in 1 2 3 4 5; do
  NAME="fullmodel_swin_t_groupcv5_svi_f${FOLD}_seed2024"
  if [[ -e "saved_models/${NAME}_model_best.pth.tar" ]]; then
    echo "Refusing to overwrite existing checkpoint: ${NAME}" >&2
    exit 1
  fi
  PYTHONPATH=. "$PYTHON_BIN" train.py \
    --gpu "$GPU" --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 \
    --data_root data --data_name CVOGL_SVI \
    --train_pth "data/CVOGL_SVI/cv5_grouped/fold${FOLD}_train.pth" \
    --val_pth "data/CVOGL_SVI/cv5_grouped/fold${FOLD}_val.pth" \
    --trogeo_backbone swin_t --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi \
    --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian \
    --gaussian_sigma 25 --gaussian_sigma_x 50 --crgpe_outer_sigma 50 --crgpe_outer_sigma_x 100 \
    --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024 --beta 1.0 --print_freq 50 \
    --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 \
    --max_epoch 25 --lr 1e-4 --savename "$NAME"
  test -f "saved_models/${NAME}_model_best.pth.tar"
  rm -f "saved_models/${NAME}_checkpoint.pth.tar"
done
