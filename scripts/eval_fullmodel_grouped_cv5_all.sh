#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"

evaluate_dataset() {
  local dataset="$1" prefix="$2" extra_args="$3"
  for FOLD in 1 2 3 4 5; do
    local name="${prefix}_f${FOLD}_seed2024"
    local split_dir="data/${dataset}/cv5_grouped"
    local checkpoint="saved_models/${name}_model_best.pth.tar"
    test -f "$checkpoint"
    # shellcheck disable=SC2086
    PYTHONPATH=. "$PYTHON_BIN" train.py --gpu "$GPU" --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 \
      --data_root data --data_name "$dataset" --train_pth "${split_dir}/fold${FOLD}_train.pth" \
      --val_pth "${split_dir}/fold${FOLD}_val.pth" --trogeo_backbone swin_t \
      --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi --trogeo_position_mode hisym_crgpe \
      --trogeo_click_map_mode gaussian $extra_args --dadpe_mode none --amr_pe_mode none \
      --standard_rng --seed 2024 --beta 1.0 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 \
      --pretrain "$checkpoint" --val --savename "${name}_cvval"
    # shellcheck disable=SC2086
    PYTHONPATH=. "$PYTHON_BIN" train.py --gpu "$GPU" --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 \
      --data_root data --data_name "$dataset" --train_pth "${split_dir}/fold${FOLD}_train.pth" \
      --val_pth "${split_dir}/fold${FOLD}_val.pth" --test_pth "data/${dataset}/${dataset}_test.pth" \
      --trogeo_backbone swin_t --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi \
      --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian $extra_args \
      --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024 --beta 1.0 \
      --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 \
      --pretrain "$checkpoint" --test --savename "${name}_cvtest"
  done
}

evaluate_dataset CVOGL_DroneAerial fullmodel_swin_t_groupcv5_drone '--gaussian_sigma 25 --crgpe_outer_sigma 50'
evaluate_dataset CVOGL_SVI fullmodel_swin_t_groupcv5_svi '--gaussian_sigma 25 --gaussian_sigma_x 50 --crgpe_outer_sigma 50 --crgpe_outer_sigma_x 100'
