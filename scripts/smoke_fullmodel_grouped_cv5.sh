#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"

if [[ ! -f data/CVOGL_DroneAerial/cv5_grouped/manifest.json ]]; then
  PYTHONPATH=. "$PYTHON_BIN" tools/make_cvogl_grouped_cv5.py --data_root data --datasets CVOGL_DroneAerial
fi
if [[ ! -f data/CVOGL_SVI/cv5_grouped/manifest.json ]]; then
  PYTHONPATH=. "$PYTHON_BIN" tools/make_cvogl_grouped_cv5.py --data_root data --datasets CVOGL_SVI
fi
PYTHONPATH=. "$PYTHON_BIN" tools/check_cvogl_grouped_cv5.py --data_root data
PYTHONPATH=. "$PYTHON_BIN" tools/smoke_cvogl_grouped_cv5.py --dataset CVOGL_DroneAerial --gpu "$GPU"
PYTHONPATH=. "$PYTHON_BIN" tools/smoke_cvogl_grouped_cv5.py --dataset CVOGL_SVI --gpu "$GPU"

run_one_epoch() {
  local dataset="$1" name="$2" extra_args="$3"
  # shellcheck disable=SC2086
  PYTHONPATH=. "$PYTHON_BIN" train.py --gpu "$GPU" --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 \
    --data_root data --data_name "$dataset" --train_pth "data/${dataset}/cv5_grouped/fold1_train.pth" \
    --val_pth "data/${dataset}/cv5_grouped/fold1_val.pth" --trogeo_backbone swin_t \
    --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi --trogeo_position_mode hisym_crgpe \
    --trogeo_click_map_mode gaussian $extra_args --dadpe_mode none --amr_pe_mode none \
    --standard_rng --seed 2024 --beta 1.0 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 \
    --max_epoch 1 --lr 1e-4 --savename "$name"
  rm -f "saved_models/${name}_checkpoint.pth.tar" "saved_models/${name}_model_best.pth.tar"
}

run_one_epoch CVOGL_DroneAerial fullmodel_swin_t_groupcv5_smoke_drone_seed2024 '--gaussian_sigma 25 --crgpe_outer_sigma 50'
run_one_epoch CVOGL_SVI fullmodel_swin_t_groupcv5_smoke_svi_seed2024 '--gaussian_sigma 25 --gaussian_sigma_x 50 --crgpe_outer_sigma 50 --crgpe_outer_sigma_x 100'
echo 'Grouped CV5 smoke COMPLETE'
