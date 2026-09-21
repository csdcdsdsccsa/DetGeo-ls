#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
DATA_ROOT="${VIGOR_DATA_ROOT:-data-1/VIGOR-Building}"
NAME="trogeo_ms_vigor_building_full_hisym_crgpe_swin_t_seed13_v2"

COMMON=(
  --gpu 0 --num_workers 24 --batch_size 12 --emb_size 768 --img_size 640
  --data_root "$DATA_ROOT" --data_name VIGOR_Building
  --train_pth "$DATA_ROOT/train.pth" --val_pth "$DATA_ROOT/val.pth" --test_pth "$DATA_ROOT/test.pth"
  --trogeo_backbone swin_t --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi
  --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian
  --gaussian_sigma 25 --gaussian_sigma_x 50 --crgpe_outer_sigma 50 --crgpe_outer_sigma_x 100
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 13
  --trogeo_optimizer adamw --train_weight_decay 5e-4 --lr 1e-4
  --beta 1.0 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 --print_freq 25
)

PYTHONPATH=. "$PYTHON_BIN" tools/test_vigor_dynamic_grid.py
PYTHONPATH=. "$PYTHON_BIN" tools/smoke_vigor_full_model.py
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --savename "$NAME"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
