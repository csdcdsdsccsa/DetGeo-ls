#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
NAME="fullmodel_unshared_swin_t_drone_naturalrng_seed2024"
BEST="saved_models/${NAME}_model_best.pth.tar"

COMMON=(
  --gpu "$GPU" --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial
  --trogeo_backbone swin_t --trogeo_unshared_backbone
  --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi
  --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian
  --gaussian_sigma 25 --crgpe_outer_sigma 50
  --dadpe_mode none --amr_pe_mode none
  --standard_rng --seed 2024
  --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

if [[ -e "$BEST" ]]; then
  echo "Refusing to overwrite existing best checkpoint: $BEST" >&2
  exit 1
fi

PYTHONPATH=. "$PYTHON_BIN" tools/test_hisym_core_ring_gpe.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_fullmodel_unshared_swin_t.py --forward --device "cuda:${GPU%%,*}"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 --savename "$NAME"
test -f "$BEST"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "$BEST" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
