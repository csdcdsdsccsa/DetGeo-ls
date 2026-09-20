#!/usr/bin/env bash
# One-epoch runs for both datasets; formal scripts start only after both pass.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --trogeo_backbone resnet50 --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi
  --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024
  --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)
PYTHONPATH=. "$PYTHON_BIN" tools/test_fullmodel_resnet50.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_click_maps.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_hisym_core_ring_gpe.py

smoke_one() {
  local data_name="$1" name="$2"; shift 2
  rm -f "saved_models/${name}_checkpoint.pth.tar" "saved_models/${name}_model_best.pth.tar"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --data_root data --data_name "$data_name" "$@" --max_epoch 1 --lr 1e-4 --savename "$name"
  test -f "saved_models/${name}_model_best.pth.tar"
  rm -f "saved_models/${name}_checkpoint.pth.tar" "saved_models/${name}_model_best.pth.tar"
  echo "[ResNet50 Full Model smoke] PASS ${data_name}"
}
smoke_one CVOGL_DroneAerial smoke_fullmodel_resnet50_drone_naturalrng_seed2024 --gaussian_sigma 25 --crgpe_outer_sigma 50
smoke_one CVOGL_SVI smoke_fullmodel_resnet50_svi_naturalrng_seed2024 --gaussian_sigma 25 --gaussian_sigma_x 50 --crgpe_outer_sigma 50 --crgpe_outer_sigma_x 100
echo '[ResNet50 Full Model smoke] BOTH PASSED'
