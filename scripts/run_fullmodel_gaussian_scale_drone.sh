#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
RESULT_DIR="results/gaussian_scale"
mkdir -p "$RESULT_DIR"
run_scale() {
  local core="$1" outer="$2" name="fullmodel_gausscale_drone_g${1}_seed2024"
  local best="saved_models/${name}_model_best.pth.tar"
  [[ ! -e "$best" ]] || { echo "Refusing to overwrite: $best" >&2; exit 1; }
  local common=(--gpu "$GPU" --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian --gaussian_bank 15,20,25,30,35 --gaussian_sigma "$core" --crgpe_outer_sigma "$outer" --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024 --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0)
  echo "Drone G${core}: core=${core}x${core} outer=${outer}x${outer}"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${common[@]}" --max_epoch 25 --lr 1e-4 --savename "$name" 2>&1 | tee "${RESULT_DIR}/${name}_train.log"
  test -f "$best"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${common[@]}" --pretrain "$best" --val --savename "${name}_val" 2>&1 | tee "${RESULT_DIR}/${name}_val.log"
  rm -f "saved_models/${name}_checkpoint.pth.tar"
}
run_scale 15 30; run_scale 20 40; run_scale 25 50; run_scale 30 60; run_scale 35 70
echo 'Drone Gaussian-scale sweep COMPLETE'
