#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
mkdir -p logs saved_models
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial
  --trogeo_backbone swin_t --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi
  --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian
  --gaussian_sigma 25 --dadpe_mode none --amr_pe_mode none
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

run_one() {
  local label="$1" guidance="$2" cross_scale="$3"
  local name="pre0d_fullmodel_${label}_swin_t_drone_seed2024"
  local best="saved_models/${name}_model_best.pth.tar"
  if [[ -e "$best" || -e "saved_models/${name}_checkpoint.pth.tar" ]]; then
    echo "Refusing to overwrite existing checkpoint for $name" >&2
    return 1
  fi
  echo "START $name guidance=$guidance cross_scale=$cross_scale"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" \
    --guidance_mode "$guidance" --cross_scale_module "$cross_scale" \
    --max_epoch 25 --lr 1e-4 --savename "$name" \
    > "logs/${name}.log" 2>&1
  if [[ ! -s "$best" ]]; then
    echo "Missing validation-best checkpoint: $best" >&2
    return 1
  fi
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" \
    --guidance_mode "$guidance" --cross_scale_module "$cross_scale" \
    --pretrain "$best" --test --savename "${name}_test" \
    > "logs/${name}_test.log" 2>&1
  echo "DONE $name"
}

run_one direct_ca direct amhcsfi_res
run_one coarse_only coarse amhcsfi_res
run_one fine_only fine amhcsfi_res
run_one original_csfi bi csfi
