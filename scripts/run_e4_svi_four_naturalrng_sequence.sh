#!/usr/bin/env bash
# Four DroneAerial ablations transferred to CVOGL_SVI, in the requested order.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
DATA_DIR="data/CVOGL_SVI"
for SPLIT in train val test; do
  test -f "${DATA_DIR}/CVOGL_SVI_${SPLIT}.pth" || {
    echo "Missing SVI ${SPLIT} split" >&2
    exit 1
  }
done

COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_SVI --trogeo_backbone swin_t
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024
  --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

run_one() {
  local name="$1" variant="$2" position="$3" map="$4"
  local -a args=("${COMMON[@]}" --trogeo_ms_det_variant "$variant"
                 --trogeo_position_mode "$position" --trogeo_click_map_mode "$map")
  if [[ "$map" == gaussian ]]; then
    args+=(--gaussian_sigma 25)
  fi
  if [[ -e "saved_models/${name}_model_best.pth.tar" ]]; then
    echo "Refusing to overwrite existing best checkpoint: ${name}" >&2
    exit 1
  fi
  echo "[SVI queue] START ${name}"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${args[@]}" --max_epoch 25 --lr 1e-4 --savename "$name"
  test -f "saved_models/${name}_model_best.pth.tar"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${args[@]}" \
    --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test"
  rm -f "saved_models/${name}_checkpoint.pth.tar"
  echo "[SVI queue] DONE ${name}"
}

run_one trogeo_ms_e4_amhcsfi_res_bi_hisym_crgpe_core25_outer50_naturalrng_swin_t_svi_seed2024 \
  h2_ind_amhcsfi_res_bi hisym_crgpe gaussian
run_one trogeo_ms_e4_h2_ind_amhcsfi_res_bi_currentpe_distance_naturalrng_swin_t_svi_seed2024 \
  h2_ind_amhcsfi_res_bi current distance
run_one trogeo_ms_e4_h2_ind_bi_nocsfi_currentpe_distance_naturalrng_swin_t_svi_seed2024 \
  h2_ind_bi_nocsfi current distance
run_one trogeo_ms_e4_h2_ind_corr2s_currentpe_distance_naturalrng_swin_t_svi_seed2024 \
  h2_ind_corr2s current distance

echo '[SVI queue] ALL FOUR COMPLETE'
