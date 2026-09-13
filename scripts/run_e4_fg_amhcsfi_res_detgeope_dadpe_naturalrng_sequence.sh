#!/usr/bin/env bash
# FG-Res + original DetGeo front-end PE / DADPE, with ordinary natural RNG.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --trogeo_ms_det_variant h2_ind_fg_amhcsfi_res
  --trogeo_position_mode detgeo --trogeo_click_map_mode distance
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

run_experiment() {
  local name="$1"
  local dadpe_mode="$2"
  if ! PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --dadpe_mode "$dadpe_mode" \
    --max_epoch 25 --lr 1e-4 --savename "$name"; then
    echo "[queue] ${name} training failed; skipping its test and continuing" >&2
    return 0
  fi
  if [[ ! -f "saved_models/${name}_model_best.pth.tar" ]]; then
    echo "[queue] ${name} has no best checkpoint; skipping its test and continuing" >&2
    return 0
  fi
  if ! PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --dadpe_mode "$dadpe_mode" \
    --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test"; then
    echo "[queue] ${name} test failed; continuing" >&2
  fi
  rm -f "saved_models/${name}_checkpoint.pth.tar"
}

if ! PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_dadpe.py; then
  echo "[queue] DADPE sanity failed; continuing with independently guarded experiments" >&2
fi
run_experiment "trogeo_ms_e4_fg_amhcsfi_res_detgeope_naturalrng_swin_t_drone_seed2024" none
run_experiment "trogeo_ms_e4_fg_amhcsfi_res_dadpe_b_naturalrng_swin_t_drone_seed2024" input
run_experiment "trogeo_ms_e4_fg_amhcsfi_res_msdadpe_d_naturalrng_swin_t_drone_seed2024" multiscale
