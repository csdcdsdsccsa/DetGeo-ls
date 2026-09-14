#!/usr/bin/env bash
# Bi-Res + fixed / adaptive multi-range DetGeo position field.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi
  --trogeo_position_mode detgeo --trogeo_click_map_mode distance --dadpe_mode none
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

run_experiment() {
  local mode="$1"
  local name="$2"
  if ! PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --amr_pe_mode "$mode" \
      --max_epoch 25 --lr 1e-4 --savename "$name"; then
    echo "[queue] ${name} training failed; skipping test and continuing" >&2
    return 0
  fi
  if [[ ! -f "saved_models/${name}_model_best.pth.tar" ]]; then
    echo "[queue] ${name} has no best checkpoint; skipping test and continuing" >&2
    return 0
  fi
  if ! PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --amr_pe_mode "$mode" \
      --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test"; then
    echo "[queue] ${name} test failed; continuing" >&2
  fi
  rm -f "saved_models/${name}_checkpoint.pth.tar"
}

if ! PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_amr_pe.py; then
  echo '[queue] AMR-PE sanity failed; continuing with guarded experiments' >&2
fi

run_experiment fixed "trogeo_ms_e4_amhcsfi_res_bi_amrpe_fixed_naturalrng_swin_t_drone_seed2024"
run_experiment adaptive "trogeo_ms_e4_amhcsfi_res_bi_amrpe_adaptive_naturalrng_swin_t_drone_seed2024"
