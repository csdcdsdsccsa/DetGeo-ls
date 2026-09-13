#!/usr/bin/env bash
# Bridge an already-running legacy single-head queue: skip attempted variants,
# but run later variants that were never reached after an earlier failure.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PIPELINE_LOG="logs/fg_amhcsfi_singlehead.pipeline"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

run_if_unattempted() {
  local variant="$1"
  local name="trogeo_ms_e4_${variant}_standardrng_swin_t_drone_seed2024"
  if grep -Eq "trogeo_ms_det_variant.*${variant}" "$PIPELINE_LOG"; then
    echo "[queue] ${variant} was already attempted by the original queue; not retrying"
    return 0
  fi
  echo "[queue] ${variant} was never reached; launching it now"
  if ! PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
    --trogeo_ms_det_variant "$variant" --savename "$name"; then
    echo "[queue] ${variant} training failed; continuing" >&2
    return 0
  fi
  if [[ ! -f "saved_models/${name}_model_best.pth.tar" ]]; then
    echo "[queue] ${variant} has no best checkpoint; continuing" >&2
    return 0
  fi
  if ! PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_ms_det_variant "$variant" \
    --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test"; then
    echo "[queue] ${variant} test failed; continuing" >&2
  fi
  rm -f "saved_models/${name}_checkpoint.pth.tar"
}

run_if_unattempted h2_ind_fg_amhcsfi_res_s4
run_if_unattempted h2_ind_fg_amhcsfi_res_afuse
