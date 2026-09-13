#!/usr/bin/env bash
# FG-AMHCSFI-Res single-head ablation: S3 -> S4 -> adaptive fusion.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

if ! PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_fg_amhcsfi_singlehead.py --batch_size 2; then
  echo "[queue] single-head sanity failed; continuing with independently guarded experiments" >&2
fi

for VARIANT in h2_ind_fg_amhcsfi_res_s3 h2_ind_fg_amhcsfi_res_s4 h2_ind_fg_amhcsfi_res_afuse; do
  NAME="trogeo_ms_e4_${VARIANT}_standardrng_swin_t_drone_seed2024"
  if ! PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
    --trogeo_ms_det_variant "$VARIANT" --savename "$NAME"; then
    echo "[queue] ${VARIANT} training failed; skipping its test and continuing" >&2
    continue
  fi
  if [[ ! -f "saved_models/${NAME}_model_best.pth.tar" ]]; then
    echo "[queue] ${VARIANT} has no best checkpoint; skipping its test and continuing" >&2
    continue
  fi
  if ! PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_ms_det_variant "$VARIANT" \
    --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"; then
    echo "[queue] ${VARIANT} test failed; continuing" >&2
  fi
  rm -f "saved_models/${NAME}_checkpoint.pth.tar"
done
