#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
COMMON_ARGS=(
  --gpu 0
  --checkpoint saved_models/model_droneaerial_bs8_model_best.pth.tar
  --data-root data
  --data-name CVOGL_DroneAerial
  --epochs 3
  --batch-size 4
  --num-workers 8
  --lr 1e-4
)

mkdir -p logs outputs/oacfr_e05

"${PYTHON_BIN}" train_oacfr.py "${COMMON_ARGS[@]}" \
  --search-scales 64 \
  --output-dir outputs/oacfr_e05/64_only_3e \
  > logs/oacfr_e05_64_only_3e.log 2>&1

"${PYTHON_BIN}" train_oacfr.py "${COMMON_ARGS[@]}" \
  --search-scales 32,64,128 \
  --output-dir outputs/oacfr_e05/coarse_to_fine_3e \
  > logs/oacfr_e05_coarse_to_fine_3e.log 2>&1

printf 'complete\n' > outputs/oacfr_e05/run_status.txt
