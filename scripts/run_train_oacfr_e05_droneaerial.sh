#!/usr/bin/env bash
set -euo pipefail

python train_oacfr.py \
  --gpu 0 \
  --checkpoint saved_models/model_droneaerial_bs8_model_best.pth.tar \
  --data-root data \
  --data-name CVOGL_DroneAerial \
  --epochs 3 \
  --batch-size 4 \
  --num-workers 8 \
  --search-scales 32,64,128 \
  --lr 1e-4 \
  --output-dir outputs/oacfr_e05/coarse_to_fine_3e
