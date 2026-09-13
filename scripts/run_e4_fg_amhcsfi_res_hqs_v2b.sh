#!/usr/bin/env bash
# HQS-v2b: only ranker inputs differ from HQS-v2a; no automatic test run.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
VARIANT="h2_ind_fg_amhcsfi_res"
BASE_NAME="trogeo_ms_e4_${VARIANT}_standardrng_swin_t_drone_seed2024_rerun1"
BASE_CKPT="saved_models/${BASE_NAME}_model_best.pth.tar"
NAME="trogeo_ms_e4_${VARIANT}_hqs_v2b_seed2024"
[[ -f "$BASE_CKPT" ]] || { echo "Missing FG-Res checkpoint: $BASE_CKPT" >&2; exit 1; }

PYTHONPATH=. "$PYTHON_BIN" tools/test_hqs_v2b.py
PYTHONPATH=. "$PYTHON_BIN" train.py \
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 \
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t \
  --trogeo_ms_det_variant "$VARIANT" --standard_rng --seed 2024 --beta 1.0 --print_freq 50 \
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 \
  --hqs_v2b --hqs_lr 1e-3 --hqs_rank_epsilon 0.03 --hqs_rank_threshold 0.0 \
  --max_epoch 10 --pretrain "$BASE_CKPT" --savename "$NAME"

PYTHONPATH=. "$PYTHON_BIN" train.py \
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 \
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t \
  --trogeo_ms_det_variant "$VARIANT" --standard_rng --seed 2024 --beta 1.0 --print_freq 50 \
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 \
  --hqs_v2b --hqs_rank_epsilon 0.03 --hqs_rank_threshold 0.0 --hqs_v2b_rank_diag \
  --hqs_v2a_rank_diag_prefix results/hqs_v2b_rank_diag \
  --pretrain "saved_models/${NAME}_model_best.pth.tar" --val --savename "${NAME}_rank_diag"
