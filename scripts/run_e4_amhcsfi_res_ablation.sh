#!/usr/bin/env bash
# Ordinary CSFI -> zero-start AMHCSFI-Res in Bi, CG, then FG paths.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
# Keep a rerun separate from any pre-existing log directory/checkpoint namespace.
RUN_TAG="${RUN_TAG:-rerun1}"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --standard_rng
  --seed 2024 --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0)
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_amhcsfi_res.py --batch_size 2
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_cross_scale_hier.py --batch_size 7
for VARIANT in h2_ind_amhcsfi_res_bi h2_ind_cg_amhcsfi_res h2_ind_fg_amhcsfi_res; do
  NAME="trogeo_ms_e4_${VARIANT}_standardrng_swin_t_drone_seed2024_${RUN_TAG}"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
    --trogeo_ms_det_variant "$VARIANT" --savename "$NAME"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_ms_det_variant "$VARIANT" \
    --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
  rm -f "saved_models/${NAME}_checkpoint.pth.tar"
done
touch logs/trogeo_ms_e4_amhcsfi_res_ablation.SUCCESS
