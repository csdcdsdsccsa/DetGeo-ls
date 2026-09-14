#!/usr/bin/env bash
# Bi-Res + DetGeo PE QCC-A: confidence times learned localization quality.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
NAME="trogeo_ms_e4_amhcsfi_res_bi_detgeope_qcc_a_naturalrng_swin_t_drone_seed2024"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi_qcc_a
  --trogeo_position_mode detgeo --trogeo_click_map_mode distance --dadpe_mode none
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
  --qcc_quality_weight 0.2 --qcc_rank_weight 0.1 --qcc_rank_epsilon 0.03 --qcc_fusion_iou 0.5)
PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_click_maps.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_amhcsfi_res.py --batch_size 2
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_qcc.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_cross_scale_hier.py --batch_size 7
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 --savename "$NAME"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
