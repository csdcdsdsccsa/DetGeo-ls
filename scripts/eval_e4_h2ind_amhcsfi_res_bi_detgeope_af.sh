#!/usr/bin/env bash
# Inference-only Bi-Res + DetGeo-PE agreement fusion.  Reuses its best checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
CHECKPOINT="saved_models/trogeo_ms_e4_amhcsfi_res_bi_detgeope_naturalrng_swin_t_drone_seed2024_model_best.pth.tar"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi
  --trogeo_position_mode detgeo --trogeo_click_map_mode distance --dadpe_mode none --amr_pe_mode none
  --agreement_fusion_decode --h3_iou_threshold 0.5
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0)
PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_click_maps.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_amhcsfi_res.py --batch_size 2
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "$CHECKPOINT" --val \
  --savename "trogeo_ms_e4_amhcsfi_res_bi_detgeope_af_iou05_val"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "$CHECKPOINT" --test \
  --savename "trogeo_ms_e4_amhcsfi_res_bi_detgeope_af_iou05_test"
