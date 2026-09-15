#!/usr/bin/env bash
# Inference-only QCC-AF decoder using the already-trained QCC-A best checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
CHECKPOINT="saved_models/trogeo_ms_e4_amhcsfi_res_bi_detgeope_qcc_a_naturalrng_swin_t_drone_seed2024_model_best.pth.tar"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi_qcc_af
  --trogeo_position_mode detgeo --trogeo_click_map_mode distance --dadpe_mode none
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
  --qcc_quality_weight 0.2 --qcc_rank_weight 0.1 --qcc_rank_epsilon 0.03 --qcc_fusion_iou 0.5)
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_qcc.py --gpu_variant h2_ind_amhcsfi_res_bi_qcc_af
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "$CHECKPOINT" --val \
  --savename "trogeo_ms_e4_amhcsfi_res_bi_qcc_af_from_qcca_iou05_val"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "$CHECKPOINT" --test \
  --savename "trogeo_ms_e4_amhcsfi_res_bi_qcc_af_from_qcca_iou05_test"
