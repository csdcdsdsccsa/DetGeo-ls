#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(--gpu 0 --num_workers 16 --max_epoch 25 --lr 1e-4 --prompt_lr 1e-4 --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --seed 13 --beta 1.0 --standard_rng --test --print_freq 50)
echo '===== P13 A Square test ====='
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --savename p13_standard_a_square_seed13_test --pretrain saved_models/p13_standard_a_square_seed13_model_best.pth.tar
echo '===== P13 B SAM refined test ====='
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --savename p13_standard_b_sam_refined_seed13_test --sam_refined_pe --pretrain saved_models/p13_standard_b_sam_refined_seed13_model_best.pth.tar
echo '===== P13 C HiSym PAE test ====='
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --savename p13_standard_c_hisym_pae_seed13_test --hisym_pae --pretrain saved_models/p13_standard_c_hisym_pae_seed13_model_best.pth.tar
echo '===== P13 D SAM + HiSym PAE test ====='
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --savename p13_standard_d_sam_hisym_pae_seed13_test --sam_refined_pe --hisym_pae --pretrain saved_models/p13_standard_d_sam_hisym_pae_seed13_model_best.pth.tar
