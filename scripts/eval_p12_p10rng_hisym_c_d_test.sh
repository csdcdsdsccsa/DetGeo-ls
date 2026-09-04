#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(--gpu 0 --num_workers 16 --max_epoch 25 --lr 1e-4 --prompt_lr 1e-4 --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --seed 13 --beta 1.0 --original_rng_matched --test --print_freq 50)
echo '===== P12 C-new HiSym PAE test ====='
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --savename p12_p10rng_hisym_pae_c_seed13_test --hisym_pae --pretrain saved_models/p12_p10rng_hisym_pae_c_seed13_model_best.pth.tar
echo '===== P12 D-new SAM + HiSym PAE test ====='
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --savename p12_p10rng_sam_hisym_pae_d_seed13_test --sam_refined_pe --hisym_pae --pretrain saved_models/p12_p10rng_sam_hisym_pae_d_seed13_model_best.pth.tar
