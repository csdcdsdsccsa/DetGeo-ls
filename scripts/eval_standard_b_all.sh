#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
declare -A variants=([e0]=none [e1_msst]=msst [e2_core]=core [e3_b1]=b1 [e4_b2]=b2)
for key in e0 e1_msst e2_core e3_b1 e4_b2; do
  savename="standard_b_${key}_seed13"
  cmd=("$PYTHON_BIN" train.py --gpu 0 --num_workers 16 --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename "${savename}_test" --seed 13 --beta 1.0 --standard_rng --pretrain "saved_models/${savename}_model_best.pth.tar" --test --print_freq 50)
  if [[ "${variants[$key]}" != none ]]; then cmd+=(--b_variant "${variants[$key]}"); fi
  PYTHONPATH=. "${cmd[@]}" > "logs/${savename}_test.log" 2>&1
done
