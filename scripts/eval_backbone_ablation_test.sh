#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
for exp in darknet53_noshare darknet53_shared resnet50_shared; do
  savename="detgeo_${exp}"
  PYTHONPATH=. "$PYTHON_BIN" train.py --gpu 0 --num_workers 16 --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --backbone_exp "$exp" --savename "${savename}_test" --seed 13 --beta 1.0 --standard_rng --pretrain "saved_models/${savename}_model_best.pth.tar" --test --print_freq 50 > "logs/${savename}_test.log" 2>&1
done
