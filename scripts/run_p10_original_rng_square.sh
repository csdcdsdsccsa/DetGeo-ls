#!/usr/bin/env bash
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PYTHONPATH=. "$PYTHON_BIN" train.py --gpu 0 --num_workers 24 --max_epoch 25 --lr 1e-4 --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename p10_original_rng_square_seed13 --seed 13 --beta 1.0 --original_rng_matched --rng_probe --print_freq 50 > logs/p10_original_rng_square_seed13.log 2>&1
