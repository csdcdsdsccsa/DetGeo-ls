#!/usr/bin/env bash
# E1 -> E2 -> E3 -> E4 training, validation-best test once per checkpoint,
# then E5 H3 decoding of the same E4 checkpoint.  Run inside screen/tmux.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50)

run_train_and_test() {
  local experiment="$1"
  local variant="$2"
  local name="trogeo_ms_${experiment}_swin_t_drone_seed2024"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 \
    --trogeo_ms_det_variant "$variant" --savename "$name"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_ms_det_variant "$variant" \
    --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test"
  rm -f "saved_models/${name}_checkpoint.pth.tar"
}

run_train_and_test e1_correct63 correct63
run_train_and_test e2_multigrid b_multigrid
run_train_and_test e3_h2_shared h2_shared
run_train_and_test e4_h2_ind h2_ind

# E5 is an inference-only decode of exactly E4's best checkpoint.
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --trogeo_ms_det_variant h3_ind \
  --pretrain saved_models/trogeo_ms_e4_h2_ind_swin_t_drone_seed2024_model_best.pth.tar \
  --test --savename trogeo_ms_e5_h3_ind_swin_t_drone_seed2024_test
