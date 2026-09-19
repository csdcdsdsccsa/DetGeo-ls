#!/usr/bin/env bash
# Freshly rerun the two unfinished SVI DetGeo-PE controls after the B/A/H queue.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_SVI --trogeo_backbone swin_t
  --trogeo_position_mode detgeo --trogeo_click_map_mode distance
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024
  --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_click_maps.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_cross_scale_hier.py --batch_size 7
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_detgeo2s.py --data_name CVOGL_SVI

run_one() {
  local name="$1" variant="$2"
  if [[ -e "saved_models/${name}_model_best.pth.tar" ]]; then
    echo "Refusing to overwrite existing best checkpoint: ${name}" >&2
    exit 1
  fi
  echo "[SVI remaining DetGeo-PE] START ${name}"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" \
    --trogeo_ms_det_variant "$variant" --max_epoch 25 --lr 1e-4 --savename "$name"
  test -f "saved_models/${name}_model_best.pth.tar"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" \
    --trogeo_ms_det_variant "$variant" --pretrain "saved_models/${name}_model_best.pth.tar" \
    --test --savename "${name}_test"
  rm -f "saved_models/${name}_checkpoint.pth.tar"
  echo "[SVI remaining DetGeo-PE] DONE ${name}"
}

# This is deliberately a new savename, so no paused checkpoint can be resumed.
run_one trogeo_ms_e4_h2_ind_bi_nocsfi_detgeope_distance_naturalrng_fresh_swin_t_svi_seed2024 \
  h2_ind_bi_nocsfi
run_one trogeo_ms_e4_h2_ind_detgeo2s_detgeope_distance_naturalrng_fresh_swin_t_svi_seed2024 \
  h2_ind_detgeo2s
echo '[SVI remaining DetGeo-PE] BOTH COMPLETE'
