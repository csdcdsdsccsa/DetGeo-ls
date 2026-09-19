#!/usr/bin/env bash
# One-epoch structural/loss smoke runs for the four missing DroneAerial cells.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024
  --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

PYTHONPATH=. "$PYTHON_BIN" tools/test_factorial_ablation_variants.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_detgeo2s.py --data_name CVOGL_DroneAerial
PYTHONPATH=. "$PYTHON_BIN" tools/test_hisym_core_ring_gpe.py
PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_cross_scale_hier.py --batch_size 7

smoke_one() {
  local code="$1" variant="$2" position="$3" click_map="$4"
  local name="smoke_drone_factorial_${code}_naturalrng_swin_t_seed2024"
  local -a args=("${COMMON[@]}" --trogeo_ms_det_variant "$variant"
                 --trogeo_position_mode "$position" --trogeo_click_map_mode "$click_map")
  if [[ "$position" == "hisym_crgpe" ]]; then
    args+=(--gaussian_sigma 25 --crgpe_outer_sigma 50)
  fi
  if [[ -e "saved_models/${name}_checkpoint.pth.tar" || -e "saved_models/${name}_model_best.pth.tar" ]]; then
    echo "Refusing to overwrite existing smoke artifact: ${name}" >&2
    exit 1
  fi
  echo "[Drone factorial smoke] START ${code}"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${args[@]}" --max_epoch 1 --lr 1e-4 --savename "$name"
  test -f "saved_models/${name}_model_best.pth.tar"
  # Smoke artifacts are intentionally discarded: every formal run starts fresh.
  rm -f "saved_models/${name}_checkpoint.pth.tar" "saved_models/${name}_model_best.pth.tar"
  echo "[Drone factorial smoke] PASS ${code}"
}

smoke_one 010 h2_ind_detgeo2s_amhcsfi_res current distance
smoke_one 001 h2_ind_detgeo2s hisym_crgpe gaussian
smoke_one 011 h2_ind_detgeo2s_amhcsfi_res hisym_crgpe gaussian
smoke_one 101 h2_ind_bi_nocsfi hisym_crgpe gaussian
echo '[Drone factorial smoke] ALL FOUR PASSED'
