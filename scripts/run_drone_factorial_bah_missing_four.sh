#!/usr/bin/env bash
# DroneAerial full-factorial B/A/H ablations: 010 -> 001 -> 011 -> 101.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024
  --beta 1.0 --print_freq 50 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

preflight() {
  PYTHONPATH=. "$PYTHON_BIN" tools/test_factorial_ablation_variants.py
  PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_detgeo2s.py --data_name CVOGL_DroneAerial
  PYTHONPATH=. "$PYTHON_BIN" tools/test_hisym_core_ring_gpe.py
  PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_cross_scale_hier.py --batch_size 7
}

run_one() {
  local code="$1" variant="$2" position="$3" click_map="$4"
  local name="trogeo_ms_e4_factorial_${code}_naturalrng_swin_t_drone_seed2024"
  local -a args=("${COMMON[@]}" --trogeo_ms_det_variant "$variant"
                 --trogeo_position_mode "$position" --trogeo_click_map_mode "$click_map")
  if [[ "$position" == "hisym_crgpe" ]]; then
    args+=(--gaussian_sigma 25 --crgpe_outer_sigma 50)
  fi
  if [[ -e "saved_models/${name}_model_best.pth.tar" ]]; then
    echo "Refusing to overwrite existing best checkpoint: ${name}" >&2
    exit 1
  fi
  echo "[Drone factorial] START ${code}: ${name}"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${args[@]}" --max_epoch 25 --lr 1e-4 --savename "$name"
  test -f "saved_models/${name}_model_best.pth.tar"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${args[@]}" \
    --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test"
  rm -f "saved_models/${name}_checkpoint.pth.tar"
  echo "[Drone factorial] DONE ${code}: ${name}"
}

preflight
run_one 010 h2_ind_detgeo2s_amhcsfi_res current distance
run_one 001 h2_ind_detgeo2s hisym_crgpe gaussian
run_one 011 h2_ind_detgeo2s_amhcsfi_res hisym_crgpe gaussian
run_one 101 h2_ind_bi_nocsfi hisym_crgpe gaussian
echo '[Drone factorial] ALL FOUR COMPLETE'

# The original SVI three-control queue was intentionally stopped after its
# completed Bi-Res row.  These are fresh runs of only its remaining two rows.
bash scripts/run_e4_svi_remaining_detgeope_naturalrng.sh
