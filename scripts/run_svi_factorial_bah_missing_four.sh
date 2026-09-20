#!/usr/bin/env bash
# CVOGL_SVI full-factorial B/A/H ablations: 010 -> 001 -> 011 -> 101.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_SVI --trogeo_backbone swin_t
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024
  --beta 1.0 --print_freq 50
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0
)

preflight() {
  echo "[SVI factorial] Running structural checks..."
  PYTHONPATH=. "$PYTHON_BIN" tools/test_factorial_ablation_variants.py
  PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_detgeo2s.py --data_name CVOGL_SVI
  PYTHONPATH=. "$PYTHON_BIN" tools/test_trogeo_click_maps.py
  PYTHONPATH=. "$PYTHON_BIN" tools/test_hisym_core_ring_gpe.py
  PYTHONPATH=. "$PYTHON_BIN" tools/test_e4_cross_scale_hier.py --batch_size 7
  echo "[SVI factorial] Preflight PASS"
}

run_one() {
  local code="$1" variant="$2" position="$3" click_map="$4" tag="$5"
  local name="trogeo_ms_e4_svi_factorial_${code}_${tag}_naturalrng_swin_t_seed2024"
  local -a args=("${COMMON[@]}" --trogeo_ms_det_variant "$variant" \
    --trogeo_position_mode "$position" --trogeo_click_map_mode "$click_map")

  # SVI anisotropic geometry: core=(sigma_y,sigma_x)=(25,50), outer=(50,100).
  if [[ "$position" == "hisym_crgpe" ]]; then
    args+=(--gaussian_sigma 25 --gaussian_sigma_x 50 \
      --crgpe_outer_sigma 50 --crgpe_outer_sigma_x 100)
  fi
  if [[ -e "saved_models/${name}_model_best.pth.tar" ]]; then
    echo "Refusing to overwrite existing best checkpoint: ${name}" >&2
    exit 1
  fi

  echo "[SVI factorial] START ${code}: variant=${variant}, position=${position}, map=${click_map}"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${args[@]}" --max_epoch 25 --lr 1e-4 --savename "$name"
  test -f "saved_models/${name}_model_best.pth.tar"
  echo "[SVI factorial] TEST ${code} using validation-best checkpoint"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${args[@]}" \
    --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test"
  rm -f "saved_models/${name}_checkpoint.pth.tar"
  echo "[SVI factorial] DONE ${code}: ${name}"
}

preflight
run_one 010 h2_ind_detgeo2s_amhcsfi_res current distance amhcsfi_res_current_distance
run_one 001 h2_ind_detgeo2s hisym_crgpe gaussian hisym_crgpe_core25x50_outer50x100
run_one 011 h2_ind_detgeo2s_amhcsfi_res hisym_crgpe gaussian amhcsfi_res_hisym_crgpe_core25x50_outer50x100
run_one 101 h2_ind_bi_nocsfi hisym_crgpe gaussian bi_hisym_crgpe_core25x50_outer50x100
echo "[SVI factorial] ALL FOUR COMPLETE"
