#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
RESULT_DIR="results/gaussian_scale"
mkdir -p "$RESULT_DIR"
for dataset in drone svi; do for scale in 15 20 25 30 35; do test -f "saved_models/fullmodel_gausscale_${dataset}_g${scale}_seed2024_model_best.pth.tar"; done; done
evaluate_drone() {
  local core="$1" outer="$2" name="fullmodel_gausscale_drone_g${1}_seed2024"
  PYTHONPATH=. "$PYTHON_BIN" train.py --gpu "$GPU" --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian --gaussian_bank 15,20,25,30,35 --gaussian_sigma "$core" --crgpe_outer_sigma "$outer" --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024 --beta 1.0 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test" 2>&1 | tee "${RESULT_DIR}/${name}_test.log"
}
evaluate_svi() {
  local cy="$1" cx="$2" oy="$3" ox="$4" name="fullmodel_gausscale_svi_g${1}_seed2024"
  PYTHONPATH=. "$PYTHON_BIN" train.py --gpu "$GPU" --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 --data_root data --data_name CVOGL_SVI --trogeo_backbone swin_t --trogeo_ms_det_variant h2_ind_amhcsfi_res_bi --trogeo_position_mode hisym_crgpe --trogeo_click_map_mode gaussian --gaussian_bank 15,20,25,30,35 --gaussian_sigma "$cy" --gaussian_sigma_x "$cx" --crgpe_outer_sigma "$oy" --crgpe_outer_sigma_x "$ox" --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024 --beta 1.0 --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 --pretrain "saved_models/${name}_model_best.pth.tar" --test --savename "${name}_test" 2>&1 | tee "${RESULT_DIR}/${name}_test.log"
}
evaluate_drone 15 30; evaluate_drone 20 40; evaluate_drone 25 50; evaluate_drone 30 60; evaluate_drone 35 70
evaluate_svi 15 30 30 60; evaluate_svi 20 40 40 80; evaluate_svi 25 50 50 100; evaluate_svi 30 60 60 120; evaluate_svi 35 70 70 140
PYTHONPATH=. "$PYTHON_BIN" tools/summarize_fullmodel_gaussian_scale.py --log_dir "$RESULT_DIR" --output_dir results
