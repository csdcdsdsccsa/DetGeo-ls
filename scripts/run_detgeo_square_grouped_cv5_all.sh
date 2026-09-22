#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PYTHONPATH=. "$PYTHON_BIN" tools/check_cvogl_grouped_cv5.py --data_root data
bash scripts/run_detgeo_square_grouped_cv5_drone.sh
bash scripts/run_detgeo_square_grouped_cv5_svi.sh
echo 'DetGeo Square Natural-RNG Grouped CV5 COMPLETE'
