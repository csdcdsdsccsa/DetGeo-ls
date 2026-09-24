#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PYTHONPATH=. "$PYTHON_BIN" tools/check_fullmodel_gaussian_scale_configs.py
bash scripts/run_fullmodel_gaussian_scale_drone.sh
bash scripts/run_fullmodel_gaussian_scale_svi.sh
echo 'ALL 10 GAUSSIAN-SCALE EXPERIMENTS COMPLETE'
