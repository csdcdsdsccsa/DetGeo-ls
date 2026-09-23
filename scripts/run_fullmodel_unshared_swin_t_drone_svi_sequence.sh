#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/run_fullmodel_unshared_swin_t_drone_naturalrng.sh
bash scripts/run_fullmodel_unshared_swin_t_svi_naturalrng.sh
