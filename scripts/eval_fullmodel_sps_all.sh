#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/eval_fullmodel_sps_drone.sh
bash scripts/eval_fullmodel_sps_svi.sh
