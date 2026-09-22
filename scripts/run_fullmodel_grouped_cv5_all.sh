#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_fullmodel_grouped_cv5_drone.sh
bash scripts/run_fullmodel_grouped_cv5_svi.sh
