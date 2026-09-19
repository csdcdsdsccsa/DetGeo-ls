#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_corr1s_s4_drone_naturalrng.sh
bash scripts/run_corr1s_s4_svi_naturalrng.sh
