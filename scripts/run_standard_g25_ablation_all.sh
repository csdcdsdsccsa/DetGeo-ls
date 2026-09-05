#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_standard_g25_ag.sh
bash scripts/run_standard_g25_cc.sh
bash scripts/run_standard_g25_ag_cc.sh
bash scripts/eval_standard_g25_ablation_test.sh
