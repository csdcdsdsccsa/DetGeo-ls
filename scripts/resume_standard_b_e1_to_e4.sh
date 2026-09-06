#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# E0 completed before an E1 dataset-selection bug was fixed; preserve E0 and resume E1 onward.
bash scripts/run_standard_b_e1_msst.sh
bash scripts/run_standard_b_e2_core.sh
bash scripts/run_standard_b_e3_b1.sh
bash scripts/run_standard_b_e4_b2.sh
bash scripts/eval_standard_b_all.sh
