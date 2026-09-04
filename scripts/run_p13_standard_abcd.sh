#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_p13_standard_a_square.sh
bash scripts/run_p13_standard_b_sam_refined.sh
bash scripts/run_p13_standard_c_hisym_pae.sh
bash scripts/run_p13_standard_d_full.sh
