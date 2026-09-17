#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_e4_h2ind_amhcsfi_res_bi_hisymgpe_naturalrng.sh
bash scripts/run_e4_h2ind_amhcsfi_res_bi_dgrpe_v2_naturalrng.sh
bash scripts/run_e4_h2ind_amhcsfi_res_bi_hisym_agpe_naturalrng.sh
