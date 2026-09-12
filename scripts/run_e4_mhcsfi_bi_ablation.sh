#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash run_e4_h2ind_amhcsfi_bi.sh
bash run_e4_h2ind_mhcsfi_bi.sh
bash run_e4_h2ind_csfi_bi_rccd.sh
bash run_e4_h2ind_amhcsfi_res_bi.sh
bash run_e4_amhcsfi_res_guidance_ablation.sh
