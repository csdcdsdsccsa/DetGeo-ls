#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash run_e4_h2ind_csfi_bi_detgeope.sh
bash run_e4_h2ind_csfi_bi_detgeope_g25.sh
