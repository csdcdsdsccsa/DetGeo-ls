#!/usr/bin/env bash
# Preserve the already-running P1 rerun; launch MH-CSFI only after it exits.
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f '[t]rain.py.*h2_ind_pqra' >/dev/null; do
  sleep 60
done
bash scripts/run_e4_mhcsfi_bi_ablation.sh
