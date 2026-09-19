#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
DATA_DIR="data/CVOGL_SVI"
for SPLIT in train val test; do
  test -f "${DATA_DIR}/CVOGL_SVI_${SPLIT}.pth" || {
    echo "Missing CVOGL_SVI ${SPLIT} split" >&2
    exit 1
  }
done

echo "[1/3] SVI Bi-Res + DetGeo-PE + distance"
bash scripts/run_e4_h2ind_amhcsfi_res_bi_detgeope_svi_naturalrng.sh

echo "[2/3] SVI Bi-NoCSFI + DetGeo-PE + distance"
bash scripts/run_e4_h2ind_bi_nocsfi_detgeope_svi_naturalrng.sh

echo "[3/3] SVI Two-scale DetGeo + DetGeo-PE + distance"
bash scripts/run_e4_h2ind_detgeo2s_detgeope_svi_naturalrng.sh

echo "SVI THREE DetGeo-PE EXPERIMENTS COMPLETE"
