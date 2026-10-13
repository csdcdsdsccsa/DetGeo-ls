#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/fullmodel_svi_seed2024}"
"$PYTHON_BIN" train.py train --dataset CVOGL_SVI --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR"
"$PYTHON_BIN" train.py test --dataset CVOGL_SVI --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" --checkpoint "$OUTPUT_DIR/best.pth.tar"
