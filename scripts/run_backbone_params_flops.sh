#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_CSV="${OUTPUT_CSV:-results/backbone_params_flops_no_weights.csv}"
OUTPUT_MD="${OUTPUT_MD:-results/backbone_params_flops_no_weights.md}"
OUTPUT_LOG="${OUTPUT_LOG:-logs/backbone_params_flops_no_weights.log}"

"$PYTHON_BIN" -c 'import fvcore' >/dev/null 2>&1 || {
  echo "fvcore is not installed in $PYTHON_BIN" >&2
  exit 1
}

mkdir -p results logs
PYTHONPATH=. "$PYTHON_BIN" tools/profile_backbone_params_flops.py \
  --device cpu --output_csv "$OUTPUT_CSV" --output_md "$OUTPUT_MD" \
  2>&1 | tee "$OUTPUT_LOG"
