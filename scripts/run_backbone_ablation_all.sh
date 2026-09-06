#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PYTHONPATH=. "$PYTHON_BIN" tools/test_backbone_ablation.py > logs/backbone_ablation_smoke.log 2>&1
bash scripts/run_backbone_darknet53_noshare.sh
bash scripts/run_backbone_darknet53_shared.sh
bash scripts/run_backbone_resnet50_shared.sh
bash scripts/eval_backbone_ablation_test.sh
