#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_standard_gaussian_field_msg.sh
bash scripts/run_standard_gaussian_field_msg_ag.sh
bash scripts/run_standard_gaussian_field_full.sh
