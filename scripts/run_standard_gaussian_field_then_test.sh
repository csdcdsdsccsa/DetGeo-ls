#!/usr/bin/env bash
# Start only after run_standard_gaussian_field_all.sh has exited successfully.
set -euo pipefail
cd "$(dirname "$0")/.."

while pgrep -f '[r]un_standard_gaussian_field_all.sh' >/dev/null; do
  sleep 60
done

bash scripts/eval_standard_gaussian_field_test.sh
