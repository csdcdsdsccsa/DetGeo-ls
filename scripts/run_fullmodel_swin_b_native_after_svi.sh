#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# The preceding projected Swin-B SVI job was launched by the existing four-run
# sequence.  Wait for both its training and validation-best test process to
# exit before allocating the same two GPUs to the Native pair.
PRECEDING='fullmodel_bires_amhcsfi_res_hisym_crgpe_core25x50_outer50x100_swin_b_svi_naturalrng_seed2024'
while pgrep -f "[t]rain.py.*${PRECEDING}" >/dev/null; do
  sleep 20
done

bash scripts/run_fullmodel_hisym_crgpe_swin_b_native_drone_naturalrng.sh
bash scripts/run_fullmodel_hisym_crgpe_swin_b_native_svi_naturalrng.sh
