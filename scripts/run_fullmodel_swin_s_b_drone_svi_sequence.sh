#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# E1 -> E2 -> E3 -> E4. Each child script trains 25 epochs, then tests its
# validation-selected checkpoint exactly once before the following experiment.
bash run_fullmodel_hisym_crgpe_swin_s_naturalrng.sh
bash run_fullmodel_hisym_crgpe_swin_s_svi_naturalrng.sh
bash run_fullmodel_hisym_crgpe_swin_b_drone_naturalrng.sh
bash run_fullmodel_hisym_crgpe_swin_b_svi_naturalrng.sh
