#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_p12_p10rng_hisym_pae_c.sh
bash scripts/run_p12_p10rng_sam_hisym_pae_d.sh
