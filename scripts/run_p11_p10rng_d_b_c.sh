#!/usr/bin/env bash
# Sequential P11 ablations: D (full) -> B (SAM P0 refinement) -> C (RGB-P only).
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_p11_p10rng_full.sh
bash scripts/run_p11_p10rng_sam_refined_pe.sh
bash scripts/run_p11_p10rng_rgbp_interaction.sh
