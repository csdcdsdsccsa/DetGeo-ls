# DetGeo experiment log

This file is the experiment memory shared through GitHub. Results below were produced before this directory was initialized as a Git repository, so they are recorded as a verified pre-history snapshot rather than being assigned invented commit hashes.

## Current research state

- Task: rerank DetGeo NMS Top-5 proposals.
- Dataset: `CVOGL_DroneAerial`.
- Detector checkpoint: `saved_models/model_droneaerial_bs8_model_best.pth.tar`.
- Detector: frozen during ranker training.
- Candidate configuration: `K=5`, NMS IoU `0.5`, RoI size `7`.
- Selection threshold of primary interest: IoU `0.5`.
- Test split: not used in the ranking experiments below.
- Current conclusion: the Top-5 pool has substantial oracle headroom; conservative residual reranking works, while independent reranking is destructive. Lightweight Cross-Attention increases both rescues and degradations and has not yet improved the best validation accuracy.
- OA-CFRGeo branch status: the first frozen-backbone anchor-free comparison is complete. A 64-only head reached 51.35% and a 32->64->128 coarse-to-fine head reached 50.49% Acc@0.5, both below the 56.01% DetGeo baseline.

## E00 — NMS Top-5 coverage

**Goal:** Establish whether reranking has useful candidates to select.

**Configuration**

- Split reported here: validation, 923 samples
- `K=5`
- NMS IoU: `0.5`

**Validation results**

| Selection | Acc@0.25 | Acc@0.5 |
|---|---:|---:|
| Original DetGeo Top-1 | 60.78% | 56.01% |
| Top-5 oracle | 87.76% | 82.99% |

**Conclusion:** The candidate pool is valuable. At IoU 0.5, the oracle headroom is 26.98 percentage points, so the main problem is candidate selection rather than proposal coverage.

## E01 — Independent GAP + MLP ranker

**Goal:** Test whether a small independent MLP can select the best Top-5 proposal.

**Configuration**

- Epochs: `3`
- Features: pooled query feature + pooled candidate feature + detector confidence
- Score: standalone MLP score, allowed to replace the detector ordering
- Target: candidate with maximum IoU

**Best validation result (epoch 3)**

| Acc@0.25 | Acc@0.5 | Rescue@0.5 | Degradation@0.5 | Net gain |
|---:|---:|---:|---:|---:|
| 56.45% | 47.24% | 73 | 154 | -81 |

**Conclusion:** Failed. The ranker learned to swap candidates but destroyed many already-correct detector Top-1 predictions. This motivated residual scoring and conservative labels.

## E02 — Raw-logit residual MLP with conservative labels

**Goal:** Preserve correct DetGeo predictions and learn only a score correction.

**Method**

```text
final_score = detector_logit + delta_score
```

- The residual head's last layer is zero-initialized, so epoch 0 exactly equals DetGeo.
- If original Top-1 IoU is above 0.5, target candidate 1.
- Otherwise, if Top-5 contains a candidate above 0.5, target the best-IoU candidate.
- If no candidate reaches 0.5, preserve candidate 1.

**Best validation result (epoch 2)**

| Acc@0.25 | Acc@0.5 | Rescue@0.5 | Degradation@0.5 | Net gain |
|---:|---:|---:|---:|---:|
| 61.32% | 56.77% | 11 | 4 | +7 |

**Conclusion:** The conservative residual formulation works and sharply reduces destructive swaps.

## E03 — Z-score residual MLP

**Goal:** Test score normalization before residual correction.

**Configuration**

- Residual alpha: `1.0`
- Detector Top-5 logits normalized by per-sample z-score
- Conservative labels retained

**Best validation result (epoch 1)**

| Acc@0.25 | Acc@0.5 | Rescue@0.5 | Degradation@0.5 | Net gain |
|---:|---:|---:|---:|---:|
| 61.21% | **56.88%** | 18 | 10 | +8 |

**Conclusion:** This is the best recorded validation Acc@0.5 so far, although it is more aggressive than the raw-logit residual MLP.

## E04 — Raw-logit residual Cross-Attention

**Date:** 2026-08-31

**Goal:** Replace GAP + MLP matching with lightweight spatial Query-to-Candidate Cross-Attention while retaining the E02 residual and conservative-label framework.

**Code changes**

- `model/ranker.py`: added `ResidualCrossAttentionRanker`.
- `train_ranker.py`: added `residual-cross-attention` score mode and attention arguments.

**Architecture**

```text
query spatial tokens -> Q
candidate RoI tokens -> K, V
softmax(QK^T / sqrt(d))V -> mean pool -> small MLP -> delta_score
final_score = detector_logit + delta_score
```

**Configuration**

- Epochs: `3`
- Batch size: `8`
- Learning rate: `1e-4`
- Match dimension: `128`
- Attention heads: `4`
- Attention hidden dimension: `64`
- Residual alpha: `1.0`
- Trainable ranker parameters: `205,185`
- Target mode: conservative at IoU `0.5`

**Command**

```bash
/root/miniconda3/envs/detgeo/bin/python train_ranker.py \
  --gpu 0 \
  --checkpoint saved_models/model_droneaerial_bs8_model_best.pth.tar \
  --data-root data \
  --data-name CVOGL_DroneAerial \
  --epochs 3 \
  --batch-size 8 \
  --num-workers 8 \
  --candidate-count 5 \
  --nms-iou 0.5 \
  --roi-size 7 \
  --lr 1e-4 \
  --score-mode residual-cross-attention \
  --target-mode conservative \
  --target-iou 0.5 \
  --residual-alpha 1.0 \
  --match-dim 128 \
  --attention-heads 4 \
  --attention-hidden-dim 64 \
  --output-dir outputs/ranking_round4/residual_cross_attention_raw_conservative_top5_3e
```

**Validation results**

| Epoch | Acc@0.25 | Acc@0.5 | Rescue@0.5 | Degradation@0.5 | Net gain |
|---:|---:|---:|---:|---:|---:|
| 0 / DetGeo | 60.78% | 56.01% | 0 | 0 | 0 |
| **1** | **61.32%** | **56.77%** | **15** | **8** | **+7** |
| 2 | 60.67% | 55.80% | 15 | 17 | -2 |
| 3 | 59.91% | 55.04% | 17 | 26 | -9 |

**Artifacts on the experiment server**

```text
outputs/ranking_round4/residual_cross_attention_raw_conservative_top5_3e/ranker_model_best.pth.tar
logs/ranking_round4_residual_cross_attention_raw_conservative_top5_3e.log
```

These generated artifacts are intentionally excluded from Git.

**Conclusion:** Cross-Attention changed the selection behavior but did not improve net accuracy over E02. Compared with E02 it produced four additional rescues and four additional degradations, leaving the same net gain. Validation degradation after epoch 1 indicates rapid overfitting. E03 remains the best recorded Acc@0.5 at 56.88%.

## Template for the next experiment

```markdown
## EXX — Short name

**Date:** YYYY-MM-DD

**Git base:** commit or tag

**Goal:**

**Changed files:**

**Dataset/splits:**

**Checkpoint:**

**Configuration:**

**Command:**

**Results:** Acc@0.25, Acc@0.5, Rescue, Degradation, net gain

**Test split used:** yes/no

**Conclusion:** keep/reject/next question
```

## E05 — Frozen DetGeo anchor-free coarse-to-fine search

**Date:** 2026-09-01

**Git base:** `2381256`

**Goal:** Run the first minimal experiment from the OA-CFRGeo proposal before adding SAM, structured tokens, recurrent evidence updates, confidence gates, or hard-negative learning. Compare a single 64x64 feature-map anchor-free detector (E05-B) against 32->64->128 coarse-to-fine search with the same anchor-free head (E05-C).

**Changed files**

- `model/oacfr_modules.py`: cosine search, coarse-to-fine prior propagation, and anchor-free center/LTRB/quality head.
- `model/oacfr_geo.py`: frozen DetGeo feature extractor and E05 model variants.
- `model/oacfr_loss.py`: spatial search, center heatmap, GIoU/LTRB, and quality losses.
- `utils/oacfr_utils.py`: anchor-free decoding and localization metrics.
- `train_oacfr.py`: deterministic train/validation entry point and checkpoint selection.
- `scripts/run_train_oacfr_e05_droneaerial.sh`: standalone E05-C command.
- `scripts/run_compare_oacfr_e05_droneaerial.sh`: sequential E05-B/E05-C comparison.

**Dataset and splits**

- Dataset: `CVOGL_DroneAerial`
- Train: 4,343 samples
- Validation: 923 samples
- Model selection: best validation `Acc@0.5`
- Test split used: **no**

**Checkpoint:** `saved_models/model_droneaerial_bs8_model_best.pth.tar`

- DetGeo checkpoint tensors loaded by matching name and shape: `576/584`
- Frozen DetGeo parameters: `73,652,015`
- E05-B trainable parameters: `2,627,590`
- E05-C trainable parameters: `2,956,296`

The existing Darknet implementation already exposes the 32x32, 64x64, and 128x128 feature maps. The original DetGeo path uses its 64x64 map, so the recorded 56.01% baseline remains the structural E05-A reference. E05-B and E05-C replace the detector head; they are detector experiments, not Top-5 rerankers, so Rescue/Degradation are not applicable.

**Common configuration**

- Frozen DetGeo query and reference backbones
- Query representation: original GAP global vector
- Match dimension: `256`
- Search temperature: `0.07`
- Anchor-free outputs: center heatmap + LTRB distances + quality
- Epochs: `3`
- Batch size: `4`
- Optimizer: AdamW
- Learning rate / weight decay: `1e-4` / `1e-4`
- Seed: `13`
- Image size: `1024`

**Exact command**

```bash
PYTHON_BIN=/root/miniconda3/envs/detgeo/bin/python \
  bash scripts/run_compare_oacfr_e05_droneaerial.sh
```

The comparison script executes these scale settings in order:

```text
E05-B: --search-scales 64
E05-C: --search-scales 32,64,128
```

**Validation results**

| Variant | Epoch | Acc@0.25 | Acc@0.5 | Mean IoU | Center accuracy |
|---|---:|---:|---:|---:|---:|
| Original DetGeo | 0 | 60.78% | **56.01%** | — | — |
| E05-B 64-only | 1 | 56.99% | 49.51% | 40.24% | 24.38% |
| E05-B 64-only | 2 | 56.23% | 49.84% | 39.95% | 23.51% |
| E05-B 64-only | **3** | **56.99%** | **51.35%** | **41.83%** | 23.62% |
| E05-C 32->64->128 | 1 | 53.30% | 45.18% | 35.81% | 10.40% |
| E05-C 32->64->128 | 2 | 54.71% | 47.35% | 36.86% | 9.43% |
| E05-C 32->64->128 | **3** | **56.45%** | **50.49%** | **39.47%** | 12.46% |

Best-result comparison at IoU 0.5:

- E05-B vs DetGeo: `51.35 - 56.01 = -4.66` percentage points.
- E05-C vs DetGeo: `50.49 - 56.01 = -5.52` percentage points.
- E05-C vs E05-B: `50.49 - 51.35 = -0.87` percentage points (8 fewer correct validation samples).

**Artifacts on the experiment server**

```text
outputs/oacfr_e05/64_only_3e/{config.json,history.json,model_best.pth.tar}
outputs/oacfr_e05/coarse_to_fine_3e/{config.json,history.json,model_best.pth.tar}
logs/oacfr_e05_64_only_3e.log
logs/oacfr_e05_coarse_to_fine_3e.log
```

Generated logs, outputs, and checkpoints remain excluded from Git.

**Conclusion:** Reject this E05 coarse-to-fine formulation as an improvement over DetGeo. The 64-only anchor-free replacement does not reproduce the original detector, and adding 32->64->128 propagation loses another 0.87 percentage points. Do not add E06-E09 complexity on top of this result yet. The next controlled question should first recover the baseline with a truly head-preserving multi-scale adapter or residual detector head, then test whether coarse-to-fine features add value without discarding DetGeo's trained 9-anchor head.

## E05-D — Anchor-free checkpoint decode diagnostic

**Date:** 2026-09-01

**Git base:** `75607c8` (E05 checkpoint); diagnostic code added in this commit.

**Goal:** Diagnose the completed E05-B/E05-C checkpoints without training or changing their weights. Separate the effect of default `Center x Quality` ranking, center-only ranking, and LTRB box decoding at the ground-truth center grid cell.

**Changed files**

- `scripts/analyze_oacfr_e05_decode.py`: reproducible validation-only decoder diagnostic, including a repository-root import fix for uninstalled checkouts.
- `EXPERIMENTS.md`: this completed diagnostic record.

**Dataset and split**

- Dataset: `CVOGL_DroneAerial`
- Split: validation, 923 samples
- Test split used: **no**

**Checkpoints**

- E05-B: `outputs/oacfr_e05/64_only_3e/model_best.pth.tar`
- E05-C: `outputs/oacfr_e05/coarse_to_fine_3e/model_best.pth.tar`

**Exact commands**

```bash
PYTHONPATH=. /root/miniconda3/envs/detgeo/bin/python \
  scripts/analyze_oacfr_e05_decode.py \
  --gpu 0 --data-root data --data-name CVOGL_DroneAerial \
  --checkpoint outputs/oacfr_e05/64_only_3e/model_best.pth.tar \
  --output-dir outputs/oacfr_e05/decode_diag_64_only_best_v3 \
  --search-scales 64

PYTHONPATH=. /root/miniconda3/envs/detgeo/bin/python \
  scripts/analyze_oacfr_e05_decode.py \
  --gpu 0 --data-root data --data-name CVOGL_DroneAerial \
  --checkpoint outputs/oacfr_e05/coarse_to_fine_3e/model_best.pth.tar \
  --output-dir outputs/oacfr_e05/decode_diag_coarse_to_fine_best_v3 \
  --search-scales 32,64,128
```

`oracle_center` is a diagnostic only: it selects the target center grid cell but still uses the model's predicted LTRB distances. It is not a deployable result.

| Variant | Decode mode | Acc@0.25 | Acc@0.5 | Mean IoU | Exact center | Mean center error |
|---|---|---:|---:|---:|---:|---:|
| E05-B 64-only | Center x Quality | 56.99% | 51.35% | 41.83% | 23.62% | 174.7 px |
| E05-B 64-only | Center only | 57.42% | 51.35% | 42.05% | 23.62% | 171.0 px |
| E05-B 64-only | Oracle center | 97.83% | **88.52%** | 71.74% | 100.00% | 6.1 px |
| E05-C 32->64->128 | Center x Quality | 56.45% | 50.49% | 39.47% | 12.46% | 172.8 px |
| E05-C 32->64->128 | Center only | 56.12% | 49.95% | 39.05% | 11.48% | 175.1 px |
| E05-C 32->64->128 | Oracle center | 96.32% | **84.07%** | 68.13% | 100.00% | 3.1 px |

Additional center tolerance evidence: E05-B `Center x Quality` is only 20.48% / 41.28% / 52.55% within 8 / 16 / 32 pixels; E05-C is 26.54% / 41.60% / 52.87%. Selected Quality is poorly calibrated against final IoU (mean absolute error 0.365 for E05-B and 0.360 for E05-C), but removing it does not improve E05-B Acc@0.5 and reduces E05-C by five samples. It is not the primary failure mode.

**Artifacts on the experiment server**

```text
outputs/oacfr_e05/decode_diag_64_only_best_v3/{config.json,summary.json,per_sample.jsonl}
outputs/oacfr_e05/decode_diag_coarse_to_fine_best_v3/{config.json,summary.json,per_sample.jsonl}
logs/e05_decode_64_only_v3.log
logs/e05_decode_coarse_to_fine_v3.log
```

Generated artifacts remain excluded from Git.

**Conclusion:** The E05 anchor-free LTRB regression is strong at the correct center, while center selection is the dominant failure: correcting only the center changes E05-B from 474/923 to 817/923 validation hits at IoU > 0.5, and E05-C from 466/923 to 776/923. The 32->64->128 formulation is worse than 64-only both in deployable accuracy and oracle-center box quality. Do not spend a 20-epoch budget on E05. E06 should preserve the trained DetGeo 9-anchor detection head exactly at initialization, add a zero-initialized residual multi-scale adapter to its 64x64 feature, and require epoch-0 per-sample decode equality with the frozen DetGeo baseline before training.

## E06 — Baseline-preserving parallel multi-scale residual adapter

**Date:** 2026-09-01

**Git base:** `0a6ffbd`

**Goal:** Test the first head-preserving E06 formulation indicated by E05-D. Keep the trained DetGeo 64x64, 9-anchor detector head frozen; add only a parallel 32/64/128 reference-feature residual adapter. The experiment is invalid unless epoch-0 logits exactly equal the original DetGeo for every validation sample.

**Changed files**

- `model/e06_geo.py`: frozen DetGeo-compatible baseline path plus 32->64 and 128->64 projected query-aware contexts, fused through `F_final = F_base + alpha * DeltaF` with `alpha=0` at initialization.
- `train_e06.py`: deterministic E06 train/validation entry point, checkpoint loading, full-validation zero-init equality guard, and artifact logging.
- `EXPERIMENTS.md`: this completed experiment record.

**Dataset and split**

- Dataset: `CVOGL_DroneAerial`
- Train / validation: 4,343 / 923 samples
- Model selection: best validation `Acc@0.5`
- Test split used: **no**

**Checkpoint and initialization**

- Base checkpoint: `saved_models/model_droneaerial_bs8_model_best.pth.tar`
- Loaded DetGeo tensors: `584/584`
- Frozen components: query backbone, reference Darknet, click preprocessing, original 64x64 mappings, cross-view path, and original 9-anchor `fcn_out`.
- Trainable E06 parameters: `10,359,297`
- Zero-init guard: `923/923` validation samples had bitwise-identical 45-channel logits; maximum absolute difference: `0.0`.

**Configuration and command**

- Parallel residual inputs: query-conditioned 32x32 and 128x128 reference contexts, resized to 64x64 and concatenated with the original 64x64 fused feature.
- Residual scale `alpha`: learnable, initialized to `0`.
- Epochs / batch size: `3` / `4`
- Optimizer: AdamW, learning rate / weight decay: `1e-4` / `1e-4`
- Seed / image size: `13` / `1024`

```bash
PYTHONPATH=. /root/miniconda3/envs/detgeo/bin/python train_e06.py \
  --gpu 0 --data-root data --data-name CVOGL_DroneAerial \
  --checkpoint saved_models/model_droneaerial_bs8_model_best.pth.tar \
  --output-dir outputs/e06_parallel_multiscale_3e_v1 \
  --epochs 3 --batch-size 4 --num-workers 8 --lr 1e-4 --seed 13
```

**Validation results**

| Variant | Epoch | Acc@0.25 | Acc@0.5 | Mean IoU | Center accuracy | Residual alpha |
|---|---:|---:|---:|---:|---:|---:|
| Original DetGeo | 0 | **60.78%** | **56.01%** | — | — | — |
| E06 parallel residual | 1 | 60.35% | 55.15% | 44.26% | 25.14% | -0.002374 |
| E06 parallel residual | **2** | **60.46%** | **55.58%** | **44.54%** | **25.57%** | -0.002631 |
| E06 parallel residual | 3 | 60.13% | 55.58% | 44.30% | 25.14% | -0.001930 |

Best-result comparison at IoU 0.5: `55.58 - 56.01 = -0.43` percentage points, or four fewer correct validation samples (`513/923` vs. the recorded DetGeo baseline).

**Artifacts on the experiment server**

```text
outputs/e06_parallel_multiscale_3e_v1/{config.json,zero_init.json,zero_init_per_sample.jsonl,history.json,model_best.pth.tar}
logs/e06_parallel_multiscale_3e_v1.log
```

Generated logs, outputs, and checkpoints remain excluded from Git.

**Conclusion:** The baseline-preserving mechanism works exactly and avoids E05's destructive head replacement, but this first parallel multi-scale adapter does not exceed DetGeo in three epochs. Do not claim a gain or extend this configuration to 20 epochs. The next controlled question, if pursued, is a sequential coarse-to-fine residual adapter using the same frozen head, zero-init equality guard, checkpoint, split, budget, and metrics; it must be compared directly against this E06 parallel result.
