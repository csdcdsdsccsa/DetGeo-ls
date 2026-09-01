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
