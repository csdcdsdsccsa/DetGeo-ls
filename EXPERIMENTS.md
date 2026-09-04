# SAM prompt experiment log

## P01: frozen original YOLO head with SAM PromptFusion

- Base commit: `ff257e8` (upstream original DetGeo baseline).
- Detector: unchanged original 9-anchor YOLO head, original target construction,
  loss, decoding, and validation protocol.
- New input: offline SAM ViT-B mask, Gaussian(click, sigma=25), and their
  product, fused as a zero-initialized residual after the original 4-to-3 click
  projection.
- Initialization gate: `SAM_PROMPT_ZERO_INIT_OK max_abs_diff=0.0` on the remote
  RTX 5090 using `model_droneaerial_bs8_model_best.pth.tar`.
- Trainable parameters: `prompt_fusion` only. All original DetGeo modules,
  including the YOLO head, remain frozen and in evaluation mode.
- Budget: 5 epochs, seed 13, batch size 8, GPU 0. Select only by validation
  Acc@0.50; do not evaluate on test while choosing the design.
- Result: best checkpoint was epoch 1 (zero-based epoch 0), with validation
  Acc@0.50/Acc@0.25/Mean IoU/Center Accuracy =
  56.66%/61.54%/45.26%/25.46%. The matched original checkpoint scored
  56.01%/60.78%/44.76%/25.14%, a +0.65 percentage-point Acc@0.50 gain.
- Decision: validation-only positive signal. Do not use test data or unfreeze
  additional DetGeo modules until this result is replicated with a second seed.

## P02: full fine-tuning, square PE versus SAM-Gaussian PE

- Design: replace, rather than augment, the original square click map. The new
  position map is `Gaussian + PromptFusion(Gaussian, SAM, Gaussian*SAM)` and is
  concatenated with RGB before the unchanged original 4-to-3 projection.
- Controls: same original checkpoint, seed 13, batch size 8, augmentation,
  10-epoch budget, original YOLO head/loss/decode, and validation protocol.
- Learning rates: pretrained DetGeo parameters use `1e-5`; newly initialized
  `prompt_fusion` uses `1e-4`. Test data is excluded.
- Square control result: completed 10 epochs. Best validation
  Acc@0.50/Acc@0.25/Mean IoU/Center Accuracy =
  54.93%/60.13%/44.27%/23.84%.
- SAM-Gaussian result: stopped by request after completing validation for epochs
  0--8 (before epoch 9 completed), preserving the best checkpoint. An
  independent validation-only rerun of that checkpoint scored
  55.90%/61.11%/44.70%/25.57%.
- Comparison: at the selected checkpoint SAM-Gaussian is +0.97 percentage
  points Acc@0.50 over the matched square control. This is a single-seed,
  validation-only early-stopped result; do not use the test set for further
  design selection or claim a final generalization improvement without replication.
- Final test (explicitly authorized after checkpoint selection): the square
  control scored Acc@0.50/Acc@0.25/Mean IoU/Center Accuracy =
  56.42%/60.74%/45.44%/29.39%. The selected SAM-Gaussian checkpoint scored
  58.48%/63.00%/46.55%/29.50%, respectively: +2.06/+2.26/+1.11/+0.11 percentage
  points. Test split SAM masks were generated offline with the same point-prompt
  ViT-B procedure used for train and validation.

## P03: scratch 10-epoch position-encoding comparison

- Both conditions start without a CVOGL_DroneAerial DetGeo checkpoint. Model
  construction retains only the original ResNet18 ImageNet and Darknet53 YOLO
  initializations.
- Controls: seed 13, batch size 8, 10 epochs, base learning rate `1e-4`, and
  unchanged original YOLO head/loss/decode. SAM PromptFusion also uses `1e-4`.
- Outputs are isolated as `scratch10_square_seed13` and
  `scratch10_sam_gaussian_seed13`; no test evaluation is used to choose either
  checkpoint.
- Checkpoint selection: independent validation of the selected Square checkpoint
  scored Acc@0.50/Acc@0.25/Mean IoU/Center Accuracy =
  46.59%/54.06%/37.76%/21.13%; the selected SAM-Gaussian checkpoint scored
  52.76%/59.48%/41.45%/19.72%. Each corresponding selected checkpoint was then
  evaluated exactly once on test after explicit authorization.
- Final test: Square scored Acc@0.50/Acc@0.25/Mean IoU/Center Accuracy =
  48.30%/53.75%/38.15%/24.97%. SAM-Gaussian scored
  56.22%/61.77%/43.88%/26.52%, a +7.92/+8.02/+5.73/+1.55 percentage-point change.

## P04: scratch 25-epoch position-encoding comparison

- P04 repeats P03 from the same backbone-only initialization, extending the
  budget to 25 epochs. It does not load a CVOGL_DroneAerial DetGeo checkpoint.
- Both conditions use seed 13, batch size 8, base learning rate `1e-4`, and the
  unchanged original YOLO head/loss/decode; SAM PromptFusion uses `1e-4`.
- Outputs are isolated as `scratch25_square_seed13` and
  `scratch25_sam_gaussian_seed13`. Select checkpoints exclusively by validation
  before any test evaluation.
- Selected-checkpoint validation: Square scored
  Acc@0.50/Acc@0.25/Mean IoU/Center Accuracy =
  52.22%/58.07%/42.24%/22.54%; SAM-Gaussian scored
  58.50%/63.27%/46.42%/26.44%.
- Final test (one evaluation per validation-selected checkpoint): Square scored
  54.98%/59.51%/43.77%/29.80%; SAM-Gaussian scored
  60.74%/64.95%/48.37%/30.63%, a +5.76/+5.44/+4.60/+0.83 percentage-point change.

## P05: one-epoch worker-count reproduction probe

- Square-only diagnostic with the unchanged non-SAM path, no task checkpoint,
  seed 13, batch size 8, learning rate `1e-4`, and `num_workers=8` to match the
  original 25-epoch run. It runs exactly one epoch and is compared to the
  original epoch-0 log; it is not a new performance result.
- Result: the epoch-0 validation Acc@0.50 was 29.1441%, exactly matching the
  original log. The logged batch-50 and later epoch-0 metrics also matched,
  confirming the non-SAM code path and initialization reproduce when worker
  count is restored to 8. The earlier worker-6 Square trajectory is therefore a
  different augmentation/randomness trajectory, not evidence of a changed
  Square model implementation.

## P06: formal 25-epoch workers-8 comparison

- This is the designated main-table protocol. P04 (workers=6) remains a
  controlled augmentation-randomness diagnostic and is not used as the formal
  Square baseline.
- Both conditions use GPU 0, batch size 8, `num_workers=8`, seed 13, learning
  rate `1e-4`, 25 epochs, beta 1.0, and no CVOGL_DroneAerial DetGeo checkpoint.
  They retain only the original ImageNet ResNet18 and YOLO Darknet initializations.
- Outputs are isolated as `scratch25_worker8_square_seed13` and
  `scratch25_worker8_sam_gaussian_seed13`; select by validation before a single
  final test evaluation of each selected checkpoint.
- Selected-checkpoint validation: Square scored
  Acc@0.50/Acc@0.25/Mean IoU/Center Accuracy =
  56.01%/60.78%/44.76%/25.14%, exactly reproducing the original 25-epoch
  Square result. SAM-Gaussian scored 60.67%/64.90%/47.88%/27.41%, a
  +4.66/+4.12/+3.12/+2.27 percentage-point change.
- Final test (one evaluation per validation-selected checkpoint): Square scored
  56.73%/60.53%/45.23%/28.88%, exactly reproducing the original checkpoint test.
  SAM-Gaussian scored 61.66%/67.52%/49.98%/34.22%, a
  +4.93/+6.99/+4.75/+5.34 percentage-point change.

## P07: 25-epoch workers-24 comparison

- P07 preserves the P06 protocol exactly, changing only `num_workers` from 8 to
  24. Both conditions use GPU 0, batch size 8, seed 13, learning rate `1e-4`, 25
  epochs, beta 1.0, and no CVOGL_DroneAerial DetGeo checkpoint.
- Outputs are isolated as `scratch25_worker24_square_seed13` and
  `scratch25_worker24_sam_gaussian_seed13`; select by validation before any
  final test evaluation.
- Selected-checkpoint validation: Square scored
  Acc@0.50/Acc@0.25/Mean IoU/Center Accuracy =
  55.04%/59.59%/43.87%/26.22%; SAM-Gaussian scored
  60.02%/64.25%/47.21%/27.52%, a +4.98/+4.66/+3.34/+1.30 percentage-point change.
- Final test (one evaluation per validation-selected checkpoint): Square scored
  57.76%/61.87%/45.87%/30.52%; SAM-Gaussian scored
  59.82%/64.95%/48.58%/30.94%, a +2.06/+3.08/+2.71/+0.42 percentage-point change.
- Re-evaluation with `num_workers=16` (batch size 8, the same selected
  checkpoints) reproduced the same four test metrics exactly for both
  conditions. This confirms that the test worker count changes throughput only
  for this deterministic evaluation protocol.

## P08: Gaussian-only positional-encoding ablation

- P08 isolates the contribution of changing DetGeo's original square click map
  to a Gaussian map. It preserves the P07 scratch protocol: GPU 0, batch size
  8, `num_workers=24`, seed 13, learning rate `1e-4`, 25 epochs, beta 1.0, and
  no CVOGL_DroneAerial DetGeo checkpoint.
- `--gaussian_only` uses `GaussianPromptDataset` and `DetGeoGaussian`: it
  generates the Gaussian map from the click point but neither loads SAM masks
  nor creates/uses PromptFusion. The YOLO detection head is unchanged.
- Selected-checkpoint validation (evaluated with batch size 8 and
  `num_workers=16`): Acc@0.50/Acc@0.25/Mean IoU/Center Accuracy =
  59.15%/63.81%/46.68%/27.41%.
- Final test (one evaluation of that validation-selected checkpoint, batch size
  8 and `num_workers=16`): 63.21%/67.42%/50.10%/32.17%.

## P09: RNG-controlled Gaussian-only vs. SAM-Gaussian

- P09 changes no model or dataset behavior. It isolates only the random
  trajectory: separate but identically seeded DataLoader generators control
  train shuffle and worker seeds; `seed_worker` also seeds Python, NumPy,
  PyTorch, OpenCV, and Albumentations augmentation RNGs.
- The global runtime RNG is reset after model/optimizer construction, so SAM's
  additional PromptFusion initialization cannot perturb later training
  randomness. `--rng_probe` logs the first three batch indices, augmented
  bboxes, and satellite-image sums before formal training begins.
- The one-epoch probes matched exactly for both methods on all three logged
  batches: sample indices, augmented bbox heads, and satellite-image sums.
- Selected-checkpoint validation (batch size 8, `num_workers=16`):
  Gaussian-only scored 58.72%/63.38%/46.31%/27.63%; SAM-Gaussian scored
  60.67%/64.68%/48.03%/27.74%, a +1.95/+1.30/+1.72/+0.11 percentage-point change.
- Final test (one evaluation per validation-selected checkpoint, batch size 8,
  `num_workers=16`): Gaussian-only scored 57.25%/62.90%/46.73%/28.37%;
  SAM-Gaussian scored 61.36%/65.47%/49.13%/29.29%, a
  +4.11/+2.57/+2.40/+0.92 percentage-point change.
- `scripts/run_p09_rngctrl_square.sh` applies the exact same P09 seeds,
  workers, batch size, and RNG probe to the unmodified DetGeo square click-map
  baseline. It is the final matched baseline for the Square -> Gaussian -> SAM
  ablation sequence.
- Selected-checkpoint validation (batch size 8, `num_workers=16`): Square
  scored 57.64%/62.51%/45.44%/25.03%; Gaussian-only scored
  58.72%/63.38%/46.31%/27.63%, a +1.08/+0.87/+0.87/+2.60 percentage-point
  change. SAM-Gaussian then added +1.95/+1.30/+1.72/+0.11 points over
  Gaussian-only.
- Final test (one evaluation per validation-selected checkpoint, batch size 8,
  `num_workers=16`): Square scored 56.83%/61.15%/44.91%/27.13%;
  Gaussian-only scored 57.25%/62.90%/46.73%/28.37%, a
  +0.42/+1.75/+1.82/+1.24 percentage-point change. SAM-Gaussian then added
  +4.11/+2.57/+2.40/+0.92 points over Gaussian-only.

## P10: original-RNG-matched three-stage ablation

- P10 preserves the original DetGeo DataLoader behavior: no explicit loader
  generator, worker initializer, or post-construction runtime RNG reset. It
  therefore targets the historical original-DetGeo trajectory rather than the
  new P09 trajectory.
- Only the SAM route isolates the extra PromptFusion initialization: it saves
  the post-DetGeo CPU torch RNG state, initializes PromptFusion normally, and
  restores that state. PromptFusion weights remain initialized; downstream
  DataLoader RNG starts at the same state as Square/Gaussian.
- The P10 Square one-epoch probe reproduced the prior
  `scratch25_worker24_square_seed13` epoch-0 training batches and validation
  metrics exactly. That completed 25-epoch run is therefore the P10 Square
  baseline: validation 55.04%/59.59%/43.87%/26.22%, test
  57.76%/61.87%/45.87%/30.52%.
- Gaussian-only reproduced its original-trajectory results: validation
  59.15%/63.81%/46.68%/27.41%, test 63.21%/67.42%/50.10%/32.17%.
- RNG-isolated SAM-Gaussian: validation 61.21%/65.98%/48.54%/29.25%, test
  60.95%/65.88%/48.69%/30.42%. Relative to Gaussian-only this is
  +2.06/+2.17/+1.86/+1.84 points on validation and
  -2.26/-1.54/-1.41/-1.75 points on test.

## P11: P10-RNG adaptive multi-mask SAM-Gaussian

- P11 preserves P10's original DataLoader RNG behavior. The Adaptive module
  passes an RNG-isolation unit test and starts exactly as `P = G_25`; its
  three-batch probe matches the P10 original trajectory.
- The offline `sam_multimask` cache contains all three point-prompted SAM
  candidates and scores for train/val/test (4,343/923/973 samples).
- Selected-checkpoint validation (batch size 8, `num_workers=16`):
  59.05%/63.71%/46.79%/26.54%.
- Final test (one evaluation of that validation-selected checkpoint, batch size
  8, `num_workers=16`): 60.23%/64.23%/47.73%/32.48%.
