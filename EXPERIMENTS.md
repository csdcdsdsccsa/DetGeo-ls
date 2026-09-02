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
