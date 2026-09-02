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
  validation-only early-stopped result; do not use the test set or claim a final
  generalization improvement without replication.
