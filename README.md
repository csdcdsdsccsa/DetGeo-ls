# Full Model — standalone

This project contains only the fixed CVOGL Full Model: **shared Swin-T +
Bi-Guidance + AMHCSFI-Res + HiSym-CRGPE** with two 64×64 YOLO heads. It does
not include the source repository's model variants, PE switches, ablation
flags, RNG probes, or experimental trainers.

The model class and initialization order match the original
`TROGeoMSDetectionAblation` Full Model (`h2_ind_amhcsfi_res_bi`). DroneAerial
uses core/outer Gaussian sigmas **25×25 / 50×50**; SVI uses **25×50 / 50×100**.
Both use the official `train/val/test.pth` splits, image size 1024, shared
Swin-T pretrained ImageNet-1K weights, batch 7, 24 workers, 25 epochs,
Adam at 1e-4 with 0.1 decay every 10 epochs, and seed 2024. Training selects
the checkpoint by validation Acc@0.50; testing strictly loads that checkpoint.
The `--gpu` argument selects one GPU; extra visible GPUs are not used.

## Data layout

Point `DATA_ROOT` at a directory containing:

```
data/
  CVOGL_DroneAerial/
    CVOGL_DroneAerial_train.pth
    CVOGL_DroneAerial_val.pth
    CVOGL_DroneAerial_test.pth
    query/
    satellite/
  CVOGL_SVI/
    CVOGL_SVI_train.pth
    CVOGL_SVI_val.pth
    CVOGL_SVI_test.pth
    query/
    satellite/
```

## Run

On a CUDA host with the required packages installed:

```bash
DATA_ROOT=/root/autodl-tmp/ls/data bash scripts/run_drone.sh
DATA_ROOT=/root/autodl-tmp/ls/data bash scripts/run_svi.sh
```

Set `PYTHON_BIN` and `OUTPUT_DIR` to override the Python executable and run
directory. An existing `best.pth.tar` is protected against an accidental
training overwrite. The scripts train once and then test the validation-best
checkpoint. Results appear in `metrics.jsonl` and `test_metrics.json`.

For evaluation of an existing source checkpoint, use the fixed architecture:

```bash
python train.py test --dataset CVOGL_DroneAerial --data-root data \
  --output-dir runs/drone_eval --checkpoint /path/to/source_model_best.pth.tar
```

## RNG protocol

This is the original `--standard_rng` / natural-consumption path: seed Python,
NumPy, PyTorch, and CUDA once at startup, then construct ordinary shuffled
DataLoaders and the model. There is **no** private `torch.Generator`,
`worker_init_fn`, RNG padding/saving/restoring, or reseed after construction.
cuDNN benchmark is off and deterministic is on, matching the source script;
PyTorch deterministic-algorithms mode is **not** enabled. Identical seeds do
not guarantee bitwise-identical retraining across CUDA executions.

## Source parity checks

From this directory, with the source repository as a sibling:

```bash
python tests/test_source_parity.py
python tests/test_protocol_parity.py
```

The first checks all model state-dict keys, initialized tensors, and parameter
count for both dataset configurations without downloading Swin weights; the
second checks the fixed loss, decoder, and four metrics against the source.
These are implementation checks, **not** new accuracy results.
