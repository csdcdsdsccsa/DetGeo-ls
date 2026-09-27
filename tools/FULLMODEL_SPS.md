# Frozen Full Model + SPS evaluation

This entry point does not train or modify the Full Model. It strictly loads
the two official-split validation-best checkpoints named in the launch scripts.
It uses the existing dataset, anchors and `select_two_heads(fusion=False)`.

## Dependencies and data

Use the existing `detgeo` environment. Fetch the unmodified official CVOS SAM
implementation (source revision used for the cloud smoke test):

```bash
mkdir -p third_party checkpoints
git clone https://github.com/zqwlearning/CVOS-Code.git third_party/cvos_sps
git -C third_party/cvos_sps checkout 289602a033d5353d06987a2d56a808e9c938cdad
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
```

Original data must be under `data/CVOGL_DroneAerial` and `data/CVOGL_SVI`.
Masks must be under `data-Seg/<dataset>/mask-satellite` (an extra `CVOGL-Seg`
directory is also supported). On the current server, `data-Seg` is a symlink
to the extracted `CVOGL-Seg` directory. Weights and data are not included in Git.

## Run

```bash
PYTHONPATH=. /root/miniconda3/envs/detgeo/bin/python tools/eval_fullmodel_sps.py \
  --data_name CVOGL_DroneAerial --split val \
  --checkpoint saved_models/trogeo_ms_e4_amhcsfi_res_bi_hisym_crgpe_core25_outer50_naturalrng_swin_t_drone_seed2024_model_best.pth.tar \
  --max_samples 5
bash scripts/eval_fullmodel_sps_all.sh
```

The sequence is Drone val/test, then SVI val/test. All split masks are audited
before inference; missing masks or unexpected sizes fail rather than skip.
Smoke outputs go to `results/fullmodel_sps/smoke`; full evaluations go to
`results/fullmodel_sps`, with per-sample CSVs, split JSONs and `summary.csv`.

SPS follows official `sam_prompt.py`: ImageNet-normalized satellite tensor
directly enters SAM ViT-H's image encoder; prompts are the selected bbox plus
its positive center point; single-mask logits are bilinearly enlarged fourfold
and thresholded at 0.5. This is intentionally not SamPredictor preprocessing.
Metrics are per-sample IoU, Dice, absolute area error and centroid distance.
Meter conversion uses the satellite filename's coverage width divided by 1024.
Original localization metrics are also reported for consistency checks.
