"""Generate point-prompted SAM masks for DetGeo query samples.

Masks are stored by split and dataset index, so repeated uses of a query image
with different clicks cannot overwrite one another.  The generator is resumable:
existing PNGs are retained unless --overwrite is supplied.
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser(description="Create offline point-prompted SAM masks")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-name", default="CVOGL_DroneAerial")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_b", choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vis-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mask_path(data_dir, split, index):
    return os.path.join(data_dir, "sam_mask", split, "{0:06d}.png".format(index))


def save_visualization(image_bgr, mask, click_xy, output_path):
    overlay = image_bgr.copy()
    green = np.array((0, 255, 0), dtype=np.uint8)
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * green).astype(np.uint8)
    x, y = int(click_xy[0]), int(click_xy[1])
    cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(output_path, overlay)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    data_dir = os.path.join(args.data_root, args.data_name)
    query_dir = os.path.join(data_dir, "query")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    predictor = SamPredictor(sam.to(device=args.device))
    rng = random.Random(args.seed)

    for split in args.splits:
        metadata_path = os.path.join(data_dir, "{0}_{1}.pth".format(args.data_name, split))
        samples = torch.load(metadata_path, map_location="cpu")
        output_dir = os.path.join(data_dir, "sam_mask", split)
        vis_dir = os.path.join(data_dir, "sam_vis", split)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        vis_indices = set(rng.sample(range(len(samples)), min(args.vis_count, len(samples))))

        generated = 0
        skipped = 0
        for index, sample in enumerate(samples):
            _, query_name, _, _, click_xy, _, _, _ = sample
            output_path = mask_path(data_dir, split, index)
            if os.path.isfile(output_path) and not args.overwrite:
                skipped += 1
                continue

            image_path = os.path.join(query_dir, query_name)
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            click = np.asarray(click_xy, dtype=np.float32)
            if not (0 <= click[0] < width and 0 <= click[1] < height):
                raise ValueError("out-of-bounds click for {0}: {1} in {2}x{3}".format(query_name, click, width, height))

            predictor.set_image(image_rgb)
            masks, scores, _ = predictor.predict(
                point_coords=click.reshape(1, 2),
                point_labels=np.asarray((1,), dtype=np.int32),
                multimask_output=True,
            )
            best_index = int(np.argmax(scores))
            best_mask = masks[best_index].astype(bool)
            if not cv2.imwrite(output_path, best_mask.astype(np.uint8) * 255):
                raise IOError("failed to write {0}".format(output_path))
            if index in vis_indices:
                save_visualization(image_bgr, best_mask, click_xy, os.path.join(vis_dir, "{0:06d}.jpg".format(index)))
            generated += 1
            if generated % 100 == 0:
                print("{0}: generated {1}/{2}, last score={3:.4f}".format(split, generated, len(samples), float(scores[best_index])), flush=True)

        print("{0}: generated={1}, skipped={2}, total={3}".format(split, generated, skipped, len(samples)), flush=True)


if __name__ == "__main__":
    main()
