"""Diagnose E05 anchor-free validation decoding without retraining.

This reports three decode modes from the same trained checkpoint:
``center_x_quality`` (the trained default), ``center_only``, and
``oracle_center`` (the ground-truth grid cell with the model's LTRB output).
The last mode isolates box regression from center selection; it is not a
deployable result.  Results are written as a compact summary and per-sample
JSONL so the selection and regression errors remain separately inspectable.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from model.oacfr_geo import OACFRGeoE05
from model.oacfr_loss import aligned_iou, decode_ltrb_at_indices
from train_oacfr import make_loader, set_seed


MODES = ("center_x_quality", "center_only", "oracle_center")
CENTER_RADII = (8, 16, 32)


def parse_args():
    parser = argparse.ArgumentParser(description="E05 checkpoint decode diagnostic")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-name", default="CVOGL_DroneAerial")
    parser.add_argument("--checkpoint", required=True,
                        help="E05 model_best.pth.tar, not the frozen DetGeo checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--search-scales", required=True, choices=("64", "32,64,128"))
    parser.add_argument("--match-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--print-freq", type=int, default=100)
    return parser.parse_args()


def target_indices(boxes, height, width, image_size):
    """Return the exact grid-cell convention used by E05 training targets."""
    stride_x = float(image_size) / width
    stride_y = float(image_size) / height
    target_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5 / stride_x).long().clamp(0, width - 1)
    target_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5 / stride_y).long().clamp(0, height - 1)
    return torch.stack((target_y, target_x), dim=1)


def predicted_indices(center, quality, mode, target):
    height, width = center.shape[-2:]
    if mode == "oracle_center":
        return target
    map_to_rank = center if mode == "center_only" else center * quality
    flat = map_to_rank.flatten(1).argmax(dim=1)
    return torch.stack((torch.div(flat, width, rounding_mode="floor"), flat % width), dim=1)


def decode(outputs, boxes, image_size, mode):
    center = outputs["center_logits"].sigmoid()
    quality = outputs["quality_logits"].sigmoid()
    height, width = center.shape[-2:]
    target = target_indices(boxes, height, width, image_size)
    indices = predicted_indices(center, quality, mode, target)
    decoded_boxes = decode_ltrb_at_indices(outputs["ltrb"], indices, image_size)
    decoded_boxes = decoded_boxes.clamp(0, image_size - 1)
    iou = aligned_iou(decoded_boxes, boxes)
    batch = torch.arange(boxes.shape[0], device=boxes.device)
    selected_center = center[batch, 0, indices[:, 0], indices[:, 1]]
    selected_quality = quality[batch, 0, indices[:, 0], indices[:, 1]]
    stride_x = float(image_size) / width
    stride_y = float(image_size) / height
    selected_xy = torch.stack(((indices[:, 1].to(boxes.dtype) + 0.5) * stride_x,
                               (indices[:, 0].to(boxes.dtype) + 0.5) * stride_y), dim=1)
    target_xy = torch.stack(((boxes[:, 0] + boxes[:, 2]) * 0.5,
                             (boxes[:, 1] + boxes[:, 3]) * 0.5), dim=1)
    center_error = (selected_xy - target_xy).square().sum(dim=1).sqrt()
    return {
        "indices": indices,
        "boxes": decoded_boxes,
        "iou": iou,
        "center_error_px": center_error,
        "center_exact": (indices == target).all(dim=1),
        "center_score": selected_center,
        "quality_score": selected_quality,
    }


def empty_stats():
    return {
        "samples": 0, "iou_sum": 0.0, "correct_025": 0, "correct_050": 0,
        "center_exact": 0, "center_error_sum": 0.0,
        **{"center_within_%dpx" % radius: 0 for radius in CENTER_RADII},
        "quality_iou_sum": 0.0, "quality_abs_error_sum": 0.0,
    }


def summarize(stats):
    count = max(stats.pop("samples"), 1)
    return {
        "sample_count": count,
        "acc_025": stats.pop("correct_025") / count,
        "acc_050": stats.pop("correct_050") / count,
        "mean_iou": stats.pop("iou_sum") / count,
        "center_exact_accuracy": stats.pop("center_exact") / count,
        "mean_center_error_px": stats.pop("center_error_sum") / count,
        "center_within_px": {
            str(radius): stats.pop("center_within_%dpx" % radius) / count
            for radius in CENTER_RADII
        },
        "mean_selected_quality": stats.pop("quality_iou_sum") / count,
        "mean_abs_quality_minus_iou": stats.pop("quality_abs_error_sum") / count,
    }


def update(stats, result):
    iou = result["iou"]
    error = result["center_error_px"]
    quality = result["quality_score"]
    stats["samples"] += iou.numel()
    stats["iou_sum"] += float(iou.sum())
    stats["correct_025"] += int((iou > 0.25).sum())
    stats["correct_050"] += int((iou > 0.5).sum())
    stats["center_exact"] += int(result["center_exact"].sum())
    stats["center_error_sum"] += float(error.sum())
    stats["quality_iou_sum"] += float(quality.sum())
    stats["quality_abs_error_sum"] += float((quality - iou).abs().sum())
    for radius in CENTER_RADII:
        stats["center_within_%dpx" % radius] += int((error <= radius).sum())


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this checkpoint diagnostic")
    scales = tuple(int(value) for value in args.search_scales.split(","))
    set_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(args, "val", False)
    model = torch.nn.DataParallel(OACFRGeoE05(
        match_dim=args.match_dim, search_scales=scales,
        temperature=args.temperature, freeze_backbone=True)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.module.freeze_backbone(True)
    model.eval()
    stats = {mode: empty_stats() for mode in MODES}
    config = vars(args).copy()
    config["parsed_search_scales"] = scales
    config["checkpoint_epoch"] = checkpoint.get("epoch")
    config["checkpoint_best_acc_050"] = checkpoint.get("best_acc_050")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch_number, (query, reference, click_map, boxes, sample_ids) in enumerate(loader, 1):
            query = query.to(device, non_blocking=True)
            reference = reference.to(device, non_blocking=True)
            click_map = click_map.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True).clamp(0, args.img_size - 1)
            outputs = model(query, reference, click_map)
            decoded = {mode: decode(outputs, boxes, args.img_size, mode) for mode in MODES}
            for mode, result in decoded.items():
                update(stats[mode], result)
            for position, sample_id in enumerate(sample_ids.tolist()):
                record = {"sample_id": int(sample_id), "gt_box": boxes[position].tolist(), "modes": {}}
                for mode, result in decoded.items():
                    record["modes"][mode] = {
                        "grid_yx": result["indices"][position].tolist(),
                        "box": result["boxes"][position].tolist(),
                        "iou": float(result["iou"][position]),
                        "center_error_px": float(result["center_error_px"][position]),
                        "center_exact": bool(result["center_exact"][position]),
                        "center_score": float(result["center_score"][position]),
                        "quality_score": float(result["quality_score"][position]),
                    }
                handle.write(json.dumps(record) + "\n")
            if batch_number % args.print_freq == 0 or batch_number == len(loader):
                print("validated %d/%d batches" % (batch_number, len(loader)), flush=True)
    summary = {
        "experiment": "E05_checkpoint_decode_diagnostic",
        "split": "validation",
        "test_split_used": False,
        "modes": {
            "center_x_quality": "deployable E05 decoding: sigmoid(center) * sigmoid(quality)",
            "center_only": "ablation: select by sigmoid(center) only",
            "oracle_center": "diagnostic only: use GT center grid cell and model LTRB",
        },
        "metrics": {mode: summarize(mode_stats) for mode, mode_stats in stats.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
