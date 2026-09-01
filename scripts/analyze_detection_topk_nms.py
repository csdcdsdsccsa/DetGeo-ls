"""Detection-head Top-K proposal analysis after IoU-based NMS.

All 9 x H x W raw DetGeo proposals are decoded with the repository's original
equations.  Per image, torchvision NMS is then applied without a score cutoff;
the remaining score-sorted proposals are used for Top-K recall statistics.
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.ops import nms
from torchvision.transforms import Compose, Normalize, ToTensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_loader import RSDataset
from model.DetGeo import DetGeo
from model.loss import build_target
from utils.checkpoint import load_pretrain
from utils.utils import eval_iou_acc


ANCHORS = {
    "CVOGL_DroneAerial": "37,41, 78,84, 96,215, 129,129, 194,82, "
                         "198,179, 246,280, 395,342, 550,573",
    "CVOGL_SVI": "37,41, 78,84, 96,215, 129,129, 194,82, "
                  "198,179, 246,280, 395,342, 550,573",
}
THRESHOLDS = (0.25, 0.50)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-name", default="CVOGL_DroneAerial", choices=ANCHORS)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"),
                        help="Dataset split to evaluate; keep test untouched until final evaluation.")
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--k-values", default="1,3,5,10,20")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--print-freq", type=int, default=25)
    return parser.parse_args()


def bbox_iou_xyxy(boxes, target):
    inter_x1 = torch.maximum(boxes[:, 0], target[0])
    inter_y1 = torch.maximum(boxes[:, 1], target[1])
    inter_x2 = torch.minimum(boxes[:, 2], target[2])
    inter_y2 = torch.minimum(boxes[:, 3], target[3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area_target = (target[2] - target[0]).clamp(min=0) * (target[3] - target[1]).clamp(min=0)
    return inter / (area_boxes + area_target - inter + 1e-16)


def decode_all_candidates(pred_one, anchors_full, image_size):
    """Decode every raw anchor/grid candidate; no confidence filtering or NMS."""
    anchor_count, _, height, width = pred_one.shape
    if height != width:
        raise ValueError("DetGeo evaluation expects a square detection map")
    total = anchor_count * height * width
    flat_indices = torch.arange(total, device=pred_one.device)
    cells_per_anchor = height * width
    anchor_idx = torch.div(flat_indices, cells_per_anchor, rounding_mode="floor")
    cell_idx = flat_indices % cells_per_anchor
    gj = torch.div(cell_idx, width, rounding_mode="floor")
    gi = cell_idx % width
    scores = pred_one[:, 4].reshape(-1)

    grid_stride = image_size // height
    scaled_anchors = anchors_full / grid_stride
    x = (pred_one[anchor_idx, 0, gj, gi].sigmoid() + gi) * grid_stride
    y = (pred_one[anchor_idx, 1, gj, gi].sigmoid() + gj) * grid_stride
    w = torch.exp(pred_one[anchor_idx, 2, gj, gi]) * scaled_anchors[anchor_idx, 0] * grid_stride
    h = torch.exp(pred_one[anchor_idx, 3, gj, gi]) * scaled_anchors[anchor_idx, 1] * grid_stride
    boxes = torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=1)
    return scores, flat_indices, anchor_idx, gi, gj, boxes


def first_true_rank(mask):
    found = torch.nonzero(mask, as_tuple=False)
    return None if len(found) == 0 else int(found[0, 0]) + 1


def image_names(dataset, index):
    row = dataset.data_list[index]
    return {"query_image": str(row[1]), "satellite_image": str(row[2])}


def main():
    args = parse_args()
    if not 0 <= args.nms_iou <= 1:
        raise ValueError("--nms-iou must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint analysis")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed_all(args.seed + 3)

    k_values = sorted({int(value) for value in args.k_values.split(",")})
    if not k_values or k_values[0] < 1:
        raise ValueError("--k-values must contain positive integers")
    max_k = max(k_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = RSDataset(args.data_root, args.data_name, args.split, args.img_size, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        pin_memory=True, num_workers=args.num_workers)
    model = torch.nn.DataParallel(DetGeo()).cuda()
    args.pretrain = args.checkpoint
    model = load_pretrain(model, args, logging)
    model.eval()

    anchors = np.array([float(value.strip()) for value in ANCHORS[args.data_name].split(",")])
    anchors_full = torch.tensor(anchors.reshape(-1, 2)[::-1].copy(), dtype=torch.float32).cuda()
    nms_counts = {threshold: {k: 0 for k in k_values} for threshold in THRESHOLDS}
    raw_top1_counts = {threshold: 0 for threshold in THRESHOLDS}
    original_counts = {threshold: 0 for threshold in THRESHOLDS}
    first_ranks = {threshold: [] for threshold in THRESHOLDS}
    kept_counts = []
    records_path = output_dir / "per_sample.jsonl"

    with records_path.open("w", encoding="utf-8") as records:
        with torch.no_grad():
            for batch_idx, (query_imgs, rs_imgs, mat_clickxy, gt_boxes, indices) in enumerate(loader):
                query_imgs = query_imgs.cuda(non_blocking=True)
                rs_imgs = rs_imgs.cuda(non_blocking=True)
                mat_clickxy = mat_clickxy.cuda(non_blocking=True)
                gt_boxes = torch.clamp(gt_boxes.cuda(non_blocking=True), min=0, max=args.img_size - 1)
                pred_raw, _ = model(query_imgs, rs_imgs, mat_clickxy)
                pred = pred_raw.view(pred_raw.shape[0], 9, 5, pred_raw.shape[2], pred_raw.shape[3])

                # Independent repository evaluation for raw Top-1 verification.
                _, target_anchor_gi_gj = build_target(gt_boxes, anchors_full, args.img_size, pred.shape[3])
                _, _, _, original_each_acc, _, _ = eval_iou_acc(
                    pred, gt_boxes, anchors_full, target_anchor_gi_gj[:, 1],
                    target_anchor_gi_gj[:, 2], args.img_size,
                    iou_threshold_list=[0.5, 0.25])
                original_counts[0.50] += int(original_each_acc[0].sum().item())
                original_counts[0.25] += int(original_each_acc[1].sum().item())

                for item in range(pred.shape[0]):
                    scores, flat_indices, anchor_idx, gi, gj, boxes = decode_all_candidates(
                        pred[item], anchors_full, args.img_size)
                    raw_order = torch.argsort(scores, descending=True, stable=True)
                    raw_ious = bbox_iou_xyxy(boxes[raw_order[:1]], gt_boxes[item])
                    for threshold in THRESHOLDS:
                        raw_top1_counts[threshold] += int(bool(raw_ious[0] > threshold))

                    keep = nms(boxes, scores, args.nms_iou)
                    kept_counts.append(int(keep.numel()))
                    keep = keep[:max_k]
                    kept_ious = bbox_iou_xyxy(boxes[keep], gt_boxes[item])
                    record = {
                        "sample_index": int(indices[item]),
                        **image_names(dataset, int(indices[item])),
                        "ground_truth_xyxy": [float(x) for x in gt_boxes[item].cpu().tolist()],
                        "grid_size": [int(pred.shape[3]), int(pred.shape[4])],
                        "raw_candidate_count": int(scores.numel()),
                        "nms_iou_threshold": args.nms_iou,
                        "nms_retained_count": kept_counts[-1],
                        "candidates_after_nms": [],
                    }
                    for rank in range(keep.numel()):
                        raw_index = keep[rank]
                        record["candidates_after_nms"].append({
                            "rank_after_nms": rank + 1,
                            "confidence_logit": float(scores[raw_index].cpu()),
                            "flat_index": int(flat_indices[raw_index].cpu()),
                            "anchor_index": int(anchor_idx[raw_index].cpu()),
                            "gi": int(gi[raw_index].cpu()),
                            "gj": int(gj[raw_index].cpu()),
                            "bbox_xyxy": [float(x) for x in boxes[raw_index].cpu().tolist()],
                            "iou": float(kept_ious[rank].cpu()),
                        })
                    for threshold in THRESHOLDS:
                        rank = first_true_rank(kept_ious > threshold)
                        first_ranks[threshold].append(rank)
                        record["first_correct_rank_after_nms_iou_%s" % threshold] = rank
                        for k in k_values:
                            if bool((kept_ious[:k] > threshold).any()):
                                nms_counts[threshold][k] += 1
                    records.write(json.dumps(record, ensure_ascii=False) + "\n")

                if batch_idx % args.print_freq == 0:
                    print("[%d/%d] processed" %
                          (min((batch_idx + 1) * args.batch_size, len(dataset)), len(dataset)), flush=True)

    total = len(dataset)
    for threshold in THRESHOLDS:
        if raw_top1_counts[threshold] != original_counts[threshold]:
            raise RuntimeError(
                "Raw Top-1 mismatch at IoU>%.2f: decoded=%d, original=%d" %
                (threshold, raw_top1_counts[threshold], original_counts[threshold]))

    rows = []
    summary = {
        "experiment": "detection_head_topk_after_nms",
        "checkpoint": str(Path(args.checkpoint)),
        "split": args.split,
        "sample_count": total,
        "data_name": args.data_name,
        "image_size": args.img_size,
        "k_values": k_values,
        "raw_candidates_per_sample": int(9 * pred.shape[3] * pred.shape[4]),
        "nms_iou_threshold": args.nms_iou,
        "raw_top1_alignment_with_original_eval_iou_acc": {},
        "nms_retained_candidates": {
            "mean": float(np.mean(kept_counts)), "min": int(np.min(kept_counts)),
            "max": int(np.max(kept_counts)),
        },
        "proposal_recall_after_nms": {},
        "first_correct_rank_after_nms": {},
    }
    for threshold in THRESHOLDS:
        threshold_key = "iou_gt_%s" % threshold
        summary["raw_top1_alignment_with_original_eval_iou_acc"][threshold_key] = {
            "decoded_raw_hits": raw_top1_counts[threshold],
            "original_eval_hits": original_counts[threshold],
            "accuracy": raw_top1_counts[threshold] / total,
            "exact_match": True,
        }
        by_k = {}
        top1_errors = total - nms_counts[threshold][1]
        for k in k_values:
            hits = nms_counts[threshold][k]
            rescued = hits - nms_counts[threshold][1]
            by_k[str(k)] = {
                "hits": hits, "recall": hits / total,
                "rescued_from_nms_top1": rescued,
                "rescuable_nms_top1_errors": None if top1_errors == 0 else rescued / top1_errors,
            }
            rows.append({
                "iou_threshold": threshold, "k": k, "hits": hits, "sample_count": total,
                "proposal_recall_after_nms": hits / total,
                "rescued_from_nms_top1": rescued,
                "rescuable_nms_top1_errors": "" if top1_errors == 0 else rescued / top1_errors,
            })
        summary["proposal_recall_after_nms"][threshold_key] = by_k
        ranks = first_ranks[threshold]
        summary["first_correct_rank_after_nms"][threshold_key] = {
            "within_max_k": sum(rank is not None for rank in ranks),
            "not_found_within_max_k": sum(rank is None for rank in ranks),
            "rank_histogram": {str(rank): sum(value == rank for value in ranks)
                               for rank in range(1, max_k + 1)},
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote %s" % output_dir)


if __name__ == "__main__":
    main()
