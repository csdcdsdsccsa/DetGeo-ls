"""Detection-head Top-K proposal analysis for a trained DetGeo checkpoint.

This script deliberately does not alter the model or train it.  It ranks every
raw detection-head confidence (9 anchors x H x W) without NMS, decodes the
candidates with the same equations as ``utils.utils.eval_iou_acc``, and reports
whether at least one proposal reaches the requested IoU thresholds.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
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
    parser.add_argument("--gpu", default="0", help="CUDA device id(s), e.g. 0")
    parser.add_argument("--checkpoint", required=True, help="DetGeo checkpoint .pth.tar")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-name", default="CVOGL_DroneAerial", choices=ANCHORS)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--k-values", default="1,3,5,10,20")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--print-freq", type=int, default=25)
    return parser.parse_args()


def bbox_iou_xyxy(boxes, target):
    """IoU of N xyxy boxes against one xyxy target, matching the strict test rule."""
    inter_x1 = torch.maximum(boxes[:, 0], target[0])
    inter_y1 = torch.maximum(boxes[:, 1], target[1])
    inter_x2 = torch.minimum(boxes[:, 2], target[2])
    inter_y2 = torch.minimum(boxes[:, 3], target[3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area_target = (target[2] - target[0]).clamp(min=0) * (target[3] - target[1]).clamp(min=0)
    return inter / (area_boxes + area_target - inter + 1e-16)


def decode_topk(pred_one, k, anchors_full, image_size):
    """Decode raw top-k logits of one [A,5,H,W] prediction, with no NMS."""
    anchor_count, _, height, width = pred_one.shape
    if height != width:
        raise ValueError("DetGeo evaluation expects a square detection map")
    total = anchor_count * height * width
    if k > total:
        raise ValueError("K=%d exceeds the %d raw proposals" % (k, total))

    confidence = pred_one[:, 4].reshape(-1)
    scores, flat_indices = torch.topk(confidence, k=k, largest=True, sorted=True)
    cells_per_anchor = height * width
    anchor_idx = torch.div(flat_indices, cells_per_anchor, rounding_mode="floor")
    cell_idx = flat_indices % cells_per_anchor
    gj = torch.div(cell_idx, width, rounding_mode="floor")
    gi = cell_idx % width

    grid_stride = image_size // height
    scaled_anchors = anchors_full / grid_stride
    tx = pred_one[anchor_idx, 0, gj, gi]
    ty = pred_one[anchor_idx, 1, gj, gi]
    tw = pred_one[anchor_idx, 2, gj, gi]
    th = pred_one[anchor_idx, 3, gj, gi]
    x = (tx.sigmoid() + gi) * grid_stride
    y = (ty.sigmoid() + gj) * grid_stride
    w = torch.exp(tw) * scaled_anchors[anchor_idx, 0] * grid_stride
    h = torch.exp(th) * scaled_anchors[anchor_idx, 1] * grid_stride
    boxes = torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=1)
    return scores, flat_indices, anchor_idx, gi, gj, boxes


def sample_names(dataset, index):
    row = dataset.data_list[index]
    return {"query_image": str(row[1]), "satellite_image": str(row[2])}


def first_true_rank(mask):
    found = torch.nonzero(mask, as_tuple=False)
    return None if len(found) == 0 else int(found[0, 0]) + 1


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this checkpoint analysis")
    os_environ = __import__("os").environ
    os_environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os_environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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
    dataset = RSDataset(args.data_root, args.data_name, "test", args.img_size, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        pin_memory=True, num_workers=args.num_workers)

    model = torch.nn.DataParallel(DetGeo()).cuda()
    # load_pretrain only needs args.pretrain and a logging-compatible object.
    args.pretrain = args.checkpoint
    model = load_pretrain(model, args, __import__("logging"))
    model.eval()

    anchors_full = np.array([float(x.strip()) for x in ANCHORS[args.data_name].split(",")])
    anchors_full = torch.tensor(anchors_full.reshape(-1, 2)[::-1].copy(), dtype=torch.float32).cuda()
    counts = {threshold: {k: 0 for k in k_values} for threshold in THRESHOLDS}
    original_counts = {threshold: 0 for threshold in THRESHOLDS}
    first_ranks = {threshold: [] for threshold in THRESHOLDS}
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

                # This is the repository's original test computation.  Its result
                # is accumulated independently and must exactly equal our K=1.
                _, target_anchor_gi_gj = build_target(gt_boxes, anchors_full, args.img_size, pred.shape[3])
                _, _, _, original_each_acc, _, _ = eval_iou_acc(
                    pred, gt_boxes, anchors_full, target_anchor_gi_gj[:, 1],
                    target_anchor_gi_gj[:, 2], args.img_size,
                    iou_threshold_list=[0.5, 0.25])
                original_counts[0.50] += int(original_each_acc[0].sum().item())
                original_counts[0.25] += int(original_each_acc[1].sum().item())

                for item_in_batch in range(pred.shape[0]):
                    scores, flat_indices, anchor_idx, gi, gj, boxes = decode_topk(
                        pred[item_in_batch], max_k, anchors_full, args.img_size)
                    ious = bbox_iou_xyxy(boxes, gt_boxes[item_in_batch])
                    hit_masks = {threshold: ious > threshold for threshold in THRESHOLDS}
                    record = {
                        "sample_index": int(indices[item_in_batch]),
                        **sample_names(dataset, int(indices[item_in_batch])),
                        "ground_truth_xyxy": [float(x) for x in gt_boxes[item_in_batch].cpu().tolist()],
                        "grid_size": [int(pred.shape[3]), int(pred.shape[4])],
                        "candidates": [],
                    }
                    for rank in range(max_k):
                        record["candidates"].append({
                            "rank": rank + 1,
                            "confidence_logit": float(scores[rank].cpu()),
                            "flat_index": int(flat_indices[rank].cpu()),
                            "anchor_index": int(anchor_idx[rank].cpu()),
                            "gi": int(gi[rank].cpu()),
                            "gj": int(gj[rank].cpu()),
                            "bbox_xyxy": [float(x) for x in boxes[rank].cpu().tolist()],
                            "iou": float(ious[rank].cpu()),
                        })
                    for threshold in THRESHOLDS:
                        rank = first_true_rank(hit_masks[threshold])
                        first_ranks[threshold].append(rank)
                        record["first_correct_rank_iou_%s" % threshold] = rank
                        for k in k_values:
                            if bool(hit_masks[threshold][:k].any()):
                                counts[threshold][k] += 1
                    records.write(json.dumps(record, ensure_ascii=False) + "\n")

                if batch_idx % args.print_freq == 0:
                    completed = min((batch_idx + 1) * args.batch_size, len(dataset))
                    print("[%d/%d] processed" % (completed, len(dataset)), flush=True)

    total = len(dataset)
    for threshold in THRESHOLDS:
        if counts[threshold][1] != original_counts[threshold]:
            raise RuntimeError(
                "K=1 mismatch at IoU>%.2f: Top-K=%d, original eval_iou_acc=%d" %
                (threshold, counts[threshold][1], original_counts[threshold]))

    summary_rows = []
    summary = {
        "experiment": "detection_head_raw_topk_without_nms",
        "checkpoint": str(Path(args.checkpoint)),
        "split": "test",
        "sample_count": total,
        "data_name": args.data_name,
        "image_size": args.img_size,
        "k_values": k_values,
        "nms": False,
        "k1_alignment_with_original_eval_iou_acc": {},
        "proposal_recall": {},
        "first_correct_rank": {},
    }
    for threshold in THRESHOLDS:
        key = "iou_gt_%s" % threshold
        k1_accuracy = counts[threshold][1] / total
        summary["k1_alignment_with_original_eval_iou_acc"][key] = {
            "topk_hits": counts[threshold][1],
            "original_eval_hits": original_counts[threshold],
            "topk_accuracy": k1_accuracy,
            "original_eval_accuracy": original_counts[threshold] / total,
            "exact_match": True,
        }
        recall_rows = {}
        top1_errors = total - counts[threshold][1]
        for k in k_values:
            hits = counts[threshold][k]
            rescued = hits - counts[threshold][1]
            recall_rows[str(k)] = {
                "hits": hits,
                "recall": hits / total,
                "rescued_from_top1": rescued,
                "rescuable_top1_errors": None if top1_errors == 0 else rescued / top1_errors,
            }
            summary_rows.append({
                "iou_threshold": threshold, "k": k, "hits": hits, "sample_count": total,
                "proposal_recall": hits / total, "rescued_from_top1": rescued,
                "rescuable_top1_errors": "" if top1_errors == 0 else rescued / top1_errors,
            })
        summary["proposal_recall"][key] = recall_rows
        ranks = first_ranks[threshold]
        rank_histogram = {str(rank): sum(value == rank for value in ranks) for rank in range(1, max_k + 1)}
        summary["first_correct_rank"][key] = {
            "within_max_k": sum(rank is not None for rank in ranks),
            "not_found_within_max_k": sum(rank is None for rank in ranks),
            "rank_histogram": rank_histogram,
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote %s" % output_dir)


if __name__ == "__main__":
    main()
