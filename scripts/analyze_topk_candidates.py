"""Offline Top-K candidate analysis for a trained DetGeo checkpoint.

This script intentionally does not modify DetGeo or train a new network.  It
uses the query-to-satellite attention map already produced by DetGeo, applies
local-maximum suppression, and reports whether any of the K highest-scoring
candidate centres lies inside the ground-truth satellite bounding box.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# Permit ``python scripts/analyze_topk_candidates.py`` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_loader import RSDataset
from model.DetGeo import DetGeo


DEFAULT_K_VALUES = (1, 3, 5, 10, 20)


def parse_k_values(value: str) -> Tuple[int, ...]:
    k_values = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not k_values or k_values[0] < 1:
        raise argparse.ArgumentTypeError("K values must be positive integers, e.g. 1,3,5,10,20")
    return k_values


def local_maximum_candidates(
    attention: torch.Tensor,
    max_k: int,
    nms_kernel: int,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return candidate scores and image-space centre coordinates for one map."""
    if attention.ndim != 2:
        raise ValueError(f"Expected a 2-D attention map, got {tuple(attention.shape)}")
    if nms_kernel < 1 or nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer")

    height, width = attention.shape
    pooled = F.max_pool2d(
        attention[None, None], kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2
    )[0, 0]
    local_maxima = attention >= pooled
    suppressed = attention.masked_fill(~local_maxima, float("-inf")).flatten()
    if int(torch.isfinite(suppressed).sum()) < max_k:
        raise RuntimeError("NMS left fewer candidates than requested; use a smaller NMS kernel")

    scores, flat_indices = torch.topk(suppressed, k=max_k, largest=True, sorted=True)
    rows = torch.div(flat_indices, width, rounding_mode="floor")
    cols = flat_indices % width
    centres_x = (cols.float() + 0.5) * (float(image_size) / width)
    centres_y = (rows.float() + 0.5) * (float(image_size) / height)
    centres = torch.stack((centres_x, centres_y), dim=1)
    return scores, centres, torch.stack((rows, cols), dim=1)


def candidate_hits(centres: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    """A candidate is a hit when its centre lies inside [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = gt_bbox.tolist()
    return (
        (centres[:, 0] >= x1)
        & (centres[:, 0] <= x2)
        & (centres[:, 1] >= y1)
        & (centres[:, 1] <= y2)
    )


def make_record(
    dataset: RSDataset,
    sample_index: int,
    gt_bbox: torch.Tensor,
    scores: torch.Tensor,
    centres: torch.Tensor,
    grid_positions: torch.Tensor,
    hits: torch.Tensor,
) -> Dict[str, object]:
    item = dataset.data_list[sample_index]
    return {
        "sample_index": int(sample_index),
        "query_image": str(item[1]),
        "satellite_image": str(item[2]),
        "gt_bbox_xyxy": [round(float(value), 4) for value in gt_bbox.tolist()],
        "gt_candidate_rank": int(hits.nonzero(as_tuple=False)[0, 0]) + 1 if bool(hits.any()) else None,
        "candidates": [
            {
                "rank": rank + 1,
                "score": round(float(scores[rank]), 8),
                "centre_xy": [round(float(value), 4) for value in centres[rank].tolist()],
                "grid_yx": [int(value) for value in grid_positions[rank].tolist()],
                "centre_in_gt": bool(hits[rank]),
            }
            for rank in range(len(scores))
        ],
    }


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    model = torch.nn.DataParallel(DetGeo()).cuda().eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DetGeo Top-K candidate recall analysis")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-name", default="CVOGL_DroneAerial", choices=("CVOGL_DroneAerial", "CVOGL_SVI"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--k-values", type=parse_k_values, default=DEFAULT_K_VALUES)
    parser.add_argument("--nms-kernel", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because the existing DetGeo implementation uses CUDA tensors internally")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transform = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    dataset = RSDataset(
        data_root=args.data_root,
        data_name=args.data_name,
        split_name="test",
        img_size=args.image_size,
        transform=transform,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = load_model(args.checkpoint)
    max_k = max(args.k_values)
    hit_counts = {k: 0 for k in args.k_values}
    rescuable_counts = {k: 0 for k in args.k_values if k > 1}
    records_path = args.output_dir / "per_sample.jsonl"

    with records_path.open("w", encoding="utf-8") as records_file, torch.no_grad():
        for batch_number, (query_images, satellite_images, click_maps, gt_boxes, sample_indices) in enumerate(loader):
            _, attention_maps = model(
                query_images.cuda(non_blocking=True),
                satellite_images.cuda(non_blocking=True),
                click_maps.cuda(non_blocking=True),
            )
            for index_in_batch in range(attention_maps.shape[0]):
                scores, centres, grid_positions = local_maximum_candidates(
                    attention_maps[index_in_batch], max_k, args.nms_kernel, args.image_size
                )
                gt_box = gt_boxes[index_in_batch].cpu()
                hits = candidate_hits(centres.cpu(), gt_box)
                for k in args.k_values:
                    hit_counts[k] += int(hits[:k].any())
                for k in rescuable_counts:
                    rescuable_counts[k] += int((not bool(hits[0])) and bool(hits[:k].any()))
                record = make_record(
                    dataset,
                    int(sample_indices[index_in_batch]),
                    gt_box,
                    scores.cpu(),
                    centres.cpu(),
                    grid_positions.cpu(),
                    hits.cpu(),
                )
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (batch_number + 1) % 20 == 0 or batch_number + 1 == len(loader):
                print(f"processed {min((batch_number + 1) * args.batch_size, len(dataset))}/{len(dataset)} samples", flush=True)

    total = len(dataset)
    top1_hits = hit_counts[1]
    rows: List[Dict[str, object]] = []
    for k in args.k_values:
        rows.append(
            {
                "k": k,
                "candidate_hits": hit_counts[k],
                "candidate_recall": hit_counts[k] / total,
                "candidate_recall_percent": 100.0 * hit_counts[k] / total,
                "rescuable_top1_errors": rescuable_counts.get(k, 0),
                "rescuable_top1_error_rate": (
                    rescuable_counts[k] / (total - top1_hits) if k > 1 and total > top1_hits else 0.0
                ),
                "rescuable_top1_error_rate_percent": (
                    100.0 * rescuable_counts[k] / (total - top1_hits) if k > 1 and total > top1_hits else 0.0
                ),
            }
        )

    summary = {
        "experiment": "DetGeo offline Top-K candidate recall analysis",
        "checkpoint": str(args.checkpoint),
        "dataset": args.data_name,
        "split": "test",
        "num_samples": total,
        "attention_map_shape": [int(attention_maps.shape[1]), int(attention_maps.shape[2])],
        "candidate_definition": "NMS local maxima of DetGeo attention map; candidate centre inside GT bbox",
        "nms_kernel": args.nms_kernel,
        "top1_candidate_hits": top1_hits,
        "top1_candidate_misses": total - top1_hits,
        "results": rows,
        "per_sample_file": records_path.name,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nCandidate Recall@K (candidate centre inside GT box)")
    for row in rows:
        resc = "-" if row["k"] == 1 else f"{row['rescuable_top1_error_rate_percent']:.2f}%"
        print(
            f"K={row['k']:>2}: Recall={row['candidate_recall_percent']:.2f}% "
            f"({row['candidate_hits']}/{total}), rescuable Top-1 errors={resc}"
        )
    print(f"\nWrote {args.output_dir / 'summary.json'} and {records_path}")


if __name__ == "__main__":
    main()
