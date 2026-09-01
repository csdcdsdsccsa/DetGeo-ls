"""First-round frozen-DetGeo NMS Top-K candidate ranking experiment.

The detector creates its real Top-K NMS proposals.  It stays in eval mode with
all parameters frozen.  The script supports the original independent scorer
and a detector-score residual scorer with conservative Acc@IoU labels.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from dataset.data_loader import RSDataset
from model.DetGeo import DetGeo
from model.ranker import (
    CandidateRanker,
    NormalizedResidualCandidateRanker,
    ResidualCandidateRanker,
    ResidualCrossAttentionRanker,
)
from utils.proposal_utils import nms_topk, pairwise_iou_with_gt, roi_align_candidates


ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"
THRESHOLDS = (0.25, 0.50)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen DetGeo checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-name", default="CVOGL_DroneAerial")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument(
        "--score-mode",
        choices=("standalone", "residual", "residual-zscore", "residual-cross-attention"),
        default="standalone")
    parser.add_argument("--target-mode", choices=("best-iou", "conservative"), default="best-iou")
    parser.add_argument("--target-iou", type=float, default=0.5)
    parser.add_argument("--residual-alpha", type=float, default=1.0)
    parser.add_argument("--score-eps", type=float, default=1e-6)
    parser.add_argument("--match-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-hidden-dim", type=int, default=64)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(args, split, shuffle):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Keep both splits unaugmented: boxes must be real frozen-detector proposals.
    dataset = RSDataset(args.data_root, args.data_name, split, args.image_size, transform, augment=False)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True)


def make_anchors(device):
    values = np.array([float(value.strip()) for value in ANCHORS.split(",")])
    return torch.tensor(values.reshape(-1, 2)[::-1].copy(), dtype=torch.float32, device=device)


def load_detector(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    detector = torch.nn.DataParallel(DetGeo()).to(device)
    detector.load_state_dict(checkpoint["state_dict"], strict=True)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    return detector


def frozen_candidates(detector, batch, anchors, args, device):
    query, reference, click_map, gt_boxes, _ = batch
    query = query.to(device, non_blocking=True)
    reference = reference.to(device, non_blocking=True)
    click_map = click_map.to(device, non_blocking=True)
    gt_boxes = gt_boxes.to(device, non_blocking=True).clamp_(0, args.image_size - 1)
    with torch.no_grad():
        outbox, _, query_features, reference_features, _ = detector(
            query, reference, click_map, return_features=True)
        prediction = outbox.view(outbox.shape[0], 9, 5, outbox.shape[-2], outbox.shape[-1])
        boxes, scores = nms_topk(
            prediction, anchors, args.image_size, args.candidate_count, args.nms_iou,
            return_logits=args.score_mode in (
                "residual", "residual-zscore", "residual-cross-attention"))
        candidate_features = roi_align_candidates(reference_features, boxes, args.image_size, args.roi_size)
        candidate_ious = pairwise_iou_with_gt(boxes, gt_boxes)
    return query_features, candidate_features, scores, boxes, candidate_ious


def new_metrics():
    return {
        "count": 0,
        "loss_sum": 0.0,
        "selection_correct": 0,
        "non_top1_targets": 0,
        "original": {threshold: 0 for threshold in THRESHOLDS},
        "oracle": {threshold: 0 for threshold in THRESHOLDS},
        "ranker": {threshold: 0 for threshold in THRESHOLDS},
        "rescued": {threshold: 0 for threshold in THRESHOLDS},
        "degraded": {threshold: 0 for threshold in THRESHOLDS},
    }


def build_targets(candidate_ious, args):
    best_candidate = candidate_ious.argmax(dim=1)
    if args.target_mode == "best-iou":
        return best_candidate
    top1_correct = candidate_ious[:, 0] > args.target_iou
    has_correct_candidate = (candidate_ious > args.target_iou).any(dim=1)
    should_correct = (~top1_correct) & has_correct_candidate
    return torch.where(should_correct, best_candidate, torch.zeros_like(best_candidate))


def update_metrics(metrics, loss, candidate_ious, rank_scores, targets):
    selected = rank_scores.argmax(dim=1)
    selected_ious = candidate_ious.gather(1, selected[:, None]).squeeze(1)
    original_ious = candidate_ious[:, 0]
    oracle_ious = candidate_ious.max(dim=1).values
    batch_size = candidate_ious.shape[0]
    metrics["count"] += batch_size
    metrics["loss_sum"] += float(loss.detach()) * batch_size
    metrics["selection_correct"] += int((selected == targets).sum())
    metrics["non_top1_targets"] += int((targets != 0).sum())
    for threshold in THRESHOLDS:
        original_hit = original_ious > threshold
        ranker_hit = selected_ious > threshold
        metrics["original"][threshold] += int(original_hit.sum())
        metrics["oracle"][threshold] += int((oracle_ious > threshold).sum())
        metrics["ranker"][threshold] += int(ranker_hit.sum())
        metrics["rescued"][threshold] += int(((~original_hit) & ranker_hit).sum())
        metrics["degraded"][threshold] += int((original_hit & (~ranker_hit)).sum())


def summarise(metrics):
    count = metrics["count"]
    summary = {
        "sample_count": count,
        "loss": metrics["loss_sum"] / count,
        "target_selection_accuracy": metrics["selection_correct"] / count,
        "non_top1_target_count": metrics["non_top1_targets"],
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        rescued = metrics["rescued"][threshold]
        degraded = metrics["degraded"][threshold]
        summary["thresholds"][str(threshold)] = {
            "original_top1_accuracy": metrics["original"][threshold] / count,
            "oracle_top5_accuracy": metrics["oracle"][threshold] / count,
            "ranker_top1_accuracy": metrics["ranker"][threshold] / count,
            "rescued_samples": rescued,
            "degraded_samples": degraded,
            "net_gain_samples": rescued - degraded,
        }
    return summary


def run_epoch(detector, ranker, loader, optimizer, anchors, args, device, epoch=None):
    is_train = optimizer is not None
    ranker.train(is_train)
    metrics = new_metrics()
    for batch_index, batch in enumerate(loader):
        query_features, candidate_features, scores, boxes, candidate_ious = frozen_candidates(
            detector, batch, anchors, args, device)
        rank_scores = ranker(query_features, candidate_features, scores, boxes, args.image_size)
        target_index = build_targets(candidate_ious, args)
        loss = F.cross_entropy(rank_scores, target_index)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        update_metrics(metrics, loss, candidate_ious, rank_scores, target_index)
        if batch_index % args.print_freq == 0:
            phase = "train" if is_train else "val"
            print("%s epoch=%s [%d/%d] loss=%.4f" % (
                phase, epoch if epoch is not None else "-", min((batch_index + 1) * args.batch_size, len(loader.dataset)),
                len(loader.dataset), float(loss.detach())), flush=True)
    return summarise(metrics)


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.residual_alpha < 0:
        raise ValueError("--residual-alpha must be non-negative")
    if args.score_eps <= 0:
        raise ValueError("--score-eps must be positive")
    if args.match_dim <= 0 or args.attention_heads <= 0 or args.attention_hidden_dim <= 0:
        raise ValueError("Cross-attention dimensions and head count must be positive")
    if args.match_dim % args.attention_heads != 0:
        raise ValueError("--match-dim must be divisible by --attention-heads")
    if args.candidate_count != 5:
        raise ValueError("The first-round baseline is intentionally fixed to NMS Top-5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen DetGeo feature extraction")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    (output_dir / "config.json").write_text(json.dumps(args_dict, indent=2), encoding="utf-8")

    train_loader = make_loader(args, "train", shuffle=True)
    val_loader = make_loader(args, "val", shuffle=False)
    detector = load_detector(args.checkpoint, device)
    if args.score_mode == "residual-cross-attention":
        ranker = ResidualCrossAttentionRanker(
            match_dim=args.match_dim, num_heads=args.attention_heads,
            hidden_dim=args.attention_hidden_dim, residual_alpha=args.residual_alpha).to(device)
    elif args.score_mode == "residual-zscore":
        ranker = NormalizedResidualCandidateRanker(
            hidden_dim=args.hidden_dim, residual_alpha=args.residual_alpha,
            score_eps=args.score_eps).to(device)
    elif args.score_mode == "residual":
        ranker = ResidualCandidateRanker(hidden_dim=args.hidden_dim).to(device)
    else:
        ranker = CandidateRanker(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    anchors = make_anchors(device)
    best_val_acc50 = float("-inf")
    history = []

    print("Frozen DetGeo parameters: %d" % sum(parameter.numel() for parameter in detector.parameters()))
    print("Trainable ranker parameters: %d" % sum(parameter.numel() for parameter in ranker.parameters()))
    print("Score mode: %s; target mode: %s@%.2f; residual alpha: %.3f" % (
        args.score_mode, args.target_mode, args.target_iou, args.residual_alpha), flush=True)

    if args.score_mode in ("residual", "residual-zscore", "residual-cross-attention"):
        with torch.no_grad():
            epoch0_val = run_epoch(detector, ranker, val_loader, None, anchors, args, device, epoch=0)
        for threshold in THRESHOLDS:
            values = epoch0_val["thresholds"][str(threshold)]
            if values["ranker_top1_accuracy"] != values["original_top1_accuracy"]:
                raise RuntimeError("Zero-initialized residual ranker does not reproduce the detector baseline")
        baseline_record = {"epoch": 0, "train": None, "val": epoch0_val}
        history.append(baseline_record)
        best_val_acc50 = epoch0_val["thresholds"]["0.5"]["ranker_top1_accuracy"]
        baseline_state = {"epoch": 0, "ranker_state_dict": ranker.state_dict(),
                          "optimizer": optimizer.state_dict(), "val_metrics": epoch0_val,
                          "args": args_dict}
        torch.save(baseline_state, output_dir / "ranker_model_best.pth.tar")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(baseline_record, indent=2), flush=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(detector, ranker, train_loader, optimizer, anchors, args, device, epoch)
        with torch.no_grad():
            val_metrics = run_epoch(detector, ranker, val_loader, None, anchors, args, device, epoch)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        acc50 = val_metrics["thresholds"]["0.5"]["ranker_top1_accuracy"]
        state = {"epoch": epoch, "ranker_state_dict": ranker.state_dict(),
                 "optimizer": optimizer.state_dict(), "val_metrics": val_metrics, "args": args_dict}
        torch.save(state, output_dir / "ranker_checkpoint.pth.tar")
        if acc50 > best_val_acc50:
            best_val_acc50 = acc50
            torch.save(state, output_dir / "ranker_model_best.pth.tar")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)
    print("Best validation Ranker Acc@0.5: %.4f" % best_val_acc50, flush=True)


if __name__ == "__main__":
    main()
