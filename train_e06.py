"""Train E06 while proving its epoch-0 output equals frozen DetGeo."""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from dataset.data_loader import RSDataset
from model.DetGeo import DetGeo
from model.e06_geo import E06ResidualMultiScaleDetGeo
from model.loss import build_target, yolo_loss
from utils.utils import eval_iou_acc


ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"


def parse_args():
    parser = argparse.ArgumentParser(description="E06 baseline-preserving residual multi-scale experiment")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-name", default="CVOGL_DroneAerial")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--print-freq", type=int, default=100)
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
        ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = RSDataset(args.data_root, args.data_name, split, args.img_size,
                        transform=transform, augment=(split == "train"))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      pin_memory=True, drop_last=False, num_workers=args.num_workers)


def load_matching(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["state_dict"]
    target = model.state_dict()
    matched = {key: value for key, value in source.items()
               if key in target and target[key].shape == value.shape}
    model.load_state_dict(matched, strict=False)
    return checkpoint, len(matched), len(source)


def anchors_for(device):
    anchors = np.array([float(value.strip()) for value in ANCHORS.split(",")])
    return torch.tensor(anchors.reshape(-1, 2)[::-1].copy(), dtype=torch.float32, device=device)


def metrics(prediction, boxes, anchors, image_size):
    reshaped = prediction.view(prediction.shape[0], 9, 5, prediction.shape[-2], prediction.shape[-1])
    _, best = build_target(boxes, anchors, image_size, reshaped.shape[-1])
    accuracy, center_accuracy, mean_iou, each_accuracy, _, _ = eval_iou_acc(
        reshaped, boxes, anchors, best[:, 1], best[:, 2], image_size,
        iou_threshold_list=[0.5, 0.25])
    return reshaped, {
        "correct_050": int(each_accuracy[0].sum()),
        "correct_025": int(each_accuracy[1].sum()),
        "center_correct": int(round(float(center_accuracy) * boxes.shape[0])),
        "iou_sum": float(mean_iou.detach()) * boxes.shape[0],
        "acc_050": float(accuracy[0]),
    }, best


def empty_totals():
    return {"samples": 0, "loss": 0.0, "geo_loss": 0.0, "cls_loss": 0.0,
            "correct_050": 0, "correct_025": 0, "center_correct": 0, "iou_sum": 0.0}


def summarize(totals):
    count = max(totals["samples"], 1)
    return {"sample_count": totals["samples"], "loss": totals["loss"] / count,
            "geo_loss": totals["geo_loss"] / count, "cls_loss": totals["cls_loss"] / count,
            "acc_050": totals["correct_050"] / count,
            "acc_025": totals["correct_025"] / count,
            "mean_iou": totals["iou_sum"] / count,
            "center_accuracy": totals["center_correct"] / count}


def run_epoch(model, loader, anchors, args, device, optimizer=None, epoch=0):
    training = optimizer is not None
    model.train(training)
    totals = empty_totals()
    for index, (query, reference, click_map, boxes, _) in enumerate(loader, 1):
        query = query.to(device, non_blocking=True)
        reference = reference.to(device, non_blocking=True)
        click_map = click_map.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True).clamp(0, args.img_size - 1)
        with torch.set_grad_enabled(training):
            prediction, _ = model(query, reference, click_map)
            reshaped, batch_metrics, best = metrics(prediction, boxes, anchors, args.img_size)
            geo_loss, cls_loss = yolo_loss(reshaped, build_target(
                boxes, anchors, args.img_size, reshaped.shape[-1])[0], anchors, best, args.img_size)
            loss = cls_loss + args.beta * geo_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0)
                optimizer.step()
        batch_size = boxes.shape[0]
        totals["samples"] += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["geo_loss"] += float(geo_loss.detach()) * batch_size
        totals["cls_loss"] += float(cls_loss.detach()) * batch_size
        for key in ("correct_050", "correct_025", "center_correct", "iou_sum"):
            totals[key] += batch_metrics[key]
        if index % args.print_freq == 0 or index == len(loader):
            print("%s epoch=%d %d/%d loss=%.4f alpha=%.6f" % (
                "train" if training else "val", epoch, index, len(loader), float(loss.detach()),
                float(model.module.residual_scale.detach())), flush=True)
    return summarize(totals)


def verify_zero_init(model, checkpoint, loader, args, device, output_dir):
    baseline = torch.nn.DataParallel(DetGeo()).to(device)
    baseline.load_state_dict(checkpoint["state_dict"], strict=True)
    baseline.eval()
    model.eval()
    maximum_error = 0.0
    equal_logit_samples = 0
    sample_count = 0
    with (output_dir / "zero_init_per_sample.jsonl").open("w", encoding="utf-8") as handle, torch.no_grad():
        for query, reference, click_map, _, sample_ids in loader:
            query, reference = query.to(device), reference.to(device)
            click_map = click_map.to(device)
            baseline_output, _ = baseline(query, reference, click_map)
            e06_output, _ = model(query, reference, click_map)
            difference = (baseline_output - e06_output).abs()
            maximum_error = max(maximum_error, float(difference.max()))
            per_sample_equal = difference.flatten(1).eq(0).all(dim=1)
            equal_logit_samples += int(per_sample_equal.sum())
            sample_count += query.shape[0]
            for sample_id, equal in zip(sample_ids.tolist(), per_sample_equal.tolist()):
                handle.write(json.dumps({"sample_id": int(sample_id), "logits_equal": bool(equal)}) + "\n")
    result = {"sample_count": sample_count, "logits_equal_samples": equal_logit_samples,
              "max_abs_logit_difference": maximum_error,
              "residual_scale": float(model.module.residual_scale.detach())}
    if equal_logit_samples != sample_count or maximum_error != 0.0:
        raise RuntimeError("E06 initialization does not exactly reproduce DetGeo: %s" % result)
    return result


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E06")
    set_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(args, "train", True)
    val_loader = make_loader(args, "val", False)
    model = torch.nn.DataParallel(E06ResidualMultiScaleDetGeo(freeze_baseline=True)).to(device)
    checkpoint, loaded, source_count = load_matching(model, args.checkpoint)
    model.module.freeze_baseline(True)
    config = vars(args).copy()
    config.update({"loaded_detgeo_tensors": "%d/%d" % (loaded, source_count),
                   "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                               if parameter.requires_grad)})
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Loaded DetGeo tensors: %d/%d" % (loaded, source_count), flush=True)
    zero_init = verify_zero_init(model, checkpoint, val_loader, args, device, output_dir)
    (output_dir / "zero_init.json").write_text(json.dumps(zero_init, indent=2), encoding="utf-8")
    print("Zero-init equality: %s" % json.dumps(zero_init), flush=True)
    anchors = anchors_for(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    history, best_acc = [], float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, anchors, args, device, optimizer, epoch)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, anchors, args, device, None, epoch)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
                  "residual_scale": float(model.module.residual_scale.detach())}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)
        state = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "best_acc_050": max(best_acc, val_metrics["acc_050"]), "config": config}
        torch.save(state, output_dir / "checkpoint.pth.tar")
        if val_metrics["acc_050"] > best_acc:
            best_acc = val_metrics["acc_050"]
            shutil.copyfile(output_dir / "checkpoint.pth.tar", output_dir / "model_best.pth.tar")
    print("Best validation Acc@0.5: %.6f" % best_acc, flush=True)


if __name__ == "__main__":
    main()
