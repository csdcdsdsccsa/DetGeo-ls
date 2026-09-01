"""Train and validate the minimal E05 coarse-to-fine anchor-free model."""

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
from model.oacfr_geo import OACFRGeoE05
from model.oacfr_loss import E05Loss
from utils.oacfr_utils import localization_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="E05 coarse-to-fine anchor-free experiment")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-name", default="CVOGL_DroneAerial")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--search-scales", default="32,64,128")
    parser.add_argument("--match-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--search-weight", type=float, default=1.0)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--quality-weight", type=float, default=1.0)
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
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = RSDataset(
        data_root=args.data_root,
        data_name=args.data_name,
        split_name=split,
        img_size=args.img_size,
        transform=transform,
        augment=(split == "train"),
    )
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        pin_memory=True, drop_last=False, num_workers=args.num_workers)


def load_detgeo_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["state_dict"]
    target = model.state_dict()
    matched = {
        key: value for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    model.load_state_dict(matched, strict=False)
    return len(matched), len(source)


def empty_totals():
    return {
        "samples": 0,
        "loss": 0.0,
        "search_loss": 0.0,
        "center_loss": 0.0,
        "box_loss": 0.0,
        "quality_loss": 0.0,
        "correct_025": 0,
        "correct_050": 0,
        "center_correct": 0,
        "iou_sum": 0.0,
    }


def summarize(totals):
    count = max(totals["samples"], 1)
    return {
        "sample_count": totals["samples"],
        "loss": totals["loss"] / count,
        "search_loss": totals["search_loss"] / count,
        "center_loss": totals["center_loss"] / count,
        "box_loss": totals["box_loss"] / count,
        "quality_loss": totals["quality_loss"] / count,
        "acc_025": totals["correct_025"] / count,
        "acc_050": totals["correct_050"] / count,
        "mean_iou": totals["iou_sum"] / count,
        "center_accuracy": totals["center_correct"] / count,
    }


def run_epoch(model, loader, criterion, device, optimizer=None, epoch=None, print_freq=100):
    training = optimizer is not None
    model.train(training)
    totals = empty_totals()
    phase = "train" if training else "val"

    for batch_index, (query_imgs, reference_imgs, click_map, boxes, _) in enumerate(loader):
        query_imgs = query_imgs.to(device, non_blocking=True)
        reference_imgs = reference_imgs.to(device, non_blocking=True)
        click_map = click_map.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True).clamp(0, loader.dataset.img_size - 1)

        with torch.set_grad_enabled(training):
            outputs = model(query_imgs, reference_imgs, click_map)
            losses = criterion(outputs, boxes)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0)
                optimizer.step()

        with torch.no_grad():
            metrics = localization_metrics(outputs, boxes, loader.dataset.img_size)
        batch_size = query_imgs.shape[0]
        totals["samples"] += batch_size
        totals["loss"] += float(losses["total"].detach()) * batch_size
        totals["search_loss"] += float(losses["search"].detach()) * batch_size
        totals["center_loss"] += float(losses["center"].detach()) * batch_size
        totals["box_loss"] += float(losses["box"].detach()) * batch_size
        totals["quality_loss"] += float(losses["quality"].detach()) * batch_size
        totals["correct_025"] += int(metrics["correct_025"].sum())
        totals["correct_050"] += int(metrics["correct_050"].sum())
        totals["center_correct"] += int(metrics["center_correct"].sum())
        totals["iou_sum"] += float(metrics["iou"].sum())

        if batch_index % print_freq == 0:
            print("%s epoch=%s [%d/%d] loss=%.4f" % (
                phase, epoch if epoch is not None else "-",
                min((batch_index + 1) * loader.batch_size, len(loader.dataset)),
                len(loader.dataset), float(losses["total"].detach())), flush=True)
    return summarize(totals)


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    scales = tuple(int(value.strip()) for value in args.search_scales.split(",") if value.strip())
    if scales not in ((64,), (32, 64, 128)):
        raise ValueError("E05 comparison supports --search-scales 64 or 32,64,128")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E05")
    set_seed(args.seed)
    device = torch.device("cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["parsed_search_scales"] = scales
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_loader = make_loader(args, "train", True)
    val_loader = make_loader(args, "val", False)
    model = OACFRGeoE05(
        match_dim=args.match_dim,
        search_scales=scales,
        temperature=args.temperature,
        freeze_backbone=True,
    )
    model = torch.nn.DataParallel(model).to(device)
    loaded, source_count = load_detgeo_checkpoint(model, args.checkpoint)
    model.module.freeze_backbone(True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = E05Loss(
        image_size=args.img_size,
        search_weight=args.search_weight,
        center_weight=args.center_weight,
        box_weight=args.box_weight,
        quality_weight=args.quality_weight,
    )
    print("Loaded DetGeo tensors: %d/%d" % (loaded, source_count), flush=True)
    print("Frozen backbone parameters: %d" % sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad), flush=True)
    print("Trainable E05 parameters: %d" % sum(
        parameter.numel() for parameter in trainable), flush=True)
    print("Search scales: %s" % (scales,), flush=True)

    history = []
    best_acc_050 = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, epoch, args.print_freq)
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, criterion, device, None, epoch, args.print_freq)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc_050": max(best_acc_050, val_metrics["acc_050"]),
            "search_scales": scales,
            "config": config,
        }
        checkpoint_path = output_dir / "checkpoint.pth.tar"
        torch.save(checkpoint, checkpoint_path)
        if val_metrics["acc_050"] > best_acc_050:
            best_acc_050 = val_metrics["acc_050"]
            shutil.copyfile(checkpoint_path, output_dir / "model_best.pth.tar")

    print("Best validation Acc@0.5: %.6f" % best_acc_050, flush=True)


if __name__ == "__main__":
    main()
