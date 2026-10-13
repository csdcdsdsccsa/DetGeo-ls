"""Train or evaluate the standalone, fixed Swin-T Full Model."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from fullmodel import FullModel
from fullmodel.data import CVOGLFullModelDataset
from fullmodel.objectives import anchors_tensor, decode_fullmodel, fullmodel_loss, per_sample_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'test'))
    parser.add_argument('--dataset', choices=('CVOGL_DroneAerial', 'CVOGL_SVI'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='validation-best checkpoint for test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=7)
    parser.add_argument('--num-workers', type=int, default=24)
    parser.add_argument('--max-epoch', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_epoch < 1 or args.lr <= 0:
        parser.error('batch-size/max-epoch/lr must be positive and num-workers nonnegative')
    if args.command == 'test' and args.checkpoint is None:
        parser.error('--checkpoint is required for test')
    return args


def seed_global_rng(seed):
    # Match the original train.py; no private DataLoader generator, worker
    # seeding callback, RNG restore, or post-construction reseed.
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataset_for(args, split):
    transform = Compose([ToTensor(), Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return CVOGLFullModelDataset(args.data_root, args.dataset, split,
                                 transform=transform, augment=split == 'train')


def make_model(dataset):
    if dataset == 'CVOGL_DroneAerial':
        return FullModel(core_sigma_y=25, core_sigma_x=25,
                         outer_sigma_y=50, outer_sigma_x=50)
    return FullModel(core_sigma_y=25, core_sigma_x=50,
                     outer_sigma_y=50, outer_sigma_x=100)


def unpack(batch, device):
    query, satellite, click, boxes, _ = batch
    return (query.to(device, non_blocking=True), satellite.to(device, non_blocking=True),
            click.to(device, non_blocking=True), boxes.to(device, non_blocking=True).clamp_(0, 1023))


def run_epoch(model, loader, anchors, device, optimizer=None, epoch=None):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(4, dtype=torch.float64)
    loss_total = 0.0
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        query, satellite, click, boxes = unpack(batch, device)
        with torch.set_grad_enabled(training):
            outputs, _ = model(query, satellite, click)
            loss = fullmodel_loss(outputs, boxes, anchors) if training else None
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                selected = decode_fullmodel(outputs, anchors)
                metrics = per_sample_metrics(selected, boxes)
                totals += metrics.sum(dim=0).double().cpu()
        n = query.shape[0]
        sample_count += n
        if loss is not None:
            loss_total += loss.item() * n
        if training and batch_index % 50 == 0:
            print('epoch={} batch={}/{} loss={:.4f} acc50={:.4f}'.format(
                epoch, batch_index, len(loader), loss.item(), (totals[0] / sample_count).item()), flush=True)
    if sample_count == 0:
        raise RuntimeError('empty split')
    values = (100.0 * totals / sample_count).tolist()
    result = dict(zip(('acc50', 'acc25', 'miou', 'center_acc'), values))
    if training:
        result['loss'] = loss_total / sample_count
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('training and full-resolution evaluation require CUDA')
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.set_device(device)
    seed_global_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / 'best.pth.tar'
    if args.command == 'train':
        for existing in (best_path, args.output_dir / 'metrics.jsonl'):
            if existing.exists():
                raise FileExistsError('refusing to mix with or overwrite {}'.format(existing))
        config = {name: str(value) if isinstance(value, Path) else value
                  for name, value in vars(args).items()}
        config['architecture'] = 'shared_swin_t_bi_guidance_amhcsfi_res_hisym_crgpe'
        config['rng_protocol'] = 'standard_rng_natural_consumption'
        (args.output_dir / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    loader_kwargs = dict(pin_memory=True, drop_last=False, num_workers=args.num_workers)
    if args.command == 'train':
        train_data = dataset_for(args, 'train')
        val_data = dataset_for(args, 'val')
        test_data = dataset_for(args, 'test')
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_data, batch_size=args.batch_size * 2,
                                shuffle=False, **loader_kwargs)
        # Source train.py also constructs the official test split before model
        # initialization; do not evaluate it for checkpoint selection.
        _test_loader = DataLoader(test_data, batch_size=args.batch_size * 2,
                                  shuffle=False, **loader_kwargs)
    else:
        test_loader = DataLoader(dataset_for(args, 'test'), batch_size=args.batch_size * 2,
                                 shuffle=False, **loader_kwargs)

    # DataParallel keeps source-compatible state_dict keys (module.*).
    model = torch.nn.DataParallel(make_model(args.dataset),
                                  device_ids=[args.gpu], output_device=args.gpu).to(device)
    anchors = anchors_tensor(device)
    if args.command == 'test':
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        result = run_epoch(model, test_loader, anchors, device)
        print('test {}'.format(json.dumps(result, ensure_ascii=False)), flush=True)
        (args.output_dir / 'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    best_acc50 = -float('inf')
    for epoch in range(args.max_epoch):
        lr = args.lr * (0.1 ** (epoch // 10))
        for group in optimizer.param_groups:
            group['lr'] = lr
        train_result = run_epoch(model, train_loader, anchors, device, optimizer, epoch)
        val_result = run_epoch(model, val_loader, anchors, device)
        print('epoch={} train={} val={}'.format(epoch, json.dumps(train_result),
                                              json.dumps(val_result)), flush=True)
        with (args.output_dir / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'epoch': epoch, 'train': train_result, 'val': val_result}) + '\n')
        if val_result['acc50'] > best_acc50:
            best_acc50 = val_result['acc50']
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                        'best_loss': best_acc50, 'optimizer': optimizer.state_dict()}, best_path)
            print('best epoch={} val_acc50={:.2f} checkpoint={}'.format(
                epoch + 1, best_acc50, best_path), flush=True)


if __name__ == '__main__':
    main()
