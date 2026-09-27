"""Per-sample CVOGL-Seg metrics; no training dependencies."""
from pathlib import Path
import numpy as np


def resolve_seg_mask_dir(seg_root, data_name):
    candidates = [Path(seg_root) / data_name / 'mask-satellite',
                  Path(seg_root) / 'CVOGL-Seg' / data_name / 'mask-satellite']
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(str(candidates))


def mask_filename(record):
    bbox = np.asarray(record[5], dtype=np.int32)
    return '{}--bbox({},{},{},{}).jpg'.format(Path(record[2]).stem, *bbox)


def get_meters_per_pixel(satellite_name, image_size=1024):
    parts = Path(satellite_name).stem.split('_')
    if len(parts) < 4:
        raise ValueError(satellite_name)
    value = float(parts[-2]) / image_size
    if not np.isfinite(value) or value <= 0:
        raise ValueError(satellite_name)
    return value


def bbox_to_mask(bbox, height=1024, width=1024):
    bbox = np.asarray(bbox)
    if not np.isfinite(bbox).all():
        raise ValueError('nonfinite bbox')
    x1, y1 = np.floor(bbox[:2]).astype(int)
    x2, y2 = np.ceil(bbox[2:]).astype(int)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def mask_centroid(mask):
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else (0., 0.)


def segmentation_metrics(pred, gt, meters_per_pixel):
    if pred.shape != gt.shape:
        raise ValueError('mask shapes differ')
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    p, g = int(pred.sum()), int(gt.sum())
    return dict(iou=float(intersection / max(union, 1)),
                dice=float(2 * intersection / max(p + g, 1)),
                aae_m2=abs(p - g) * meters_per_pixel ** 2,
                me_m=float(np.linalg.norm(np.subtract(mask_centroid(pred), mask_centroid(gt))) * meters_per_pixel))
