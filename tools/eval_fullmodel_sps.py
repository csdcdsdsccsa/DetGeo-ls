"""Frozen Full Model + official CVOS bbox/center-point SPS, never trains."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--seg_root', default='data-Seg')
    parser.add_argument('--data_name', required=True, choices=['CVOGL_DroneAerial', 'CVOGL_SVI'])
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sam_checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output_dir', default='results/fullmodel_sps')
    parser.add_argument('--save_masks', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0, help='smoke only; 0 evaluates full split')
    parser.add_argument('--preflight_only', action='store_true')
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error('--max_samples must be nonnegative')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import Compose, ToTensor, Normalize
    from dataset.trogeo_loader import TROGeoRSDataset
    from utils.sps_metrics import (resolve_seg_mask_dir, mask_filename, bbox_to_mask,
                                   get_meters_per_pixel, segmentation_metrics)
    from utils.multiscale_detection import select_two_heads, eval_decoded_boxes
    svi = args.data_name == 'CVOGL_SVI'
    dataset = TROGeoRSDataset(args.data_root, args.data_name, args.split,
        img_size=1024, transform=Compose([ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225])]),
        augment=False, aug_mode='current', click_map_mode='gaussian',
        gaussian_sigma=25., gaussian_sigma_x=50. if svi else None)
    mask_dir = resolve_seg_mask_dir(args.seg_root, args.data_name)
    if not len(dataset):
        raise ValueError('empty split')
    # Audit ALL records, even for smoke. Never silently skip missing masks.
    for record in dataset.data_list:
        path = mask_dir / mask_filename(record)
        gt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(path)
        if gt.shape != (1024, 1024):
            raise ValueError('{}: {}'.format(path, gt.shape))
        get_meters_per_pixel(record[2])
    print('Mask preflight: {} / {} records OK; {}'.format(len(dataset), len(dataset), mask_dir), flush=True)
    for path in (args.checkpoint, args.sam_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if args.preflight_only:
        return
    sys.path.insert(0, str(ROOT / 'third_party' / 'cvos_sps'))
    from segment_anything import sam_model_registry
    from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(emb_size=768, backbone='swin_t',
        variant='h2_ind_amhcsfi_res_bi', position_mode='hisym_crgpe', dadpe_mode='none', amr_pe_mode='none',
        gaussian_sigma=25., gaussian_sigma_x=50. if svi else None,
        crgpe_outer_sigma=50., crgpe_outer_sigma_x=100. if svi else None)).cuda()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    del ckpt
    model.eval().requires_grad_(False)
    sam = sam_model_registry['vit_h'](checkpoint=args.sam_checkpoint).cuda().eval().requires_grad_(False)
    anchors = torch.tensor(np.array([37,41,78,84,96,215,129,129,194,82,198,179,246,280,395,342,550,573],
                                   dtype=np.float32).reshape(-1, 2)[::-1].copy(), device='cuda')
    tag = 'svi' if svi else 'drone'
    out = Path(args.output_dir)
    if args.max_samples:
        out = out / 'smoke'
    out.mkdir(parents=True, exist_ok=True)
    rows, localization = [], []
    count = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)
    with torch.inference_mode():
        for index in range(count):
            query, satellite, click, bbox, sample_index = dataset[index]
            query, satellite = query.unsqueeze(0).cuda(), satellite.unsqueeze(0).cuda()
            click = torch.as_tensor(click).unsqueeze(0).cuda()
            predictions, _ = model(query, satellite, click)
            for stage in ['stage3', 'stage4']:
                assert predictions[stage].shape == (1, 45, 64, 64)
            final_box, _ = select_two_heads(predictions['stage3'].view(1,9,5,64,64),
                predictions['stage4'].view(1,9,5,64,64), anchors, 1024, fusion=False)
            localization.append([float(x) for x in eval_decoded_boxes(final_box,
                torch.as_tensor(bbox).unsqueeze(0).cuda(), 1024, grid_size=64)])
            points = ((final_box[:, :2] + final_box[:, 2:]) / 2).unsqueeze(1)
            sparse, dense = sam.prompt_encoder(points=(points, torch.ones((1,1), dtype=torch.int64, device='cuda')),
                                               boxes=final_box, masks=None)
            low, quality = sam.mask_decoder(image_embeddings=sam.image_encoder(satellite),
                image_pe=sam.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense, multimask_output=False)
            masks = F.interpolate(low, scale_factor=4, mode='bilinear', align_corners=False)
            assert masks.shape == (1,1,1024,1024)
            assert torch.isfinite(masks).all() and torch.isfinite(quality).all()
            pred_mask = (masks[0,0] > .5).cpu().numpy().astype(np.uint8)
            record = dataset.data_list[sample_index]
            gt = (cv2.imread(str(mask_dir / mask_filename(record)), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            box = final_box[0].cpu().numpy()
            mpp = get_meters_per_pixel(record[2])
            row = dict(sample_index=sample_index, query_name=record[1], satellite_name=record[2])
            row.update(dict(zip(['gt_x1','gt_y1','gt_x2','gt_y2'], np.asarray(record[5]).tolist())))
            row.update(dict(zip(['pred_x1','pred_y1','pred_x2','pred_y2'], box.tolist())))
            for prefix, mask in [('bbox', bbox_to_mask(box)), ('sps', pred_mask)]:
                row.update({prefix + '_' + key: value for key, value in segmentation_metrics(mask, gt, mpp).items()})
            row.update(sam_predicted_iou=float(quality.item()), pred_mask_pixels=int(pred_mask.sum()),
                       gt_mask_pixels=int(gt.sum()), empty_sps_mask=not bool(pred_mask.any()), meters_per_pixel=mpp)
            rows.append(row)
            if args.save_masks:
                dest = out / 'masks' / tag / args.split
                dest.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest / '{}--{}--{}.png'.format(index, Path(record[1]).stem, Path(record[2]).stem)), pred_mask * 255)
            if index % 50 == 0:
                print('{}/{}'.format(index + 1, count), flush=True)
    summary = dict(dataset=args.data_name, split=args.split, count=count, full_split=count == len(dataset),
                   checkpoint=args.checkpoint, sam_checkpoint=args.sam_checkpoint,
                   protocol='CVOS image_encoder(ImageNet-normalized satellite); bbox + center positive point; logits > 0.5',
                   data_root=str(Path(args.data_root).resolve()), mask_dir=str(mask_dir.resolve()),
                   localization=dict(zip(['Acc@0.50','Acc@0.25','Mean IoU','Center Acc'], (np.mean(localization, axis=0)*100).tolist())))
    for prefix in ['bbox', 'sps']:
        summary[prefix] = {key: float(np.mean([r[prefix+'_'+key] for r in rows])) * (100 if key in ['iou','dice'] else 1)
                          for key in ['iou','dice','aae_m2','me_m']}
    if not all(np.isfinite(float(r[k])) for r in rows for k in r if k not in ['query_name','satellite_name']):
        raise ValueError('nonfinite sample metric')
    with (out / '{}_{}_samples.csv'.format(tag,args.split)).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / '{}_{}_summary.json'.format(tag,args.split)).write_text(json.dumps(summary, indent=2))
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset','split','method','mIoU','mDice','AAE_m2','ME_m'])
        for path in sorted(out.glob('*_summary.json')):
            data = json.loads(path.read_text())
            for prefix, method in [('bbox','Full Model'),('sps','Full Model + SPS')]:
                writer.writerow([data['dataset'],data['split'],method,*[data[prefix][key] for key in ['iou','dice','aae_m2','me_m']]])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
