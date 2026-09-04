"""Generate resumable three-candidate point-prompted SAM masks."""
import argparse
import os

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--data-name', default='CVOGL_DroneAerial')
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--sam-model-type', default='vit_b', choices=('vit_b', 'vit_l', 'vit_h'))
    parser.add_argument('--splits', nargs='+', default=('train', 'val', 'test'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    data_dir, query_dir = os.path.join(args.data_root, args.data_name), os.path.join(args.data_root, args.data_name, 'query')
    predictor = SamPredictor(sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(args.device))
    for split in args.splits:
        samples = torch.load(os.path.join(data_dir, '{}_{}.pth'.format(args.data_name, split)), map_location='cpu')
        output_dir = os.path.join(data_dir, 'sam_multimask', split)
        os.makedirs(output_dir, exist_ok=True)
        made = skipped = 0
        for index, sample in enumerate(samples):
            output = os.path.join(output_dir, '{:06d}.npz'.format(index))
            if os.path.isfile(output) and not args.overwrite:
                skipped += 1; continue
            _, query_name, _, _, click_xy, _, _, _ = sample
            image = cv2.imread(os.path.join(query_dir, query_name), cv2.IMREAD_COLOR)
            if image is None: raise FileNotFoundError(query_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            click = np.asarray(click_xy, dtype=np.float32)
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(point_coords=click.reshape(1, 2), point_labels=np.asarray((1,), dtype=np.int32), multimask_output=True)
            np.savez_compressed(output, masks=masks.astype(np.uint8), scores=scores.astype(np.float32))
            made += 1
            if made % 100 == 0: print('{}: {}/{}'.format(split, made, len(samples)), flush=True)
        print('{}: generated={}, skipped={}, total={}'.format(split, made, skipped, len(samples)), flush=True)


if __name__ == '__main__':
    main()
