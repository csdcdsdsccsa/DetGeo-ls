"""GPU forward/backward smoke test for the actual VIGOR Full Model."""

import numpy as np
import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.multiscale_detection_loss import two_head_yolo_loss


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('this smoke test requires CUDA')
    torch.manual_seed(13)
    device = torch.device('cuda:0')
    anchors_str = ('137,82, 144,164, 479,243, 255,537, 73,202, '
                   '242,117, 175,359, 259,260, 74,108')
    anchors = torch.tensor(np.asarray([float(value) for value in anchors_str.split(',')], dtype=np.float32)
                           .reshape(-1, 2)[::-1].copy(), device=device)
    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe', gaussian_sigma=25, gaussian_sigma_x=50,
        crgpe_outer_sigma=50, crgpe_outer_sigma_x=100).to(device).train()
    query = torch.randn(1, 3, 256, 512, device=device)
    satellite = torch.randn(1, 3, 640, 640, device=device)
    click = torch.zeros(1, 256, 512, device=device)
    click[:, 128, 256] = 1.0
    boxes = torch.tensor([[160., 160., 400., 420.]], device=device)
    predictions, _ = model(query, satellite, click)
    assert tuple(predictions['stage3'].shape) == (1, 45, 40, 40)
    assert tuple(predictions['stage4'].shape) == (1, 45, 40, 40)
    p3 = predictions['stage3'].view(1, 9, 5, 40, 40)
    p4 = predictions['stage4'].view(1, 9, 5, 40, 40)
    geo, cls = two_head_yolo_loss(p3, p4, boxes, anchors, 640)
    (geo + cls).backward()
    print('VIGOR Full Model GPU smoke passed: stage3/stage4=[1,45,40,40]; backward passed')


if __name__ == '__main__':
    main()
