"""Small CPU forward-equivalence smoke; full-resolution training still needs CUDA."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('SOURCE_REPO', ROOT.parent / 'DetGeo-sam-prompt-yolo'))
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(SOURCE))

from fullmodel import FullModel
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def main():
    original_swin_t = models.swin_t

    def no_download(*args, **kwargs):
        kwargs['weights'] = None
        return original_swin_t(*args, **kwargs)

    for dataset, query_width, sigma_x, outer_x in (
        ('DroneAerial', 128, 25, 50), ('SVI', 256, 50, 100),
    ):
        with patch.object(models, 'swin_t', side_effect=no_download):
            torch.manual_seed(2024)
            original = TROGeoMSDetectionAblation(
                emb_size=768, backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
                position_mode='hisym_crgpe', gaussian_sigma=25,
                gaussian_sigma_x=sigma_x, crgpe_outer_sigma=50,
                crgpe_outer_sigma_x=outer_x)
            torch.manual_seed(2024)
            standalone = FullModel(25, sigma_x, 50, outer_x)
        original.eval()
        standalone.eval()
        torch.manual_seed(42)
        query = torch.randn(1, 3, 128, query_width)
        satellite = torch.randn(1, 3, 256, 256)
        yy = torch.arange(128, dtype=torch.float32)[:, None]
        xx = torch.arange(query_width, dtype=torch.float32)[None, :]
        click = torch.exp(-((yy - 30).square() / (2 * 25 ** 2) +
                            (xx - 20).square() / (2 * sigma_x ** 2)))[None]
        with torch.inference_mode():
            old_outputs, _ = original(query, satellite, click)
            new_outputs, _ = standalone(query, satellite, click)
        assert old_outputs.keys() == new_outputs.keys()
        for name in old_outputs:
            torch.testing.assert_close(new_outputs[name], old_outputs[name], rtol=0, atol=0)
        print('{}: source and standalone forward outputs are bitwise identical'.format(dataset))


if __name__ == '__main__':
    main()
