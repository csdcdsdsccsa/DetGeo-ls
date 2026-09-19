"""CPU regression checks for TROGeo distance and Gaussian click maps."""
import math
import numpy as np
import torch
from dataset.trogeo_loader import TROGeoRSDataset
from model.detgeo_position_embedding import DetGeoPositionEmbedding


def main():
    height = width = 256
    center_h, center_w = 128, 128
    distance = TROGeoRSDataset.make_click_map((height, width), center_h, center_w, 'distance', 25.0)
    gaussian = TROGeoRSDataset.make_click_map((height, width), center_h, center_w, 'gaussian', 25.0)
    if distance.shape != gaussian.shape or gaussian.shape != (height, width):
        raise RuntimeError('click-map shape mismatch')
    if distance.dtype != np.float32 or gaussian.dtype != np.float32:
        raise RuntimeError('click maps must be float32')
    if gaussian[center_h, center_w] != 1.0 or not np.all((gaussian >= 0.0) & (gaussian <= 1.0)):
        raise RuntimeError('invalid Gaussian range or center value')
    if not np.isclose(gaussian[center_h, center_w + 25], math.exp(-0.5), atol=1e-6):
        raise RuntimeError('Gaussian sigma=25 value at radius 25 is incorrect')
    if not np.allclose(distance, TROGeoRSDataset.make_click_map((height, width), center_h, center_w), atol=0.0):
        raise RuntimeError('default distance map changed')
    svi = TROGeoRSDataset.make_click_map((256, 512), 128, 256, 'gaussian', 25.0, 50.0)
    if not np.isclose(svi[128, 256], 1.0, atol=1e-6):
        raise RuntimeError('SVI anisotropic Gaussian center is incorrect')
    if not np.isclose(svi[153, 256], math.exp(-0.5), atol=1e-6):
        raise RuntimeError('SVI anisotropic Gaussian sigma_y=25 check failed')
    if not np.isclose(svi[128, 306], math.exp(-0.5), atol=1e-6):
        raise RuntimeError('SVI anisotropic Gaussian sigma_x=50 check failed')
    front_end = DetGeoPositionEmbedding().eval()
    with torch.no_grad():
        output = front_end(torch.from_numpy(gaussian).unsqueeze(0).unsqueeze(0).repeat(2, 4, 1, 1))
    if tuple(output.shape) != (2, 3, height, width):
        raise RuntimeError('DetGeo position front end must map 4 channels to 3 channels')
    print('TROGeo click-map sanity passed: isotropic_r25={:.6f} svi_y25={:.6f} svi_x50={:.6f}'.format(
        gaussian[center_h, center_w + 25], svi[153, 256], svi[128, 306]), flush=True)


if __name__ == '__main__':
    main()
