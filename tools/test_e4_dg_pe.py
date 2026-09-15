"""Focused DG/RDG-PE checks: exact zero residual, geometry, and gradients."""

import argparse
import torch

from model.dg_position_embedding import DGPositionEmbedding, RDGPositionEmbedding, recover_gaussian_geometry


def gaussian(batch, height, width, sigma, device):
    rows = torch.arange(height, device=device).view(1, height, 1).float()
    cols = torch.arange(width, device=device).view(1, 1, width).float()
    cx = torch.tensor([width // 3 + i for i in range(batch)], device=device).view(batch, 1, 1).float()
    cy = torch.tensor([height // 2 - i for i in range(batch)], device=device).view(batch, 1, 1).float()
    return torch.exp(-((cols - cx).square() + (rows - cy).square()) / (2.0 * sigma * sigma))


def check(module_type, device):
    sigma = 25.0
    module = module_type(gaussian_sigma=sigma).to(device).train()
    g = gaussian(2, 96, 80, sigma, device)
    inputs = torch.cat((torch.randn(2, 3, 96, 80, device=device), g.unsqueeze(1)), dim=1)
    base = module.base_encoder(inputs)
    output = module(inputs)
    error = (output - base).abs().max().item()
    assert output.shape == (2, 3, 96, 80), output.shape
    assert error == 0.0, 'zero-init residual must be exact, got {}'.format(error)
    geometry = recover_gaussian_geometry(g, sigma)
    assert all(value.shape == (2, 96, 80) and torch.isfinite(value).all() for value in geometry)
    loss = module(inputs).square().mean()
    loss.backward()
    grad = module.output_projection.weight.grad
    assert grad is not None and grad.abs().sum().item() > 0.0
    print('{}: shape={} identity_error={:.8f} final_projection_grad={:.6f}'.format(
        module_type.__name__, tuple(output.shape), error, grad.abs().mean().item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    check(DGPositionEmbedding, device)
    check(RDGPositionEmbedding, device)
