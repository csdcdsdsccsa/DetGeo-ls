"""CPU sanity checks for the frozen Bi-Res ACR refinement head."""

import torch

from model.acr_head import ACRRefinementHead


def main():
    torch.manual_seed(7)
    head = ACRRefinementHead(channels=384, hidden_dim=128, roi_size=5, num_heads=4).eval()
    assert torch.count_nonzero(head.alpha_head.weight) == 0
    assert torch.count_nonzero(head.alpha_head.bias) == 0
    assert torch.count_nonzero(head.box_head.weight) == 0
    assert torch.count_nonzero(head.box_head.bias) == 0
    feature3, feature4 = torch.randn(2, 384, 64, 64), torch.randn(2, 384, 64, 64)
    box3 = torch.tensor([[100., 120., 380., 440.], [50., 60., 300., 420.]])
    box4 = torch.tensor([[110., 125., 390., 450.], [70., 55., 310., 410.]])
    score3, score4 = torch.tensor([0.4, 0.7]), torch.tensor([0.6, 0.3])
    state = head(feature3, feature4, box3, box4, score3, score4, torch.tensor([0.6, 0.8]), 1024)
    expected = (score3[:, None] * box3 + score4[:, None] * box4) / (score3 + score4)[:, None]
    error = (state['refined_box'] - expected).abs().max().item()
    assert error < 1e-5, 'zero-init ACR must reproduce agreement fusion, max error={:.8f}'.format(error)
    state['refined_box'].sum().backward()
    assert any(parameter.grad is not None for parameter in head.parameters())
    print('ACR sanity passed: zero-init agreement fusion max_abs_diff={:.8f}'.format(error))


if __name__ == '__main__':
    main()
