"""CPU checks for HQS-v2b's quality-aware ranker."""

import torch

from model.TROGeo_ms_detection_ablation import HeadPredictionQualityRanker


def main():
    torch.manual_seed(7)
    ranker = HeadPredictionQualityRanker()
    feat3 = torch.randn(4, 384, requires_grad=True)
    feat4 = torch.randn(4, 384, requires_grad=True)
    stats = torch.rand(4, 7, requires_grad=True)
    logit = ranker(feat3.detach(), feat4.detach(), stats.detach())
    assert logit.shape == (4,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor([1., 0., 1., 0.]))
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in ranker.parameters())
    assert feat3.grad is None and feat4.grad is None and stats.grad is None
    try:
        ranker(feat3.detach(), feat4.detach(), torch.rand(4, 6))
    except RuntimeError:
        pass
    else:
        raise AssertionError('HQS-v2b accepted invalid quality-statistic shape')
    print('HQS-v2b unit checks passed')


if __name__ == '__main__':
    main()
