"""CPU unit and RNG-trajectory checks for HQS-v2a."""
import torch
import torch.nn.functional as F

from model.TROGeo_ms_detection_ablation import HeadPairwiseRanker, HeadQualitySelector


def main():
    torch.manual_seed(7)
    ranker = HeadPairwiseRanker()
    feat3 = torch.randn(4, 384, requires_grad=True)
    feat4 = torch.randn(4, 384, requires_grad=True)
    score3 = torch.randn(4, requires_grad=True)
    score4 = torch.randn(4, requires_grad=True)
    logit = ranker(feat3.detach(), feat4.detach(), score3.detach(), score4.detach())
    if logit.shape != (4,):
        raise RuntimeError('unexpected rank-logit shape {}'.format(tuple(logit.shape)))
    F.binary_cross_entropy_with_logits(logit, torch.tensor([1., 0., 1., 0.])).backward()
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in ranker.parameters()):
        raise RuntimeError('HQS-v2a ranker did not receive gradients')
    if any(value.grad is not None for value in (feat3, feat4, score3, score4)):
        raise RuntimeError('HQS-v2a inputs were not detached')

    torch.manual_seed(2024)
    _ = HeadQualitySelector()
    expected_post = torch.get_rng_state()
    torch.manual_seed(2024)
    pre = torch.get_rng_state()
    _padding = HeadQualitySelector()
    padded_post = torch.get_rng_state()
    del _padding
    torch.set_rng_state(pre)
    _ = HeadPairwiseRanker()
    torch.set_rng_state(padded_post)
    if not torch.equal(expected_post, torch.get_rng_state()):
        raise RuntimeError('HQS-v2a post-construction RNG does not match HQS-v1')
    print('HQS-v2a unit and RNG checks passed')


if __name__ == '__main__':
    main()
