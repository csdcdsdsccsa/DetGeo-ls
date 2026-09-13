"""CPU checks for HQS-v1's isolated selector behavior."""
import torch

from model.TROGeo_ms_detection_ablation import HeadQualitySelector


def main():
    torch.manual_seed(7)
    selector = HeadQualitySelector()
    feat3 = torch.randn(3, 384, requires_grad=True)
    feat4 = torch.randn(3, 384, requires_grad=True)
    score3 = torch.randn(3, requires_grad=True)
    score4 = torch.randn(3, requires_grad=True)
    logits = selector(feat3.detach(), feat4.detach(), score3.detach(), score4.detach())
    if logits.shape != (3, 2):
        raise RuntimeError('unexpected HQS logits shape {}'.format(tuple(logits.shape)))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in selector.parameters()):
        raise RuntimeError('HQS did not receive gradients')
    if any(value.grad is not None for value in (feat3, feat4, score3, score4)):
        raise RuntimeError('HQS inputs were not detached')
    print('HQS-v1 unit checks passed')


if __name__ == '__main__':
    main()
