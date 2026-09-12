"""Unit tests for training-only RCCD confidence consensus."""

import torch

from model.multiscale_detection_loss import rccd_consensus_loss


def test_identical_predictions_zero_loss():
    torch.manual_seed(1)
    pred = torch.randn(2, 9, 5, 64, 64)
    loss, diagnostics = rccd_consensus_loss(pred, pred.clone(), temperature=2.0, weight_temperature=0.2)
    assert torch.isfinite(loss) and abs(loss.item()) < 1e-5, loss.item()
    assert abs(diagnostics['rccd_weight3'].item() - diagnostics['rccd_weight4'].item()) < 1e-5


def test_reliable_branch_gets_larger_teacher_weight():
    pred3, pred4 = torch.zeros(1, 9, 5, 64, 64), torch.zeros(1, 9, 5, 64, 64)
    pred3[0, 0, 4, 10, 10] = 20.0
    loss, diagnostics = rccd_consensus_loss(pred3, pred4, temperature=2.0, weight_temperature=0.2)
    assert torch.isfinite(loss)
    assert diagnostics['rccd_reliability3'] > diagnostics['rccd_reliability4']
    assert diagnostics['rccd_weight3'] > diagnostics['rccd_weight4']


def test_rccd_only_updates_confidence_channel():
    torch.manual_seed(2)
    pred3 = torch.randn(2, 9, 5, 64, 64, requires_grad=True)
    pred4 = torch.randn(2, 9, 5, 64, 64, requires_grad=True)
    loss, _ = rccd_consensus_loss(pred3, pred4, temperature=2.0, weight_temperature=0.2)
    loss.backward()
    assert torch.isfinite(pred3.grad).all() and torch.isfinite(pred4.grad).all()
    assert pred3.grad[:, :, :4].abs().max().item() == 0.0
    assert pred4.grad[:, :, :4].abs().max().item() == 0.0
    assert pred3.grad[:, :, 4].abs().sum().item() > 0.0
    assert pred4.grad[:, :, 4].abs().sum().item() > 0.0


if __name__ == '__main__':
    test_identical_predictions_zero_loss()
    test_reliable_branch_gets_larger_teacher_weight()
    test_rccd_only_updates_confidence_channel()
    print('RCCD tests passed.')
