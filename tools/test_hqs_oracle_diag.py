"""CPU unit checks for the validation-only HQS Oracle diagnostic."""

import torch

from utils.multiscale_detection import analyze_two_head_oracle, select_two_heads


def main():
    torch.manual_seed(2026)
    batch = 3
    anchors = torch.tensor(
        [[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129],
         [96, 215], [78, 84], [37, 41]], dtype=torch.float32)
    pred3 = torch.randn(batch, 9, 5, 64, 64)
    pred4 = torch.randn(batch, 9, 5, 64, 64)
    target = torch.tensor([[100, 120, 400, 450], [300, 200, 650, 700], [20, 50, 900, 980]], dtype=torch.float32)
    diag = analyze_two_head_oracle(pred3, pred4, target, anchors, 1024)
    selected, _ = select_two_heads(pred3, pred4, anchors, 1024, fusion=False)
    if not torch.equal(diag['confidence_box'], selected):
        raise RuntimeError('Oracle diagnostic confidence decoder differs from select_two_heads')
    if not torch.all(diag['oracle_iou'] + 1e-7 >= diag['iou3']):
        raise RuntimeError('oracle IoU is below Head3 IoU')
    if not torch.all(diag['oracle_iou'] + 1e-7 >= diag['iou4']):
        raise RuntimeError('oracle IoU is below Head4 IoU')
    if not torch.all(diag['oracle_iou'] + 1e-7 >= diag['confidence_iou']):
        raise RuntimeError('oracle IoU is below confidence-selected IoU')
    identical = analyze_two_head_oracle(pred3, pred3.clone(), target, anchors, 1024)
    if not torch.equal(identical['confidence_box'], identical['oracle_box']):
        raise RuntimeError('identical heads must agree for Oracle and confidence selection')
    if torch.is_grad_enabled() and any(value.requires_grad for value in diag.values() if torch.is_tensor(value)):
        raise RuntimeError('Oracle diagnostic must not retain gradients')
    print('HQS Oracle diagnostic unit checks passed')


if __name__ == '__main__':
    main()
