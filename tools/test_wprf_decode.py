"""Unit checks for the inference-only winner-preserving residual fusion decoder."""

import torch

from utils.multiscale_detection import select_two_heads, select_two_heads_wprf


def make_prediction(batch, peak_anchor, peak_y, peak_x, box_logits):
    prediction = torch.full((batch, 9, 5, 64, 64), -12.0)
    for index in range(batch):
        prediction[index, peak_anchor[index], :4, peak_y[index], peak_x[index]] = torch.tensor(box_logits[index])
        prediction[index, peak_anchor[index], 4, peak_y[index], peak_x[index]] = 12.0
    return prediction


def main():
    anchors = torch.tensor(((37., 41.), (78., 84.), (96., 215.), (129., 129.), (194., 82.),
                            (198., 179.), (246., 280.), (395., 342.), (550., 573.)))
    pred3 = make_prediction(3, (0, 1, 2), (11, 19, 27), (12, 22, 33),
                            ((0.0, 0.0, 0.1, 0.1), (0.1, -0.1, 0.2, -0.1), (-0.1, 0.2, -0.1, 0.2)))
    pred4 = make_prediction(3, (0, 1, 2), (11, 19, 27), (13, 22, 33),
                            ((0.05, 0.0, 0.1, 0.1), (0.15, -0.1, 0.2, -0.1), (-0.05, 0.2, -0.1, 0.2)))
    # Let the second head win the middle example while preserving high agreement.
    pred4[1, 1, 4, 19, 22] = 13.0

    original, original_diag = select_two_heads(pred3, pred4, anchors, 1024)
    zero, zero_diag = select_two_heads_wprf(pred3, pred4, anchors, 1024, mode='iou', alpha_max=0.0)
    assert torch.equal(original, zero), 'alpha_max=0 must exactly reproduce confidence competition'
    assert torch.equal(original_diag['stage3_selected'], zero_diag['stage3_selected'])
    assert torch.equal(original_diag['stage4_selected'], zero_diag['stage4_selected'])

    iou_box, iou_diag = select_two_heads_wprf(pred3, pred4, anchors, 1024, mode='iou')
    full_box, full_diag = select_two_heads_wprf(pred3, pred4, anchors, 1024, mode='full')
    assert iou_box.shape == original.shape == full_box.shape == (3, 4)
    assert torch.equal(original_diag['stage3_selected'], iou_diag['stage3_selected'])
    assert torch.equal(original_diag['stage4_selected'], full_diag['stage4_selected'])
    assert torch.allclose(iou_diag['wprf_g_conf'], torch.ones_like(iou_diag['wprf_g_conf']))
    assert torch.allclose(iou_diag['wprf_g_geo'], torch.ones_like(iou_diag['wprf_g_geo']))
    for diag in (iou_diag, full_diag):
        assert 0.0 <= diag['wprf_alpha'] <= 0.25
        assert 0.0 <= diag['wprf_refined_ratio'] <= 1.0
        assert 0.0 <= diag['wprf_zero_ratio'] <= 1.0
    print('WPRF decoder checks passed: zero-residual identity and winner preservation.')


if __name__ == '__main__':
    main()
