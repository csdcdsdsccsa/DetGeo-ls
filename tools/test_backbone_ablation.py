"""GPU smoke, sharing and parameter-count checks for pure DetGeo backbone ablations."""
import gc
import torch

from model.DetGeo_backbone_ablation import DetGeoBackboneAblation


def counts(model):
    total = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for name, p in model.named_parameters() if 'darknet' in name or 'resnet50' in name)
    return total, backbone, total - backbone


def main():
    for experiment in ('darknet53_noshare', 'darknet53_shared', 'resnet50_shared'):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = DetGeoBackboneAblation(backbone_exp=experiment).cuda().train()
        if experiment == 'darknet53_noshare':
            q_param = next(model.query_darknet.parameters()); r_param = next(model.reference_darknet.parameters())
            assert model.query_darknet is not model.reference_darknet and q_param is not r_param and torch.allclose(q_param, r_param)
            with torch.no_grad(): q_param.add_(1.0)
            assert not torch.allclose(q_param, r_param)
        elif experiment == 'darknet53_shared':
            assert hasattr(model, 'shared_darknet') and not hasattr(model, 'query_darknet') and not hasattr(model, 'reference_darknet')
        else:
            assert hasattr(model, 'shared_resnet50') and not hasattr(model, 'query_resnet50') and not hasattr(model, 'reference_resnet50')
        query = torch.randn(1, 3, 256, 256, device='cuda')
        reference = torch.randn(1, 3, 1024, 1024, device='cuda')
        click = torch.randn(1, 256, 256, device='cuda')
        outbox, attention = model(query, reference, click)
        assert outbox.shape == (1, 45, 64, 64) and attention.shape == (1, 64, 64)
        outbox.mean().backward()
        assert any(p.grad is not None for p in model.parameters())
        total, backbone, non_backbone = counts(model)
        print('{} total={} backbone={} non_backbone={} peak_mib={:.1f}'.format(
            experiment, total, backbone, non_backbone, torch.cuda.max_memory_allocated() / 1024 / 1024), flush=True)
        del model, query, reference, click, outbox, attention; gc.collect()


if __name__ == '__main__':
    main()
