"""Assert that untrained SAM PromptFusion preserves pretrained DetGeo logits."""

import argparse
import torch

from model.DetGeo import DetGeo
from model.DetGeo_sam_prompt import DetGeoSAMPrompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(13)
    baseline = torch.nn.DataParallel(DetGeo()).cuda().eval()
    prompted = torch.nn.DataParallel(DetGeoSAMPrompt()).cuda().eval()
    checkpoint = torch.load(args.pretrain, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    baseline.load_state_dict(state_dict, strict=True)
    incompatible = prompted.load_state_dict(state_dict, strict=False)
    unexpected_prompt_keys = [key for key in incompatible.unexpected_keys if not key.startswith("module.prompt_fusion.")]
    if unexpected_prompt_keys:
        raise RuntimeError("unexpected keys: {0}".format(unexpected_prompt_keys))

    query = torch.randn(args.batch_size, 3, 256, 256, device="cuda")
    reference = torch.randn(args.batch_size, 3, 1024, 1024, device="cuda")
    original_click = torch.randn(args.batch_size, 256, 256, device="cuda")
    gaussian = torch.rand(args.batch_size, 256, 256, device="cuda")
    mask = torch.randint(0, 2, (args.batch_size, 256, 256), device="cuda", dtype=torch.int64).float()
    with torch.no_grad():
        baseline_logits, _ = baseline(query, reference, original_click)
        prompted_logits, _ = prompted(query, reference, original_click, gaussian, mask, gaussian * mask)
    maximum_difference = (baseline_logits - prompted_logits).abs().max().item()
    if not torch.equal(baseline_logits, prompted_logits):
        raise AssertionError("zero-init equality failed: max_abs_diff={0}".format(maximum_difference))
    print("SAM_PROMPT_ZERO_INIT_OK max_abs_diff={0}".format(maximum_difference))


if __name__ == "__main__":
    main()
