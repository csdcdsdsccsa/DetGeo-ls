"""Check initial P_new equals Gaussian and the replacement model runs."""

import argparse
import torch

from model.DetGeo_sam_prompt import DetGeoSAMPrompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(13)
    prompted = torch.nn.DataParallel(DetGeoSAMPrompt()).cuda().eval()
    checkpoint = torch.load(args.pretrain, map_location="cpu")
    state_dict = checkpoint["state_dict"]
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
        position_map = prompted.module.prompt_fusion(gaussian.unsqueeze(1), mask.unsqueeze(1), (gaussian * mask).unsqueeze(1))
        logits, _ = prompted(query, reference, original_click, gaussian, mask, gaussian * mask)
    maximum_difference = (position_map - gaussian.unsqueeze(1)).abs().max().item()
    if not torch.equal(position_map, gaussian.unsqueeze(1)):
        raise AssertionError("P_new must equal Gaussian at zero init: max_abs_diff={0}".format(maximum_difference))
    if logits.shape != (args.batch_size, 45, 64, 64):
        raise AssertionError("unexpected YOLO output shape: {0}".format(tuple(logits.shape)))
    print("SAM_POSITION_REPLACEMENT_INIT_OK max_abs_diff={0}".format(maximum_difference))


if __name__ == "__main__":
    main()
