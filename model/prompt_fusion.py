import torch
import torch.nn as nn


class PromptFusion(nn.Module):
    """Zero-initialized residual fusion; zero init preserves pretrained DetGeo."""

    def __init__(self, hidden_channels=16):
        super().__init__()
        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.rgb_fusion = nn.Sequential(
            nn.Conv2d(3 + hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 3, 1, bias=True),
        )
        nn.init.zeros_(self.rgb_fusion[-1].weight)
        nn.init.zeros_(self.rgb_fusion[-1].bias)

    def forward(self, base_query_rgb, gaussian_map, sam_mask, masked_gaussian):
        prompts = torch.cat((gaussian_map, sam_mask, masked_gaussian), dim=1)
        prompt_features = self.prompt_encoder(prompts)
        delta_rgb = self.rgb_fusion(torch.cat((base_query_rgb, prompt_features), dim=1))
        return base_query_rgb + delta_rgb
