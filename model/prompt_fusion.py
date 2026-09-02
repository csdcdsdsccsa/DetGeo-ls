import torch
import torch.nn as nn


class PromptFusion(nn.Module):
    """SAM-Gaussian adaptive positional encoding: P_new = G + delta_position."""

    def __init__(self, hidden_channels=16):
        super().__init__()
        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )
        nn.init.zeros_(self.prompt_encoder[-1].weight)
        nn.init.zeros_(self.prompt_encoder[-1].bias)

    def forward(self, gaussian_map, sam_mask, masked_gaussian):
        prompts = torch.cat((gaussian_map, sam_mask, masked_gaussian), dim=1)
        delta_position = self.prompt_encoder(prompts)
        return gaussian_map + delta_position
