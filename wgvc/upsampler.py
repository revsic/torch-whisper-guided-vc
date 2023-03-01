from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsampler(nn.Module):
    """Whisper to signal-level upsampler.
    """
    def __init__(self, channels: int, kernels: int, scales: List[int], leak: float):
        """Initializer.
        Args:
            channels: size of the input channels.
            kernels: size of the convolutional kernels.
            scales: upsampling scales.
            leak: leaky relu coefficient.
        """
        super().__init__()
        self.scales = scales
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernels, padding=kernels // 2, bias=False))
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv2d(1, 1, (1, scale * 2 + 1), padding=(0, scale), bias=False)),
                nn.LeakyReLU(leak))
            for scale in scales])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Upsample the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensor.
        Returns:
            [torch.float32; [B, C, T x prod(scales)]], upsampled.
        """
        # [B, 1, C, T]
        x = self.conv(inputs)[:, None]
        for scale, conv in zip(self.scales, self.upsamples):
            # [B, 1, C, T x scale]
            x = conv(
                F.interpolate(x, scale_factor=(1, scale), mode='nearest'))
        # [B, C, T x prod(scales)]
        return x.squeeze(1)
