import torch
import torch.nn as nn


class ConvGLU(nn.Module):
    """Dropout - Conv1d - GLU and layer normalization.
    """
    def __init__(self,
                 channels: int,
                 kernels: int,
                 dropout: float):
        """Initializer.
        Args:
            channels: size of the input channels.
            kernels: size of the convolutional kernels.
            dropout: dropout rate.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels * 2, kernels, padding=kernels // 2),
            nn.GLU(dim=1))

    def forward(self, inputs: torch.Tensor):
        """Transform the inputs with given conditions.
        Args:
            inputs: [torch.float32; [B, channels, T]], input channels.
        Returns:
            [torch.float32; [B, channels, T]], transformed.
        """
        # [B, channels, T]
        return inputs + self.conv(inputs)
