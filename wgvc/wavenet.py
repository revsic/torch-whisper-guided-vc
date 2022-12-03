import torch
import torch.nn as nn


class WaveNetBlock(nn.Module):
    """WaveNet block, dilated convolution and skip connection.
    """
    def __init__(self,
                 channels: int,
                 embed: int,
                 kernels: int,
                 dilations: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            embed: size of the embedding channels.
            kernels: size of the convolutional kernel.
            dilations: dilation rates of contoluion.
        """
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                channels, channels * 2, kernels,
                padding=(kernels - 1) * dilations // 2, dilation=dilations))

        self.proj_embed = nn.utils.weight_norm(
            nn.Linear(embed, channels, bias=False))

        self.proj_res = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, 1))
        self.proj_skip = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, 1))

    def forward(self, inputs: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """Pass to the wavenet block.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensor.
            embed: [torch.float32; [B, E]], auxiliary embeddings.
        Returns:
            residual: [torch.float32; [B, C, T]], residually connected.
            skip: [torch.float32; [B, C, T]], skip connection purposed.
        """
        # [B, C, T]
        x = inputs + self.proj_embed(embed)[..., None]
        # [B, C x 2, T]
        x = self.conv(x)
        # [B, C, T]
        gate, context = x.chunk(2, dim=1)
        # [B, C, T]
        x = torch.sigmoid(gate) * torch.tanh(context)
        # [B, C, T]
        res = (x + self.proj_res(x)) * (2 ** -0.5)
        return res, self.proj_skip(x)
