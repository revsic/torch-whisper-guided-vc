import numpy as np
import torch
import torch.nn as nn


class Embedder(nn.Module):
    """Sinusoidal embedding and mapping networks.
    """
    def __init__(self, pe: int, channels: int, steps: int, layers: int):
        """Initializer.
        Args:
            pe: size of the positional encodings.
            channels: size of the output embedding.
            steps: total diffusion steps
            layers: the number of the network layers.
        """
        super().__init__()
        self.register_buffer('buffer', Embedder.sinusoidal(steps, pe))
        self.mapper = nn.Sequential(
            nn.Linear(pe, channels), nn.SiLU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.SiLU())
                for _ in range(layers - 1)])

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        """Generate the embeddings.
        Args:
            steps: [torch.long; [B]], diffusion steps.
        Returns:
            [torch.float32; [B, E]], embedding.
        """
        return self.mapper(self.buffer[steps])

    @staticmethod
    def sinusoidal(steps: int, channels: int) -> torch.Tensor:
        """Generate sinusoidal embedding introduced by Vaswani et al., 2017.
        Args:
            steps: S, total diffusion steps.
            channels: C, size of the sinusoidal positional embeddings.
        Returns:
            [torch.float32; [S, C]], sinusoidal positional embedding.
        """
        # [S]
        pos = torch.arange(steps)
        # [C // 2]
        i = torch.arange(0, channels, 2)
        # [S, C // 2]
        context = pos[:, None] * torch.exp(-np.log(10000) * i / channels)[None]
        # [S, C]
        return torch.stack(
                [torch.sin(context), torch.cos(context)], dim=-1
            ).reshape(steps, channels)
