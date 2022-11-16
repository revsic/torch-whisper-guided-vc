from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Convolutional residual block.
    """
    def __init__(self, channels: int, kernels: int, blocks: int):
        """Initializer.
        Args:
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            blocks: the number of the convolution blocks before residual connection.
        """
        super().__init__()
        self.block = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(channels, channels, kernels, padding=kernels // 2),
                nn.ReLU(),
                nn.BatchNorm1d(channels))
            for _ in range(blocks)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input 1d feature map.
        Returns:
            [torch.float32; [B, C, T]], residually connected.
        """
        return inputs + self.block(inputs)


class AuxResidualBlock(nn.Module):
    """Convolutional residual block with auxiliary contexts.
    """
    def __init__(self, channels: int, kernels: int, aux: int, context: int):
        """Initializer.
        Args:
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            aux: size of the auxiliary contexts.
            context: size of the context vectors.
        """
        super().__init__()
        self.pre_interp = nn.Sequential(
            nn.Conv1d(context, context, kernels, padding=kernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels))

        self.post_interp = nn.Conv1d(context, channels, 1)

        self.preblock = nn.Sequential(
            nn.Conv1d(channels, channels, kernels, padding=kernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels))
        
        self.proj = nn.Linear(aux, channels, bias=False)

        self.postblock = nn.Sequential(
            nn.Conv1d(channels, channels, kernels, padding=kernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels))

    def forward(self,
                inputs: torch.Tensor,
                aux: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input 1D feature map.
            aux: [torch.float32; [B, `aux`]], auxiliary embedding.
            context: [torch.float32; [B, `context`, T']], context vectors.
        Returns:
            [torch.float32; [B, C, T]], residually connected.
        """
        # [B, context, T]
        context = F.interpolate(
            self.pre_interp(context), size=inputs.shape[-1], mode='linear')
        # [B, C, T]
        context = self.post_interp(context)
        # [B, C, T]
        return inputs + self.postblock(
            self.preblock(inputs + context) + self.proj(aux)[..., None])


class AuxSequential(nn.Module):
    """Sequential wrapper for auxiliary input passing.
    """
    def __init__(self, lists: List[nn.Module]):
        """Initializer.
        Args:
            lists: module lists.
        """
        super().__init__()
        self.lists = nn.ModuleList(lists)

    def forward(self, inputs: torch.Tensor, *aux) -> torch.Tensor:
        """Chaining outputs with auxiliary inputs.
        """
        x = inputs
        for module in self.lists:
            x = module(x, *aux)
        return x
