import torch
import torch.nn as nn

from .resblock import AuxResidualBlock, AuxSequential


class UNet(nn.Module):
    """Spectrogram U-Net for noise estimator.
    """
    def __init__(self,
                 mel: int,
                 channels: int,
                 kernels: int,
                 aux: int,
                 context: int,
                 stages: int,
                 blocks: int):
        """Initializer.
        Args:
            mel: size of the mel filter channels.
            channels: size of the hidden channels.
            kernels: size of the convolutional kernels.
            aux: size of the auxiliary channels.
            context: size of the context channels.
            stages: the number of the resolution scales.
            blocks: the number of the residual blocks in each stages.
        """
        super().__init__()
        self.proj = nn.Conv1d(mel, channels, 1)
        self.dblocks = nn.ModuleList([
            AuxSequential([
                AuxResidualBlock(channels * 2 ** i, kernels, aux, context)
                for _ in range(blocks)])
            for i in range(stages - 1)])

        self.downsamples = nn.ModuleList([
            # half resolution
            nn.Conv1d(channels * 2 ** i, channels * 2 ** (i + 1), kernels, 2, padding=kernels // 2)
            for i in range(stages - 1)])

        self.neck = AuxResidualBlock(
            channels * 2 ** (stages - 1), kernels, aux, context)

        self.upsamples = nn.ModuleList([
            # double resolution
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels * 2 ** i, channels * 2 ** (i - 1), kernels, padding=kernels // 2))
            for i in range(stages - 1, 0, -1)])

        self.ublocks = nn.ModuleList([
            AuxSequential([
                AuxResidualBlock(channels * 2 ** i, kernels, aux, context)
                for _ in range(blocks)])
            for i in range(stages - 2, -1, -1)])

        self.proj_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, mel, 1))

    def forward(self,
                inputs: torch.Tensor,
                aux: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """Spectrogram U-net.
        Args:
            inputs: [torch.float32; [B, mel, T]], input tensor, spectrogram.
            aux: [torch.float32; [B, aux]], auxiliary informations, times.
            context: [torch.float32; [B, context, T']], contextual features.
        Returns:
            [torch.float32; [B, mel, T]], transformed.
        """
        # [B, C, T]
        x = self.proj(inputs)
        # (stages - 1) x [B, C x 2^i, T / 2^i]
        internals = []
        for dblock, downsample in zip(self.dblocks, self.downsamples):
            # [B, C x 2^i, T / 2^i]
            x = dblock(x, aux, context)
            internals.append(x)
            # [B, C x 2^(i + 1), T / 2^(i + 1)]
            x = downsample(x)
        # [B, C x 2^stages, T / 2^stages]
        x = self.neck(x, aux, context)
        for i, ublock, upsample in zip(reversed(internals), self.ublocks, self.upsamples):
            # [B, C x 2^i, T / 2^i]
            x = ublock(upsample(x) + i, aux, context)
        # [B, mel, T]
        return self.proj_out(x)
