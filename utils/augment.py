from typing import Optional

import torch
import torch.nn.functional as F

from wgvc import WhisperWrapper


class AugmentedWhisper:
    """Wrapping Whisper for augmentation.
    """
    def __init__(self, whisper: WhisperWrapper, smin: int, smax: int, std: float):
        """Initializer.
        Args:
            whisper: Whisper wrapper.
            smin, smax: size minimum and maximum, (both-inclusive).
            std: standard deviation of additional noise.
        """
        self.whisper = whisper
        self.smin, self.smax = smin, smax
        self.std = std
        # alias
        self.mel, _ = whisper.fbank.shape

    def _augment(self, mel: torch.Tensor, size: int) -> torch.Tensor:
        """Augment the single datum.
        Args:
            mel: [torch.float32; [mel, S]], spectrogram.
            size: target size.
        Returns:
            [torch.float32; [mel, S]], augmented.
        """
        # [S, min(mel, size)]
        sr = F.interpolate(mel.T[None], size=size, mode='linear')[0, :, :self.mel]
        if self.mel > size:
            # [S, mel - size], replication padding
            rep = sr[:, -1:].repeat(1, self.mel - size)
            # [S, mel]
            sr = torch.cat([sr, rep + torch.randn_like(rep) * self.std], dim=-1)
        return sr.T

    def augment(self, mel: torch.Tensor, sizes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Spectrogram resizing based augmentation.
        Args:
            mel: [torch.float32; [B, mel, S]], mel-spectrogram.
            sizes: [torch.long; [B]], size of the extended spectrogram.
        Returns:
            [torch.float32; [B, mel, S]], resized.
        """
        if sizes is None:
            # B
            bsize, _, _ = mel.shape
            # [B]
            sizes = torch.randint(self.smin, self.smax + 1, (bsize,), device=mel.device)
        # [B, mel, S]
        return torch.stack([self._augment(m, s.item()) for m, s in zip(mel, sizes)], dim=0)

    def forward(self,
                audio: torch.Tensor,
                sizes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Whisper feature with spectrogram-resize augmentation.
        Args:
            audio: [torch.float32; [B, T]], audio, [-1, 1]-ranged.
            sizes: [torch.long; [B]], size of the resized spectrogram.
        Returns:
            [torch.float32; [B, d_model, T' // hop]], encoded features.
        """
        # resample
        audio = self.whisper.resample(audio)
        # T
        _, timesteps = audio.shape
        # [B, S], zero-padding
        audio = F.pad(audio, (0, self.whisper.samples - timesteps))
        # [B, mel, S // hop]
        logmel = self.whisper.preproc(audio)
        # [B, mel, S // hop], augmentatoin
        aug = self.augment(logmel, sizes=sizes)
        # [B, S // hop, d_model], encoder only
        outputs = self.whisper.model.encoder(aug).last_hidden_state
        # [B, d_model, T // hop]
        return outputs[:, :timesteps // self.whisper.hop].transpose(1, 2)
