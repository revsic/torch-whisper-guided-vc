from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from wgvc import WhisperGuidedVC


class TrainingWrapper:
    """Training wrapper.
    """
    def __init__(self, model: WhisperGuidedVC, config: Config):
        """Initializer.
        Args:
            model: whisper-guided vc model.
            config: training configurations.
        """
        self.model = model
        self.config = config

    def random_segment(self, speeches: np.ndarray, lengths: np.ndarray) \
            -> np.ndarray:
        """Segment audio into fixed sized array.
        Args:
            speeches: [np.float32; [B, T]], speech audio signal.
            lengths: [np.long; [B]], speech lengths.
        Returns:
            [np.float32; [B, seglen]], segmented speech.
        """
        # alias
        seglen = self.config.train.seglen
        # [B]
        start = np.random.randint(np.maximum(lengths - seglen, 1))
        # [B, seglen]
        return np.stack(
            [np.pad(n[s:s + seglen], [0, max(seglen - len(n), 0)])
             for n, s in zip(speeches, start)])

    def compute_loss(self, sid: torch.Tensor, speeches: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss.
        Args:
            sid: [torch.long; [B]], speaker ids.
            speeches: [torch.float32; [B, seglen]], segmetned speech.
        Returns:
            loss and dictionaries.
        """
        # B
        bsize, = sid.shape
        # [B], zero-based
        steps = torch.randint(
            self.config.model.steps, (bsize,), device=sid.device)
        # [B, seglen], [B]
        base_mean, base_std = self.model.diffusion(speeches, steps)
        # [B, seglen]
        base = base_mean + torch.randn_like(base_mean) * base_std[:, None]
        # [B, spk]
        spkembed = self.model.spkembed(sid)
        # for classifier-free guidance
        null = torch.rand(bsize, device=sid.device) < self.config.train.null_prob
        spkembed[null] = self.model.nullspk
        # normalize
        spkembed = F.normalize(spkembed, dim=-1)
        # [B, seglen]
        denoised = self.model.denoise(speeches, spkembed, steps)
        # []
        noise_estim = (base - denoised).abs().mean()

        # [1 + S]
        logsnr, _ = self.model.scheduler()
        # [1 + S]
        alphas_bar = torch.sigmoid(logsnr)
        # [], prior loss
        schedule_loss = torch.log(
            torch.clamp_min(1 - alphas_bar[-1], 1e-7)) + torch.log(
                torch.clamp_min(alphas_bar[0], 1e-7))

        # []
        loss = noise_estim - schedule_loss
        losses = {
            'noise-estim': noise_estim.item(),
            'schedule-loss': schedule_loss.item()}
        return loss, losses, {
            'base': base.cpu().detach().numpy(),
            'denoised': denoised.cpu().detach().numpy(),
            'alphas-bar': alphas_bar.cpu().detach().numpy()}
