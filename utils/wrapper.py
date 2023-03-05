from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from wgvc import WhisperGuidedVC

from .augment import AugmentedWhisper


class TrainingWrapper:
    """Training wrapper.
    """
    def __init__(self, model: WhisperGuidedVC, config: Config, device: torch.device):
        """Initializer.
        Args:
            model: whisper-guided vc model.
            config: training configurations.
            device: device.
        """
        self.model = model
        self.config = config
        self.device = device
        # for augmentation purpose
        self.a_whisper = AugmentedWhisper(
            model.whisper, config.train.smin, config.train.smax, config.train.std)
        # windows
        self.windows = [
            torch.hann_window(fft, device=device)
            for fft in config.fft]

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
        # pre-compute
        with torch.no_grad():
            # [B, d_model, T' / hop]
            context = self.a_whisper.forward(speeches)
            # [B, S]
            pitch = self.model.analyze_pitch(speeches)

        # B, T
        bsize, timesteps, = speeches.shape
        # [B, aux, seglen]
        context = self.model.blend(context, pitch)[..., :timesteps]

        # [B], zero-based
        steps = torch.randint(
            self.config.model.steps, (bsize,), device=self.device)
        # [B, seglen], [B]
        base_mean, base_std = self.model.diffusion(speeches, steps)
        # [B, seglen]
        base = base_mean + torch.randn_like(base_mean) * base_std[:, None]

        nulls = self.config.train.null_size
        # [1, spk]
        nullembed = F.normalize(self.model.nullspk[None], dim=-1)
        # uncondition
        uncond = self.model.denoise(base[:nulls], nullembed, steps[:nulls])

        # [B, spk]
        spkembed = F.normalize(self.model.spkembed(sid), dim=-1)
        # condition
        cond = self.model.denoise(
            base[nulls:], spkembed[nulls:], steps[nulls:], context=context[nulls:])

        # [B, T]
        denoised = torch.cat([uncond, cond], dim=0)
        # []
        noise_estim = (speeches - denoised).abs().mean()

        # pack
        pack = torch.cat([speeches, denoised], dim=0)
        # spectrogram guiding
        mss = 0.
        for win in self.windows:
            # fft
            fft, = win.shape
            # [B x 2, fft, T // (fft // 4)]
            stft = torch.stft(pack, fft, window=win, return_complex=True).abs()
            # [B, fft, T // (fft // 4)]
            fft_s, fft_d = stft.clamp_min(1e-5).log().chunk(2, dim=0)
            # []
            mss = mss + (fft_s - fft_d).square().mean()

        # []
        loss = noise_estim + mss
        losses = {
            'diffusion-loss': noise_estim.item(),
            'noise-estim': noise_estim.item(),
            'mss': mss.item()}
        return loss, losses, {
            'seg': speeches.cpu().detach().numpy(),
            'base': base.cpu().detach().numpy(),
            'denoised': denoised.cpu().detach().numpy()}
