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
            for fft in config.train.fft]

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
        # B
        bsize, = lengths.shape
        # [B]
        start = torch.rand(bsize, device=self.device) * (lengths - seglen).clamp_min(0)
        # [B, seglen]
        return torch.stack([
            F.pad(n[s:s + seglen], [0, max(seglen - len(n), 0)])
            for n, s in zip(speeches, start.long())])

    def compute_loss(self, sid: torch.Tensor, speeches: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss.
        Args:
            sid: [torch.long; [B]], speaker ids.
            speeches: [torch.float32; [B, T]], segmetned speech.
        Returns:
            loss and dictionaries.
        """
        # pre-compute
        with torch.no_grad():
            nulls = self.config.train.null_size
            # B, T
            bsize, seglen = speeches.shape
            # [B - nulls, T]
            seg_c = speeches[nulls:]
            # [B - nulls, d_model, T' / hop]
            context = self.a_whisper.forward(seg_c)
            # [B - nulls, S]
            pitch = self.model.analyze_pitch(seg_c)

        # [B - nulls, aux, T - alpha]
        context = self.model.blend(context, pitch)

        # B
        bsize, = sid.shape
        # [B], zero-based
        steps = torch.randint(
            self.config.model.steps, (bsize,), device=self.device)
        # [B, T], [B]
        base_mean, base_std = self.model.diffusion(speeches, steps)
        # [B, T]
        base = base_mean + torch.randn_like(base_mean) * base_std[:, None]

        # [nulls, spk]
        nullembed = F.normalize(self.model.nullspk[None], dim=-1).repeat(nulls, 1)
        # uncondition
        uncond = self.model.denoise(base[:nulls], nullembed, steps[:nulls])

        # T - alpha
        _, _, ctxsteps = context.shape
        # [B - nulls, spk]
        spkembed = F.normalize(self.model.spkembed(sid[nulls:]), dim=-1)
        # condition
        cond = self.model.denoise(
            base[nulls:, :ctxsteps], spkembed, steps[nulls:], context=context)
        # [B - nulls, T]
        cond = F.pad(cond, [0, seglen - ctxsteps])

        # [B, T]
        denoised = torch.cat([uncond, cond], dim=0)
        # [B]
        noise_estim = (speeches - denoised).abs()

        # metric purpose
        estim_u, estim_c = noise_estim.split([nulls, bsize - nulls], dim=0)
        estim_u, estim_c = estim_u.mean().item(), estim_c.mean().item()
        # [B]
        noise_estim = noise_estim.mean()

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
        # mean
        mss = mss / len(self.windows)

        # []
        loss = noise_estim + mss
        losses = {
            'loss': loss.item(),
            'estim': noise_estim.item(),
            'estim-c': estim_c,
            'estim-u': estim_u,
            'mss': mss.item()}
        return loss, losses, {
            'base': base.cpu().detach().numpy(),
            'denoised': denoised.cpu().detach().numpy()}
