from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Config
from .embedder import Embedder
from .scheduler import Scheduler
from .unet import UNet
from .wav2vec2 import Wav2Vec2Wrapper


class WhisperGuidedVC(nn.Module):
    """Whisper-guided DDPM-based Voice conversion.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configurations.
        """
        super().__init__()
        self.w = config.w
        self.steps = config.steps
        self.embedder = Embedder(
            config.pe,
            config.embeddings,
            config.steps,
            config.mappings)

        self.scheduler = Scheduler(
            config.steps,
            config.internals,
            config.logit_min,
            config.logit_max)

        self.spkembed = nn.Embedding(config.num_spk, config.spk)
        # for classifier-free guidance
        self.register_buffer('nullspk', torch.randn(config.spk))
        self.register_buffer('nullcontext', torch.randn(config.context))

        self.wav2vec2 = Wav2Vec2Wrapper(
            config.w2v_name,
            config.sr,
            config.w2v_lin)

        self.unet = UNet(
            config.mel,
            config.channels,
            config.kernels,
            config.embeddings + config.spk,
            config.context,
            config.stages,
            config.blocks)

    def forward(self,
                context: torch.Tensor,
                spkid: torch.Tensor,
                signal: Optional[torch.Tensor] = None,
                use_tqdm: bool = False) \
            -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Generated waveform conditioned on mel-spectrogram.
        Args:
            context: [torch.float32; [B, mel, T]], context mel-spectrogram.
            spkid: [torch.long; [B]], speaker id.
            signal: [torch.float32; [B, mel, T]], initial noise.
            use_tqdm: use tqdm range or not.
        Returns:
            [torch.float32; [B, mel, T]], denoised result.
            S x [np.float32; [B, mel, T]], internal representations.
        """
        # [B, spk]
        spk = self.spkembed(spkid)
        # [B, mel, T]
        signal = signal or torch.randn_like(context)
        # [B, context, T], contextualized.
        context = self.wav2vec2(context)
        # S x [B, mel, T]
        ir = [signal.cpu().detach().numpy()]
        # zero-based step
        ranges = range(self.steps - 1, -1, -1)
        if use_tqdm:
            ranges = tqdm(ranges)
        for step in ranges:
            # [1]
            step = torch.tensor([step], device=signal.device)
            # [B, mel, T], [B]
            mean, std = self.inverse(signal, context, spk, step)
            # [B, mel, T]
            signal = mean + torch.randn_like(mean) * std[:, None, None]
            ir.append(signal.cpu().detach().numpy())
        # [B, mel, T]
        return signal, ir

    def diffusion(self,
                  signal: torch.Tensor,
                  steps: torch.Tensor,
                  next_: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffusion process.
        Args:
            signal: [torch.float32; [B, mel, T]], input signal.
            steps: [torch.long; [B]], t, target diffusion steps, zero-based.
            next_: whether move single steps or multiple steps.
                if next_, signal is z_{t - 1}, otherwise signal is z_0.
        Returns:
            [torch.float32; [B, mel, T]], z_{t}, diffused mean.
            [torch.float32; [B]], standard deviation.
        """
        # [S + 1]
        logsnr, betas = self.scheduler()
        if next_:
            # [B], one-based sample
            beta = betas[steps + 1]
            # [B, mel, T], [B]
            return (1. - beta[:, None, None]).sqrt() * signal, beta.sqrt()
        # [S + 1]
        alphas_bar = torch.sigmoid(logsnr)
        # [B], one-based sample
        alpha_bar = alphas_bar[steps + 1]
        # [B, mel, T], [B]
        return alpha_bar[:, None, None].sqrt() * signal, (1 - alpha_bar).sqrt()

    def inverse(self,
                signal: torch.Tensor,
                context: torch.Tensor,
                spk: torch.Tensor,
                steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse process, single step denoise.
        Args:
            signal: [torch.float32; [B, mel, T]], input signal, z_{t}.
            context: [torch.float32; [B, context, T]], contextualized.
            spk: [torch.float32; [B, spk]], speaker vectors.
            steps: [torch.long; [B]], t, diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, mel, T]], waveform mean, z_{t - 1}
            [torch.float32; [B]], waveform std.
        """
        # B, _, T
        bsize, _, timestep = signal.shape
        # [S + 1]
        logsnr, betas = self.scheduler()
        # [S + 1]
        alphas, alphas_bar = 1. - betas, torch.sigmoid(logsnr)
        # [B, mel, T]
        cond = self.denoise(signal, context, spk, steps)
        # [B, mel, T]
        uncond = self.denoise(
            signal,
            self.nullcontext[None, :, None].repeat(bsize, 1, timestep),
            self.nullspk[None].repeat(bsize),
            steps)
        # [B, mel, T], classifier-free guidance
        denoised = (1 + self.w) * cond - self.w * uncond
        # [B], make one-based
        prev, steps = steps, steps + 1
        # [B, mel, T]
        mean = alphas_bar[prev, None, None].sqrt() * betas[steps, None, None] / (
                1 - alphas_bar[steps, None, None]) * denoised + \
            alphas[steps, None, None].sqrt() * (1. - alphas_bar[prev, None, None]) / (
                1 - alphas_bar[steps, None, None]) * signal
        # [B]
        var = (1 - alphas_bar[prev]) / (1 - alphas_bar[steps]) * betas[steps]
        return mean, var.sqrt()

    def denoise(self,
                signal: torch.Tensor,
                context: torch.Tensor,
                spk: torch.Tensor,
                steps: torch.Tensor) -> torch.Tensor:
        """Denoise the signal w.r.t. outpart signal.
        Args:
            signal: [torch.float32; [B, mel, T]], input signal.
            context: [torch.float32; [B, context, T]], contextualized.
            spk: [torch.float32; [B, spk]], speaker vectors.
            steps: [torch.long; [B]], diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, mel, T]], denoised signal.
        """
        # [B, embeddings + spk]
        embed = torch.cat([self.embedder(steps), spk], dim=-1)
        # [B, mel, T]
        return self.unet(signal, embed, context)

    def save(self, path: str, optim: Optional[torch.optim.Optimizer] = None):
        """Save the models.
        Args:
            path: path to the checkpoint.
            optim: optimizer, if provided.
        """
        dump = {'model': self.state_dict(), 'config': vars(self.config)}
        if optim is not None:
            dump['optim'] = optim.state_dict()
        torch.save(dump, path)

    def load_(self, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints inplace.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        self.load_state_dict(states['model'])
        if optim is not None:
            optim.load_state_dict(states['optim'])

    @classmethod
    def load(cls, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        config = Config()
        for key, val in states['config'].items():
            if not hasattr(config, key):
                import warnings
                warnings.warn(f'unidentified key {key}')
                continue
            setattr(config, key, val)
        # construct
        wgvc = cls(config)
        wgvc.load_(states, optim)
        return wgvc
