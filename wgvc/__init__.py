from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import Config
from .convglu import ConvGLU
from .embedder import Embedder
from .nacf import NACF
from .upsampler import Upsampler
from .wavenet import WaveNetBlock
from .whisper import WhisperWrapper


class WhisperGuidedVC(nn.Module):
    """Whisper-guided DDPM-based Voice conversion.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configurations.
        """
        super().__init__()
        self.config = config
        # alias
        self.w = config.w
        self.tau = config.tau
        self.steps = config.steps

        # [1 + steps]
        a = torch.arange(1 + config.steps) / config.steps
        f = ((a + config.s) / (1 + config.s) * np.pi * 0.5).cos().square()
        self.register_buffer('alphas_bar', f / f[0], persistent=False)
        # [1 + steps]
        self.register_buffer('betas', F.pad(1 - f[1:] / f[:-1], [1, 0]), persistent=False)

        # models
        self.proj_signal = nn.utils.weight_norm(
            nn.Conv1d(1, config.channels, 1))

        self.embedder = Embedder(
            config.pe,
            config.embeddings,
            config.steps,
            config.mappings)

        self.spkembed = nn.Embedding(config.num_spk, config.spk)
        # for classifier-free guidance
        self.register_buffer('nullspk', torch.randn(config.spk))

        self.whisper = WhisperWrapper(
            config.whisper_name,
            config.sr)

        # alias
        d_model = self.whisper.model.config.d_model
        self.encoder = nn.Sequential(*(
            [nn.Conv1d(d_model, config.aux, 1)]
            + [
                ConvGLU(config.aux, config.kernels, config.dropout)
                for _ in range(config.encoders)]))

        self.nacf = NACF(config.sr)
        self.enc_pitch = nn.Sequential(
            nn.Conv1d(1, config.aux, 1),
            nn.ReLU(),
            nn.Conv1d(
                config.aux, config.aux, config.upkernels,
                padding=config.upkernels // 2, groups=config.aux, bias=False))

        self.factor = config.sr / self.whisper.resample.new_freq

        self.upsampler = Upsampler(
            config.aux,
            config.upkernels,
            config.upscales,
            config.leak)

        self.blocks = nn.ModuleList([
            WaveNetBlock(
                config.channels,
                config.embeddings + config.spk,
                config.aux,
                config.kernels,
                config.dilations ** j)
            for _ in range(config.cycles)
            for j in range(config.layers)])
        
        self.bias = nn.Parameter(torch.zeros(config.channels, 1))

        self.proj_out = nn.Sequential(
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(config.channels, config.channels, 1)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(config.channels, 1, 1)),
            nn.Tanh())

    def analyze_pitch(self, signal: torch.Tensor) -> torch.Tensor:
        """Analyze the relative pitch in log-2 scale.
        Args:
            signal: [torch.float32; [B, T]], input signal, [-1, 1]-ranged.
        Returns:
            [torch.float32; [B, T / 100]], pitch sequence.
        """
        # [B, S]
        pitch = self.nacf.forward(signal)
        # [B]
        median = torch.stack([p[p > 0].median() for p in pitch])
        # [B, S]
        return torch.where(pitch > 0, (pitch / median[:, None]).log2(), 0)

    def blend(self, context: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        """Generate the condition of the wavenet from context and pitch.
        Args:
            context: [torch.float32; [B, d_model, T' / hop]], context features.
            pitch: [torch.float32; [B, S]], pitch sequence.
        Returns:
            [torch.float32; [B, aux, T]], condition features.
        """
        # T' / hop
        _, _, timesteps = context.shape
        # [B, 1, T' / hop]
        pitch = F.interpolate(pitch[:, None], size=timesteps, mode='linear')
        # [B, aux, T' // hop]
        cond = self.encoder(context) + self.enc_pitch(pitch)
        # [B, aux, T // hop]
        cond = F.interpolate(cond, scale_factor=self.factor, mode='nearest')
        # [B, aux, T]
        return self.upsampler(cond)

    def forward(self,
                context: torch.Tensor,
                spkid: torch.Tensor,
                signal: Optional[torch.Tensor] = None,
                use_tqdm: bool = False) \
            -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Generated waveform conditioned on mel-spectrogram.
        Args:
            context: [torch.float32; [B, T]], context audio signal.
            spkid: [torch.long; [B]], speaker id.
            signal: [torch.float32; [B, T]], initial noise.
            use_tqdm: use tqdm range or not.
        Returns:
            [torch.float32; [B, T]], denoised result.
            S x [np.float32; [B, T]], internal representations.
        """
        # _, T
        _, timesteps = context.shape
        # [B, spk]
        spk = F.normalize(self.spkembed(spkid), dim=-1)
        # [B, T]
        signal = signal or torch.randn_like(context)
        # apply temperature
        signal = signal * self.tau ** -0.5

        # [B, S]
        pitch = self.analyze_pitch(context)
        # [B, d_model, _ // hop]
        context = self.whisper(context)
        # [B, aux, T']
        context = self.blend(context, pitch)
        # _, _, T'
        _, _, ctxstep = context.shape
        # [B, T']
        signal = signal[:, :ctxstep]
        # S x [B, T']
        ir = [signal.cpu().detach().numpy()]
        # zero-based step
        ranges = range(self.steps - 1, -1, -1)
        if use_tqdm:
            ranges = tqdm(ranges)
        for step in ranges:
            # [1]
            step = torch.tensor([step], device=signal.device)
            # [B, T'], [B]
            mean, std = self.inverse(signal, context, spk, step)
            # [B, T']
            signal = mean + torch.randn_like(mean) * std[:, None]
            ir.append(signal.cpu().detach().numpy())
        # [B, T]
        return signal, ir

    def diffusion(self,
                  signal: torch.Tensor,
                  steps: torch.Tensor,
                  next_: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffusion process.
        Args:
            signal: [torch.float32; [B, T]], input signal.
            steps: [torch.long; [B]], t, target diffusion steps, zero-based.
            next_: whether move single steps or multiple steps.
                if next_, signal is z_{t - 1}, otherwise signal is z_0.
        Returns:
            [torch.float32; [B, T]], z_{t}, diffused mean.
            [torch.float32; [B]], standard deviation.
        """
        # [S + 1]
        betas = self.betas
        if next_:
            # [B], one-based sample
            beta = betas[steps + 1]
            # [B, mel, T], [B]
            return (1. - beta[:, None, None]).sqrt() * signal, beta.sqrt()
        # [S + 1]
        alphas_bar = self.alphas_bar
        # [B], one-based sample
        alpha_bar = alphas_bar[steps + 1]
        # [B, T], [B]
        return alpha_bar[:, None].sqrt() * signal, (1 - alpha_bar).sqrt()

    def inverse(self,
                signal: torch.Tensor,
                context: torch.Tensor,
                spk: torch.Tensor,
                steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse process, single step denoise.
        Args:
            signal: [torch.float32; [B, T]], input signal, z_{t}.
            context: [torch.float32; [B, d_model, T]], contextual feature
            spk: [torch.float32; [B, spk]], speaker vectors, normalized.
            steps: [torch.long; [B]], t, diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, T]], waveform mean, z_{t - 1}
            [torch.float32; [B]], waveform std.
        """
        # [S + 1]
        betas = self.betas
        # [S + 1]
        alphas, alphas_bar = 1. - betas, self.alphas_bar
        # [B, T]
        cond = self.denoise(signal, spk, steps, context)
        # [1, spk]
        nullspk = F.normalize(self.nullspk, dim=-1)[None]
        # [B, T]
        uncond = self.denoise(signal, nullspk, steps)
        # [B, T], classifier-free guidance
        denoised = (1 + self.w) * cond - self.w * uncond
        # [B], make one-based
        prev, steps = steps, steps + 1
        # [B, T]
        mean = alphas_bar[prev, None].sqrt() * betas[steps, None] / (
                1 - alphas_bar[steps, None]) * denoised + \
            alphas[steps, None].sqrt() * (1. - alphas_bar[prev, None]) / (
                1 - alphas_bar[steps, None]) * signal
        # [B]
        var = (1 - alphas_bar[prev]) / (1 - alphas_bar[steps]) * betas[steps]
        return mean, var.sqrt()

    def denoise(self,
                signal: torch.Tensor,
                spk: torch.Tensor,
                steps: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Denoise the signal w.r.t. outpart signal.
        Args:
            signal: [torch.float32; [B, T]], input signal.
            spk: [torch.float32; [B, spk]], speaker vectors.
            steps: [torch.long; [B]], diffusion steps, zero-based.
            context: [torch.float32; [B, d_model, T]], contextual feature.
        Returns:
            [torch.float32; [B, T]], denoised signal.
        """
        # [B, C, T]
        x = self.proj_signal(signal[:, None])
        # [B, E]
        embed = torch.cat([self.embedder(steps), spk], dim=-1)
        # L x [B, C, T]
        skips = 0.
        for block in self.blocks:
            # [B, C, T], [B, C, T]
            x, skip = block(x, embed, context)
            skips = skips + skip
        # [B, T]
        return self.proj_out(
            skips * (len(self.blocks) ** -0.5) + self.bias).squeeze(dim=1)

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
