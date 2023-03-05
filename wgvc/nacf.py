"""
Copyright (C) https://github.com/praat/praat
 
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF


class NACF(nn.Module):
    """Normalized autocorrelation-based Pitch estimation, reimplementation of Praat.
    """
    def __init__(self,
                 sr: int,
                 frame_time: float = 0.01,
                 freq_min: float = 75.,
                 freq_max: float = 600.,
                 down_sr: int = 16000,
                 k: int = 15,
                 thold_silence: float = 0.03,
                 thold_voicing: float = 0.45,
                 cost_octave: float = 0.01,
                 cost_jump: float = 0.35,
                 cost_vuv: float = 0.14,
                 median_win: Optional[int] = 5):
        """Initializer.
        Args:
            sr: sampling rate.
            frame_time: duration of the frame.
            freq_min, freq_max: frequency min and max.
            down_sr: downsampling sr for fast computation.
            k: the maximum number of the candidates.
        """
        super().__init__()
        self.sr = sr
        self.strides = int(down_sr * frame_time)
        self.fmin, self.fmax = freq_min, freq_max
        self.tmax = int(down_sr // freq_min)
        self.tmin = int(down_sr // freq_max)
        self.down_sr = down_sr
        self.k = min(max(k, int(freq_max / freq_min)), self.tmax - self.tmin)

        # set windows based on tau-max
        self.w = int(2 ** np.ceil(np.log2(self.tmax))) + 1
        self.register_buffer(
            'window', torch.hann_window(self.w), persistent=False)
        # [w + 1], normalized autocorrelation of the window
        r = torch.fft.rfft(self.window, self.w * 2, dim=-1)
        # [w]
        ws = torch.fft.irfft(r * r.conj(), dim=-1)[:self.w]
        ws = ws / ws[0]
        self.register_buffer('nacw', ws, persistent=False)

        # alias
        self.thold_silence = thold_silence
        self.thold_voicing = thold_voicing
        self.cost_octave = cost_octave
        # correction
        c = 1.  # c = 0.01 * down_sr
        self.cost_jump = cost_jump * c
        self.cost_vuv = cost_vuv * c
        self.median_win = median_win

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Estimate the pitch from the input signal.
        Args:
            inputs: [torch.float32; [..., T]], input signal, [-1, 1]-ranged.
        Returns:
            [torch.float32; [..., S]], estimated pitch sequence.
        """
        x = AF.resample(inputs, self.sr, self.down_sr)
        # [..., T / strides, w]
        frames = F.pad(x, [0, self.w]).unfold(-1, self.w, self.strides)
        # [..., T / strides, w]
        frames = (frames - frames.mean(dim=-1)[..., None]) * self.window

        # [...]
        global_peak = inputs.abs().amax(dim=-1)
        # [..., T / strides]
        local_peak = frames.abs().amax(dim=-1)
        # [..., T / strides]
        intensity = torch.where(
            local_peak > global_peak[..., None],
            torch.tensor(1., device=x.device),
            local_peak / global_peak[..., None])

        # [..., T / strides, w + 1]
        fft = torch.fft.rfft(frames, self.w * 2, dim=-1)
        # [..., T / strides, w]
        acf = torch.fft.irfft(fft * fft.conj(), dim=-1)[..., :self.w]
        # [..., T / strides, w], normalized autocorrelation
        nacf = acf / (acf[..., :1] * self.nacw)
        # [..., T / strides, tmax + 1]
        nacf = nacf[..., :self.tmax + 1]

        # [..., T / strides, tmax], x[i + 1] - x[i]
        d = nacf.diff(dim=-1)
        # [..., T / strides, tmax - 1], inc & dec
        localmax = (d[..., :-1] >= 0) & (d[..., 1:] <= 0)
        # [..., T / strides, tmax - 1]
        flag = localmax & (nacf[..., 1:-1] > 0.5 * self.thold_voicing)

        # [..., T / strides, tmax - 1], parabolic interpolation
        n, c, p = nacf[..., 2:], nacf[..., 1:-1], nacf[..., :-2]
        dr =  0.5 * (n - p) / (2. * c - n - p)
        # [tmax - 1]
        a = torch.arange(self.tmax - 1, device=dr.device)
        # [..., T / strides, tmax - 1]
        freqs = self.down_sr / (1 + (dr + a).clamp_min(0.))
        ## TODO: sinc interpolation, depth=30
        logits = nacf[..., 1:self.tmax]
        # reflect logits of high values (for short windows)
        logits = logits.where(logits <= 1., 1 / logits)
        # additional penalty
        logits = logits - self.cost_octave * (self.fmin / freqs).log2()
        # masking
        FLOOR = -1e5
        logits.masked_fill_(~flag, FLOOR)
        # [..., T / strides, k], [..., T / strides, k], topk
        logits, indices = logits.topk(self.k, dim=-1)
        # [..., T / strides, k]
        freqs = freqs.gather(-1, indices)

        ## TODO: maximize sinc interpolation, depth=4
        logits, freqs = logits, freqs
        # [..., T / strides]
        logits_uv = 2. - intensity / (
            self.thold_silence / (1. + self.thold_voicing))
        logits_uv = self.thold_voicing + logits_uv.clamp_min(0.)
        # [..., T / strides, k]
        voiced = (logits > FLOOR) & (freqs < self.fmax)
        # [..., T / strides, k]
        delta = torch.where(
            ~voiced,
            logits_uv[..., None],
            logits - self.cost_octave * (self.fmax / freqs).log2())

        # [..., T / strides - 1, k, k]
        trans = self.cost_jump * (
            freqs[..., :-1, :, None] / freqs[..., 1:, None, :]).log2().abs()
        # both voiceless
        trans.masked_fill_(
            ~voiced[..., :-1, :, None] & ~voiced[..., 1:, None, :], 0.)
        # voice transition
        trans.masked_fill_(
            voiced[..., :-1, :, None] != voiced[..., 1:, None, :], self.cost_vuv)

        # S(=T / strides)
        steps = delta.shape[-2]
        # [..., k]
        value = delta[..., 0, :]
        # [..., S, k]
        ptrs = torch.zeros_like(delta)
        for i in range(1, steps):
            # [..., k]
            value, ptrs[..., i, :] = (
                value[..., None]
                - trans[..., i - 1, :, :] + delta[..., i, None, :]).max(dim=-2)
        # [..., S], backtracking
        states = torch.zeros_like(delta[..., 0], dtype=torch.long)
        # initial state
        states[..., -1] = value.argmax(dim=-1)
        for i in range(steps - 2, -1, -1):
            # [...]
            states[..., i] = ptrs[..., i + 1, :].gather(
                -1, states[..., i + 1, None]).squeeze(dim=-1)

        # [..., T / strides, 1], sampling
        freqs = freqs.gather(-1, states[..., None])
        # masking unvoiced
        freqs.masked_fill_(~voiced.gather(-1, states[..., None]), 0.)
        # [..., T / strides]
        f0 = freqs.squeeze(dim=-1)
        # median pool
        if self.median_win is not None:
            w = self.median_win // 2
            # replication
            f0 = torch.cat(
                [f0[..., :1]] * w + [f0] + [f0[..., -1:]] * w, dim=-1)
            f0 = torch.median(
                f0.unfold(-1, self.median_win, 1),
                dim=-1).values
        return f0
