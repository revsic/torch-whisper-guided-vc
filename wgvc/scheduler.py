from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PDense(nn.Module):
    """Linear projection with positive weights.
    """
    def __init__(self, inputs: int, outputs: int):
        """Initializer.
        Args:
            inputs: size of the input channels.
            outputs: size of the output channels.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(outputs))
        # sample from U(-1/sqrt(inputs), 1/sqrt(inputs)), reference nn.Linear.
        torch.nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the inputs with positive weights.
        Args:
            inputs: [torch.float32; [..., I]], input tensor.
        Returns:
            [torch.float32; [..., O]], output tensor.
        """
        return torch.matmul(inputs, F.softplus(self.weights)) + self.bias


class Scheduler(nn.Module):
    """Log-SNR estimator, reference variational diffusion models, kingma et al., 2021.
    """
    def __init__(self, steps: int, channels: int, logit_min: float, logit_max: float):
        """Initializer.
        Args:
            steps: the number of the diffusion steps.
            channels: size of the hidden units.
            logit_min: lower bound of log-SNR.
            logit_max: upper bound of log-SNR.
        """
        super().__init__()
        self.steps = steps
        # use only positive dense.
        self.affine = PDense(1, 1)
        self.proj = nn.Sequential(
            PDense(1, channels), nn.Sigmoid(), PDense(channels, 1))
        # logit min, max
        self.logit_min = nn.Parameter(torch.tensor(logit_min, dtype=torch.float32))
        self.logit_max = nn.Parameter(torch.tensor(logit_max, dtype=torch.float32))

    def forward(self) -> Tuple[torch.Tensor, Tuple]:
        """Compute beta values.
        Returns:
            [torch.float32; [1 + steps]], log-SNR.
            [torch.float32; [1 + steps]], beta values.
        """
        device = self.affine.weights.device
        # [1 + S]
        times = torch.arange(
            0, self.steps + 1, dtype=torch.float32, device=device) / self.steps
        # [1 + S, 1]
        aff = self.affine(times[:, None])
        # [1 + S]
        sched = (aff + self.proj(aff)).squeeze(dim=-1)
        # [S + 1], normalized in range [0, 1], monotonically increase
        sched = (sched - sched[0]) / (sched[-1] - sched[0] + 1e-7)
        # [S + 1], range [logit_min, logit_max]
        nlogsnr = sched * (self.logit_max - self.logit_min) + self.logit_min
        # [S + 1], monotonically decrease
        alpha_bar = torch.sigmoid(-nlogsnr)
        # [S + 1]
        beta = 1. - alpha_bar / (F.pad(alpha_bar, [1, -1], value=1.) + 1e-7)
        # [S + 1]
        return -nlogsnr, beta
