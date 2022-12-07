from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Scheduler(nn.Module):
    """Fixed noise scheduler, reference from DiffWave, Kong et al., 2021.
    """
    def __init__(self, steps: int, start: float, end: float):
        """Initializer.
        Args:
            steps: the number of the diffusion steps.
            start: beta start.
            end: beta end.
        """
        super().__init__()
        self.register_buffer(
            'beta',
            torch.linspace(start, end, steps, dtype=torch.float32),
            persistent=False)

    def forward(self) -> Tuple[torch.Tensor, Tuple]:
        """Compute beta values.
        Returns:
            [torch.float32; [1 + steps]], log-SNR.
            [torch.float32; [1 + steps]], beta values.
        """
        # [1 + steps]
        beta = F.pad(self.beta, [1, 0])
        # alpha_bar
        logsnr = torch.cumprod(1 - beta, dim=-1).logit()
        return logsnr, beta
