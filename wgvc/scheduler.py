from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Scheduler(nn.Module):
    """Fixed noise scheduler, reference from Improved DDPM, Nichol & Dhariwal, 2021.
    """
    def __init__(self, steps: int, s: float = 0.008):
        """Initializer.
        Args:
            steps: the number of the diffusion steps.
            s: offset parameter for preventing too small beta.
        """
        super().__init__()
        # [steps + 1]
        a = torch.arange(1 + steps) / steps
        f = ((a + s) / (1 + s) * np.pi * 0.5).cos().square()
        self.register_buffer('alpha', f / f[0], persistent=False)

    def forward(self) -> Tuple[torch.Tensor, Tuple]:
        """Compute beta values.
        Returns:
            [torch.float32; [1 + steps]], log-SNR.
            [torch.float32; [1 + steps]], beta values.
        """
        # alias
        a = self.alpha
        # [1 + steps]
        logsnr = a.logit()
        # [1 + steps]
        beta = F.pad(1 - a[1:] / a[:-1], [1, 0])
        return logsnr, beta
