import torch
from torch import nn
import torch.nn.functional as F

from nanovllm import kernels
from nanovllm.kernels.swiglu import silu_and_mul


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()
        self.backend = kernels.backend

    @torch.compile
    def silu_and_mul_compiled(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "triton":
            return silu_and_mul(x)
        return self.silu_and_mul_compiled(x)
