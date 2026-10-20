# Forward-only SiLU-and-multiply, adapted from Liger-Kernel (src/liger_kernel/ops/swiglu.py, commit 67d7222):
#
#   Copyright 2024 LinkedIn Corporation. All Rights Reserved.
#   Licensed under the BSD 2-Clause License, see
#   https://github.com/linkedin/Liger-Kernel/blob/main/LICENSE
#
# Changes from Liger: no backward, no gate_multiplier, no Blackwell tiled variant, and the input is
# nano-vllm's SiluAndMul input as it is: one tensor [rows, 2 * n] holding [gate | up] in the last dim,
# read in place (no chunk / copy). Numerics as nanovllm/layers/activation.py runs eagerly:
#   out = silu(gate.float()).to(dtype) * up     (silu in fp32 rounded to bf16, then a bf16 multiply)
# which is also what Liger's kernel does.
import torch
import triton
import triton.language as tl

from nanovllm.kernels.rms_norm import calculate_settings


@triton.jit
def _silu_and_mul_kernel(out_ptr, x_ptr, x_row_stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_ptr += row * x_row_stride
    out_ptr += row * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    gate = tl.load(x_ptr + cols, mask=mask, other=0)
    up = tl.load(x_ptr + n_cols + cols, mask=mask, other=0)
    g = gate.to(tl.float32)
    out = (g * tl.sigmoid(g)).to(up.dtype) * up
    tl.store(out_ptr + cols, out, mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """silu(x[..., :n]) * x[..., n:] for x [..., 2 * n] with a contiguous last dim."""
    if x.dim() != 2 or x.stride(-1) != 1:
        x = x.reshape(-1, x.shape[-1]).contiguous()  # not hit: nano-vllm passes 2D contiguous input
    rows, n = x.shape[0], x.shape[1] // 2
    out = torch.empty(rows, n, dtype=x.dtype, device=x.device)
    if rows:
        block, num_warps = calculate_settings(n)
        _silu_and_mul_kernel[(rows,)](out, x, x.stride(0), n, BLOCK_SIZE=block, num_warps=num_warps)
    return out
