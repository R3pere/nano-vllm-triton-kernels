# Forward-only RMSNorm and Add+RMSNorm, adapted from Liger-Kernel
# (src/liger_kernel/ops/rms_norm.py and fused_add_rms_norm.py, commit 67d7222):
#
#   Copyright 2024 LinkedIn Corporation. All Rights Reserved.
#   Licensed under the BSD 2-Clause License, see
#   https://github.com/linkedin/Liger-Kernel/blob/main/LICENSE
#   Liger's rms_norm.py in turn credits Unsloth (Apache-2.0) for the row-per-program structure.
#
# Changes from Liger: no RSTD buffer and no backward (inference only), no offset / Gemma / casting-mode
# options, no autotune or blocked variant, and the numerics follow nano-vllm's torch code
# (nanovllm/layers/layernorm.py) instead of a Liger casting mode:
#   rms_norm:           y = (x.float() * rsqrt(mean(x.float()^2) + eps)).to(dtype) * w
#                       = Liger casting_mode="llama" (fp32 rstd and normalisation, cast, weight multiply in x's dtype).
#   fused_add_rms_norm: s = x.float() + r.float(); returns (rms_norm of the fp32 s, s.to(dtype)).
#                       No Liger mode matches: Liger adds x + r in bf16 (rounding s) and normalises the rounded
#                       sum, nano-vllm normalises the unrounded fp32 sum and only rounds the residual it returns.
#
# One program per row, the whole row in registers (BLOCK = next_power_of_2(n_cols)) and num_warps from
# Liger's calculate_settings. Everything is fixed by the shape, so CUDA graph capture never triggers tuning.
import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt


def calculate_settings(n: int) -> tuple[int, int]:
    """Liger's ops/utils.py calculate_settings (CUDA branch): block = next power of two of the row length."""
    block = triton.next_power_of_2(n)
    assert block <= 65536, f"row of {n} elements is too long for one program"
    num_warps = 32 if block >= 32768 else 16 if block >= 8192 else 8 if block >= 2048 else 4
    return block, num_warps


@triton.jit
def _rms_norm_kernel(
    y_ptr, x_ptr, w_ptr,
    rows_per_group, x_group_stride, x_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # x may be a strided 3D view [T, H, n] (q/k norm on a slice of the qkv output): row = t * H + h.
    # y is contiguous [T * H, n].
    row = tl.program_id(0).to(tl.int64)
    x_ptr += (row // rows_per_group) * x_group_stride + (row % rows_per_group) * x_row_stride
    y_ptr += row * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + cols, mask=mask, other=0)
    w = tl.load(w_ptr + cols, mask=mask, other=0)
    xf = x.to(tl.float32)
    rstd = rsqrt(tl.sum(xf * xf, axis=0) / n_cols + eps)
    y = (xf * rstd).to(x.dtype) * w
    tl.store(y_ptr + cols, y, mask=mask)


@triton.jit
def _fused_add_rms_norm_kernel(
    y_ptr, s_ptr, x_ptr, r_ptr, w_ptr,
    x_row_stride, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # y and s are contiguous [rows, n].
    row = tl.program_id(0).to(tl.int64)
    x_ptr += row * x_row_stride
    r_ptr += row * r_row_stride
    y_ptr += row * n_cols
    s_ptr += row * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + cols, mask=mask, other=0)
    r = tl.load(r_ptr + cols, mask=mask, other=0)
    w = tl.load(w_ptr + cols, mask=mask, other=0)
    s = x.to(tl.float32) + r.to(tl.float32)
    tl.store(s_ptr + cols, s.to(x.dtype), mask=mask)
    rstd = rsqrt(tl.sum(s * s, axis=0) / n_cols + eps)
    y = (s * rstd).to(x.dtype) * w
    tl.store(y_ptr + cols, y, mask=mask)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim. x: [rows, n] or [T, H, n] with a contiguous last dim (any other strides)."""
    n = x.shape[-1]
    if x.dim() != 3 and x.dim() != 2 or x.stride(-1) != 1:
        x = x.reshape(-1, n).contiguous()
    if x.dim() == 2:
        groups, rows_per_group, group_stride, row_stride = x.shape[0], 1, x.stride(0), 0
    else:
        groups, rows_per_group, group_stride, row_stride = x.shape[0], x.shape[1], x.stride(0), x.stride(1)
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = groups * rows_per_group
    if rows:
        block, num_warps = calculate_settings(n)
        _rms_norm_kernel[(rows,)](y, x, w, rows_per_group, group_stride, row_stride, n, eps,
                                  BLOCK_SIZE=block, num_warps=num_warps)
    return y


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(RMSNorm(x + residual), x + residual), the sum in fp32. x, residual: [rows, n] with a contiguous last dim."""
    assert x.shape == residual.shape and x.dim() == 2 and x.stride(-1) == 1 and residual.stride(-1) == 1
    rows, n = x.shape
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    s = torch.empty_like(x, memory_format=torch.contiguous_format)
    if rows:
        block, num_warps = calculate_settings(n)
        _fused_add_rms_norm_kernel[(rows,)](y, s, x, residual, w, x.stride(0), residual.stride(0), n, eps,
                                            BLOCK_SIZE=block, num_warps=num_warps)
    return y, s
