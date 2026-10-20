"""Tiled Triton matmul: C = A @ B, A [M, K], B [K, N], fp32 or bf16 inputs, fp32 accumulation.

Structure follows Triton tutorial 03 (matrix multiplication). Not wired into the model yet (M5 adds
the SiLU*up epilogue). B may be any strided view, e.g. `weight.t()` of an nn.Linear weight [N, K].
"""
import torch
import triton
import triton.language as tl


def _config(bm, bn, bk, warps, stages, group_m=8):
    return triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": group_m},
                         num_warps=warps, num_stages=stages)


# Small search space. Configs that need more shared memory than the GPU has (sm86: 99 KB per block)
# fail to compile and the autotuner skips them (timing = inf). Example: fp32 16x64x128 with 4 stages
# keeps 3 tiles of (16*128 + 128*64) * 4 B = 40 KB in flight = 120 KB.
CONFIGS = [
    _config(128, 128, 32, 4, 4),
    _config(128, 128, 32, 8, 3),
    _config(128, 64, 32, 4, 4),
    _config(64, 128, 32, 4, 4),
    _config(64, 64, 32, 4, 4),
    _config(128, 256, 32, 8, 3),
    _config(64, 64, 64, 4, 3),
    _config(16, 64, 128, 4, 4),  # tl.dot needs BLOCK_M >= 16; for decode (M = 1..16)
]


# key: retune when the shape or the fp32 precision mode changes. The autotuner also adds the argument
# dtypes to the key, so fp32 and bf16 are tuned separately.
@triton.autotune(configs=CONFIGS, key=["M", "N", "K", "INPUT_PRECISION"])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    INPUT_PRECISION: tl.constexpr,  # "ieee" or "tf32" for fp32 inputs; None for bf16
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # One program computes one BLOCK_M x BLOCK_N tile of C.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # GROUP_M swizzle: walk tiles in column-major order inside groups of GROUP_M tile-rows, so
    # programs running at the same time share A rows and B columns in L2.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row/column indices of this tile. Out-of-range rows/columns wrap with % so loads stay unmasked
    # along M and N; their results are thrown away by the masked store at the end.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # K tail: zero-fill past the end so it adds nothing to the dot product.
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_left, other=0.0)
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor, input_precision: str = "ieee") -> torch.Tensor:
    """a [M, K] @ b [K, N] -> [M, N] in a's dtype. input_precision only matters for fp32 inputs:
    "ieee" = true fp32 FMA, "tf32" = tensor cores with tf32-rounded inputs."""
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0], (a.shape, b.shape)
    assert a.dtype == b.dtype and a.dtype in (torch.float32, torch.bfloat16), (a.dtype, b.dtype)
    assert input_precision in ("ieee", "tf32")
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        INPUT_PRECISION=input_precision if a.dtype == torch.float32 else None,
    )
    return c
