# Fused q/k RMSNorm + RoPE + paged KV-cache write (Qwen3 attention, everything between qkv_proj and flash-attn).
#
# Replaces four kernels per layer (q_norm, k_norm, rotary_emb, store_kvcache) with one. It reads q and k straight
# from the qkv_proj output (strided [T, H, D] views, no contiguous copy) and v, and writes:
#   q_out [T, Hq, D]  normed + rotated q, contiguous (what flash-attn takes)
#   k_cache/v_cache   normed + rotated k and raw v at slot_mapping[t] (rows with slot -1 are skipped, as store_kvcache does)
#   k_out [T, Hkv, D] normed + rotated k, only when asked (prefill without prefix cache reads k from here, not the cache)
#
# Numerics follow nano-vllm's torch code as written (and as it runs eagerly), with its two roundings:
#   RMSNorm (layers/layernorm.py rms_forward): n = (x.float() * rsqrt(mean(x.float()^2) + eps)).to(bf16) * w
#   RoPE (layers/rotary_embedding.py apply_rotary_emb): cos, sin = cos_sin_cache[pos] (fp32), halves x1 = n[:D/2],
#     x2 = n[D/2:]; y1 = x1*cos - x2*sin, y2 = x2*cos + x1*sin in fp32, then .to(bf16)
# The kernel is launched with enable_fp_fusion=False: torch computes x1*cos and x2*sin as separate rounded products,
# a fused multiply-add would round once and differ in the last fp32 bit. The sum of squares is reduced in a different
# order than PyTorch's reduce kernel, so a normalised value can still land on the other side of a bf16 rounding
# boundary now and then (1 ulp, 2 after the weight multiply), as in M2's rms_norm.
#
# Program mapping: grid = (T, Hkv // KV_PER_PROG). A program takes one token and KV_PER_PROG kv heads: those k and v
# heads and the R = Hq // Hkv q heads that share each of them (GQA). Heads are loaded as two halves of D/2, so RoPE
# needs no shuffle in registers, and cos/sin are read once per program. KV_PER_PROG and num_warps depend only on the
# model's shape and T only sets the grid (it is never a kernel argument), so a new batch size never compiles a new
# specialisation: CUDA graph safe, no autotune.
import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt


@triton.jit
def _norm_rope(x_ptrs, w_ptr, cos, sin, eps, HALF: tl.constexpr):
    """x_ptrs: [rows, HALF] pointers to the first halves of rows of length 2*HALF. Returns the two bf16 halves."""
    d = tl.arange(0, HALF)
    x1 = tl.load(x_ptrs).to(tl.float32)
    x2 = tl.load(x_ptrs + HALF).to(tl.float32)
    rstd = rsqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / (2 * HALF) + eps)
    w1 = tl.load(w_ptr + d)
    w2 = tl.load(w_ptr + HALF + d)
    n1 = ((x1 * rstd[:, None]).to(tl.bfloat16) * w1[None, :]).to(tl.float32)
    n2 = ((x2 * rstd[:, None]).to(tl.bfloat16) * w2[None, :]).to(tl.float32)
    y1 = n1 * cos[None, :] - n2 * sin[None, :]
    y2 = n2 * cos[None, :] + n1 * sin[None, :]
    return y1.to(tl.bfloat16), y2.to(tl.bfloat16)


@triton.jit
def _qk_norm_rope_kv_kernel(
    q_out_ptr, k_out_ptr, q_ptr, k_ptr, v_ptr, q_stride, k_stride, v_stride,
    positions_ptr, slot_mapping_ptr, q_w_ptr, k_w_ptr, cos_sin_ptr, cos_sin_stride, k_cache_ptr, v_cache_ptr, eps,
    HKV: tl.constexpr, R: tl.constexpr, KV_PER_PROG: tl.constexpr, HALF: tl.constexpr,
    WRITE_K: tl.constexpr, STORE_CACHE: tl.constexpr,
):
    D: tl.constexpr = 2 * HALF
    t = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1) * KV_PER_PROG
    d = tl.arange(0, HALF)
    pos = tl.load(positions_ptr + t)
    cos = tl.load(cos_sin_ptr + pos * cos_sin_stride + d)
    sin = tl.load(cos_sin_ptr + pos * cos_sin_stride + HALF + d)

    # q heads g*R ... (g+KV_PER_PROG)*R - 1
    hq = g * R + tl.arange(0, KV_PER_PROG * R)
    y1, y2 = _norm_rope(q_ptr + t * q_stride + hq[:, None] * D + d[None, :], q_w_ptr, cos, sin, eps, HALF)
    out = q_out_ptr + t * (HKV * R * D) + hq[:, None] * D + d[None, :]
    tl.store(out, y1)
    tl.store(out + HALF, y2)

    # kv heads g ... g+KV_PER_PROG-1
    hk = g + tl.arange(0, KV_PER_PROG)
    y1, y2 = _norm_rope(k_ptr + t * k_stride + hk[:, None] * D + d[None, :], k_w_ptr, cos, sin, eps, HALF)
    if WRITE_K:
        out = k_out_ptr + t * (HKV * D) + hk[:, None] * D + d[None, :]
        tl.store(out, y1)
        tl.store(out + HALF, y2)
    if STORE_CACHE:
        slot = tl.load(slot_mapping_ptr + t).to(tl.int64)
        if slot != -1:
            off = slot * (HKV * D) + hk[:, None] * D + d[None, :]
            tl.store(k_cache_ptr + off, y1)
            tl.store(k_cache_ptr + off + HALF, y2)
            dv = tl.arange(0, D)
            v = tl.load(v_ptr + t * v_stride + hk[:, None] * D + dv[None, :])
            tl.store(v_cache_ptr + slot * (HKV * D) + hk[:, None] * D + dv[None, :], v)


# Fixed launch heuristic: a program = 1 token x 4 kv heads (8 q heads), 4 warps. Chosen from a sweep of kv heads
# per program {1,2,4,8} x num_warps {1,2,4} (M3 report): best or within 0.1 us of best at T <= 32 (decode), ~8% behind
# the best (8 heads, 2 warps) at T = 8192.
KV_PER_PROG = 4
NUM_WARPS = 4


def qk_norm_rope_kv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor,
    q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float, cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor | None, write_k: bool,
    kv_per_prog: int | None = None, num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """q: [T, Hq, D], k, v: [T, Hkv, D], each with a contiguous head (any token stride: views of the qkv output work).
    positions: [T] int64. cos_sin_cache: [max_pos, 1, D] fp32 (cos | sin). k_cache, v_cache: [blocks, block_size,
    Hkv, D] contiguous, or empty (then nothing is stored). slot_mapping: [T] int32, -1 = don't store.
    Returns (q_out, k_out or None)."""
    T, Hq, D = q.shape
    Hkv = k.shape[1]
    kv_per_prog = kv_per_prog or KV_PER_PROG
    num_warps = num_warps or NUM_WARPS
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16 and k.shape == v.shape == (T, Hkv, D)
    assert Hq % Hkv == 0 and Hkv % kv_per_prog == 0 and D % 2 == 0
    for x in (q, k, v):
        assert x.stride(2) == 1 and x.stride(1) == D
    assert positions.shape == (T,) and cos_sin_cache.dtype == torch.float32 and cos_sin_cache.shape[-1] == D
    store = k_cache.numel() > 0
    if store:
        assert k_cache.is_contiguous() and v_cache.is_contiguous() and k_cache.shape[-2:] == (Hkv, D)
        assert slot_mapping.shape == (T,)
    q_out = torch.empty(T, Hq, D, dtype=q.dtype, device=q.device)
    k_out = torch.empty(T, Hkv, D, dtype=k.dtype, device=k.device) if write_k else None
    if T:
        _qk_norm_rope_kv_kernel[(T, Hkv // kv_per_prog)](
            q_out, k_out if write_k else q_out, q, k, v, q.stride(0), k.stride(0), v.stride(0),
            positions, slot_mapping if store else positions, q_weight, k_weight, cos_sin_cache, cos_sin_cache.stride(0),
            k_cache if store else q_out, v_cache if store else q_out, eps,
            HKV=Hkv, R=Hq // Hkv, KV_PER_PROG=kv_per_prog, HALF=D // 2, WRITE_K=write_k, STORE_CACHE=store,
            num_warps=num_warps, enable_fp_fusion=False,
        )
    return q_out, k_out
