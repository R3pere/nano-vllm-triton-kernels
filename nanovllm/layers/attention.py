import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.kernels.qk_norm_rope_kv import qk_norm_rope_kv
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # kernel_backend "triton_fused": (q_norm, k_norm, rotary_emb) of the model's attention, set by it (a plain tuple,
        # so nn.Module doesn't register the modules twice). q and k then arrive un-normed and un-rotated, and
        # forward does norm + rope + KV write in one kernel: it needs slot_mapping, which pre_attn can't read.
        self.norm_rope = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor | None = None):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if self.norm_rope is not None:
            q_norm, k_norm, rotary_emb = self.norm_rope
            # prefill without prefix cache reads k from here; everything else reads it from the cache
            write_k = context.is_prefill and context.block_tables is None
            q, k_out = qk_norm_rope_kv(q, k, v, positions, q_norm.weight, k_norm.weight, q_norm.eps,
                                       rotary_emb.cos_sin_cache, k_cache, v_cache, context.slot_mapping, write_k)
            if write_k:
                k = k_out
        elif k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode: every sequence brings the same number of query tokens (1, or 1+k when verifying guesses)
            num_seqs = context.context_lens.size(0)
            q = q.reshape(num_seqs, -1, self.num_heads, self.head_dim)
            # causal mask is aligned to the bottom right: query i of n sees the cache up to context_len-n+i
            o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
            o = o.view(-1, self.num_heads, self.head_dim)
        return o
