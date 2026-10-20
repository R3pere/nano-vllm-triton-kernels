"""Correctness of nanovllm/kernels/qk_norm_rope_kv.py (fused q/k RMSNorm + RoPE + paged KV write).

Reference: the torch composition the kernel replaces, run eagerly (torch.compiler.set_stance("force_eager")), i.e.
the code exactly as written: q_norm(q), k_norm(k) (layers/layernorm.py), rotary_emb(positions, q, k)
(layers/rotary_embedding.py), store_kvcache(k, v, k_cache, v_cache, slot_mapping) (layers/attention.py).
Qwen3-0.6B sizes (16 q heads, 8 kv heads, head_dim 128, rope_theta 1e6, max_position 40960), bf16, q/k/v are the
strided views of a qkv_proj output. Cases: tokens 1/7/128/4097; positions random (non-contiguous, up to 40959) or
0..T-1; slot_mapping distinct random slots of a 2*T-slot cache with every 5th token (4, 9, ...) set to -1. Both
sides start from the same random cache contents.

The end-to-end output can't get a plain ulp bound: RoPE subtracts (x1*cos - x2*sin), and where the result nearly
cancels, a 1-ulp difference in a normed input (the sum-of-squares order differs from PyTorch's, as in M2) is many
ulps of the small output (seen: 11 ulps on 0.023, from inputs ~0.3). So the two stages are checked separately:
  norm stage   run the kernel with an identity rotation table (cos = 1, sin = 0, all positions 0): its outputs are
               then exactly its normed q and k. Within MAX_ULP = 2 bf16 ulps of the eager q_norm / k_norm (M2's limit).
  rope stage   the kernel's real output must equal, bitwise, the eager rotary_emb applied to the kernel's own normed
               q and k (from the norm stage): the rotation itself (fp32, no FMA contraction) adds no difference.
  KV write     k_cache rows at the written slots equal k_out bitwise, v_cache rows equal v bitwise, and every row
               of either cache that slot_mapping doesn't name is untouched (slot -1 skipped).

`python tests/test_kernels_qk_rope_kv.py` prints, per case and output, how many elements are bitwise equal and the
largest ulp difference end to end against the eager torch code, the compile backend (q/k norm and rope through
torch.compile) and the triton backend (M2's rms_norm kernel, rope through torch.compile), plus the norm stage.

Usage: pytest tests/test_kernels_qk_rope_kv.py   (or: python tests/test_kernels_qk_rope_kv.py)
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.kernels.qk_norm_rope_kv import qk_norm_rope_kv
from nanovllm.kernels.rms_norm import rms_norm
from nanovllm.layers.attention import store_kvcache
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding

MAX_ULP = 2
TOKENS = [1, 7, 128, 4097]
EPS = 1e-6
HEAD_DIM, Q_HEADS, KV_HEADS, MAX_POS, THETA, BLOCK = 128, 16, 8, 40960, 1e6, 256


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in bf16 steps between two bf16 tensors (bit patterns mapped to a monotonic integer line)."""
    def line(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (line(a) - line(b)).abs()


def stats(a, b):
    d = ulp_diff(a, b)
    return int((d == 0).sum()), d.numel(), int(d.max()) if d.numel() else 0


_rope = None


def rope():
    global _rope
    if _rope is None:
        _rope = RotaryEmbedding(HEAD_DIM, HEAD_DIM, MAX_POS, THETA).cuda()
    return _rope


def eager(fn, *args):
    with torch.compiler.set_stance("force_eager"):
        return fn(*args)


def make_inputs(tokens, random_positions):
    g = torch.Generator(device="cuda").manual_seed(tokens * 2 + random_positions)
    norms = []
    for _ in range(2):
        norm = RMSNorm(HEAD_DIM, eps=EPS).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(1 + 0.1 * torch.randn(HEAD_DIM, device="cuda", generator=g))
        norms.append(norm)
    qkv = torch.randn(tokens, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM, device="cuda", generator=g).to(torch.bfloat16)
    q, k, v = qkv.split([Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
    q, k, v = q.view(-1, Q_HEADS, HEAD_DIM), k.view(-1, KV_HEADS, HEAD_DIM), v.view(-1, KV_HEADS, HEAD_DIM)
    if random_positions:
        positions = torch.randint(0, MAX_POS, (tokens,), device="cuda", generator=g)
    else:
        positions = torch.arange(tokens, device="cuda")
    num_blocks = (2 * tokens + BLOCK - 1) // BLOCK
    slots = torch.randperm(num_blocks * BLOCK, device="cuda", generator=g)[:tokens].to(torch.int32)
    slots[4::5] = -1
    cache = torch.randn(2, num_blocks, BLOCK, KV_HEADS, HEAD_DIM, device="cuda", generator=g).to(torch.bfloat16)
    return dict(norms=norms, q=q, k=k, v=v, positions=positions, slots=slots, cache=cache)


def run_fused(x, identity_rope=False):
    cache = x["cache"].clone()
    positions, cos_sin = x["positions"], rope().cos_sin_cache
    if identity_rope:
        positions = torch.zeros_like(positions)
        cos_sin = torch.cat([torch.ones(HEAD_DIM // 2), torch.zeros(HEAD_DIM // 2)]).view(1, 1, HEAD_DIM).cuda()
    q_out, k_out = qk_norm_rope_kv(x["q"], x["k"], x["v"], positions, x["norms"][0].weight, x["norms"][1].weight, EPS,
                                   cos_sin, cache[0], cache[1], x["slots"], write_k=True)
    return dict(q=q_out, k_out=k_out, k_cache=cache[0], v_cache=cache[1])


def run_reference(x, mode):
    """mode: "eager" (torch code as written), "compile" (compile backend), "triton" (triton backend)."""
    cache = x["cache"].clone()
    q_norm, k_norm = x["norms"]
    if mode == "triton":
        qr, kr = rope()(x["positions"], rms_norm(x["q"], q_norm.weight, EPS), rms_norm(x["k"], k_norm.weight, EPS))
    elif mode == "compile":
        qr, kr = rope()(x["positions"], q_norm(x["q"]), k_norm(x["k"]))
    else:
        qr, kr = eager(rope(), x["positions"], eager(q_norm, x["q"]), eager(k_norm, x["k"]))
    store_kvcache(kr, x["v"], cache[0], cache[1], x["slots"])
    return dict(q=qr, k_out=kr, k_cache=cache[0], v_cache=cache[1])


def written_rows(x, cache):
    return cache.view(-1, KV_HEADS, HEAD_DIM)[x["slots"][x["slots"] >= 0].long()]


CASES = [(t, rp) for t in TOKENS for rp in (True, False)]
IDS = [f"tokens={t} positions={'random' if rp else 'arange'}" for t, rp in CASES]


@pytest.mark.parametrize("tokens,random_positions", CASES, ids=IDS)
def test_norm_stage(tokens, random_positions):
    x = make_inputs(tokens, random_positions)
    normed = run_fused(x, identity_rope=True)
    for name, out, norm, inp in (("q", normed["q"], x["norms"][0], x["q"]), ("k", normed["k_out"], x["norms"][1], x["k"])):
        worst = stats(out, eager(norm, inp))[2]
        assert worst <= MAX_ULP, f"{name} norm: {worst} bf16 ulps from the eager torch code"


@pytest.mark.parametrize("tokens,random_positions", CASES, ids=IDS)
def test_rope_stage_bitwise(tokens, random_positions):
    x = make_inputs(tokens, random_positions)
    normed, out = run_fused(x, identity_rope=True), run_fused(x)
    qr, kr = eager(rope(), x["positions"], normed["q"], normed["k_out"])
    assert torch.equal(out["q"], qr)
    assert torch.equal(out["k_out"], kr)


@pytest.mark.parametrize("tokens,random_positions", CASES, ids=IDS)
def test_kv_write(tokens, random_positions):
    x = make_inputs(tokens, random_positions)
    out = run_fused(x)
    real = x["slots"] >= 0
    assert torch.equal(written_rows(x, out["k_cache"]), out["k_out"][real])
    assert torch.equal(written_rows(x, out["v_cache"]), x["v"][real])
    untouched = torch.ones(x["cache"].shape[1] * BLOCK, dtype=torch.bool, device="cuda")
    untouched[x["slots"][real].long()] = False
    for i, name in enumerate(("k_cache", "v_cache")):
        before = x["cache"][i].view(-1, KV_HEADS, HEAD_DIM)[untouched]
        assert torch.equal(out[name].view(-1, KV_HEADS, HEAD_DIM)[untouched], before), name


if __name__ == "__main__":
    torch._dynamo.config.recompile_limit = 64    # as in model_runner.py
    inference = torch.inference_mode(); inference.__enter__()    # as ModelRunner runs the model
    modes = ("eager", "compile", "triton")
    print("end to end: kernel output vs each backend's composition (caches: the written rows only); "
          "norm stage: kernel's normed q/k vs eager q_norm/k_norm")
    print(f"{'case':36s} {'output':8s}" + "".join(f" {'vs ' + m + ': equal':>25s} {'max':>4s}" for m in modes + ("norm stage",)))
    for (tokens, rp), name in zip(CASES, IDS):
        x = make_inputs(tokens, rp)
        out, normed = run_fused(x), run_fused(x, identity_rope=True)
        refs = [run_reference(x, m) for m in modes]
        norm_ref = {"q": eager(x["norms"][0], x["q"]), "k_out": eager(x["norms"][1], x["k"])}
        for o in out:
            line = f"{name:36s} {o:8s}"
            for r in refs:
                a, b = (written_rows(x, out[o]), written_rows(x, r[o])) if o.endswith("cache") else (out[o], r[o])
                eq, numel, worst = stats(a, b)
                line += f" {eq:>11d}/{numel:<13d} {worst:4d}"
            if o in norm_ref:
                eq, numel, worst = stats(normed[o], norm_ref[o])
                line += f" {eq:>11d}/{numel:<13d} {worst:4d}"
            print(line)
