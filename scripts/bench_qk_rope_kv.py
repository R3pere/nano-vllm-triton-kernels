"""Microbenchmark of the attention pre-segment (everything between qkv_proj and flash-attn) for one layer, at the
model's sizes, per kernel backend:
  compile       q_norm, k_norm (RMSNorm through @torch.compile), rotary_emb (@torch.compile), store_kvcache
  triton        q_norm, k_norm as nanovllm/kernels/rms_norm.py, rotary_emb (@torch.compile), store_kvcache
  triton_fused  nanovllm/kernels/qk_norm_rope_kv.py (one kernel)
Input: a qkv_proj output [T, (Hq + 2*Hkv)*D] (q, k, v are strided views of it), random positions, distinct random
slots in a paged cache. Output: rotated q; normed + rotated k and v in the cache. Decode-style (no k_out): the
fused kernel only writes k to the cache, the others also leave the rotated k tensor (they need it for the store).

Bytes the segment must move (e = 2, bf16), per token: read q, k, v (Hq + 2*Hkv)*D*e, its position (8) and slot (4),
its cos/sin row D*4 (fp32); write q_out Hq*D*e and the k, v cache rows 2*Hkv*D*e. Plus the two norm weights 2*D*e
once. Qwen3-0.6B (Hq 16, Hkv 8, D 128): 8192 + 12 + 512 + 4096 + 4096 = 16,908 B per token + 512 B.
Effective GB/s = these bytes / time; the separate-kernel backends move more (intermediates).

Timing as scripts/bench_norm_act.py: do_bench (L2 flushed before each call, includes host dispatch when it is longer
than the GPU work) and do_bench_cudagraph (GPU time only), --repeats rounds with the backends interleaved, medians.

Usage: python scripts/bench_qk_rope_kv.py [--model PATH] [--max-tokens 8192] [--repeats 3] [--out result.json]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import triton

from nanovllm.kernels.qk_norm_rope_kv import qk_norm_rope_kv
from nanovllm.kernels.rms_norm import rms_norm
from nanovllm.layers.attention import store_kvcache
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding

BACKENDS = ["compile", "triton", "triton_fused"]
BLOCK = 256


def load_shape(model):
    with open(os.path.join(model, "config.json")) as f:
        cfg = json.load(f)
    return dict(hq=cfg["num_attention_heads"], hkv=cfg["num_key_value_heads"], d=cfg["head_dim"], eps=cfg["rms_norm_eps"],
                theta=cfg["rope_theta"], max_pos=cfg["max_position_embeddings"])


def segment_bytes(T, s, write_k=False):
    per_token = (s["hq"] + 2 * s["hkv"]) * s["d"] * 2 + 8 + 4 + s["d"] * 4 + (s["hq"] + 2 * s["hkv"]) * s["d"] * 2
    return T * (per_token + (s["hkv"] * s["d"] * 2 if write_k else 0)) + 2 * s["d"] * 2


def make_segments(s, max_tokens):
    """{backend: fn(inputs, write_k)} and make_inputs(T) for one layer's pre-attention segment."""
    norms = []
    for _ in range(2):
        norm = RMSNorm(s["d"], eps=s["eps"]).cuda().to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(1 + 0.1 * torch.randn(s["d"], device="cuda"))
        norms.append(norm)
    q_norm, k_norm = norms
    rope = RotaryEmbedding(s["d"], s["d"], s["max_pos"], s["theta"]).cuda()
    blocks = 2 * max_tokens // BLOCK + 1
    cache = torch.zeros(2, blocks, BLOCK, s["hkv"], s["d"], device="cuda", dtype=torch.bfloat16)
    k_cache, v_cache = cache[0], cache[1]
    eps = s["eps"]

    def make_inputs(T):
        qkv = torch.randn(T, (s["hq"] + 2 * s["hkv"]) * s["d"], device="cuda", dtype=torch.bfloat16)
        q, k, v = qkv.split([s["hq"] * s["d"], s["hkv"] * s["d"], s["hkv"] * s["d"]], dim=-1)
        q, k, v = q.view(T, s["hq"], s["d"]), k.view(T, s["hkv"], s["d"]), v.view(T, s["hkv"], s["d"])
        positions = torch.randint(0, s["max_pos"], (T,), device="cuda")
        slots = torch.randperm(blocks * BLOCK, device="cuda")[:T].to(torch.int32)
        return q, k, v, positions, slots

    def compile_(x, write_k):
        q, k, v, positions, slots = x
        qr, kr = rope(positions, q_norm.rms_forward(q), k_norm.rms_forward(k))
        store_kvcache(kr, v, k_cache, v_cache, slots)
        return qr, kr

    def triton_(x, write_k):
        q, k, v, positions, slots = x
        qr, kr = rope(positions, rms_norm(q, q_norm.weight, eps), rms_norm(k, k_norm.weight, eps))
        store_kvcache(kr, v, k_cache, v_cache, slots)
        return qr, kr

    def fused(x, write_k):
        q, k, v, positions, slots = x
        return qk_norm_rope_kv(q, k, v, positions, q_norm.weight, k_norm.weight, eps, rope.cos_sin_cache,
                               k_cache, v_cache, slots, write_k)

    return {"compile": compile_, "triton": triton_, "triton_fused": fused}, make_inputs


def bench(fn, warmup, rep):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def bench_graph(fn, rep):
    return triton.testing.do_bench_cudagraph(fn, rep=rep, return_mode="median")


@torch.inference_mode()    # as ModelRunner runs the model
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=25, help="do_bench warmup, ms")
    p.add_argument("--rep", type=int, default=100, help="do_bench measuring time, ms")
    p.add_argument("--out")
    args = p.parse_args()
    torch._dynamo.config.recompile_limit = 64    # as in model_runner.py

    s = load_shape(args.model)
    fns, make_inputs = make_segments(s, args.max_tokens)
    tokens = [2 ** i for i in range(args.max_tokens.bit_length()) if 2 ** i <= args.max_tokens]
    inputs = {T: make_inputs(T) for T in tokens}
    for T in tokens:    # compile (and JIT) everything before timing
        for b in BACKENDS:
            fns[b](inputs[T], False)
    torch.cuda.synchronize()

    runs, graph_runs = {}, {}
    for r in range(args.repeats):
        for T in tokens:
            for b in BACKENDS:
                fn = lambda: fns[b](inputs[T], False)
                runs.setdefault((b, T), []).append(bench(fn, args.warmup, args.rep))
                torch.cuda.synchronize()
                graph_runs.setdefault((b, T), []).append(bench_graph(fn, args.rep))
        print(f"repeat {r + 1}/{args.repeats} done", flush=True)

    results = []
    print(f"{'tokens':>6s} {'bytes':>10s} " + " ".join(f"{b + ' us':>15s} {'graph us':>8s} {'GB/s':>6s}" for b in BACKENDS))
    for T in tokens:
        nbytes = segment_bytes(T, s)
        line = f"{T:6d} {nbytes:10d} "
        for b in BACKENDS:
            ms, gms = statistics.median(runs[b, T]), statistics.median(graph_runs[b, T])
            results.append({"backend": b, "tokens": T, "ms": ms, "ms_runs": runs[b, T], "graph_ms": gms,
                            "graph_ms_runs": graph_runs[b, T], "bytes": nbytes, "gbps": nbytes / ms / 1e6,
                            "graph_gbps": nbytes / gms / 1e6})
            line += f"{ms * 1e3:15.2f} {gms * 1e3:8.2f} {nbytes / ms / 1e6:6.1f} "
        print(line, flush=True)

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.mem,clocks.max.sm,temperature.gpu,clocks.sm",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out = {"model": args.model, "shape": s, "args": vars(args), "torch": torch.__version__, "triton": triton.__version__,
           "gpu": smi, "bytes_formula": "T*((Hq+2Hkv)*D*2 read + 8 + 4 + D*4 + (Hq+2Hkv)*D*2 written) + 2*D*2",
           "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
