"""Microbenchmark of the three L1/L2 ops at the model's sizes: eager torch / torch.compile / Triton (nanovllm/kernels).

  rms_norm      x [T, hidden]                          bytes = 2*T*hidden*e + hidden*e          (read x, write y, read w)
  add_rms_norm  x, residual [T, hidden]                bytes = 4*T*hidden*e + hidden*e          (read x, r; write y, new r; read w)
  silu_and_mul  x [T, 2*intermediate] -> [T, inter.]   bytes = 3*T*intermediate*e               (read gate, up; write out)
e = 2 (bf16). These are the bytes the op must move (each tensor once); effective GB/s = bytes / time. Eager torch
moves more (fp32 temporaries), Inductor's RMSNorm also writes a 4-byte rstd per row; that is not counted.

eager = the layer's torch code with torch.compile bypassed; compile = the layer as nano-vllm runs it
(@torch.compile); triton = nanovllm/kernels. Timing: triton.testing.do_bench (median of `rep` ms of calls, L2
flushed before each call, `warmup` ms first) and do_bench_cudagraph (calls replayed from a CUDA graph: GPU time
without host dispatch), repeated --repeats times with the backends interleaved; the median of the repeats is
reported. GB/s and % of ceiling use the do_bench time. Practical bandwidth ceiling: the same do_bench on a large
tensor copy (torch copy_), measured warm, before and after the ops.

Usage: python scripts/bench_norm_act.py [--model PATH] [--max-tokens 8192] [--repeats 3] [--out result.json]
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

from nanovllm.kernels.rms_norm import rms_norm, fused_add_rms_norm
from nanovllm.kernels.swiglu import silu_and_mul
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm

BACKENDS = ["eager", "compile", "triton"]


def uncompiled(method):
    """The original Python function behind a @torch.compile method."""
    fn = getattr(method, "_torchdynamo_orig_callable", None)
    assert fn is not None, "can't find the uncompiled function"
    return fn


def make_ops(hidden, inter, eps):
    norm = RMSNorm(hidden, eps=eps).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(hidden, device="cuda"))
    act = SiluAndMul()
    w = norm.weight
    rms_eager, add_eager = uncompiled(RMSNorm.rms_forward), uncompiled(RMSNorm.add_rms_forward)
    act_eager = uncompiled(SiluAndMul.silu_and_mul_compiled)
    bf16 = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    return {
        "rms_norm": (lambda T: (bf16(T, hidden),),
                     {"eager": lambda x: rms_eager(norm, x), "compile": norm.rms_forward, "triton": lambda x: rms_norm(x, w, eps)},
                     lambda T: (2 * T * hidden + hidden) * 2),
        "add_rms_norm": (lambda T: (bf16(T, hidden), bf16(T, hidden)),
                         {"eager": lambda x, r: add_eager(norm, x, r), "compile": norm.add_rms_forward,
                          "triton": lambda x, r: fused_add_rms_norm(x, r, w, eps)},
                         lambda T: (4 * T * hidden + hidden) * 2),
        "silu_and_mul": (lambda T: (bf16(T, 2 * inter),),
                         {"eager": lambda x: act_eager(act, x), "compile": act.silu_and_mul_compiled, "triton": silu_and_mul},
                         lambda T: 3 * T * inter * 2),
    }


def bench(fn, warmup, rep):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def bench_graph(fn, rep):
    return triton.testing.do_bench_cudagraph(fn, rep=rep, return_mode="median")


def measure_ceiling(nbytes_copy, warmup, rep, repeats):
    src = torch.empty(nbytes_copy // 2, device="cuda", dtype=torch.bfloat16).normal_()
    dst = torch.empty_like(src)
    runs = [bench(lambda: dst.copy_(src), warmup, rep) for _ in range(repeats)]
    return statistics.median(runs), runs


@torch.inference_mode()    # as ModelRunner runs the model. Without it RMSNorm.weight requires grad and Inductor
def main():                # compiles a training graph that also writes the tensors saved for backward (rstd, ...)
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=25, help="do_bench warmup, ms")
    p.add_argument("--rep", type=int, default=100, help="do_bench measuring time, ms")
    p.add_argument("--copy-mib", type=int, default=512, help="size of the copied tensor for the bandwidth ceiling")
    p.add_argument("--out")
    args = p.parse_args()
    torch._dynamo.config.recompile_limit = 64    # as in model_runner.py

    with open(os.path.join(args.model, "config.json")) as f:
        cfg = json.load(f)
    hidden, inter, eps = cfg["hidden_size"], cfg["intermediate_size"], cfg["rms_norm_eps"]
    ops = make_ops(hidden, inter, eps)
    tokens = [2 ** i for i in range(args.max_tokens.bit_length()) if 2 ** i <= args.max_tokens]

    inputs = {(op, T): make(T) for op, (make, _, _) in ops.items() for T in tokens}
    for (op, T), inp in inputs.items():    # compile (and JIT) everything before timing
        for b in BACKENDS:
            ops[op][1][b](*inp)
    torch.cuda.synchronize()

    # Practical ceiling: device-to-device copy, measured after the GPU has been busy (warm-up below) and again at
    # the end; the larger of the two medians is used. Measured first thing in a cold process it reads ~20% low.
    copy_bytes = args.copy_mib * 2 ** 20
    for _ in range(3):
        bench(lambda: [fns[b](*inputs[op, tokens[-1]]) for op, (_, fns, _) in ops.items() for b in BACKENDS], args.warmup, args.rep)
    copy_before = measure_ceiling(copy_bytes, args.warmup, args.rep, args.repeats)
    print(f"ceiling (before): copy of {args.copy_mib} MiB, {copy_before[0]:.3f} ms -> {2 * copy_bytes / copy_before[0] / 1e6:.1f} GB/s", flush=True)

    # Two timings per point: do_bench (each call launched from Python, L2 flushed before it: includes host dispatch
    # whenever that is longer than the kernel) and do_bench_cudagraph (the calls replayed from one CUDA graph, as
    # nano-vllm's decode runs them: GPU time only, L2 not flushed).
    runs, graph_runs = {}, {}
    for r in range(args.repeats):
        for op, (_, fns, _) in ops.items():
            for T in tokens:
                for b in BACKENDS:
                    inp = inputs[op, T]
                    runs.setdefault((op, b, T), []).append(bench(lambda: fns[b](*inp), args.warmup, args.rep))
                    graph_runs.setdefault((op, b, T), []).append(bench_graph(lambda: fns[b](*inp), args.rep))
        print(f"repeat {r + 1}/{args.repeats} done", flush=True)

    copy_after = measure_ceiling(copy_bytes, args.warmup, args.rep, args.repeats)
    copy_ms = min(copy_before[0], copy_after[0])
    ceiling = 2 * copy_bytes / copy_ms / 1e6
    print(f"ceiling (after): {copy_after[0]:.3f} ms -> {2 * copy_bytes / copy_after[0] / 1e6:.1f} GB/s; used {ceiling:.1f} GB/s", flush=True)

    results = []
    print(f"{'op':14s} {'tokens':>6s} " + " ".join(f"{b + ' us':>11s} {'graph us':>8s} {'GB/s':>6s} {'%ceil':>5s}" for b in BACKENDS))
    for op, (_, _, nbytes) in ops.items():
        for T in tokens:
            line = f"{op:14s} {T:6d} "
            for b in BACKENDS:
                ms, gms = statistics.median(runs[op, b, T]), statistics.median(graph_runs[op, b, T])
                gbps = nbytes(T) / ms / 1e6
                results.append({"op": op, "backend": b, "tokens": T, "ms": ms, "ms_runs": runs[op, b, T],
                                "graph_ms": gms, "graph_ms_runs": graph_runs[op, b, T],
                                "bytes": nbytes(T), "gbps": gbps, "pct_ceiling": 100 * gbps / ceiling,
                                "graph_gbps": nbytes(T) / gms / 1e6})
                line += f"{ms * 1e3:11.2f} {gms * 1e3:8.2f} {gbps:6.1f} {100 * gbps / ceiling:5.1f} "
            print(line, flush=True)

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.mem,clocks.max.sm,temperature.gpu,clocks.sm",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out = {"model": args.model, "hidden": hidden, "intermediate": inter, "eps": eps, "dtype": "bfloat16",
           "args": vars(args), "torch": torch.__version__, "triton": triton.__version__, "gpu": smi,
           "ceiling": {"what": f"torch copy_ of {args.copy_mib} MiB (read + write), faster of before/after", "ms": copy_ms,
                       "before_ms_runs": copy_before[1], "after_ms_runs": copy_after[1], "gbps": ceiling},
           "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
