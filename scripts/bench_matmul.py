"""Benchmark nanovllm/kernels/matmul.py against cuBLAS (torch.matmul / F.linear).

(a) fp32 squares, Triton input_precision="ieee" vs torch.matmul with TF32 disabled.
(b) bf16 Linear shapes from a model config.json: x [M, in] @ weight[out, in].t(), Triton vs F.linear.

Timing: triton.testing.do_bench (median of its own repeats); each case is warmed up first (the first
call also runs autotune), then do_bench runs 3 times and the median of the 3 medians is reported.

Usage: python scripts/bench_matmul.py --model <dir with config.json> --out result.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.kernels.matmul import _matmul_kernel, matmul


def gpu_state():
    q = "temperature.gpu,clocks.sm,clocks.mem,power.draw"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def bench(fn, repeats=3):
    fn()
    torch.cuda.synchronize()
    triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")  # warm-up pass, discarded
    runs = [triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median") for _ in range(repeats)]
    return statistics.median(runs), runs


def winner():
    cfg = _matmul_kernel.best_config
    timings = getattr(_matmul_kernel, "configs_timings", None) or {}
    return {
        "config": {**cfg.kwargs, "num_warps": cfg.num_warps, "num_stages": cfg.num_stages},
        # timings of all configs from the autotune run of this shape (ms; inf = did not compile/fit)
        "autotune_ms": {str(c): (t[0] if isinstance(t, list) else t) for c, t in timings.items()},
    }


def triton_case(fn):
    _matmul_kernel.configs_timings = None
    fn()  # triggers autotune for a new key
    torch.cuda.synchronize()
    return winner()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fp32-sizes", default="1024,2048,4096")
    p.add_argument("--tokens", default="1,32,512,2048")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.cuda.get_device_name(0)
    out = {"device": dev, "torch": torch.__version__, "triton": triton.__version__,
           "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "fp32": [], "bf16": []}

    # (a) fp32 squares
    print("gpu", gpu_state())
    for s in [int(x) for x in args.fp32_sizes.split(",")]:
        torch.manual_seed(0)
        a = torch.randn(s, s, device="cuda")
        b = torch.randn(s, s, device="cuda")
        win = triton_case(lambda: matmul(a, b, "ieee"))
        err = (matmul(a, b, "ieee").double() - a.double() @ b.double()).abs().max().item()
        t_tri, r_tri = bench(lambda: matmul(a, b, "ieee"))
        t_cub, r_cub = bench(lambda: torch.matmul(a, b))
        flops = 2 * s ** 3
        row = {"size": s, "triton_ms": t_tri, "triton_runs": r_tri, "cublas_ms": t_cub, "cublas_runs": r_cub,
               "triton_gflops": flops / t_tri / 1e6, "cublas_gflops": flops / t_cub / 1e6,
               "triton_max_abs_err_vs_fp64": err, "winner": win, "gpu_after": gpu_state()}
        out["fp32"].append(row)
        print(f"fp32 {s}: triton {row['triton_gflops']:.0f} GFLOPs, cublas {row['cublas_gflops']:.0f} GFLOPs, "
              f"winner {win['config']}")

    # (b) bf16 Linear shapes
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    hd = cfg.get("head_dim", h // cfg["num_attention_heads"])
    q, kv = cfg["num_attention_heads"] * hd, cfg["num_key_value_heads"] * hd
    layers = {"qkv_proj": (h, q + 2 * kv), "o_proj": (q, h), "gate_up_proj": (h, 2 * inter),
              "down_proj": (inter, h)}  # name: (in_features, out_features), tensor parallel = 1
    out["layers"] = layers
    for name, (k, n) in layers.items():
        torch.manual_seed(0)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k ** -0.5
        for m in [int(x) for x in args.tokens.split(",")]:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = F.linear(x, w)
            win = triton_case(lambda: matmul(x, w.t()))
            err = (matmul(x, w.t()).float() - ref.float()).abs().max().item()
            t_tri, r_tri = bench(lambda: matmul(x, w.t()))
            t_cub, r_cub = bench(lambda: F.linear(x, w))
            flops = 2 * m * n * k
            bytes_ = 2 * (m * k + n * k + m * n)
            row = {"layer": name, "M": m, "N": n, "K": k, "triton_ms": t_tri, "triton_runs": r_tri,
                   "cublas_ms": t_cub, "cublas_runs": r_cub,
                   "triton_tflops": flops / t_tri / 1e9, "cublas_tflops": flops / t_cub / 1e9,
                   "triton_gbps": bytes_ / t_tri / 1e6, "cublas_gbps": bytes_ / t_cub / 1e6,
                   "max_abs_diff_vs_cublas": err, "winner": win}
            out["bf16"].append(row)
            print(f"bf16 {name} M={m}: triton {t_tri*1e3:.1f} us, cublas {t_cub*1e3:.1f} us, "
                  f"ratio {t_cub/t_tri:.2f}, winner {win['config']}")
    out["gpu_end"] = gpu_state()
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
