"""Correctness of nanovllm/kernels/rms_norm.py and swiglu.py against the layers' own torch code.

Reference: RMSNorm / SiluAndMul (nanovllm/layers) with the compile backend, run eagerly
(torch.compiler.set_stance("force_eager")), i.e. the torch code exactly as written. All bf16, Qwen3-0.6B sizes:
  RMSNorm         rows 1/7/128/4097 x n 128 (q/k norm) and 1024 (hidden); also the real q/k-norm input, a strided
                  [tokens, heads, 128] view of the qkv output
  Add+RMSNorm     rows 1/7/128/4097 x n 128 and 1024; both outputs (normed, new residual) are checked
  SiluAndMul      rows 1/7/128/4097 x [gate | up] of 2 x 3072
Tolerance: every output element within MAX_ULP = 2 bf16 ulps of the reference (ulps counted on the bf16 bit
patterns). Both compute in fp32 and round to bf16 at the same points, so the only difference is the fp32
summation order of the mean of squares (and the rsqrt / sigmoid implementation). That can move the normalised
value, which is rounded to bf16 before the weight multiply, by one bf16 step; multiplied by a weight above 1 the
step can become 2 ulps of the output (seen: 12 of 4.2M elements of add_rms_norm rows=4097 n=1024). The new
residual of add_rms_norm and SiluAndMul have no reduction and are expected to be bitwise equal.

`python tests/test_kernels_norm_act.py` prints, per case, how many elements are bitwise equal and the largest ulp
difference, against both the eager torch code and the torch.compile (Inductor) version.

Usage: pytest tests/test_kernels_norm_act.py   (or: python tests/test_kernels_norm_act.py)
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.kernels.rms_norm import rms_norm, fused_add_rms_norm
from nanovllm.kernels.swiglu import silu_and_mul
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm

MAX_ULP = 2
ROWS = [1, 7, 128, 4097]
EPS = 1e-6
HIDDEN, HEAD_DIM, INTERMEDIATE, Q_HEADS, KV_HEADS = 1024, 128, 3072, 16, 8


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in bf16 steps between two bf16 tensors (bit patterns mapped to a monotonic integer line)."""
    def line(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (line(a) - line(b)).abs()


def make_norm(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    norm = RMSNorm(n, eps=EPS).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(n, device="cuda", generator=g))
    return norm, g


def randn(shape, g):
    return torch.randn(shape, device="cuda", generator=g).to(torch.bfloat16)


def eager(fn, *args):
    with torch.compiler.set_stance("force_eager"):
        return fn(*args)


# Each case returns (name, [(triton output, module callable, module args, which output)]).
def case_rms(rows, n):
    norm, g = make_norm(n, rows * 7 + n)
    x = randn((rows, n), g)
    return [(rms_norm(x, norm.weight, EPS), norm, (x,), None)]


def case_qk_norm(tokens):
    norm, g = make_norm(HEAD_DIM, tokens * 3 + 1)
    qkv = randn((tokens, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM), g)
    q, k, _ = qkv.split([Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
    q, k = q.view(-1, Q_HEADS, HEAD_DIM), k.view(-1, KV_HEADS, HEAD_DIM)
    return [(rms_norm(t, norm.weight, EPS), norm, (t,), None) for t in (q, k)]


def case_add_rms(rows, n):
    norm, g = make_norm(n, rows * 5 + n)
    x, r = randn((rows, n), g), randn((rows, n), g)
    y, s = fused_add_rms_norm(x, r, norm.weight, EPS)
    return [(y, norm, (x, r), 0), (s, norm, (x, r), 1)]


def case_silu(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = randn((rows, 2 * INTERMEDIATE), g)
    return [(silu_and_mul(x), SiluAndMul(), (x,), None)]


CASES = ([(f"rms_norm rows={r} n={n}", case_rms, (r, n)) for n in (HEAD_DIM, HIDDEN) for r in ROWS]
         + [(f"q/k norm tokens={t} (strided view)", case_qk_norm, (t,)) for t in ROWS]
         + [(f"add_rms_norm rows={r} n={n}", case_add_rms, (r, n)) for n in (HEAD_DIM, HIDDEN) for r in ROWS]
         + [(f"silu_and_mul rows={r} n={INTERMEDIATE}", case_silu, (r,)) for r in ROWS])


def compare(out, module, args, which, compiled):
    ref = module(*args) if compiled else eager(module, *args)
    if which is not None:
        ref = ref[which]
    assert out.shape == ref.shape and out.dtype == ref.dtype == torch.bfloat16
    d = ulp_diff(out, ref)
    return int((d == 0).sum()), out.numel(), int(d.max())


@pytest.mark.parametrize("name,case,args", CASES, ids=[c[0] for c in CASES])
def test_matches_torch_code(name, case, args):
    for out, module, margs, which in case(*args):
        _, _, worst = compare(out, module, margs, which, compiled=False)
        limit = 0 if which == 1 or name.startswith("silu") else MAX_ULP    # no reduction: bitwise
        assert worst <= limit, f"{name}: {worst} bf16 ulps from the eager torch code"


if __name__ == "__main__":
    torch._dynamo.config.recompile_limit = 64    # as in model_runner.py: past the limit torch.compile silently runs eagerly
    inference = torch.inference_mode(); inference.__enter__()    # as ModelRunner runs the model (Inductor then compiles an inference graph)
    print(f"{'case':42s} {'output':6s} {'vs eager: equal':>22s} {'max ulp':>7s} {'vs compile: equal':>24s} {'max ulp':>7s}")
    for name, case, args in CASES:
        for i, (out, module, margs, which) in enumerate(case(*args)):
            e_eq, numel, e_max = compare(out, module, margs, which, compiled=False)
            c_eq, _, c_max = compare(out, module, margs, which, compiled=True)
            label = ["y", "res"][which] if which is not None else ["q", "k"][i] if "q/k" in name else "y"
            print(f"{name:42s} {label:6s} {e_eq:>11d}/{numel:<10d} {e_max:7d} {c_eq:>12d}/{numel:<11d} {c_max:7d}")
