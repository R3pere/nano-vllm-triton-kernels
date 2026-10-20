"""Correctness of nanovllm/kernels/matmul.py against a float64 reference.

Tolerances (checked with torch.testing.assert_close against (A.double() @ B.double()) on the same inputs):
  fp32, input_precision="ieee": rtol=1e-4, atol=1e-4 * sqrt(K)  (fp32 accumulation-order error only)
  fp32, input_precision="tf32": rtol=1e-2, atol=1e-2 * sqrt(K)  (inputs rounded to 10-bit mantissa)
  bf16 in / bf16 out:           rtol=1.6e-2, atol=1e-2 * sqrt(K) (output rounded to bf16: 8-bit mantissa)
Inputs are randn, so |C| ~ sqrt(K); atol scales with that.

Usage: pytest tests/test_kernels_matmul.py   (or: python tests/test_kernels_matmul.py)
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.kernels.matmul import matmul

TOL = {
    ("fp32", "ieee"): (1e-4, 1e-4),
    ("fp32", "tf32"): (1e-2, 1e-2),
    ("bf16", "ieee"): (1.6e-2, 1e-2),
}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

# Includes shapes that are not multiples of any BLOCK_M/N/K (16..256), M = 1 (decode) and
# K smaller than BLOCK_K.
SHAPES = [
    (1, 1024, 4096),
    (1, 3072, 1024),
    (7, 33, 65),
    (17, 129, 257),
    (100, 300, 70),
    (127, 255, 1000),
    (512, 512, 512),
    (1000, 1000, 1000),
    (3, 5, 2),
]


def _check(M, N, K, dtype, prec, b_transposed):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=DTYPES[dtype])
    if b_transposed:  # nn.Linear layout: weight [N, K], matmul uses weight.t()
        b = torch.randn(N, K, device="cuda", dtype=DTYPES[dtype]).t()
    else:
        b = torch.randn(K, N, device="cuda", dtype=DTYPES[dtype])
    c = matmul(a, b, input_precision=prec)
    assert c.shape == (M, N) and c.dtype == a.dtype
    ref = a.double() @ b.double()
    rtol, atol = TOL[(dtype, prec)]
    torch.testing.assert_close(c.double(), ref, rtol=rtol, atol=atol * math.sqrt(K))


@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("dtype,prec", list(TOL))
@pytest.mark.parametrize("b_transposed", [False, True])
def test_matmul(M, N, K, dtype, prec, b_transposed):
    _check(M, N, K, dtype, prec, b_transposed)


def test_ieee_is_not_tf32():
    # Guard that "ieee" really avoids tf32: on K=4096 randn inputs tf32 error is ~1e-2 * sqrt(K) scale,
    # ieee is ~1e-5. If the precision flag were ignored, both would match.
    torch.manual_seed(0)
    a = torch.randn(256, 4096, device="cuda")
    b = torch.randn(4096, 256, device="cuda")
    ref = a.double() @ b.double()
    err_ieee = (matmul(a, b, "ieee").double() - ref).abs().max().item()
    err_tf32 = (matmul(a, b, "tf32").double() - ref).abs().max().item()
    assert err_ieee < 1e-2 < err_tf32, (err_ieee, err_tf32)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
