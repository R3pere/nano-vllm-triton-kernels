# Which implementation the layers use: "compile" (their @torch.compile torch code), "triton" (RMSNorm and SiluAndMul
# as rms_norm.py, swiglu.py) or "triton_fused" (triton, plus Qwen3's q/k norm, rotary and KV-cache write as the one
# kernel of qk_norm_rope_kv.py, run inside Attention.forward). ModelRunner sets it from Config.kernel_backend before
# building the model; each layer reads it once, in __init__.
backend = "compile"
