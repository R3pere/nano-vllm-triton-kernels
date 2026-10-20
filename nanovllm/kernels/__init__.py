# Which implementation RMSNorm and SiluAndMul use: "compile" (their @torch.compile torch code) or "triton"
# (rms_norm.py, swiglu.py). ModelRunner sets it from Config.kernel_backend before building the model;
# each layer reads it once, in __init__.
backend = "compile"
