# nano-vllm-triton-kernels

这个仓库在 [nano-vllm-spec-decode](https://github.com/R3pere/nano-vllm-spec-decode)（投机解码 + 混合批处理）的基础上继续做，用 Triton 手写了推理热路径上的几个算子，又把 attention 前面那一段处理合成了一个 kernel。代码都在 `nanovllm/kernels/`，用 `LLM(path, kernel_backend=...)` 或者 `bench.py --kernel-backend` 切换：

- `compile`：默认，用 torch.compile。
- `triton`：RMSNorm、Add+RMSNorm、SiLU×Mul 换成 Triton 版本。
- `triton_fused`：在 `triton` 的基础上再加融合 kernel。

具体写了这些：

- `kernels/matmul.py`：分块 matmul，带 `GROUP_M` swizzle 和 autotune，支持 fp32 / tf32 / bf16 输入，fp32 累加，B 可以是任意 stride。还没接进模型。
- `kernels/rms_norm.py`、`kernels/swiglu.py`：从 Liger-Kernel 裁出来的 forward-only 版本，直接读带 stride 的视图，不拷贝。没用 autotune，所以录 CUDA Graph 时不会触发调参。
- `kernels/qk_norm_rope_kv.py`：融合 kernel。qkv_proj 和 flash-attn 之间的活全在这一个 kernel 里做完：从 qkv 输出里读 q/k/v，按头做 RMSNorm，做 RoPE，写出 q，再按 `slot_mapping` 把 k/v 直接写进 paged KV cache。分段 CUDA Graph 里拿不到 `slot_mapping`，所以我把 norm 和 rope 挪进了 attention 的 eager 段。那里原来就要启动一次写 KV 的 kernel，现在换成融合 kernel，launch 数没变。

## 结果

RTX 3060 Laptop 6GB，Qwen3-0.6B，BF16。

| 对比 | 结果 |
|---|---|
| 融合 kernel vs torch.compile，每层 attention 前的处理 | 5 个 kernel 变成 1 个；一步 decode 从 444 个 kernel 降到 332 个 |
| 同上，ncu，T=2048 | DRAM 流量 68.1 → 39.8 MB（理论最小值的 1.76× → 1.02×），GPU 时间 234 → 129 µs |
| 同上，微基准 | 快 2.1–5.3 倍；T=8192 时 301 GB/s，是拷贝带宽上限的 95% |
| 固定 batch decode，整步 CUDA Graph 重放 | fused / compile = 0.981–0.988，batch 32 每步省 180–218 µs；墙钟 TPOT 的差别在噪声范围内 |
| Triton matmul vs cuBLAS | FP32 4096 是 1.11 倍；BF16 小 M（qkv，M=32）是 1.38 倍；M=2048 时只有 0.92–0.99 倍 |
| 正确性 | kernel 单测 24 + 24 + 55 项通过；端到端 27 个变体都通过 FP32 参考检查，`triton_fused` 的 graph 和 eager 逐 token 一致（10/10） |

单独替换 RMSNorm / SiLU 基本没有收益：大 T 时和 torch.compile 的 GPU 时间差不多（±1%），T ≤ 64 时 SiLU 还慢 18–49%。`rms_norm` 一行一个 program，在 head_dim=128 的 q/k norm 上比 Inductor 慢一倍。attention 前的处理只占 decode 图时间的 1.4–3.1%，单个算子换来换去省不出时间，真正有用的是把几次 launch 合成一次、少读写几遍显存。

测试在 `tests/test_kernels_{matmul,norm_act,qk_rope_kv}.py`，基准在 `scripts/bench_{matmul,norm_act,qk_rope_kv,decode}.py`。

## 安装

需要 Python 3.10–3.12、CUDA GPU，依赖见 `pyproject.toml`（torch、triton、transformers、flash-attn、xxhash）。

```bash
pip install -e .
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
```

---

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（MIT 许可，提交 bb823b3），模型、分页 KV 缓存、前缀缓存等基础部分来自上游。
