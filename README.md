> **`triton-ops` 分支：推理算子融合**。在下面的投机解码 + 混合批处理版本基础上，用 Triton 手写推理热路径上的算子，并把 attention 前的整段处理融合成一个 kernel。全部代码在 `nanovllm/kernels/`，用 `LLM(path, kernel_backend=...)` 或 `bench.py --kernel-backend` 切换：`compile`（默认，torch.compile）、`triton`（RMSNorm / Add+RMSNorm / SiLU×Mul 换成 Triton）、`triton_fused`（再加融合 kernel）。
>
> - **Triton matmul**（`kernels/matmul.py`）：分块 + `GROUP_M` swizzle + autotune，fp32 / tf32 / bf16 输入、fp32 累加，B 支持任意 stride。暂未接入模型。
> - **RMSNorm / Add+RMSNorm / SiLU×Mul**（`kernels/rms_norm.py`、`kernels/swiglu.py`）：从 Liger-Kernel 裁出的 forward-only kernel，直接读带 stride 的视图、不拷贝；不用 autotune，CUDA Graph 录图安全。
> - **q/k RMSNorm + RoPE + paged KV 写入融合**（`kernels/qk_norm_rope_kv.py`）：一个 kernel 完成 qkv_proj 与 flash-attn 之间的全部工作，即从 qkv 输出读 q/k/v、按头做 RMSNorm、RoPE、写 q，并按 `slot_mapping` 把 k/v 直接写进 paged KV cache。为了在分段 CUDA Graph 下拿到 `slot_mapping`，把 norm 和 rope 移进了 attention 的 eager 段，launch 数不变。
>
> 关键结果（RTX 3060 Laptop 6GB，Qwen3-0.6B，BF16）：
>
> | 场景 | 结果 |
> |---|---|
> | 融合 kernel vs torch.compile（每层 attention 前处理段） | 5 个 kernel → 1 个；每步 decode 444 → 332 个 kernel |
> | 同上，ncu，T=2048 | DRAM 流量 68.1 → 39.8 MB（理论最小值的 1.76× → 1.02×）；GPU 时间 234 → 129 µs |
> | 同上，微基准 | 提速 2.1–5.3×；T=8192 时 301 GB/s，为拷贝带宽上限的 95% |
> | 固定 batch decode，整步 CUDA Graph 重放时间 | fused / compile = 0.981–0.988，batch 32 每步省 180–218 µs；墙钟 TPOT 差异在噪声内 |
> | Triton matmul vs cuBLAS | FP32 4096：1.11×；BF16 小 M（qkv M=32）1.38×；M=2048 时 0.92–0.99× |
> | 正确性 | kernel 单测 24 + 24 + 55 项通过；端到端 27 个变体通过 FP32 裁判，`triton_fused` 的 graph 与 eager 逐 token 一致 10/10 |
>
> 负结果：RMSNorm / SiLU 的 Triton kernel 在大 T 时 GPU 时间与 torch.compile 持平（±1%），在 T ≤ 64 时 SiLU 慢 18–49%；`rms_norm` 一行一个 program，在 head_dim=128 的 q/k norm 上比 Inductor 慢一倍。这也是做融合 kernel 的原因：前处理段只占 decode 图时间的 1.4–3.1%，单算子替换省不出时间，合并 launch、减少显存往返才有收益。
>
> 测试与基准：`tests/test_kernels_{matmul,norm_act,qk_rope_kv}.py`，`scripts/bench_{matmul,norm_act,qk_rope_kv,decode}.py`。
>
> **本分支说明**：这是在上游 nano-vllm（提交 bb823b3）基础上做的个人学习项目。加入了**投机解码**（n-gram 和小模型草拟，贪心与随机采样，CUDA Graph，按接受率动态选草拟长度 k）和 **vLLM V1 式的混合批处理**（prefill 与 decode 同一步计算，分段 CUDA Graph），用 FP32 裁判、eager/图逐 token 一致、故障注入和随机压力测试保证正确性，并在 A100 + Qwen3-8B 上和 vLLM 0.15.1 做了同条件对比。**完整报告（设计取舍、负面结果、复现命令）见 [docs/REPORT.md](docs/REPORT.md)**。下面上游 logo 之后的内容是上游原文。
>
> 关键结果（A100-SXM4-80GB，目标 Qwen3-8B，草拟 Qwen3-0.6B，BF16；“3 次”为 3 次均值）：
>
> | 场景 | 负载 | 结果 | 次数 |
> |---|---|---|---|
> | n-gram 动态 k vs 不投机 | 照抄型任务，温度 0，并发 1 | 88 → 335 tok/s，**3.8×** | 3 |
> | 0.6B 草拟动态 k vs 不投机 | 普通问答，温度 0，并发 1 | 88 → 116 tok/s，1.3× | 3 |
> | 投机解码的负收益 | 普通问答，温度 0，并发 64 | n-gram 动态 k 0.94×，0.6B 草拟 k=4 0.79× | 1 |
> | 混合批处理 vs vLLM，同预算 2048 | ShareGPT 400 条，泊松 36.5 个/秒 | 吞吐 3565 vs 3688 tok/s（**96.7%**）；TTFT P99 **132 vs 190 ms** | 3 |
> | 混合批处理 vs vLLM，同预算 2048 | ShareGPT 1000 条同时到达 | 吞吐 4197 vs 4333 tok/s（96.9%）；decode 慢约 3% | 3 |
> | 正确性 | 16 个变体，10 / 30 个 prompt | 全部通过 FP32 裁判；eager 与图四对 10/10、30/30 逐 token 一致 | 1 |
>
> 复现：`bash scripts/run_p5.sh`（全部实验）和 `bash scripts/run_p5.sh --sections env correctness budget --reuse <P5b 结果目录>`（同预算对比）；每个数字对应的单条 `bench.py` 命令见 REPORT 第 5 节。
>
> 常用命令（`MODEL` 指向模型目录，默认 `~/huggingface/Qwen3-0.6B/`）：
>
> ```bash
> python tests/test_spec_unit.py                          # 投机解码单元测试（只用 CPU，不需要模型）
> MODEL=/path/to/Qwen3-0.6B python tests/test_greedy.py   # 正确性测试
> MODEL=/path/to/Qwen3-0.6B python bench.py               # 性能测试：TTFT、TPOT、吞吐
> MODEL=/path/to/Qwen3-0.6B python bench.py --workload copy --concurrency 4 --spec ngram   # 投机解码
> MODEL=/path/to/Qwen3-0.6B python scripts/bench_matrix.py                  # 整张对比表（约 1.5 小时）
> MODEL=/path/to/Qwen3-0.6B python scripts/step_cost.py                     # 验证一步比普通一步贵多少
> MODEL=/path/to/Qwen3-0.6B python scripts/profile_decode.py --spec ngram   # 一步 decode 里各部分的耗时
> ```
>
> 本分支对上游的改动：
> - `SamplingParams(temperature=0)` 表示贪心采样（上游不允许）。
> - n-gram 投机解码：`LLM(path, speculative_method="ngram", num_speculative_tokens=4)`。每一步先从前文里“照抄”猜几个 token，再让模型一次性检查，猜对的直接收下。贪心和随机采样都支持，输出与不开投机时等价。`llm.get_spec_stats()` 返回接受率等统计。检查这一步也走 CUDA Graph；同时 decode 的请求超过 `spec_max_batch_size`（默认 512 // (1+k)）时这一步不猜。只支持单卡。
> - Draft model 投机解码：`LLM(path, speculative_method="draft", draft_model=小模型路径)`，小模型逐个猜 k 个 token，大模型一次检查。两个模型共用块表，各有一份 KV cache。
> - 动态 k：`spec_dynamic_k=True`（此时 `num_speculative_tokens` 是 k 的上限，建议 8），每一步按各请求最近的接受率和启动时测出的每步耗时选 k（0、1、2、4、8），接受率低时自动不猜。
> - 混合批处理：`enable_mixed_batching=True`，`max_num_batched_tokens` 是每步 token 预算（A100 + 8B 建议 ≥ 2048）；混合步默认走分段 CUDA Graph（`mixed_cuda_graph`）。暂不支持和 draft model 同时开。
> - 序列长度到 `max_model_len` 就结束（上游不限制，CUDA Graph 下会越界出错）。
> - `bench.py` 增加了“照抄型”任务（`--workload copy`）、普通问答任务（`--workload chat`）、“随机猜”最坏情况（`--spec-proposer random`）、并发上限（`--concurrency`），以及 `--spec draft --draft-model 路径`、`--dynamic-k`。

<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)