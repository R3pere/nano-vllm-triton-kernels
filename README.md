## 算子融合

这个仓库在 [投机解码 + 混合批处理](https://github.com/R3pere/nano-vllm-spec-decode)（也就是下面第二节）的基础上继续做，用 Triton 手写了推理热路径上的几个算子，又把 attention 前面那一段处理合成了一个 kernel。代码都在 `nanovllm/kernels/`，用 `LLM(path, kernel_backend=...)` 或者 `bench.py --kernel-backend` 切换：

- `compile`：默认，用 torch.compile。
- `triton`：RMSNorm、Add+RMSNorm、SiLU×Mul 换成 Triton 版本。
- `triton_fused`：在 `triton` 的基础上再加融合 kernel。

具体写了这些：

- `kernels/matmul.py`：分块 matmul，带 `GROUP_M` swizzle 和 autotune，支持 fp32 / tf32 / bf16 输入，fp32 累加，B 可以是任意 stride。还没接进模型。
- `kernels/rms_norm.py`、`kernels/swiglu.py`：从 Liger-Kernel 裁出来的 forward-only 版本，直接读带 stride 的视图，不拷贝。没用 autotune，所以录 CUDA Graph 时不会触发调参。
- `kernels/qk_norm_rope_kv.py`：融合 kernel。qkv_proj 和 flash-attn 之间的活全在这一个 kernel 里做完：从 qkv 输出里读 q/k/v，按头做 RMSNorm，做 RoPE，写出 q，再按 `slot_mapping` 把 k/v 直接写进 paged KV cache。分段 CUDA Graph 里拿不到 `slot_mapping`，所以我把 norm 和 rope 挪进了 attention 的 eager 段。那里原来就要启动一次写 KV 的 kernel，现在换成融合 kernel，launch 数没变。

### 结果

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

## 投机解码 + 混合批处理

基于上游 nano-vllm（bb823b3）做的个人项目，加了两块：

- 投机解码：n-gram 和小模型草拟两种，支持贪心和随机采样，验证步走 CUDA Graph，按接受率动态选草拟长度 k。
- vLLM V1 式的混合批处理：prefill 和 decode 放在同一步里算，配合分段 CUDA Graph。

在 A100 + Qwen3-8B 上和 vLLM 0.15.1 用同一套客户端做了对比。设计、负面结果和每个数字的复现命令都在 [docs/REPORT.md](docs/REPORT.md)。

### 结果

A100-SXM4-80GB，目标模型 Qwen3-8B，草拟模型 Qwen3-0.6B，BF16。

| 场景 | 负载 | 结果 | 次数 |
|---|---|---|---|
| n-gram 动态 k vs 不投机 | 照抄型任务，温度 0，并发 1 | 88 → 335 tok/s，3.8× | 3 次均值 |
| 0.6B 草拟动态 k vs 不投机 | 普通问答，温度 0，并发 1 | 88 → 116 tok/s，1.3× | 3 次均值 |
| 投机解码变慢的情况 | 普通问答，温度 0，并发 64 | n-gram 动态 k 0.94×，0.6B 草拟 k=4 0.79× | 1 |
| 混合批处理 vs vLLM，预算 2048 | ShareGPT 400 条，泊松 36.5 个/秒 | 吞吐 3565 vs 3688 tok/s（96.7%），TTFT P99 132 vs 190 ms | 3 次均值 |
| 混合批处理 vs vLLM，预算 2048 | ShareGPT 1000 条同时到达 | 吞吐 4197 vs 4333 tok/s（96.9%），decode 慢约 3% | 3 次均值 |
| 正确性 | 16 个变体，10 / 30 个 prompt | 全部通过 FP32 参考检查，eager 和图的四组对照 10/10、30/30 逐 token 一致 | 1 |

全部实验用 `bash scripts/run_p5.sh` 跑，同预算对比用 `bash scripts/run_p5.sh --sections env correctness budget --reuse <P5b 结果目录>`。单个数字对应的 `bench.py` 命令见 REPORT 第 4 节。

### 用法

```python
LLM(path, speculative_method="ngram", num_speculative_tokens=4)          # n-gram 投机解码
LLM(path, speculative_method="draft", draft_model=small_model_path)      # 小模型草拟
LLM(path, speculative_method="ngram", num_speculative_tokens=8, spec_dynamic_k=True)   # 动态 k，此时 k 是上限
LLM(path, enable_mixed_batching=True, max_num_batched_tokens=2048)       # 混合批处理
```

- `SamplingParams(temperature=0)` 是贪心采样，上游不允许。
- 贪心和随机采样都支持投机解码，输出和不开投机时等价。`llm.get_spec_stats()` 返回接受率等统计。只支持单卡。
- 一步里 decode 的请求超过 `spec_max_batch_size`（默认 512 // (1+k)）时，这一步不猜。
- 动态 k 每步在 0、1、2、4、8 里选，依据是各请求最近的接受率和启动时测出的每步耗时，接受率低时会自动不猜。
- 混合批处理默认走分段 CUDA Graph（`mixed_cuda_graph`）。`max_num_batched_tokens` 是每步 token 预算，A100 + 8B 建议至少 2048。暂时不能和小模型草拟一起开。
- 序列长度到 `max_model_len` 就结束。上游没有这个限制，在 CUDA Graph 下会越界。

常用命令（`MODEL` 指向模型目录，默认 `~/huggingface/Qwen3-0.6B/`）：

```bash
python tests/test_spec_unit.py                          # 投机解码单元测试（只用 CPU，不需要模型）
MODEL=/path/to/Qwen3-0.6B python tests/test_greedy.py   # 正确性测试
MODEL=/path/to/Qwen3-0.6B python bench.py               # 性能测试：TTFT、TPOT、吞吐
MODEL=/path/to/Qwen3-0.6B python bench.py --workload copy --concurrency 4 --spec ngram   # 投机解码
MODEL=/path/to/Qwen3-0.6B python scripts/bench_matrix.py                  # 整张对比表（约 1.5 小时）
MODEL=/path/to/Qwen3-0.6B python scripts/step_cost.py                     # 验证一步比普通一步贵多少
MODEL=/path/to/Qwen3-0.6B python scripts/profile_decode.py --spec ngram   # 一步 decode 里各部分的耗时
```

`bench.py` 新增了照抄型任务（`--workload copy`）、普通问答（`--workload chat`）、随机猜的最坏情况（`--spec-proposer random`）、并发上限（`--concurrency`），以及 `--spec draft --draft-model 路径`、`--dynamic-k`。

---

以下是上游 README 原文。

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