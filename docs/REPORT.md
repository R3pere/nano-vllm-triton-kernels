# 项目报告：在 nano-vllm 上做投机解码和混合批处理

> 写给面试官，也写给以后的自己。所有数字都来自租卡结果包里的原始 JSON（结果包已提交在 `benchmarks/`，见文末），每个数字都注明硬件、模型、负载、重复次数和复现命令。

## 1. 一句话

在 nano-vllm（一个约 1500 行的精简版 vLLM）上，加了**投机解码**（n-gram 和小模型草拟两种，支持贪心和随机采样、CUDA Graph、按接受率动态选草拟长度）和 **vLLM V1 式的混合批处理**（prefill 和 decode 放进同一步，配分段 CUDA Graph），再用一套可复现的 benchmark 在 A100 + Qwen3-8B 上和 vLLM 0.15.1 做了同条件对比。

最重要的三个结果（A100-SXM4-80GB，Qwen3-8B）：
- 照抄型任务、并发 1：n-gram 动态 k 让输出速度从 88 到 335 tok/s，**3.8 倍**。
- 混合批处理和 vLLM 同 token 预算：吞吐是 vLLM 的 **96.6%～97.7%**，泊松到达时 TTFT P99 **更低**（132 vs 190 ms）。
- 正确性：所有变体通过 FP32 裁判；eager 与 CUDA Graph 四对在 8B 上逐 token 完全一致（10/10、30/30）。

## 2. 哪些是上游已有的，哪些是本项目做的

上游基线：GeeeekExplorer/nano-vllm 提交 bb823b3。

| 上游已有 | 本项目新增或修改 |
|---|---|
| Qwen3 模型、张量并行、分页 KV 缓存（块大小 256）、前缀缓存 | 贪心采样（上游只能随机采样），同一批里贪心和随机请求可以混着 |
| 调度器：有等待的请求就整步 prefill，否则整步 decode | 混合调度 `schedule_mixed`：先排 decode，剩下的预算给切块的 prefill；修了调度死锁 |
| decode 走整图 CUDA Graph，prefill 走 eager | 投机验证也走 CUDA Graph；混合步走**分段** CUDA Graph（29 段 × 若干档位） |
| `bench.py`：只报总吞吐 | 重写 `bench.py`：TTFT / TPOT / ITL（P50、P99、最长）、泊松到达、ShareGPT、vLLM 适配层（同一套客户端代码测两个引擎） |
| 无测试 | `tests/test_greedy.py`（FP32 裁判 + MUST_MATCH）、`tests/test_spec_unit.py`（纯 CPU 单元测试，含随机压力测试） |
| — | n-gram 草拟、小模型草拟（两模型共用块表）、投机采样验证 `verify_drafts`、动态 k、租卡脚本（一键装环境、一键跑全部实验、断点续跑） |

规模：引擎目录 `nanovllm/` 改了 14 个文件，+1109 / −91 行；测试、benchmark 和脚本另有约 2900 行。没有写自研 kernel，没有做模型适配。

## 3. 每个功能的设计要点和取舍

### 3.1 投机解码的骨架（n-gram）

- **做法**：每步 decode 先猜最多 k 个 token（n-gram：取末尾 4/3/2 个 token，在前文里找最近一次出现，把它后面的 token 当猜测），然后把“最后一个 token + 猜测”一次送进模型，按位置比对决定收下几个，再加上模型自己给的一个 token。
- **贪心**：猜测必须等于模型第一名。**随机采样**：标准的投机采样规则（以 min(1, p/q) 接受，拒绝后从 max(p−q, 0) 重新采样），输出分布与逐个采样完全相同。
- **KV 回滚**：没有任何回滚操作。调度器每步给每个请求预留 1+k 个槽位，被拒绝的猜测写下的 KV 下一步自然被覆盖；只让“已确认的 token 数”推进前缀缓存的登记。取舍：简单、零拷贝，代价是每个请求最多多占 1 个块。
- **CUDA Graph**：验证步单独录一族图（每请求 1+k 个 token）。有任何请求有草稿时整批补到 1+k；整批都没猜到东西时走普通 decode 图。

### 3.2 小模型草拟（draft model）

- 同一个 `ModelRunner` 里放两个模型，**共用调度器、块表和槽位**，各有一份 KV 张量。所以 BlockManager、前缀缓存、抢占都不用改，代价是每块的字节数是两者之和（KV 能放的 token 变少）。
- 第一步给小模型喂 2 个 token，之后每步喂上一步的猜测，共 k 步，每步走 CUDA Graph（eager 下 k=4 草拟要 140～200 ms，图下约 25～37 ms，本机 3060 测的）。
- **不猜的步也要让小模型跑一步写 KV**，否则它会越落越远。这是设计时没想到、实现时发现的。

### 3.3 动态 k

- 每个请求记一个接受率估计 α（带衰减的计数）；启动时实测“每个 k、每个批大小一步多久”；每步在 0/1/2/4/8 里选“期望产出 token ÷ 一步耗时”最大的 k，整批共用。
- 调试中修了三个会让它失效的问题：先验当成固定伪计数导致 α 上不去（锁死在小 k）；k=0 时永远不检验导致 α 停在 0.5（“探索饿死”，n-gram 用“免费检验”解决，小模型每 16 步试探一次）；几个 k 打平时由噪声决定（加 10% 门槛 `MIN_GAIN`）。
- 取舍：整批共用一个 k，不做每请求不同 k（那需要不规则形状，图录不了）。

### 3.4 混合批处理（统一调度）+ 分段 CUDA Graph

- **调度**：每步先排所有 decode（含投机验证，k 在调度时按剩余预算和空闲块选），剩下的 token 预算按到达顺序给 prefill，任何请求都可以切块。新请求只用空闲块，不为它抢占正在 decode 的请求。
- **执行**：所有请求的 query token 拼成一个扁平批，attention 统一走 `flash_attn_varlen_func` + 块表。
- **分段图**：attention 的形状随批结构变，录不进图；其余部分（投影、MLP、norm）只和总 token 数有关。所以每层在 attention 前后切开，28 层切 29 段，段走图、attention 走 eager，总 token 数补齐到 64 的倍数档位（这就是 vLLM V1 的 piecewise CUDA Graph 思路）。补齐的 token 不进 attention，结构上保证不写 KV。
- **取舍**：draft model 不支持混合（形状和步骤要重新设计）；档位上限 512 个 token，超过的混合步走 eager（vLLM 0.15.1 默认也只给 ≤512 个 token 的步录图，比较是对等的）。分段图让启动多约 40 秒。

## 4. 正确性怎么保证

BF16 下“和 HuggingFace 逐 token 一致”做不到：两个候选 logit 完全打平或只差一个最小刻度时，计算顺序一变就翻转，HF 自己 BF16 和 FP32 比都有 20/1280 个 token 不同。所以用了四层检查：

1. **FP32 裁判**（`tests/test_greedy.py`）：把每个变体生成的结果连同 prompt 喂给 FP32 的 HF 模型，每个生成的 token 必须是 FP32 下的第一名，或与第一名差距 ≤ 0.5。正常的 BF16 误差最大约 0.2～0.4，真 bug 的差距是 10～36。共 16 个变体（eager、图、分块 prefill、n-gram、draft、动态 k、混合……）。
2. **MUST_MATCH**（逐 token 完全相同）：本应算同一件事的四对必须一字不差——graph = eager、spec_ngram_graph = spec_ngram、spec_draft_graph = spec_draft、mixed_graph = mixed_eager。为此 eager 也按图的档位补批大小、补块表宽度，让两边的计算形状完全一样。小模型的 bug 不会让输出出错只会降低接受率，所以自己给自己草拟时还要求接受率 ≥ 0.95。
3. **故障注入（反向验证）**：每加一类检查，就故意改坏代码看它能不能抓住。抓住的例子：KV 写错一格（差距 31）、多收一个被拒绝的猜测（34）、补齐位置的 slot 填 0 覆盖别人的 KV（29）、分段图里位置缓冲区没更新（34.9）、eager 不补齐（MUST_MATCH 掉到 4/10）、小模型 prefill 不跑（接受率 0.98 → 0.48）。**抓不住的也记下来了**：decode 位置整体加 1（RoPE 只看相对位置，输出仍合理）、已缓存长度多推进 1（GPU 测试抓不到，靠单元测试抓）。
4. **单元测试 + 随机压力测试**（`tests/test_spec_unit.py`，纯 CPU）：投机采样的分布统计测试（20 万次，正确规则总变差 0.0024，故意写错 0.23）；调度器状态更新的各种边界；混合调度 300 组随机配置（随机块数、预算、k、接受个数、EOS、共享前缀、到达时间），每步检查块引用计数和“有活就不能返回空批”，共触发 1204 次抢占，全部完成、块全部归还。这个测试独立撞到了调度死锁。

A100 + 8B 上的最终状态（P5b3，`results/p5b3_extract/p5b3/correctness*.json`）：16 个变体全部通过 FP32 裁判（最大差距 0.19～0.36，HF 自己 BF16 是 0.22）；四对 MUST_MATCH 10 个 prompt 全部 10/10，30 个 prompt 全部 30/30。

## 5. 真实结果

**环境**：FunHPC 租的 A100-SXM4-80GB；nano-vllm 用 Python 3.10、torch 2.8.0+cu128、flash-attn 2.8.3；vLLM 0.15.1（torch 2.9.1，单独的 venv）。目标模型 Qwen3-8B，草拟模型 Qwen3-0.6B，BF16，`gpu_memory_utilization` 0.9。两次租卡：P5b（2026-10-15，代码 433664b）和 P5b3（2026-10-19，代码 fe47ad8）。两次之间同一配置的吞吐差 0～1%，同方向，数据可比。

**怎么复现**：所有格子由一条命令跑出（参数写在 `scripts/run_p5.py` 里，每个格子的完整 `bench.py` 命令写在对应 `.log` 的第一行）：

```bash
bash scripts/run_p5.sh                                                    # P5b：全部 157 项
bash scripts/run_p5.sh --out results/p5b3 --reuse <P5b 结果目录> --sections env correctness budget   # P5b3
python scripts/run_p5.py --out results/p5b_extract/p5 --table-only       # 重新打印汇总表
```

表里“3 次”表示重复 3 次取均值（括号里 [最小..最大]），否则是 1 次。重复 3 次的格子离散度都很小（吞吐最大最小差多数 < 1%）。

### 5.1 投机解码（P5b，代码 433664b）

负载：`copy` 是照抄型任务（给代码加注释、改拼写等，素材是上游源码），`chat` 是普通问答（几乎没有可照抄的内容）。客户端保持固定并发 c，请求数 c=1/4/16/64 时分别为 8/16/48/128，每个最多输出 768 个 token。n-gram 动态 k 上限 8；草拟模型 Qwen3-0.6B。数字是输出 tok/s 相对“不投机 + CUDA Graph”的倍数。单格命令示例：

```bash
python bench.py --model Qwen3-8B --workload copy --temperature 0 --concurrency 1 --num-seqs 8 --max-tokens 768 --spec ngram --k 8 --dynamic-k
```

| 负载 | 温度 | 并发 | 不投机 tok/s | n-gram 动态 k | 0.6B 草拟 k=4 | 0.6B 草拟动态 k |
|---|---|---|---|---|---|---|
| copy | 0 | 1 | 88（3 次） | **3.82×**（3 次） | 2.14×（3 次） | 2.40×（3 次） |
| copy | 0 | 4 | 293 | 3.63× | 1.87× | 2.00× |
| copy | 0 | 16 | 887（3 次） | 2.44×（3 次） | 1.64×（3 次） | 1.72×（3 次） |
| copy | 0 | 64 | 2387 | 1.40× | 1.15× | 1.17× |
| copy | 0.6 | 1 | 88 | 3.85× | 2.10× | 2.26× |
| copy | 0.6 | 16 | 825 | 2.60× | 1.70× | 1.68× |
| chat | 0 | 1 | 88（3 次） | 1.06×（3 次） | 1.30×（3 次） | 1.31×（3 次） |
| chat | 0 | 4 | 292 | 1.09× | 1.19× | 1.28× |
| chat | 0 | 16 | 940 | 1.02× | 1.11× | 1.21× |
| chat | 0 | 64 | 2747 | **0.94×** | **0.79×** | **0.93×** |
| chat | 0.6 | 64 | 2833 | **0.91×** | **0.72×** | **0.85×** |

接受率（copy，温度 0，并发 1）：n-gram 0.69，草拟 k=4 0.86。

- 照抄型任务 n-gram 最强，不需要小模型；低并发 3.6～3.9 倍，并发 64 仍有 1.4 倍。
- 普通问答 n-gram 基本没用；0.6B 草拟在低并发有 1.2～1.3 倍。
- 动态 k 比固定 k=4 好，在“不该猜”的场景尤其明显（chat 并发 64：0.93× vs 0.79×）。
- 说明：这一节在 P5b 的代码上测，当时 dynamo 重编译超限的问题还没修（见第 8 节），草拟模型的图模式有部分 RMSNorm 退回未编译代码，草拟这两列可能略偏低，修复后没有重测。n-gram 和不投机两列不受影响。

### 5.2 混合批处理 vs vLLM，同 token 预算（P5b3，代码 fe47ad8）

负载：ShareGPT（照 vLLM 的 `ShareGPTDataset` 抽样，prompt 平均 217 token、输出平均 202 token，`ignore_eos` 保证两边生成一样多的 token），温度 0。`all` = 1000 条同时到达；`lo` / `hi` = 400 条按泊松到达，21.9 / 36.5 个/秒（nano 上游调度测出的承载能力 24.3 个/秒的 0.9 倍和 1.5 倍）。两个引擎走完全相同的客户端循环和计时代码（进程内，不经过 HTTP）。时间单位 ms，吞吐是输出 tok/s。

```bash
python bench.py --model Qwen3-8B --engine nanovllm --workload sharegpt --dataset sharegpt_sample.json --temperature 0 --num-seqs 400 --rate 36.483 --mixed --max-num-batched-tokens 2048
python bench.py --model Qwen3-8B --engine vllm     --workload sharegpt --dataset sharegpt_sample.json --temperature 0 --num-seqs 400 --rate 36.483 --max-num-batched-tokens 2048
```

| 负载 | 预算 | 次数 | 吞吐 nano / vLLM | TTFT P99 nano / vLLM | TPOT 均值 nano / vLLM | ITL P99 nano / vLLM | ITL 最长 nano / vLLM |
|---|---|---|---|---|---|---|---|
| all | 2048 | 3 | 4197 / 4333（96.9%） | 29175 / 28744 | 109.8 / 106.7 | 183 / 185 | 191 / 192 |
| all | 16384 | 3 | 4261 / 4361（97.7%） | 28479 / 28377 | 139.5 / 136.1 | 322 / 326 | 996 / 996 |
| hi | 2048 | 3 | 3565 / 3688（96.7%） | **132 / 190** | 38.8 / 36.0 | 115 / 114 | 133 / 139 |
| hi | 16384 | 3 | 3565 / 3685（96.7%） | **146 / 195** | 38.9 / 36.1 | 116 / 112 | 147 / 152 |
| lo | 2048 | 3 | 2983 / 3089（96.6%） | **104 / 137** | 26.7 / 24.0 | 84 / 80 | 107 / 104 |
| lo | 16384 | 3 | 2985 / 3090（96.6%） | **103 / 138** | 26.7 / 23.9 | 83 / 80 | 108 / 102 |

（预算 4096、8192 各 1 次，结果夹在中间，见 `results/p5b3_extract/p5b3/tables.md`。）

- 吞吐是 vLLM 的 96.6%～97.7%，差距来自 decode 步慢 3～11%（TPOT），见第 6 节。
- 泊松负载下 TTFT P99 nano 更好；ITL 两边基本一样。
- 预算只在“全部同时到达”时起作用，两边规律一样：预算越大，TTFT P50 越小（nano 8.5 → 6.4 s），但 ITL 最长从约 190 ms 涨到约 1 s。
- 和不混合的上游调度比（`all`，P5b3 同一 session，1 次）：上游调度吞吐 4966 更高、TTFT P99 12.7 s 更低，但 TPOT 330 ms、ITL 最长 11.6 s——它先把 1000 个 prompt 全部 prefill 完才开始 decode。这是“吞吐 vs 卡顿”取舍的另一端，不是做得更好。所以 P5b 时看到的“nano 吞吐比 vLLM 高 14%”（4913 vs 4316）不能当结论引用，它比的是两种不同的调度策略。

### 5.3 预算 512 在 A100 上太小（P5b，负面结果）

本机 3060 + 0.6B 上，预算 512 能同时改善卡顿和 TTFT；搬到 A100 + 8B 就不成立。负载：随机 token prompt 1024～2048、输出 256～512、200 个请求泊松到达，温度 0.6，lo / hi = 4.5 / 7.5 个/秒。

| 配置 | 次数 | lo：吞吐 / TTFT P99 / ITL 最长 | hi：吞吐 / TTFT P99 / ITL 最长 |
|---|---|---|---|
| 上游调度 + 图 | 3 | 1377 / 337 ms / 879 ms | 1695 / 473 ms / 1693 ms |
| 混合 512 | 3 | 1130 / 10960 ms / 99 ms | 1121 / 30265 ms / 103 ms |
| 混合 2048 | 1 | 1345 / 547 ms / 191 ms | 1534 / 4433 ms / 243 ms |

混合 512 把卡顿压到 0.1 秒，但吞吐掉 18%～34%，TTFT 涨几十倍：每步最多 512 个 token，A100 喂不饱，prefill 速度跟不上到达速度，队列越排越长。**预算要按卡定**。ShareGPT 上同样：预算 512 → 2048，`all` 吞吐 3420 → 4197，`hi` TTFT P99 794 → 132 ms。

### 5.4 本机数字（补充）

RTX 3060 Laptop 6GB + Qwen3-0.6B，只作补充，这台笔记本不同时间的状态差别很大，只在同一张表内比较（`scripts/bench_matrix.py`）：
- 照抄型、温度 0、并发 1：投机 + 图 517 tok/s，不投机的图 139 tok/s，3.7 倍（P2，1 次）。
- 混合 512 + 分段图，长 prompt 泊松 1 个/秒：ITL P99 97.5 → 48.9 ms，最长卡顿 254 → 57 ms，吞吐持平（P4b，1 次）。

## 6. 和 vLLM 的差距：可能的原因（未逐项验证）

同预算下 nano 的 decode 步慢 3～11%（TPOT：lo 26.7 vs 24.0、hi 38.8 vs 36.0、all 109.8 vs 106.7 ms）。可能的原因，按我认为的可能性排序：

1. **vLLM 默认开 async scheduling**：CPU 准备下一步和 GPU 计算重叠。nano 每步是同步的，CPU 时间直接加在步长上。A100 上 8B 一步只有 20 多毫秒，1～2 ms 的 CPU 时间就是 5%～10%。
2. **torch.compile 的算子融合**：vLLM 0.15.1 对整个模型做 torch.compile（norm、激活、残差加等融合），nano 只对 RMSNorm、rotary、SiluAndMul、采样这几个函数单独编译。
3. KV 块大小不同（nano 256、vLLM 16）影响 attention kernel 的切分，影响方向不确定。

公平性上已知的其他差别：为了让两个引擎走同一套客户端循环，vLLM 以**进程内模式**运行（`scripts/vllm_adapter.py` 设了 `VLLM_ENABLE_V1_MULTIPROCESSING=0`），而正常部署时 vLLM V1 把引擎核心放在单独的进程里，让 CPU 工作和 GPU 计算重叠，所以这里测到的 vLLM 可能略低于它的最佳状态（8B 这种 GPU 占主导的负载下影响应该不大，没有单独测）；KV 容量 nano 39～40 万 token、vLLM 约 38.7 万 token（nano 多 1%～4%）；投机时 vLLM 自动关闭 async scheduling；nano 在一步超过 `512 // (1+k)` 个请求时不猜，vLLM 一直猜。所以投机解码在 ShareGPT 上不和 vLLM 比速度（两边调度和门槛都不同），只在第 5.1 节和自己的不投机比。

## 7. 负面结果和已知局限

- **n-gram 在普通对话、高并发下是负收益**：chat 并发 64，动态 k 慢 6%～9%。不是选错了 k（92% 的步选了 k=0），而是每步给 64 个请求做 n-gram 查找和选 k 的 CPU 开销约 2 ms（本机 profile），A100 上一步只有 13 ms。ShareGPT 上混合 2048 + n-gram 动态 k 也比不投机慢 3%～6%（P5b3，1 次）。改进方向：全批 α 都很低时隔几步才查；查找向量化。没做。
- **draft model 的 TTFT 约翻倍**：chat 并发 1，82 ms vs 43 ms（P5b，3 次）。小模型的 prefill 也要跑一遍，两次都是 eager，短 prompt 的 eager prefill 几乎全是 kernel 发射开销，和模型大小关系不大。对一个 768 token 的回答占约 0.6%。改进方向：小模型 prefill 走分段图。没做。
- **draft model 在高并发下慢**：chat 并发 64，草拟 k=4 慢 21%～28%，动态 k 也慢 7%～15%。
- **decode 比 vLLM 慢 3～11%**，见第 6 节。
- **draft model 不支持混合批处理**。
- **分段图让启动多约 40 秒**（本机，232 个图），层数越多图越多。
- **FP32 裁判抓不住“温和”的 bug**：例如位置编号整体偏 1（第 4 节）。
- **前缀缓存可能复用一个位置的旧小模型 KV**：只影响接受率，不影响输出，没处理。
- 第 5.1 节的草拟数字在修复 dynamo 重编译问题之前测的，没重测。

## 8. 开发中发现并修复的最有价值的几个问题

1. **8B 上 eager 和 CUDA Graph 不逐 token 一致，其实是两个真 bug。** P5b 在 A100 上 MUST_MATCH 掉到 7/10、3/10，FP32 裁判却全过，很容易被解释成“大模型的正常数值差异”然后放宽判据。回本机查清了：(a) 不开投机时 eager 的 decode 没补批大小，cuBLAS 对不同的 M 选不同 kernel（8B 的 down_proj 在批 9～11 补到 16 时结果就变）；0.6B 在批 ≤16 时恰好不变，测试只有 10 个 prompt，所以从 P0 到 P5a 一直没暴露。(b) torch.compile 对 `rms_forward` 的重编译次数超过上限 8，超限后**悄悄**退回未编译代码，舍入不同；两个形状不同的模型（8B + 0.6B）让形状数变多。修法：eager 一律补批；上限提到 64 并打开 `fail_on_recompile_limit_hit`，以后再超就报错而不是悄悄变慢变不准。测试加了 `--num-prompts 30` 让 decode 批超过 16。修后本机 30/30，A100 + 8B 10/10、30/30。**教训：判据不过时先假设是 bug，不要先放宽判据；测试规模要覆盖到真实形状。**
2. **混合调度死锁**：decode 请求要块而不够时抢占自己、插到等待队首，排在一个“prefill 做了一半、手里占着块”的请求前面；prefill 循环只看队首，拿不到块就停，占着块的请求永远轮不到。修法参考 vLLM V1：半截 prefill 优先继续，decode 缺块时先抢占半截 prefill。随机压力测试独立复现了它。
3. **改 forward 让 KV 块变少**：为分段图把每层拆成 `pre_attn` / `post_attn`，q/k/v 被局部变量多拿了一会儿，预热峰值高了约 300 MB，KV 块从 99 降到 88。KV 块数是用预热峰值反推的，所以张量的生命周期会直接变成性能。
4. **显存不够时不报错，而是悄悄变慢**：Windows/WSL 上显存满了，驱动把新分配放到系统内存。动态 k 启动时测耗时没清缓存，表现是某个配置只有正常的三分之一速度。
5. **动态 k 的“探索饿死”**：新请求 α=0.5，大批次时 k=0 最划算，于是从不检验，α 永远 0.5。n-gram 用“k=0 时拿第一个猜测和模型输出比一下”做免费、无偏的检验，吞吐 1562 → 约 2800 tok/s（本机）。

## 9. 花费和结果包

- 两次租卡共 **59 元**（P5b 44 元，其中约 15 元是实例空闲；P5b3 15 元），预算上限 100 元。
- 结果包：`benchmarks/p5_20261015_2330.tar.gz`（P5b）和 `benchmarks/p5b3_20261019_2210.tar.gz`（P5b3），校验和在 `benchmarks/SHA256SUMS`。文中引用的 `results/p5b_extract/p5/`、`results/p5b3_extract/p5b3/` 就是这两个包解开后的目录：`mkdir -p results/p5b_extract results/p5b3_extract && tar -xzf benchmarks/p5_20261015_2330.tar.gz -C results/p5b_extract && tar -xzf benchmarks/p5b3_20261019_2210.tar.gz -C results/p5b3_extract`。每个包里有每格的 JSON 和日志、`tables.md`、`env/`（nvidia-smi、两套 pip freeze、git 提交号）、`rates.json`、`sharegpt_sample.json`。
- 环境脚本 `scripts/setup_remote.sh`。
