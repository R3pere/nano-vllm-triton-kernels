# nano-vllm 投机解码与混合批处理：实现记录

这个项目在 nano-vllm（约 1500 行的精简版 vLLM）上加了两样东西：投机解码和 vLLM V1 那种 prefill/decode 混合批处理。最后在 A100 + Qwen3-8B 上和 vLLM 0.15.1 用同一套客户端跑了对比。下面的数字都能在 `benchmarks/` 的结果包里找到原始 JSON。

主要结果（A100-SXM4-80GB，Qwen3-8B）：

- 照抄型任务、并发 1，n-gram 动态 k 把输出速度从 88 tok/s 提到 335 tok/s，3.8 倍。
- 混合批处理和 vLLM 用同样的 token 预算，吞吐是 vLLM 的 96.6%～97.7%；泊松到达时 TTFT P99 比 vLLM 低（132 vs 190 ms）。
- 所有变体都通过了 FP32 参考检查，eager 和 CUDA Graph 的四组对照在 8B 上逐 token 一致（10/10、30/30）。

## 1. 改了什么

上游是 GeeeekExplorer/nano-vllm 的 bb823b3。

| 上游已有 | 我加的或改的 |
|---|---|
| Qwen3 模型、张量并行、分页 KV 缓存（块大小 256）、前缀缓存 | 贪心采样（上游只能随机采样），同一批里可以混着贪心和随机请求 |
| 调度器：有等待的请求就整步 prefill，否则整步 decode | 混合调度 `schedule_mixed`：先排 decode，剩下的预算给分块 prefill；修了一个死锁 |
| decode 用整图 CUDA Graph，prefill 用 eager | 投机验证步也走 CUDA Graph；混合步用分段 CUDA Graph（29 段，按 token 数分档） |
| `bench.py` 只报总吞吐 | 重写 `bench.py`：TTFT / TPOT / ITL 的 P50、P99、最大值，泊松到达，ShareGPT，vLLM 适配层（两个引擎跑同一份客户端代码） |
| 没有测试 | `tests/test_greedy.py`（FP32 参考 + 逐 token 对照）、`tests/test_spec_unit.py`（纯 CPU 单测和随机压力测试） |
| — | n-gram 草拟、小模型草拟（两个模型共用块表）、投机采样验证 `verify_drafts`、动态 k、租卡用的脚本（装环境、跑全部实验、断点续跑） |

`nanovllm/` 下改了 14 个文件，+1109 / −91 行；测试、benchmark 和脚本另外约 2900 行。这个分支没有写自定义 kernel，也没做新模型适配。

## 2. 设计

### 2.1 n-gram 投机解码

每步 decode 前，先取序列末尾 4/3/2 个 token 去前文里找最近一次出现的位置，把后面跟着的最多 k 个 token 当作猜测。然后把"最后一个 token + 猜测"一次送进模型，逐位比对，猜对几个收几个，再加上模型自己在第一个不一致位置给出的 token。

贪心模式下，猜测必须等于模型的 argmax。随机采样用标准的投机采样规则：以 min(1, p/q) 的概率接受，拒绝后从 max(p−q, 0) 归一化后的分布里重新采样，这样输出分布和逐个采样完全一样。

KV 不需要回滚。调度器每步给每个请求预留 1+k 个槽位，被拒绝的猜测写进去的 KV 下一步会被覆盖，前缀缓存只登记已确认的 token。这样实现简单、没有拷贝，代价是每个请求最多多占一个块。

验证步单独录了一组 CUDA Graph，每个请求 1+k 个 token。批里只要有一个请求有猜测，整批就补齐到 1+k；整批都没猜到时走普通 decode 图。

### 2.2 小模型草拟

两个模型放在同一个 `ModelRunner` 里，共用调度器、块表和槽位，各自有一份 KV 张量。好处是 BlockManager、前缀缓存、抢占逻辑都不用改；坏处是每个块的字节数是两个模型之和，能放的 token 变少了。

第一步给小模型喂 2 个 token，之后每步喂上一步的猜测，一共 k 步，每步都走 CUDA Graph。在本机 3060 上，k=4 的草拟用 eager 要 140～200 ms，用图只要 25～37 ms，所以草拟必须走图。

有个设计时没想到的地方：这一步不猜的时候，小模型也得跑一步把 KV 写上，不然它的 KV 会越落越远。

### 2.3 动态 k

每个请求维护一个带衰减的接受率估计 α。启动时实测不同 k、不同批大小下一步要多久。每步在 0/1/2/4/8 里选"期望产出 token 数 ÷ 一步耗时"最大的 k，整批共用一个 k。不做每个请求不同的 k，因为那样形状不规则，录不了图。

调试时修了三个让它失效的问题：

- 先验被当成固定的伪计数，α 涨不上去，k 一直锁在小值。
- k=0 时永远不检验，α 一直停在 0.5，我叫它"探索饿死"。n-gram 的解决办法是 k=0 时顺手拿第一个猜测和模型输出比一下，不花额外计算；小模型每 16 步强制试探一次。
- 几个 k 收益差不多时，选哪个全看噪声。加了 10% 的门槛 `MIN_GAIN`。

### 2.4 混合批处理和分段 CUDA Graph

调度：每步先排所有 decode（包括投机验证，k 在调度时按剩余预算和空闲块选），剩下的 token 预算按到达顺序给 prefill，任何请求都可以分块。新请求只用空闲块，不会为了它抢占正在 decode 的请求。

执行：所有请求的 query token 拼成一个扁平的批，attention 统一用 `flash_attn_varlen_func` 加块表。

分段图：attention 的形状跟着批的组成变，没法录图；投影、MLP、norm 只跟总 token 数有关。所以在每层 attention 前后切开，28 层切成 29 段。段走图，attention 走 eager，总 token 数补齐到 64 的倍数。这就是 vLLM V1 的 piecewise CUDA Graph 思路。补出来的 token 不进 attention，所以不会写 KV。

限制：用小模型草拟时不支持混合，因为形状和步骤都得重新设计；档位上限是 512 个 token，超过的混合步走 eager（vLLM 0.15.1 默认也只给 512 以内的步录图，所以对比是对等的）。分段图让启动多花约 40 秒。

## 3. 正确性

BF16 下没法要求和 HuggingFace 逐 token 一致。两个候选 logit 打平或者只差一个最小刻度时，计算顺序一变就会翻过来，HF 自己的 BF16 和 FP32 比都有 20/1280 个 token 不同。所以我用了四层检查。

第一层是 FP32 参考（`tests/test_greedy.py`）。把每个变体生成的结果连同 prompt 喂给 FP32 的 HF 模型，每个生成的 token 必须是 FP32 下的第一名，或者和第一名差距不超过 0.5。正常的 BF16 误差在 0.2～0.4，真 bug 的差距是 10～36。一共 16 个变体，包括 eager、图、分块 prefill、n-gram、草拟、动态 k、混合等。

第二层是逐 token 对照。本该算同一件事的四组必须完全一样：graph 和 eager、spec_ngram_graph 和 spec_ngram、spec_draft_graph 和 spec_draft、mixed_graph 和 mixed_eager。为了做到这点，eager 也按图的档位补齐批大小和块表宽度。另外小模型的 bug 不会让输出出错，只会让接受率下降，所以"自己给自己草拟"时要求接受率不低于 0.95。

第三层是故障注入。每加一类检查，我就故意改坏代码，看它能不能抓到。能抓到的：KV 写错一格（差距 31）、多收一个被拒绝的猜测（34）、补齐位置的 slot 填 0 覆盖了别人的 KV（29）、分段图里位置缓冲区没更新（34.9）、eager 不补齐（对照降到 4/10）、小模型不跑 prefill（接受率 0.98 → 0.48）。抓不到的也记下来了：decode 位置整体加 1（RoPE 只看相对位置，输出看起来仍然正常）；已缓存长度多推进 1（GPU 测试抓不到，靠单测抓）。

第四层是单测和随机压力测试（`tests/test_spec_unit.py`，只用 CPU）。投机采样做了分布统计测试，20 万次采样，正确规则的总变差 0.0024，故意写错的是 0.23。还有调度器状态更新的各种边界情况。混合调度跑了 300 组随机配置（块数、预算、k、接受个数、EOS、共享前缀、到达时间都随机），每步检查块的引用计数，以及"有活要干时不能返回空批"。一共触发了 1204 次抢占，全部跑完，块全部归还。后面说的调度死锁就是这个测试先撞到的。

A100 + 8B 的最终结果在 `results/p5b3_extract/p5b3/correctness*.json`：16 个变体全部通过 FP32 参考，最大差距 0.19～0.36（HF 自己的 BF16 是 0.22）；四组对照 10 个 prompt 都是 10/10，30 个 prompt 都是 30/30。

## 4. 实测结果

环境：FunHPC 上租的 A100-SXM4-80GB。nano-vllm 用 Python 3.10、torch 2.8.0+cu128、flash-attn 2.8.3；vLLM 0.15.1 用单独的 venv（torch 2.9.1）。目标模型 Qwen3-8B，草拟模型 Qwen3-0.6B，BF16，`gpu_memory_utilization` 0.9。

一共租了两次：第一次 2026-10-15（结果包 P5b，代码 433664b），第二次 2026-10-19（P5b3，代码 fe47ad8）。两次之间同一配置的吞吐差 0～1%，方向一致，数据可以放在一起比。

所有实验都由一条命令跑出来。参数在 `scripts/run_p5.py` 里，每个格子对应的完整 `bench.py` 命令写在各自 `.log` 的第一行。

```bash
bash scripts/run_p5.sh                                                    # P5b：全部 157 项
bash scripts/run_p5.sh --out results/p5b3 --reuse <P5b 结果目录> --sections env correctness budget   # P5b3
python scripts/run_p5.py --out results/p5b_extract/p5 --table-only       # 重新打印汇总表
```

表里标"3 次"的是 3 次取均值，括号里是最小到最大，没标的跑了 1 次。跑 3 次的格子波动都很小，吞吐的最大最小差大多不到 1%。

### 4.1 投机解码（P5b，代码 433664b）

两种负载：`copy` 是照抄型任务，比如给代码加注释、改拼写，素材是上游源码；`chat` 是普通问答，几乎没有能照抄的内容。客户端保持固定并发 c，c=1/4/16/64 时请求数分别是 8/16/48/128，每个最多输出 768 个 token。n-gram 动态 k 上限是 8，草拟模型是 Qwen3-0.6B。表里是输出 tok/s 相对"不投机 + CUDA Graph"的倍数。单个格子的命令长这样：

```bash
python bench.py --model Qwen3-8B --workload copy --temperature 0 --concurrency 1 --num-seqs 8 --max-tokens 768 --spec ngram --k 8 --dynamic-k
```

| 负载 | 温度 | 并发 | 不投机 tok/s | n-gram 动态 k | 0.6B 草拟 k=4 | 0.6B 草拟动态 k |
|---|---|---|---|---|---|---|
| copy | 0 | 1 | 88（3 次） | 3.82×（3 次） | 2.14×（3 次） | 2.40×（3 次） |
| copy | 0 | 4 | 293 | 3.63× | 1.87× | 2.00× |
| copy | 0 | 16 | 887（3 次） | 2.44×（3 次） | 1.64×（3 次） | 1.72×（3 次） |
| copy | 0 | 64 | 2387 | 1.40× | 1.15× | 1.17× |
| copy | 0.6 | 1 | 88 | 3.85× | 2.10× | 2.26× |
| copy | 0.6 | 16 | 825 | 2.60× | 1.70× | 1.68× |
| chat | 0 | 1 | 88（3 次） | 1.06×（3 次） | 1.30×（3 次） | 1.31×（3 次） |
| chat | 0 | 4 | 292 | 1.09× | 1.19× | 1.28× |
| chat | 0 | 16 | 940 | 1.02× | 1.11× | 1.21× |
| chat | 0 | 64 | 2747 | 0.94× | 0.79× | 0.93× |
| chat | 0.6 | 64 | 2833 | 0.91× | 0.72× | 0.85× |

接受率（copy，温度 0，并发 1）：n-gram 0.69，草拟 k=4 0.86。

照抄型任务上 n-gram 最好用，不需要小模型，低并发时 3.6～3.9 倍，并发 64 还有 1.4 倍。普通问答上 n-gram 基本没用，0.6B 草拟在低并发时有 1.2～1.3 倍。动态 k 比固定 k=4 好，在不该猜的场景差得更明显（chat 并发 64：0.93× vs 0.79×）。

要注意的是，这一节是在 P5b 的代码上测的，当时 dynamo 重编译超限的问题还没修（见第 7 节）。草拟模型在图模式下有一部分 RMSNorm 退回了未编译的代码，所以草拟的两列可能略偏低，修好以后没有重测。n-gram 和不投机的两列不受影响。

### 4.2 混合批处理 vs vLLM，同 token 预算（P5b3，代码 fe47ad8）

负载是 ShareGPT，抽样方法照搬 vLLM 的 `ShareGPTDataset`，prompt 平均 217 token，输出平均 202 token，开了 `ignore_eos` 保证两边生成的 token 一样多，温度 0。`all` 是 1000 条同时到达；`lo` / `hi` 是 400 条按泊松到达，分别 21.9 和 36.5 个/秒，也就是 nano 上游调度测出的承载能力（24.3 个/秒）的 0.9 倍和 1.5 倍。两个引擎跑完全相同的客户端循环和计时代码，都在进程内调用，不走 HTTP。时间单位 ms，吞吐是输出 tok/s。

```bash
python bench.py --model Qwen3-8B --engine nanovllm --workload sharegpt --dataset sharegpt_sample.json --temperature 0 --num-seqs 400 --rate 36.483 --mixed --max-num-batched-tokens 2048
python bench.py --model Qwen3-8B --engine vllm     --workload sharegpt --dataset sharegpt_sample.json --temperature 0 --num-seqs 400 --rate 36.483 --max-num-batched-tokens 2048
```

| 负载 | 预算 | 次数 | 吞吐 nano / vLLM | TTFT P99 nano / vLLM | TPOT 均值 nano / vLLM | ITL P99 nano / vLLM | ITL 最长 nano / vLLM |
|---|---|---|---|---|---|---|---|
| all | 2048 | 3 | 4197 / 4333（96.9%） | 29175 / 28744 | 109.8 / 106.7 | 183 / 185 | 191 / 192 |
| all | 16384 | 3 | 4261 / 4361（97.7%） | 28479 / 28377 | 139.5 / 136.1 | 322 / 326 | 996 / 996 |
| hi | 2048 | 3 | 3565 / 3688（96.7%） | 132 / 190 | 38.8 / 36.0 | 115 / 114 | 133 / 139 |
| hi | 16384 | 3 | 3565 / 3685（96.7%） | 146 / 195 | 38.9 / 36.1 | 116 / 112 | 147 / 152 |
| lo | 2048 | 3 | 2983 / 3089（96.6%） | 104 / 137 | 26.7 / 24.0 | 84 / 80 | 107 / 104 |
| lo | 16384 | 3 | 2985 / 3090（96.6%） | 103 / 138 | 26.7 / 23.9 | 83 / 80 | 108 / 102 |

预算 4096 和 8192 各跑了 1 次，结果落在两者之间，见 `results/p5b3_extract/p5b3/tables.md`。

吞吐是 vLLM 的 96.6%～97.7%，差在 decode 步慢了 3～11%（看 TPOT），原因见第 5 节。泊松负载下 nano 的 TTFT P99 更好，ITL 两边差不多。

预算只在"全部同时到达"时有影响，两边的规律一样：预算越大，TTFT P50 越小（nano 从 8.5 s 降到 6.4 s），但最长 ITL 从约 190 ms 涨到约 1 s。

和不混合的上游调度比（`all`，P5b3 同一次租卡，跑了 1 次）：上游调度吞吐 4966，更高，TTFT P99 12.7 s，更低，但 TPOT 330 ms，最长 ITL 11.6 s，因为它先把 1000 个 prompt 全部 prefill 完才开始 decode。这只是"吞吐 vs 卡顿"的另一个极端，不能说它更好。所以第一次租卡时看到的"nano 吞吐比 vLLM 高 14%"（4913 vs 4316）不能当结论，那次比的其实是两种不同的调度策略。

### 4.3 预算 512 在 A100 上太小（P5b）

在本机 3060 + 0.6B 上，预算 512 能同时改善卡顿和 TTFT，换到 A100 + 8B 就不行了。负载是随机 token，prompt 1024～2048，输出 256～512，200 个请求泊松到达，温度 0.6，lo / hi 分别是 4.5 / 7.5 个/秒。

| 配置 | 次数 | lo：吞吐 / TTFT P99 / ITL 最长 | hi：吞吐 / TTFT P99 / ITL 最长 |
|---|---|---|---|
| 上游调度 + 图 | 3 | 1377 / 337 ms / 879 ms | 1695 / 473 ms / 1693 ms |
| 混合 512 | 3 | 1130 / 10960 ms / 99 ms | 1121 / 30265 ms / 103 ms |
| 混合 2048 | 1 | 1345 / 547 ms / 191 ms | 1534 / 4433 ms / 243 ms |

混合 512 确实把卡顿压到了 0.1 秒，但吞吐掉了 18%～34%，TTFT 涨了几十倍。原因是每步最多 512 个 token 喂不饱 A100，prefill 的速度赶不上请求到达的速度，队列越排越长。预算得按卡来定。ShareGPT 上也是这样：预算从 512 调到 2048，`all` 的吞吐从 3420 涨到 4197，`hi` 的 TTFT P99 从 794 ms 降到 132 ms。

### 4.4 本机数字

RTX 3060 Laptop 6GB + Qwen3-0.6B，只作参考。这台笔记本不同时间的状态差别很大，只在同一张表里比较（`scripts/bench_matrix.py`）：

- 照抄型、温度 0、并发 1：投机 + 图 517 tok/s，不投机的图 139 tok/s，3.7 倍（1 次）。
- 混合 512 + 分段图，长 prompt 泊松 1 个/秒：ITL P99 从 97.5 降到 48.9 ms，最长卡顿从 254 降到 57 ms，吞吐不变（1 次）。

## 5. 和 vLLM 的差距

同预算下 nano 的 decode 步慢 3～11%（TPOT：lo 26.7 vs 24.0，hi 38.8 vs 36.0，all 109.8 vs 106.7 ms）。下面是我猜的原因，按可能性排，都没有逐项验证：

1. vLLM 默认开了 async scheduling，CPU 准备下一步和 GPU 计算是重叠的。nano 每步是同步的，CPU 时间直接加在步长上。A100 上 8B 一步只有 20 多毫秒，1～2 ms 的 CPU 时间就是 5～10%。
2. vLLM 0.15.1 对整个模型做 torch.compile，norm、激活、残差加这些会被融合；nano 只对 RMSNorm、rotary、SiluAndMul、采样这几个函数单独编译。
3. KV 块大小不同（nano 256，vLLM 16），会影响 attention kernel 的切分，影响方向说不准。

对比里还有几处不完全对等：

- 为了让两个引擎跑同一套客户端循环，vLLM 是在进程内运行的（`scripts/vllm_adapter.py` 设了 `VLLM_ENABLE_V1_MULTIPROCESSING=0`）。正常部署时 vLLM V1 会把引擎核心放到单独的进程，让 CPU 和 GPU 的工作重叠，所以这里测到的 vLLM 可能比它的最好水平低一点。8B 这种 GPU 占大头的负载影响应该不大，但我没单独测。
- KV 容量 nano 39～40 万 token，vLLM 约 38.7 万，nano 多 1～4%。
- 开投机时 vLLM 会自动关掉 async scheduling；nano 在一步超过 `512 // (1+k)` 个请求时不猜，vLLM 一直猜。所以投机解码在 ShareGPT 上没有和 vLLM 比速度，只在 4.1 节和自己的不投机版本比。

## 6. 不好的结果和已知问题

- n-gram 在普通对话、高并发下反而变慢：chat 并发 64，动态 k 慢 6～9%。问题不在选 k（92% 的步都选了 k=0），而在于每步给 64 个请求做 n-gram 查找和选 k 要花约 2 ms CPU（本机 profile），A100 上一步才 13 ms。ShareGPT 上混合 2048 + n-gram 动态 k 也比不投机慢 3～6%（P5b3，1 次）。可以改成全批 α 都很低时隔几步才查一次，或者把查找向量化，还没做。
- 小模型草拟的 TTFT 差不多翻倍：chat 并发 1，82 ms vs 43 ms（P5b，3 次）。小模型也要跑一遍 prefill，两次都是 eager，短 prompt 的 eager prefill 基本全是 kernel 启动开销，和模型大小关系不大。对一个 768 token 的回答来说只占约 0.6%。可以让小模型的 prefill 也走分段图，还没做。
- 小模型草拟在高并发下变慢：chat 并发 64，k=4 慢 21～28%，动态 k 也慢 7～15%。
- decode 比 vLLM 慢 3～11%，见第 5 节。
- 小模型草拟不支持混合批处理。
- 分段图让启动多花约 40 秒（本机，232 个图），层数越多图越多。
- FP32 参考抓不到"温和"的 bug，比如位置编号整体偏 1（见第 3 节）。
- 前缀缓存可能复用某一个位置上旧的小模型 KV。只影响接受率、不影响输出，没处理。
- 4.1 节的草拟数字是修复 dynamo 重编译问题之前测的，没重测。

## 7. 排查过的几个问题

**8B 上 eager 和 CUDA Graph 对不上，查下来是两个真 bug。** 第一次租卡时，A100 上的逐 token 对照掉到了 7/10 和 3/10，可 FP32 参考全过了。这种情况很容易被当成"大模型正常的数值差异"，然后把判据放宽。我回本机查清了原因：

- 不开投机时，eager 的 decode 没有补齐批大小。cuBLAS 会对不同的 M 选不同的 kernel，8B 的 down_proj 在批大小 9～11 补到 16 时结果就会变。0.6B 在批大小不超过 16 时碰巧不变，而测试只有 10 个 prompt，所以从一开始就没暴露出来。
- torch.compile 对 `rms_forward` 的重编译次数超过了上限 8，超过以后会悄悄退回未编译的代码，舍入方式就不一样了。两个形状不同的模型（8B + 0.6B）让形状数变多，才触发了这个上限。

修法是 eager 一律补齐批大小；重编译上限提到 64，并打开 `fail_on_recompile_limit_hit`，以后再超就直接报错，不会悄悄变慢、变不准。测试加了 `--num-prompts 30`，让 decode 批超过 16。修完后本机 30/30，A100 + 8B 10/10、30/30。这件事让我记住了：判据不过时先当成 bug 查，别急着放宽；测试规模要覆盖到真实会出现的形状。

**混合调度死锁。** decode 请求要块但不够时，会抢占自己并插到等待队列最前面，排在一个"prefill 做了一半、手里占着块"的请求前面。prefill 循环只看队首，拿不到块就停，那个占着块的请求永远轮不到。参考 vLLM V1 改成：做了一半的 prefill 优先继续，decode 缺块时先抢占做了一半的 prefill。这个问题是随机压力测试先复现出来的。

**改了 forward，KV 块变少了。** 为了分段图，我把每层拆成了 `pre_attn` / `post_attn`，q/k/v 被局部变量多引用了一会儿，预热时的显存峰值高了约 300 MB，KV 块从 99 降到 88。KV 块数是用预热峰值反推出来的，所以张量活多久会直接影响性能。

**显存不够时不报错，只是变慢。** Windows/WSL 上显存满了以后，驱动会把新分配放到系统内存里。动态 k 启动测耗时前没清缓存，结果某个配置只有正常速度的三分之一。

**动态 k 的探索饿死。** 新请求的 α 是 0.5，大批次下 k=0 最划算，于是一直不检验，α 就一直是 0.5。n-gram 改成 k=0 时拿第一个猜测和模型输出比一下，不花额外计算也没有偏差，本机吞吐从 1562 涨到约 2800 tok/s。

## 8. 花费和结果包

两次租卡一共 59 元：P5b 44 元，其中约 15 元是实例空闲；P5b3 15 元。预算上限 100 元。

结果包是 `benchmarks/p5_20261015_2330.tar.gz`（P5b）和 `benchmarks/p5b3_20261019_2210.tar.gz`（P5b3），校验和在 `benchmarks/SHA256SUMS`。文中提到的 `results/p5b_extract/p5/` 和 `results/p5b3_extract/p5b3/` 就是这两个包解开后的目录：

```bash
mkdir -p results/p5b_extract results/p5b3_extract && tar -xzf benchmarks/p5_20261015_2330.tar.gz -C results/p5b_extract && tar -xzf benchmarks/p5b3_20261019_2210.tar.gz -C results/p5b3_extract
```

每个包里有每个格子的 JSON 和日志、`tables.md`、`env/`（nvidia-smi、两套 pip freeze、git 提交号）、`rates.json`、`sharegpt_sample.json`。环境脚本是 `scripts/setup_remote.sh`。
