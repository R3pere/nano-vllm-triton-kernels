"""Fixed-batch decode benchmark: B requests with a C-token prompt are all prefilled first, then N decode steps
are timed. No arrivals, no queueing; the run is refused if the KV cache can't hold every request to the end
(no preemption). CUDA graphs on unless --enforce-eager.

Each timed step is one LLMEngine.step() (schedule, prepare, graph replay, sample, copy the B new tokens to the
CPU), wall-clock. Every request gets one token per step, so TPOT = step time; tokens/s = B / step time.
graph_replay_ms: GPU time of the decode CUDA graph alone (the model forward, CUDA events around 50 replays,
median of 3), i.e. the step without its CPU work, sampling and copies.

KV capacity: the engine allocates num_kvcache_blocks blocks of kvcache_block_size (256) tokens from what is left
of gpu_memory_utilization after the weights and the warmup peak (ModelRunner.allocate_kv_cache). A request of
C + 2 + N + 1 tokens (prompt, 2 untimed decode steps, N timed, 1 more so it doesn't finish inside the timing)
needs ceil(that / 256) blocks; B of them must fit. Qwen3-0.6B on a 6 GB RTX 3060 Laptop with the defaults below
gets 111 blocks (28,416 tokens): batch 32 fits up to ctx 637 (3 blocks each = 96), so ctx 1024 (160 blocks) doesn't.

Usage: python scripts/bench_decode.py [--kernel-backend compile|triton] [--batch 1 8 32] [--ctx 256 512]
                                      [--steps 128] [--out result.json]
"""
import argparse
import json
import math
import os
import sys
import time
from random import Random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--kernel-backend", choices=["compile", "triton"], default="compile")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--ctx", type=int, nargs="+", default=[256, 512], help="prompt length (context when decode starts)")
    p.add_argument("--steps", type=int, default=128, help="timed decode steps")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-num-batched-tokens", type=int, default=4096, help="prefill budget per step (also sets the warmup peak)")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    args = p.parse_args()

    import torch
    from nanovllm import LLM, SamplingParams
    max_bs = max(args.batch)
    llm = LLM(args.model, enforce_eager=args.enforce_eager, kernel_backend=args.kernel_backend, max_num_seqs=max_bs,
              max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    config = llm.config
    block = config.kvcache_block_size
    capacity = {"kv_blocks": config.num_kvcache_blocks, "kv_tokens": config.num_kvcache_blocks * block}
    print(f"backend {args.kernel_backend}, KV cache {capacity['kv_blocks']} blocks x {block} = {capacity['kv_tokens']} tokens", flush=True)

    rng = Random(args.seed)
    results = []
    for ctx in args.ctx:
        for bs in args.batch:
            total_len = ctx + 2 + args.steps + 1
            blocks_needed = bs * math.ceil(total_len / block)
            assert total_len <= config.max_model_len, f"ctx {ctx}: {total_len} tokens > max_model_len"
            assert blocks_needed <= config.num_kvcache_blocks, f"bs {bs} ctx {ctx}: needs {blocks_needed} KV blocks, have {config.num_kvcache_blocks}"
            preempt0 = llm.scheduler.num_preemptions
            params = SamplingParams(temperature=0.6, max_tokens=2 + args.steps + 1 + 1, ignore_eos=True)
            seqs = [llm.add_request([rng.randrange(10000) for _ in range(ctx)], params) for _ in range(bs)]
            # prefill everything (possibly over several steps), then 2 untimed decode steps
            while any(seq.num_completion_tokens == 0 for seq in seqs):
                _, n = llm.step()
                assert n > 0, "decode step before every request was prefilled"
            for _ in range(2):
                _, n = llm.step()
                assert n == -bs
            times = []
            for _ in range(args.steps):
                t = time.perf_counter()
                _, n = llm.step()    # ends with the new tokens copied to the CPU, so the GPU work is done
                times.append(time.perf_counter() - t)
                assert n == -bs, f"step produced {-n} tokens, expected {bs}"
            graph_ms = None
            if not args.enforce_eager:
                # GPU time of the model's decode graph alone (no scheduling, sampling or copies): replay the graph of
                # the last step (same inputs, rewrites the same KV slots) between CUDA events
                graph = llm.model_runner.graphs[("target", 1, bs)]
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                per_run = []
                for _ in range(3):
                    start.record()
                    for _ in range(50):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    per_run.append(start.elapsed_time(end) / 50)
                graph_ms = sorted(per_run)[1]
            while not llm.is_finished():
                llm.step()
            assert llm.scheduler.num_preemptions == preempt0, "preemption during the run"
            times_ms = sorted(t * 1e3 for t in times)
            step_ms = sum(times_ms) / len(times_ms)
            r = {"batch": bs, "ctx": ctx, "steps": args.steps, "kv_blocks_needed": blocks_needed,
                 "step_ms_mean": step_ms, "step_ms_median": times_ms[len(times_ms) // 2],
                 "step_ms_min": times_ms[0], "step_ms_max": times_ms[-1], "tokens_per_s": bs / step_ms * 1e3,
                 "graph_replay_ms": graph_ms}
            results.append(r)
            print(f"bs {bs:3d} ctx {ctx:5d}: step mean {step_ms:7.3f} ms, median {r['step_ms_median']:7.3f}, "
                  f"min {times_ms[0]:7.3f}, max {times_ms[-1]:7.3f}, {r['tokens_per_s']:8.1f} tok/s, "
                  f"graph replay {graph_ms or 0:7.3f} ms", flush=True)

    out = {"kernel_backend": args.kernel_backend, "enforce_eager": args.enforce_eager, "args": vars(args), **capacity,
           "gpu": torch.cuda.get_device_name(), "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    llm.exit()


if __name__ == "__main__":
    main()
