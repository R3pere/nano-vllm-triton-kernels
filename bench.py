"""Benchmark: prints TTFT, TPOT and throughput.

  TTFT (time to first token)   = first token time - request arrival time
  TPOT (time per output token) = (finish time - first token time) / (output tokens - 1)
  throughput                   = output tokens / total wall time
  ITL (inter-token latency)    = time between two moments a request receives new tokens, over all requests.
                                 Reported as P50, P99 and max (the longest stall any request saw). With
                                 speculative decoding one step can bring several tokens: that is one gap.

Workloads (--workload):
  random  same as upstream bench.py: random token prompts (100-1024 tokens), random output lengths
          (100-1024 tokens), EOS ignored. Note: on random prompts the model's output quickly falls
          into repeating a few tokens, so n-gram guessing works well here (acceptance ~0.8). It is
          NOT a "nothing to copy" case.
  copy    real editing tasks whose answer mostly repeats the input: "add comments to this code",
          "add type hints", "add docstrings", "fix typos in this text". The material is upstream
          nano-vllm's own source and README at commit bb823b3 (read with git, so it never changes).
          Output ends at EOS or --max-tokens.
  chat    ordinary questions whose answers don't copy the prompt (explain, write a story, list pros and
          cons, ...), 8 kinds of request x 8 topics. Output ends at EOS or --max-tokens. Little to copy
          here, so n-gram guessing mostly misses: this is the everyday case to set against "copy".
  sharegpt  real user prompts from ShareGPT, sampled the way vLLM's benchmarks do (scripts/sharegpt.py
          makes the --dataset file). Each request generates exactly as many tokens as the real answer had
          (EOS ignored), so every engine does the same work.

--engine vllm runs the same workload, client loop and timing code on vLLM (scripts/vllm_adapter.py; needs a
Python environment with vLLM installed): same model, bf16, CUDA graphs, prefix caching, --max-num-seqs,
--max-num-batched-tokens, --gpu-memory-utilization and --max-model-len. vLLM always mixes prefill chunks into
decode steps, so --mixed doesn't change it; --dynamic-k, --spec-proposer random and --no-mixed-graph are
nano-vllm only. vLLM's steps aren't split by kind ("all").

--spec-proposer random replaces the n-gram guesses with random tokens (almost never accepted): the worst
case, which measures the pure cost of checking guesses.
--spec draft --draft-model PATH lets a small model guess. --dynamic-k chooses k every step from the acceptance
rate, up to --k (with dynamic k, use a larger --k such as 8).

All requests arrive at once unless --rate is given (Poisson arrivals, requests/second).
--mixed runs decode requests and prefill chunks in the same step (enable_mixed_batching); then
--max-num-batched-tokens (the token budget per step) decides how big the prefill chunks can get.
Mixed steps use piecewise CUDA graphs unless --no-mixed-graph (P4a: eager). The result also has the mean
time of a step by kind (all decode / mixed / all prefill), measured around LLMEngine.step.

--concurrency C keeps at most C requests in flight: a new request is sent only when one finishes,
and its TTFT is counted from when it is sent. Speculative decoding helps most when few requests
run together, so test several levels.

Usage: python bench.py [--engine nanovllm|vllm] [--workload random|copy|chat|sharegpt [--dataset FILE]]
                       [--num-seqs N] [--concurrency C] [--rate R]
                       [--temperature T] [--enforce-eager] [--spec ngram|draft [--k K] [--dynamic-k]
                       [--draft-model PATH] [--spec-proposer random] [--spec-max-batch B]]
                       [--mixed [--no-mixed-graph]] [--max-num-batched-tokens N] [--out result.json]
scripts/bench_matrix.py runs the whole comparison matrix and prints the tables; scripts/run_p5.py the P5 runs.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from random import Random, expovariate, randint, seed

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MATERIAL_COMMIT = "bb823b3"    # upstream nano-vllm
CODE_FILES = [
    "nanovllm/engine/block_manager.py", "nanovllm/engine/llm_engine.py", "nanovllm/engine/model_runner.py",
    "nanovllm/engine/scheduler.py", "nanovllm/engine/sequence.py", "nanovllm/layers/attention.py",
    "nanovllm/layers/embed_head.py", "nanovllm/layers/layernorm.py", "nanovllm/layers/linear.py",
    "nanovllm/layers/rotary_embedding.py", "nanovllm/models/qwen3.py", "nanovllm/utils/loader.py", "bench.py",
]
CODE_TASKS = [
    "Add a short comment above every function and class in the following Python code. Keep everything else exactly the same and output the complete code.",
    "Add type hints to every function parameter and return value in the following Python code where they are missing. Output the complete modified code and nothing else.",
    "Add a one-line docstring to every function in the following Python code that does not have one. Output the complete code and nothing else.",
]
TEXT_TASK = "Fix any spelling or grammar mistakes in the following Markdown text and output the full corrected text, keeping all formatting and links."
CHAT_TASKS = [
    "Explain {} to a ten-year-old.",
    "Write a short story about {}.",
    "What are the pros and cons of {}?",
    "Write a poem about {}.",
    "Give me five interesting facts about {}.",
    "Write a persuasive paragraph about why {} matters.",
    "Describe a typical day of someone whose job involves {}.",
    "What questions would you ask an expert on {}, and why?",
]
CHAT_TOPICS = ["volcanoes", "the stock market", "honeybees", "black holes", "the Roman Empire",
               "machine translation", "coffee", "jazz music"]


def read_material(path):
    return subprocess.run(["git", "show", f"{MATERIAL_COMMIT}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def copy_prompts(tokenizer, n, rng):
    tasks = []
    for path in CODE_FILES:
        lines = read_material(path).splitlines()
        for i in range(0, len(lines), 40):
            chunk = "\n".join(lines[i:i + 40])
            if chunk.count("\n") >= 10:
                tasks += [f"{task}\n\n```python\n{chunk}\n```" for task in CODE_TASKS]
    readme = read_material("README.md").split("## Star History")[0]
    tasks.append(f"{TEXT_TASK}\n\n{readme}")
    rng.shuffle(tasks)
    tasks = [tasks[i % len(tasks)] for i in range(n)]
    return chat_template(tokenizer, tasks)


def chat_prompts(tokenizer, n, rng):
    tasks = [task.format(topic) for task in CHAT_TASKS for topic in CHAT_TOPICS]
    rng.shuffle(tasks)
    return chat_template(tokenizer, [tasks[i % len(tasks)] for i in range(n)])


def chat_template(tokenizer, texts):
    return [tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True, enable_thinking=False) for t in texts]


def summarize(values_s):
    ms = np.array(values_s) * 1000
    return {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p99": float(np.percentile(ms, 99)), "max": float(ms.max())}


class NanoEngine:
    """nano-vllm behind the small interface main() drives: add() returns the request's Sequence (it has
    num_completion_tokens and is_finished), step() runs one engine step and says what kind of step it was."""

    def __init__(self, args):
        from nanovllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.draft_model = (args.draft_model or args.model) if args.spec == "draft" else None
        self.llm = llm = LLM(args.model, enforce_eager=args.enforce_eager, max_model_len=args.max_model_len,
                             speculative_method=args.spec, num_speculative_tokens=args.k, spec_max_batch_size=args.spec_max_batch,
                             spec_dynamic_k=args.dynamic_k, draft_model=self.draft_model,
                             enable_mixed_batching=args.mixed, mixed_cuda_graph=not args.no_mixed_graph,
                             max_num_batched_tokens=args.max_num_batched_tokens, max_num_seqs=args.max_num_seqs,
                             gpu_memory_utilization=args.gpu_memory_utilization, kernel_backend=args.kernel_backend)
        self.tokenizer = llm.tokenizer
        self.spec = args.spec
        if args.spec == "ngram" and args.spec_proposer == "random":
            rng = Random(args.seed)
            vocab_size = llm.model_runner.config.hf_config.vocab_size
            llm.proposer.propose = lambda seq, k: [rng.randrange(vocab_size) for _ in range(max(k, 0))]

    def add(self, prompt, temperature, max_tokens, ignore_eos):
        return self.llm.add_request(prompt, self.SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos))

    def step(self):
        num_mixed = self.llm.num_mixed_steps
        _, num_tokens = self.llm.step()    # ends by copying the new tokens to the CPU, so the step's GPU work is done
        return "mixed" if self.llm.num_mixed_steps > num_mixed else "decode" if num_tokens < 0 else "prefill"

    def reset_stats(self):
        self.llm.reset_spec_stats()
        self.llm.num_mixed_steps = self.llm.scheduler.num_preemptions = 0

    def stats(self):
        config = self.llm.config
        s = {"engine_version": "nano-vllm " + git_commit(), "kv_blocks": config.num_kvcache_blocks,
             "kv_tokens": config.num_kvcache_blocks * config.kvcache_block_size, "kernel_backend": config.kernel_backend,
             "spec_max_batch": self.llm.spec_max_batch_size if self.spec else 0,
             "mixed_steps": self.llm.num_mixed_steps, "preemptions": self.llm.scheduler.num_preemptions}
        if self.spec:
            s["spec_stats"] = self.llm.get_spec_stats()
        return s


def git_commit():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return "unknown"
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("+dirty" if dirty else "")


def run_to_end(engine, prompts, temperature, max_tokens):
    reqs = [engine.add(prompt, temperature, max_tokens, False) for prompt in prompts]
    while not all(r.is_finished for r in reqs):
        engine.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--engine", choices=["nanovllm", "vllm"], default="nanovllm")
    p.add_argument("--workload", choices=["random", "copy", "chat", "sharegpt"], default="random")
    p.add_argument("--dataset", help="sharegpt workload: requests sampled by scripts/sharegpt.py (the first --num-seqs are used)")
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--input-len", type=int, nargs=2, default=[100, 1024], metavar=("MIN", "MAX"), help="random workload")
    p.add_argument("--output-len", type=int, nargs=2, default=[100, 1024], metavar=("MIN", "MAX"), help="random workload")
    p.add_argument("--max-tokens", type=int, default=768, help="copy and chat workloads: output limit")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--rate", type=float, default=0, help="requests per second; 0 = all arrive at once")
    p.add_argument("--concurrency", type=int, default=0, help="most requests in flight; 0 = no limit")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--spec", choices=["ngram", "draft"], help="speculative decoding method (default: off)")
    p.add_argument("--k", type=int, default=4, help="guessed tokens checked per step (with --dynamic-k: at most)")
    p.add_argument("--dynamic-k", action="store_true", help="choose k every step from the acceptance rate")
    p.add_argument("--draft-model", help="--spec draft: path of the model that guesses (default: --model itself)")
    p.add_argument("--spec-proposer", choices=["ngram", "random"], default="ngram", help="random = worst case, n-gram guesses replaced by random tokens")
    p.add_argument("--spec-max-batch", type=int, default=0, help="guess only when at most this many requests decode together; 0 = default")
    p.add_argument("--mixed", action="store_true", help="mixed batching: decode requests and prefill chunks in one step")
    p.add_argument("--no-mixed-graph", action="store_true", help="--mixed: run mixed steps eagerly, unpadded (P4a) instead of piecewise CUDA graphs")
    p.add_argument("--max-num-batched-tokens", type=int, default=16384, help="token budget per step")
    p.add_argument("--max-num-seqs", type=int, default=512, help="most requests running together")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--kernel-backend", choices=["compile", "triton"], default="compile", help="RMSNorm and SiluAndMul implementation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="also write the result as JSON")
    args = p.parse_args()
    if args.workload == "sharegpt" and not args.dataset:
        p.error("--workload sharegpt needs --dataset (made by scripts/sharegpt.py sample)")

    if args.engine == "vllm":
        if args.dynamic_k or args.spec_proposer != "ngram" or args.no_mixed_graph or args.spec_max_batch or args.kernel_backend != "compile":
            p.error("--dynamic-k, --spec-proposer, --no-mixed-graph, --spec-max-batch and --kernel-backend are nano-vllm only")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from vllm_adapter import VllmEngine
        engine = VllmEngine(args)
    else:
        engine = NanoEngine(args)

    seed(args.seed)
    n = args.num_seqs
    if args.workload == "random":
        prompts = [[randint(0, 10000) for _ in range(randint(*args.input_len))] for _ in range(n)]
        params = [(args.temperature, randint(*args.output_len), True) for _ in range(n)]    # (temperature, max_tokens, ignore_eos)
    elif args.workload == "sharegpt":
        with open(args.dataset) as f:
            requests = json.load(f)
        assert len(requests) >= n, f"{args.dataset} has {len(requests)} requests, {n} wanted"
        prompts = [r["prompt_token_ids"] for r in requests[:n]]
        params = [(args.temperature, r["output_len"], True) for r in requests[:n]]
    else:
        make_prompts = copy_prompts if args.workload == "copy" else chat_prompts
        prompts = make_prompts(engine.tokenizer, n, Random(args.seed))
        params = [(args.temperature, args.max_tokens, False)] * n
    arrivals = [0.0] * n
    if args.rate > 0:
        t = 0.0
        for i in range(n):
            arrivals[i] = t
            t += expovariate(args.rate)

    # Warmup. First calls with new shapes are slow (torch.compile of the sampler, kernel setup), so warm up
    # with one and with several requests. Repetitive made-up prompts make the model repeat itself, so n-gram
    # guesses get verified too; they share no text with the real prompts, so the prefix cache can't help those.
    warmup = [[100 + i % (3 + j) for i in range(64 * (j + 1))] for j in range(4)]
    run_to_end(engine, warmup[:1], args.temperature, 32)
    run_to_end(engine, warmup, args.temperature, 32)
    engine.reset_stats()

    seqs = [None] * n
    first_token = [None] * n
    finish = [None] * n
    token_times = [[] for _ in range(n)]    # when each request received new tokens (one entry per step that brought some)
    num_seen = [0] * n
    step_times = defaultdict(list)    # seconds per step, by kind of step
    running = set()
    next_req = 0
    start = time.perf_counter()
    while next_req < n or running:
        now = time.perf_counter() - start
        while next_req < n and arrivals[next_req] <= now and not (args.concurrency and len(running) >= args.concurrency):
            arrivals[next_req] = max(arrivals[next_req], now)    # with a concurrency limit, a request "arrives" when it can be sent
            seqs[next_req] = engine.add(prompts[next_req], *params[next_req])
            running.add(next_req)
            next_req += 1
        if not running:    # idle until the next request arrives
            time.sleep(arrivals[next_req] - now)
            continue
        step_start = time.perf_counter()
        kind = engine.step()
        now = time.perf_counter() - start
        step_times[kind].append(time.perf_counter() - step_start)
        for i in list(running):
            if seqs[i].num_completion_tokens > num_seen[i]:
                num_seen[i] = seqs[i].num_completion_tokens
                token_times[i].append(now)
            if first_token[i] is None and seqs[i].num_completion_tokens > 0:
                first_token[i] = now
            if seqs[i].is_finished:
                finish[i] = now
                running.remove(i)
    duration = time.perf_counter() - start

    input_tokens = sum(map(len, prompts))
    output_lens = [seq.num_completion_tokens for seq in seqs]
    output_tokens = sum(output_lens)
    stats = engine.stats()
    result = {
        "engine": args.engine,
        "engine_version": stats.pop("engine_version"),
        "model": args.model,
        "workload": args.workload,
        "dataset": args.dataset,
        "num_seqs": n,
        "concurrency": args.concurrency,
        "rate": args.rate,
        "temperature": args.temperature,
        "enforce_eager": args.enforce_eager,
        "spec": args.spec,
        "k": args.k if args.spec else 0,
        "dynamic_k": args.dynamic_k if args.spec else False,
        "draft_model": engine.draft_model,
        "spec_proposer": args.spec_proposer if args.spec == "ngram" else None,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_s": duration,
        "output_throughput": output_tokens / duration,
        "total_throughput": (input_tokens + output_tokens) / duration,
        "request_throughput": n / duration,
        "ttft_ms": summarize([first_token[i] - arrivals[i] for i in range(n)]),
        "tpot_ms": summarize([(finish[i] - first_token[i]) / max(output_lens[i] - 1, 1) for i in range(n)]),
        "itl_ms": summarize([b - a for times in token_times for a, b in zip(times, times[1:])] or [0.0]),
        # output tokens per (request, step that gave it tokens), measured here in the client, so it means the same for
        # both engines (each engine's own speculative counters count steps differently); 1.0 without speculation
        "tokens_per_arrival": output_tokens / max(sum(map(len, token_times)), 1),
        "mixed": args.mixed or args.engine == "vllm",    # vLLM always mixes prefill chunks into decode steps
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "mixed_cuda_graph": not args.no_mixed_graph,
        # mean wall time of one step by kind: all decode (incl. checking guesses), mixed, all prefill (vLLM: all)
        "step_ms": {kind: 1000 * float(np.mean(ts)) for kind, ts in step_times.items()},
        "num_steps": {kind: len(ts) for kind, ts in step_times.items()},
        **stats,
    }

    print(f"{result['engine_version']}: {n} requests  input tokens: {input_tokens}  output tokens: {output_tokens}  time: {duration:.2f}s")
    print(f"throughput: {result['output_throughput']:.1f} output tok/s, {result['total_throughput']:.1f} total tok/s, "
          f"{result['request_throughput']:.2f} requests/s")
    for name in ("ttft_ms", "tpot_ms", "itl_ms"):
        s = result[name]
        print(f"{name[:4].upper().rstrip('_')} (ms): mean {s['mean']:.1f}  p50 {s['p50']:.1f}  p99 {s['p99']:.1f}  max {s['max']:.1f}")
    print(f"mixed steps: {result.get('mixed_steps', '-')}  preemptions: {result.get('preemptions', '-')}  KV blocks: {result['kv_blocks']}")
    print("ms per step: " + "  ".join(f"{kind} {ms:.1f} ({result['num_steps'][kind]} steps)" for kind, ms in result["step_ms"].items()))
    if args.spec and result.get("spec_stats"):
        s = result["spec_stats"]
        print(f"speculative: acceptance rate {s['acceptance_rate']:.3f}  tokens per arrival {result['tokens_per_arrival']:.2f}  steps per k {s.get('k_steps', '-')}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
