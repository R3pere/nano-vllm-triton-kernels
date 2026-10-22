"""Runs the benchmark comparison matrix and prints the comparison tables.

Every run is one `python bench.py ...` in its own process (a fresh engine each time), saved as
<out>/<name>.json. Runs whose JSON already exists are skipped, so an interrupted matrix can be resumed;
delete the directory to measure again. --table-only just prints the tables from existing results.

Sections:
  copy    "copy" workload (answers mostly repeat the prompt), temperature 0 and 0.6, concurrency 1/4/16/all,
          eager / CUDA graph / speculative eager / speculative graph
  chat    "chat" workload (ordinary questions, little to copy), same grid, CUDA graph vs speculative graph
  k       copy workload, temperature 0, concurrency 4: speculative graph with k = 2, 4, 8
  random  upstream's random-token workload, 64 requests at once, all four configurations
  worst   copy workload with random guesses (almost never accepted), concurrency 1/16/all: the pure
          cost of checking guesses, against the plain runs of the copy section
  dynk    dynamic k (n-gram, k at most 8) against the CUDA graph, fixed k=4 and (copy only) fixed k=8,
          on copy and chat, temperature 0 and 0.6, concurrency 1/4/16/all; plus the random-guess worst case
  mixed   mixed batching under requests that keep arriving (Poisson): long prompts (1-2K tokens) and long
          outputs (256-512), so new prefills keep interrupting decoding. Upstream scheduling (CUDA graph)
          against mixed batching with token budgets 256/512/1024/2048 (steps that are all decode use the CUDA
          graphs, mixed steps of up to 512 tokens the piecewise graphs, bigger ones run eagerly), mixed batching
          at 512 with mixed steps eager as in P4a (mixed512_p4a), fully eager at 512, and upstream / mixed512
          with n-gram dynamic k. Reports ITL (P50 / P99 / longest stall), TTFT, throughput and the mean time of
          a decode step and of a mixed step.

Usage: MODEL=/path/to/Qwen3-0.6B python scripts/bench_matrix.py [--out results/matrix] [--sections ...] [--table-only]
                                                                [--only CONFIG ...]
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_SEQS = {1: 8, 4: 16, 16: 32, 0: 64}    # concurrency (0 = all at once) -> number of requests
TEMPERATURES = (0, 0.6)
CONFIGS = {
    "eager": ["--enforce-eager"],
    "graph": [],
    "spec_eager": ["--enforce-eager", "--spec", "ngram"],
    "spec_graph": ["--spec", "ngram"],
}
DYNK_CONFIGS = {
    "spec_graph_k8": ["--spec", "ngram", "--k", "8"],
    "spec_graph_dynk": ["--spec", "ngram", "--k", "8", "--dynamic-k"],
}
DYNK_GRID = {"copy": ["graph", "spec_graph", "spec_graph_k8", "spec_graph_dynk"], "chat": ["graph", "spec_graph", "spec_graph_dynk"]}
MIXED_RATES = (1.0, 2.0)    # requests per second
MIXED_BUDGETS = (256, 512, 1024, 2048)
MIXED_ARGS = ["--workload", "random", "--input-len", "1024", "2048", "--output-len", "256", "512", "--num-seqs", "48"]
MIXED_CONFIGS = {
    "graph": [],
    **{f"mixed{b}": ["--mixed", "--max-num-batched-tokens", str(b)] for b in MIXED_BUDGETS},
    "mixed512_p4a": ["--mixed", "--max-num-batched-tokens", "512", "--no-mixed-graph"],
    "mixed512_eager": ["--mixed", "--max-num-batched-tokens", "512", "--enforce-eager"],
    "spec_dynk": ["--spec", "ngram", "--k", "8", "--dynamic-k"],
    "mixed512_spec_dynk": ["--mixed", "--max-num-batched-tokens", "512", "--spec", "ngram", "--k", "8", "--dynamic-k"],
}
SECTIONS = ["copy", "chat", "k", "random", "worst", "dynk", "mixed"]


def grid_name(workload, t, c, config):
    return f"{workload}_T{t}_c{c}_{config}"


def grid_args(workload, t, c):
    return ["--workload", workload, "--temperature", str(t), "--concurrency", str(c), "--num-seqs", str(NUM_SEQS[c])]


def plan(sections):
    """(name, bench.py arguments) for every run of the chosen sections."""
    runs = []
    for workload, configs in (("copy", list(CONFIGS)), ("chat", ["graph", "spec_graph"])):
        if workload in sections:
            for t in TEMPERATURES:
                for c in NUM_SEQS:
                    for config in configs:
                        runs.append((grid_name(workload, t, c, config), grid_args(workload, t, c) + CONFIGS[config]))
    if "k" in sections:
        for k in (2, 8):
            runs.append((f"copy_T0_c4_spec_graph_k{k}", grid_args("copy", 0, 4) + CONFIGS["spec_graph"] + ["--k", str(k)]))
    if "random" in sections:
        for config in CONFIGS:
            runs.append((f"random_{config}", ["--workload", "random", "--num-seqs", "64", "--temperature", "0.6"] + CONFIGS[config]))
    if "worst" in sections:
        for c in (1, 16, 0):
            for config in ("eager", "graph"):    # plain runs, same as in the copy section
                runs.append((grid_name("copy", 0, c, config), grid_args("copy", 0, c) + CONFIGS[config]))
            for config in ("spec_eager", "spec_graph"):
                runs.append((grid_name("copy", 0, c, config) + "_randguess", grid_args("copy", 0, c) + CONFIGS[config] + ["--spec-proposer", "random"]))
    if "dynk" in sections:
        configs = {**CONFIGS, **DYNK_CONFIGS}
        for workload, names in DYNK_GRID.items():
            for t in TEMPERATURES:
                for c in NUM_SEQS:
                    for config in names:
                        runs.append((grid_name(workload, t, c, config), grid_args(workload, t, c) + configs[config]))
        for c in (1, 16, 0):
            runs.append((grid_name("copy", 0, c, "graph"), grid_args("copy", 0, c)))
            for config in ("spec_graph", "spec_graph_dynk"):
                runs.append((grid_name("copy", 0, c, config) + "_randguess", grid_args("copy", 0, c) + configs[config] + ["--spec-proposer", "random"]))
    if "mixed" in sections:
        for rate in MIXED_RATES:
            for config, args in MIXED_CONFIGS.items():
                runs.append((f"mixed_r{rate}_{config}", MIXED_ARGS + ["--rate", str(rate)] + args))
    seen = set()
    return [r for r in runs if not (r[0] in seen or seen.add(r[0]))]


def run_all(runs, out):
    os.makedirs(out, exist_ok=True)
    for i, (name, args) in enumerate(runs):
        path = os.path.join(out, f"{name}.json")
        if os.path.exists(path):
            continue
        print(f"[{i + 1}/{len(runs)}] {name}: bench.py {' '.join(args)}", flush=True)
        with open(os.path.join(out, f"{name}.log"), "w") as log:
            subprocess.run([sys.executable, os.path.join(ROOT, "bench.py"), *args, "--out", path], cwd=ROOT, check=True,
                           stdout=log, stderr=subprocess.STDOUT)


def load(out, name):
    path = os.path.join(out, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def cell(r):
    return f"{r['tpot_ms']['mean']:.1f}ms / {r['output_throughput']:.0f}" if r else "-"


def spec_cells(r):
    if not r:
        return ["-", "-"]
    s = r["spec_stats"]
    return [f"{s['acceptance_rate']:.2f}", f"{s['tokens_per_step']:.2f}"]


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    print("\n".join(lines) + "\n")


def conc_label(c):
    return "all 64" if c == 0 else str(c)


def print_tables(out, sections):
    for workload, configs in (("copy", list(CONFIGS)), ("chat", ["graph", "spec_graph"])):
        if workload not in sections:
            continue
        for t in TEMPERATURES:
            print(f"### {workload}, temperature {t}  (TPOT mean / output tok/s)\n")
            rows = []
            for c in NUM_SEQS:
                rs = {config: load(out, grid_name(workload, t, c, config)) for config in configs}
                gain = ""
                if rs["graph"] and rs["spec_graph"]:
                    gain = f"{rs['spec_graph']['output_throughput'] / rs['graph']['output_throughput']:.2f}x"
                rows.append([conc_label(c)] + [cell(rs[config]) for config in configs] + spec_cells(rs["spec_graph"]) + [gain])
            table(["concurrency"] + configs + ["acceptance", "tokens/step", "spec_graph vs graph"], rows)
    if "k" in sections:
        print("### k (copy, temperature 0, concurrency 4, speculative graph)\n")
        rows = []
        for k in (2, 4, 8):
            r = load(out, f"copy_T0_c4_spec_graph_k{k}" if k != 4 else grid_name("copy", 0, 4, "spec_graph"))
            rows.append([str(k), cell(r)] + spec_cells(r))
        table(["k", "TPOT / tok/s", "acceptance", "tokens/step"], rows)
    if "random" in sections:
        print("### random-token workload, 64 requests at once, temperature 0.6\n")
        rows = []
        for config in CONFIGS:
            r = load(out, f"random_{config}")
            rows.append([config, cell(r), f"{r['ttft_ms']['mean'] / 1000:.1f}s" if r else "-"] + (spec_cells(r) if config.startswith("spec") else ["", ""]))
        table(["config", "TPOT / tok/s", "TTFT mean", "acceptance", "tokens/step"], rows)
    if "worst" in sections:
        print("### worst case: random guesses (copy, temperature 0)  (TPOT mean / output tok/s)\n")
        rows = []
        for c in (1, 16, 0):
            plain = {config: load(out, grid_name("copy", 0, c, config)) for config in ("eager", "graph")}
            worst = {config: load(out, grid_name("copy", 0, c, config) + "_randguess") for config in ("spec_eager", "spec_graph")}
            row = [conc_label(c)]
            for base, spec in (("eager", "spec_eager"), ("graph", "spec_graph")):
                change = f"{worst[spec]['output_throughput'] / plain[base]['output_throughput'] - 1:+.0%}" if plain[base] and worst[spec] else "-"
                row += [cell(plain[base]), cell(worst[spec]), change]
            rows.append(row)
        table(["concurrency", "eager", "spec_eager, random guesses", "change", "graph", "spec_graph, random guesses", "change"], rows)
    if "dynk" in sections:
        for workload, names in DYNK_GRID.items():
            for t in TEMPERATURES:
                print(f"### dynamic k: {workload}, temperature {t}  (TPOT mean / output tok/s; k = steps per k)\n")
                rows = []
                for c in NUM_SEQS:
                    rs = {config: load(out, grid_name(workload, t, c, config)) for config in names}
                    dyn = rs["spec_graph_dynk"]
                    ratios = [f"{dyn['output_throughput'] / rs[b]['output_throughput']:.2f}x" if dyn and rs[b] else "-"
                              for b in names if b != "spec_graph_dynk"]
                    rows.append([conc_label(c)] + [cell(rs[config]) for config in names] + spec_cells(dyn)
                                + [k_hist(dyn)] + ratios)
                table(["concurrency"] + names + ["acceptance", "tokens/step", "k"]
                      + [f"dynk vs {b}" for b in names if b != "spec_graph_dynk"], rows)
        print("### dynamic k, worst case: random guesses (copy, temperature 0)  (TPOT mean / output tok/s)\n")
        rows = []
        for c in (1, 16, 0):
            graph = load(out, grid_name("copy", 0, c, "graph"))
            row = [conc_label(c), cell(graph)]
            for config in ("spec_graph", "spec_graph_dynk"):
                r = load(out, grid_name("copy", 0, c, config) + "_randguess")
                row += [cell(r), f"{r['output_throughput'] / graph['output_throughput'] - 1:+.0%}" if r and graph else "-"]
            row.append(k_hist(load(out, grid_name("copy", 0, c, "spec_graph_dynk") + "_randguess")))
            rows.append(row)
        table(["concurrency", "graph", "spec_graph k=4, random guesses", "change", "dynamic k, random guesses", "change", "k"], rows)
    if "mixed" in sections:
        print_mixed(out)


def print_mixed(out):
    for rate in MIXED_RATES:
        print(f"### mixed batching, {rate} requests/s (ms; throughput in output tok/s)\n")
        rows = []
        for config in MIXED_CONFIGS:
            r = load(out, f"mixed_r{rate}_{config}")
            if not r:
                rows.append([config] + ["-"] * 11)
                continue
            itl, ttft = r["itl_ms"], r["ttft_ms"]
            step_ms = r.get("step_ms", {})    # results from before P4b don't have it
            ms = lambda kind: f"{step_ms[kind]:.1f}" if step_ms.get(kind) is not None else "-"
            rows.append([config, f"{itl['p50']:.1f}", f"{itl['p99']:.1f}", f"{itl['max']:.0f}", f"{ttft['p50']:.0f}",
                         f"{ttft['p99']:.0f}", f"{r['tpot_ms']['mean']:.1f}", f"{r['output_throughput']:.0f}",
                         ms("decode"), ms("mixed"), str(r.get("mixed_steps", 0)), str(r.get("preemptions", 0))])
        table(["config", "ITL P50", "ITL P99", "ITL max", "TTFT P50", "TTFT P99", "TPOT mean", "throughput",
               "decode step", "mixed step", "mixed steps", "preemptions"], rows)


def k_hist(r):
    """Share of steps per k, e.g. "0:12% 8:88%"."""
    if not r or "k_steps" not in r.get("spec_stats", {}):
        return "-"
    steps = r["spec_stats"]["k_steps"]
    total = sum(steps.values())
    return " ".join(f"{k}:{100 * n / total:.0f}%" for k, n in steps.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "results", "matrix"))
    p.add_argument("--sections", nargs="+", choices=SECTIONS, default=SECTIONS)
    p.add_argument("--table-only", action="store_true")
    p.add_argument("--only", nargs="+", metavar="CONFIG", help="run only the runs whose name ends with _CONFIG (tables show all)")
    args = p.parse_args()
    if not args.table_only:
        runs = plan(args.sections)
        if args.only:
            runs = [r for r in runs if any(r[0].endswith(f"_{c}") for c in args.only)]
        run_all(runs, args.out)
    print_tables(args.out, args.sections)


if __name__ == "__main__":
    main()
