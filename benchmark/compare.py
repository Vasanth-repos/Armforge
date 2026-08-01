"""
Compare baseline vs optimized benchmark results.
Usage: python benchmark/compare.py
"""
import json, glob
from tabulate import tabulate

def load_latest(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)

def pct_gain(base_tps, opt_tps):
    if base_tps and opt_tps and base_tps > 0:
        gain = (opt_tps - base_tps) / base_tps * 100
        sign = "+" if gain >= 0 else ""
        return f"{sign}{gain:.0f}%"
    return "N/A"

def compare():
    baseline  = load_latest("results/bench_baseline_*.json")
    optimized = load_latest("results/bench_llama_*.json")
    sglang    = load_latest("results/bench_sglang_*.json")

    base_tps = baseline.get("avg_tps") if baseline else None

    rows = []
    if baseline:
        rows.append(["Baseline (vanilla llama.cpp)",
                     f"{base_tps} tok/s",
                     f"{baseline.get('avg_ttft_ms', 'N/A')} ms",
                     "—"])
    if optimized:
        rows.append(["llama.cpp + KleidiAI + Speculative decoding",
                     f"{optimized.get('avg_tps')} tok/s",
                     f"{optimized.get('avg_ttft_ms', 'N/A')} ms",
                     pct_gain(base_tps, optimized.get("avg_tps"))])
    if sglang:
        rows.append(["SGLang W8A8 CPU (ARM64)",
                     f"{sglang.get('avg_tps')} tok/s",
                     f"{sglang.get('avg_ttft_ms', 'N/A')} ms",
                     pct_gain(base_tps, sglang.get("avg_tps"))])

    if not rows:
        print("No benchmark results found. Run scripts/03_benchmark_baseline.sh first.")
        return

    print("\n=== ArmForge — Before vs After ===\n")
    print(tabulate(rows,
                   headers=["Configuration", "Throughput", "TTFT", "vs Baseline"],
                   tablefmt="github"))

if __name__ == '__main__':
    compare()
