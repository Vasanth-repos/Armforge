"""
Three-row comparison: baseline / KleidiAI-only / KleidiAI+speculative.
Each row isolated to show individual contribution.
Includes visual bar charts for terminal & README output, and auto-exports results/SUMMARY.md.

Usage: python benchmark/compare.py
"""
import json, glob, os
from tabulate import tabulate

def load_latest(pattern):
    files = sorted(glob.glob(pattern))
    if not files: return None
    with open(files[-1], encoding="utf-8") as f: return json.load(f)

def pct(base, val):
    if base and val and base > 0:
        g = (val - base) / base * 100
        return f"{'+'if g>=0 else ''}{g:.0f}%"
    return "N/A"

def draw_bar(val, max_val, scale=20, fill_char="#"):
    if not val or not max_val or max_val <= 0:
        return ""
    length = int((val / max_val) * scale)
    return fill_char * max(1, length)

def export_markdown_summary(table_md, tps_chart, ttft_chart):
    os.makedirs("results", exist_ok=True)
    summary_path = "results/SUMMARY.md"
    content = f"""# 📊 ArmForge Benchmark Comparison Summary

## Performance Breakdown Table

{table_md}

## ⚡ Throughput Comparison (tokens/sec — higher is better)
```text
{tps_chart}
```

## ⏱️ TTFT Latency Comparison (ms — lower is better)
```text
{ttft_chart}
```

> **Note on Speculative Decoding:** On CPU, speculative verification runs sequentially per draft step. Speculative decoding targets **TTFT reduction (-30% to -40%)**, while throughput gain vs KleidiAI-only is expected to be flat.
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nSaved Markdown Summary to: {summary_path}")

def compare():
    baseline  = load_latest("results/bench_baseline_*.json")
    kleidiai  = load_latest("results/bench_kleidiai_*.json")
    optimized = load_latest("results/bench_optimized_*.json")
    sglang    = load_latest("results/bench_sglang_*.json")

    if not any([baseline, kleidiai, optimized]):
        print("No results found. Run benchmark scripts first.")
        return

    base_tps  = baseline.get("avg_tps") if baseline else None
    base_ttft = baseline.get("avg_ttft_ms") if baseline else None

    rows = []
    if baseline:
        rows.append([
            "[1] Baseline (vanilla llama.cpp, KleidiAI OFF)",
            f"{base_tps} tok/s", f"{base_ttft} ms", "—", "—"
        ])
    if kleidiai:
        rows.append([
            "[2] + KleidiAI dotprod kernels (no speculative)",
            f"{kleidiai.get('avg_tps')} tok/s",
            f"{kleidiai.get('avg_ttft_ms')} ms",
            pct(base_tps,  kleidiai.get("avg_tps")),
            pct(base_ttft, kleidiai.get("avg_ttft_ms")),
        ])
    if optimized:
        rows.append([
            "[3] + KleidiAI + Speculative Decoding (TTFT focus)",
            f"{optimized.get('avg_tps')} tok/s",
            f"{optimized.get('avg_ttft_ms')} ms",
            pct(base_tps,  optimized.get("avg_tps")),
            pct(base_ttft, optimized.get("avg_ttft_ms")),
        ])
    if sglang:
        rows.append([
            "    SGLang W8A8 CPU [bonus]",
            f"{sglang.get('avg_tps')} tok/s",
            f"{sglang.get('avg_ttft_ms')} ms",
            pct(base_tps,  sglang.get("avg_tps")),
            "N/A",
        ])

    table_str = tabulate(rows,
        headers=["Configuration", "Throughput", "TTFT", "vs Baseline (tps)", "vs Baseline (ttft)"],
        tablefmt="github")

    print("\n=== ArmForge — Optimization Breakdown ===\n")
    print(table_str)

    # Build ASCII Charts
    tps_vals = [r.get("avg_tps") for r in [baseline, kleidiai, optimized, sglang] if r and r.get("avg_tps")]
    max_tps = max(tps_vals) if tps_vals else 1

    tps_lines = []
    if baseline and baseline.get("avg_tps"):
        tps_lines.append(f"Baseline:     {draw_bar(baseline['avg_tps'], max_tps):<20} {baseline['avg_tps']} tok/s")
    if kleidiai and kleidiai.get("avg_tps"):
        tps_lines.append(f"+KleidiAI:    {draw_bar(kleidiai['avg_tps'], max_tps):<20} {kleidiai['avg_tps']} tok/s ({pct(base_tps, kleidiai['avg_tps'])})")
    if optimized and optimized.get("avg_tps"):
        tps_lines.append(f"+Speculative: {draw_bar(optimized['avg_tps'], max_tps):<20} {optimized['avg_tps']} tok/s ({pct(base_tps, optimized['avg_tps'])})")

    ttft_vals = [r.get("avg_ttft_ms") for r in [baseline, kleidiai, optimized, sglang] if r and r.get("avg_ttft_ms")]
    max_ttft = max(ttft_vals) if ttft_vals else 1

    ttft_lines = []
    if baseline and baseline.get("avg_ttft_ms"):
        ttft_lines.append(f"Baseline:     {draw_bar(baseline['avg_ttft_ms'], max_ttft):<20} {baseline['avg_ttft_ms']} ms")
    if kleidiai and kleidiai.get("avg_ttft_ms"):
        ttft_lines.append(f"+KleidiAI:    {draw_bar(kleidiai['avg_ttft_ms'], max_ttft):<20} {kleidiai['avg_ttft_ms']} ms ({pct(base_ttft, kleidiai['avg_ttft_ms'])})")
    if optimized and optimized.get("avg_ttft_ms"):
        ttft_lines.append(f"+Speculative: {draw_bar(optimized['avg_ttft_ms'], max_ttft):<20} {optimized['avg_ttft_ms']} ms ({pct(base_ttft, optimized['avg_ttft_ms'])})")

    tps_chart_str = "\n".join(tps_lines)
    ttft_chart_str = "\n".join(ttft_lines)

    print("\n--- Throughput Comparison (tokens/sec — higher is better) ---")
    print(tps_chart_str)

    print("\n--- Latency Comparison (TTFT ms — lower is better) ---")
    print(ttft_chart_str)

    # Export to SUMMARY.md
    export_markdown_summary(table_str, tps_chart_str, ttft_chart_str)

if __name__ == '__main__':
    compare()
