"""
Benchmark tokens/sec, TTFT, throughput.

Usage:
  python benchmark/run_bench.py --mode baseline  --port 8001
  python benchmark/run_bench.py --mode kleidiai  --port 8000
  python benchmark/run_bench.py --mode optimized --port 8000
  python benchmark/run_bench.py --mode sglang    --port 30000

WARMUP: One throwaway request is sent before measurement starts.
This ensures KV cache paths and JIT code are warm for fair comparison.
"""
import time, json, argparse, statistics, os, platform, subprocess
from datetime import datetime
import requests

PROMPTS = [
    "Explain how ARM Neoverse processors accelerate AI inference.",
    "What is speculative decoding and why does it reduce latency?",
    "Describe the difference between INT8 and FP16 quantization.",
    "How does the KleidiAI library improve matrix multiplication on ARM?",
    "Write a Python function that computes Fibonacci numbers efficiently.",
]

def check_server(base_url, timeout=5):
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        return r.status_code in (200, 404)
    except Exception:
        return False

def warmup(base_url):
    """Single warmup request — not recorded in results."""
    try:
        requests.post(f"{base_url}/v1/completions",
            json={"prompt": "warmup", "max_tokens": 5, "stream": False},
            timeout=60)
        print("  Warmup OK")
    except Exception as e:
        print(f"  Warmup failed (non-fatal): {e}")

def measure_ttft(base_url, prompt, max_tokens=5, timeout=60):
    """Time To First Token — streaming."""
    start = time.perf_counter()
    try:
        with requests.post(f"{base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": True},
            stream=True, timeout=timeout) as r:
            for chunk in r.iter_lines():
                if chunk and chunk != b"data: [DONE]":
                    return time.perf_counter() - start, None
    except Exception as e:
        return None, str(e)
    return None, "no chunks received"

def measure_throughput(base_url, prompt, max_tokens=128, timeout=180):
    """Tokens per second — full generation, non-streaming."""
    start = time.perf_counter()
    try:
        r = requests.post(f"{base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": False},
            timeout=timeout)
        elapsed = time.perf_counter() - start
        data = r.json()
        tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
        return tokens / elapsed, None
    except Exception as e:
        return None, str(e)

def get_system_metadata():
    """Gather reproducible hardware & system environment metadata."""
    meta = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_cores": os.cpu_count(),
        "arm_features": [],
        "optimal_threads": "N/A"
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Features"):
                    meta["arm_features"] = line.split(":")[1].strip().split()
                    break
    except Exception:
        pass
    try:
        with open(os.path.expanduser("~/armforge/results/optimal_threads.txt")) as f:
            meta["optimal_threads"] = f.read().strip()
    except Exception:
        pass
    return meta

def run(mode, port):
    base_url = f"http://localhost:{port}"
    print(f"\n=== ArmForge Benchmark [{mode.upper()}] → {base_url} ===\n")

    if not check_server(base_url):
        print(f"ERROR: No server at {base_url}. Start server first.")
        return None

    print("Running warmup request...")
    warmup(base_url)

    ttfts, tps_list = [], []
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt[:55]}...")
        ttft, e1 = measure_ttft(base_url, prompt)
        tps,  e2 = measure_throughput(base_url, prompt)
        if ttft: ttfts.append(ttft); print(f"  TTFT:       {ttft*1000:.0f} ms")
        if tps:  tps_list.append(tps); print(f"  Throughput: {tps:.2f} tok/s")
        if e1 or e2: print(f"  Error: {e1 or e2}")

    if not tps_list:
        print("No successful measurements.")
        return None

    results = {
        "mode":        mode,
        "port":        port,
        "timestamp":   datetime.now().isoformat(),
        "system_info": get_system_metadata(),
        "avg_ttft_ms": round(statistics.mean(ttfts) * 1000, 1) if ttfts else None,
        "avg_tps":     round(statistics.mean(tps_list), 2),
        "min_tps":     round(min(tps_list), 2),
        "max_tps":     round(max(tps_list), 2),
        "samples":     len(PROMPTS),
    }

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"results/bench_{mode}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- Summary ---")
    print(f"  Avg TTFT:  {results['avg_ttft_ms']} ms")
    print(f"  Avg tok/s: {results['avg_tps']}")
    print(f"  Saved:     {fname}")
    return results

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
        choices=["baseline","kleidiai","optimized","sglang"])
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    run(args.mode, args.port)
