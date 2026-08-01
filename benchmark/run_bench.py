"""
Benchmark tokens/sec, TTFT, and throughput against any OpenAI-compatible endpoint.

Usage:
  python benchmark/run_bench.py --mode llama   # llama.cpp server on :8000
  python benchmark/run_bench.py --mode sglang  # SGLang server on :30000
  python benchmark/run_bench.py --mode baseline # baseline server on :8001
"""
import time, json, argparse, statistics, os
from datetime import datetime
import requests

PROMPTS = [
    "Explain how ARM Neoverse processors accelerate AI inference.",
    "What is speculative decoding and why does it reduce latency?",
    "Describe the difference between INT8 and FP16 quantization.",
    "How does the KleidiAI library improve matrix multiplication on ARM?",
    "Write a Python function that computes Fibonacci numbers efficiently.",
]

PORTS = {"llama": 8000, "sglang": 30000, "baseline": 8001}

def check_server(base_url, timeout=5):
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        return r.status_code in (200, 404)  # 404 = running but no /health route
    except Exception:
        return False

def measure_ttft(base_url, prompt, max_tokens=5, timeout=60):
    """Time To First Token via streaming."""
    start = time.perf_counter()
    first = None
    try:
        with requests.post(
            f"{base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": True},
            stream=True, timeout=timeout
        ) as r:
            for chunk in r.iter_lines():
                if chunk and chunk != b"data: [DONE]":
                    first = time.perf_counter() - start
                    break
    except Exception as e:
        return None, str(e)
    return first, None

def measure_throughput(base_url, prompt, max_tokens=128, timeout=180):
    """Tokens per second for full generation."""
    start = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": False},
            timeout=timeout
        )
        elapsed = time.perf_counter() - start
        data = r.json()
        tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
        return tokens / elapsed, None
    except Exception as e:
        return None, str(e)

def run(mode, model_name=None):
    port = PORTS[mode]
    base_url = f"http://localhost:{port}"
    print(f"\n=== ArmForge Benchmark [{mode.upper()}] → {base_url} ===\n")

    if not check_server(base_url):
        print(f"ERROR: No server responding at {base_url}")
        print(f"Start the server first, then retry.")
        return None

    ttfts, tps_list = [], []
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt[:55]}...")
        ttft, err1 = measure_ttft(base_url, prompt)
        tps,  err2 = measure_throughput(base_url, prompt)
        if ttft:
            ttfts.append(ttft)
            print(f"  TTFT:       {ttft*1000:.0f} ms")
        if tps:
            tps_list.append(tps)
            print(f"  Throughput: {tps:.2f} tok/s")
        if err1 or err2:
            print(f"  Error: {err1 or err2}")

    if not tps_list:
        print("No successful measurements. Check server logs.")
        return None

    results = {
        "mode":        mode,
        "model_name":  model_name or ("Llama-3.2-3B-Instruct" if mode != "sglang" else "Qwen2.5-1.5B"),
        "timestamp":   datetime.now().isoformat(),
        "platform":    "arm64",
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
    print(f"  Avg TTFT:    {results['avg_ttft_ms']} ms")
    print(f"  Avg tok/s:   {results['avg_tps']}")
    print(f"  Saved:       {fname}")
    return results

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llama", "sglang", "baseline"], default="llama")
    ap.add_argument("--model-name", type=str, default=None)
    args = ap.parse_args()
    run(args.mode, args.model_name)
