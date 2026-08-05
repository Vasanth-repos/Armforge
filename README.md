# ArmForge — ARM64 LLM Inference Optimization Suite

**Open-source ARM64 LLM inference benchmark: KleidiAI + speculative decoding on free Oracle Cloud A1.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## Project Overview

ArmForge evaluates and stacks optimizations for LLM inference on ARM Neoverse CPUs, benchmarked against a clean baseline on identical hardware:

1. **KleidiAI kernels** — built into llama.cpp via `GGML_CPU_KLEIDIAI=ON`, activating ARM-native `dotprod` / `i8mm` matrix multiplication. Primary **throughput** optimization (+30–50% tok/s).
2. **Speculative decoding** — pairs Llama-3.2-3B-Instruct with Llama-3.2-1B-Instruct (same tokenizer family). Primary **latency (TTFT)** reduction play on CPU.
3. **Thread Sweep Tuning** — auto-detects optimal thread counts (e.g. 1..4) to avoid memory bandwidth saturation on Neoverse N1.
4. **NUMA Interleave & Process Pinning** — leverages `numactl --interleave=all` for uniform memory access latency across CPU sockets/cores.
5. **SGLang W8A8 CPU (Bonus Demo)** — ARM64-native inference engine with INT8 quantization.

---

## Benchmark Isolation & Methodology

To ensure scientific credibility, optimizations are measured independently:

| Configuration | Metric Target | Rationale |
|---|---|---|
| **① Baseline** | Reference baseline | Vanilla `llama.cpp` (`GGML_CPU_KLEIDIAI=OFF`, `-ngl 0`, `--mlock`) |
| **② + KleidiAI** | Throughput win | Native ARM `dotprod` matrix kernels (`-b 512`) |
| **③ + KleidiAI + Speculative** | Latency win (TTFT) | 1B draft model overlap (`--model-draft`, `--spec-type draft-simple`) |

> **Note on CPU Speculative Decoding:** On CPU, speculative verification is sequential. Speculative decoding targets **TTFT reduction (-30-40%)**, while throughput gains over KleidiAI-only are expected to be flat or marginal.

---

## Quick Start Guide

### 🚀 Option A: One-Command Execution (Recommended)

Run the complete pipeline end-to-end with a single command:

```bash
git clone https://github.com/YOUR_USERNAME/armforge
cd armforge
bash scripts/run_all.sh
```

---

### Option B: Step-by-Step Setup

```bash
# 1. Bootstrap environment (PyTorch CPU, packages, 4GB swap)
bash scripts/00_bootstrap.sh

# 2. Build llama.cpp with KleidiAI ON and OFF (baseline)
bash scripts/01_build_llamacpp.sh
bash scripts/01b_build_baseline.sh

# 3. Download GGUF models (Llama-3.2-3B and 1B draft)
bash scripts/02_download_models.sh

# 4. Auto-tune thread count for optimal RAM bandwidth
bash scripts/03_tune_threads.sh

# 5. Run benchmarks
bash scripts/04_benchmark_baseline.sh
bash scripts/05_benchmark_kleidiai.sh
bash scripts/06_benchmark_optimized.sh

# 6. Display three-row comparison table with visual charts
source ~/armforge_env/bin/activate
python benchmark/compare.py

# 7. Launch web dashboard
bash scripts/07_start_dashboard.sh  # Open http://YOUR_IP:8080
```

### Validation

```bash
uname -m                                      # → aarch64
python3 inference/arm_features.py             # Feature report & optimal threads
grep KLEIDIAI ~/llama.cpp/build/CMakeCache.txt # → GGML_CPU_KLEIDIAI:BOOL=ON
curl http://localhost:8080/api/metrics | python3 -m json.tool
```

---

## Tech Stack
- **llama.cpp** (KleidiAI build) — [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **FastAPI / Uvicorn** — Real-time performance dashboard
- **Oracle Cloud A1** — ARM Neoverse N1, Always Free Tier
- **License**: Apache 2.0
