# ArmForge

**First open-source stack combining KleidiAI + speculative decoding on a free ARM64 cloud instance.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Project Overview

ArmForge stacks three compounding optimizations for LLM inference on ARM Neoverse CPUs, benchmarked against a clean baseline on identical hardware:

1. **KleidiAI kernels** — built into llama.cpp via `GGML_CPU_KLEIDIAI=ON`, activates ARM-native dotprod/I8MM matrix multiply
2. **Speculative decoding** — 1B draft model generates 5 tokens per step; 3B verifies in one parallel pass; output is identical to non-speculative
3. **SGLang W8A8 CPU** — ARM64-native inference engine with W8A8 INT8 quantization (aarch64 wheel available since July 2026)

## Why It Should Win

- Judges can reproduce every result — benchmark JSON files are committed
- Uses SGLang's brand-new ARM64 backend (merged May 2026)
- Correct speculative decoding with a matched draft model (Llama-3.2-1B + 3B, identical tokenizer)
- Zero cost — runs entirely on Oracle Cloud Always Free tier

## Functionality / Output

- Baseline benchmark (vanilla llama.cpp, no KleidiAI)
- Optimized benchmark (KleidiAI + speculative decoding)
- SGLang W8A8 CPU benchmark (bonus track)
- before/after comparison table via `benchmark/compare.py`
- Live dashboard at `:8080` auto-refreshing every 10s

## Setup Instructions

### Requirements
- ARM64 instance: Oracle Cloud A1 (2 OCPU, 12 GB) — always free
- OS: Ubuntu 22.04 aarch64

### Steps
```bash
git clone https://github.com/YOUR_USERNAME/armforge
cd armforge
bash scripts/00_bootstrap.sh
bash scripts/01_build_llamacpp.sh
bash scripts/02_download_models.sh
bash scripts/03_benchmark_baseline.sh
bash scripts/04_benchmark_optimized.sh
source ~/armforge_env/bin/activate && python benchmark/compare.py
bash scripts/05_start_dashboard.sh
```

### Validation
```bash
uname -m                          # → aarch64
python3 inference/arm_features.py # shows detected features
curl http://localhost:8080/api/metrics | python3 -m json.tool
```

## Tech Stack
- llama.cpp (KleidiAI build) — https://github.com/ggml-org/llama.cpp
- SGLang 0.5.16+ — https://github.com/sgl-project/sglang
- FastAPI / uvicorn — dashboard
- huggingface_hub — model download
- Oracle Cloud A1 — ARM Neoverse N1, always free
