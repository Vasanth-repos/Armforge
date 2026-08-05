# ArmForge — Complete Build Specification (v3 Corrected)

> **AI Agent & Builder Specification:** Execute every shell block in strict order. All files are defined inline. Each phase must succeed before proceeding.

---

## Project Summary

* **Name:** ArmForge  
* **Track:** ARM AI Optimization Challenge 2026 — Cloud AI (Track 2)  
* **Tagline:** Open-source ARM64 LLM inference benchmark: KleidiAI + speculative decoding on free Oracle Cloud A1  
* **License:** Apache 2.0 | **Cost:** $0 | **Python:** 3.12 | **OS:** Canonical Ubuntu 22.04 aarch64  

---

## Corrections Applied vs v2

| # | Issue | Old Approach | Corrected Approach |
|---|---|---|---|
| 1 | Speculative decoding framing | Claimed throughput win on CPU | Framed as TTFT/latency play; isolated from KleidiAI in results |
| 2 | Benchmark warmup | No warmup — first result skewed | Warmup call before recording any measurements |
| 3 | Thread tuning | Fixed `nproc-1` | Sweep 1,2,3,4 threads; auto-select best before final benchmark |
| 4 | Model locking | No mlock | `--mlock` added to prevent swap thrash on 12 GB instance |
| 5 | GPU layer ambiguity | Implicit | `-ngl 0` explicit on all server calls |
| 6 | KleidiAI batch size | Default batch | `-b 512` added to activate dotprod kernel paths |
| 7 | SGLang in comparison | In primary benchmark table | Demoted to optional bonus; not in main comparison |
| 8 | Result isolation | KleidiAI+speculative combined | Three-row table: baseline / +KleidiAI / +KleidiAI+speculative |
| 9 | Oracle region note | Missing | Added: use `ap-singapore-1` or `eu-frankfurt-1` if A1 unavailable |
| 10 | Batch size in benchmark | 128 tokens fixed | Configurable; throughput measured per-token correctly |

---

## Repository Structure

```
armforge/
├── LICENSE
├── README.md
├── BUILD_SPEC.md              ← Consolidated build specification (v3)
├── about_the_project.md       ← Submission story
├── ArmForge_Project_Submission_Details.docx
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   ├── run_all.sh             ← 1-command master execution script
│   ├── 00_bootstrap.sh
│   ├── 01_build_llamacpp.sh
│   ├── 01b_build_baseline.sh  ← separate baseline build
│   ├── 02_download_models.sh
│   ├── 03_tune_threads.sh     ← thread sweep before benchmarking
│   ├── 04_benchmark_baseline.sh
│   ├── 05_benchmark_kleidiai.sh
│   ├── 06_benchmark_optimized.sh
│   └── 07_start_dashboard.sh
├── inference/
│   ├── arm_features.py
│   ├── speculative_server.py
│   ├── sglang_server.py
│   └── demo_chat.py           ← Interactive CLI chat client
├── benchmark/
│   ├── run_bench.py           ← Records system reproducibility metadata
│   └── compare.py             ← Generates SUMMARY.md and ASCII bar charts
├── dashboard/
│   ├── app.py                 ← FastAPI streaming endpoint & metrics
│   └── templates/
│       └── index.html         ← Interactive browser playground & UI
└── results/                   ← Benchmark JSON outputs & SUMMARY.md
```

---

## Infrastructure Requirements

### Primary: Oracle Cloud A1 (Always Free)
```
Shape:   VM.Standard.A1.Flex
OCPUs:   2 (set manually)
Memory:  12 GB (set manually)
OS:      Canonical Ubuntu 22.04 aarch64
Storage: 100 GB boot volume
Ports:   Open 22, 8000, 8001, 8080, 30000

Capacity Note: If A1 is out of capacity in your home region, select
ap-singapore-1, eu-frankfurt-1, or ap-tokyo-1.
```

---

## Benchmark & Execution Pipeline

```bash
# Option A: One-Shot Execution
bash scripts/run_all.sh

# Option B: Step-by-Step Sequence
bash scripts/00_bootstrap.sh           # PyTorch CPU, swap, dependencies
bash scripts/01_build_llamacpp.sh      # Build llama.cpp with KleidiAI ON
bash scripts/01b_build_baseline.sh     # Build llama.cpp baseline (KleidiAI OFF)
bash scripts/02_download_models.sh     # Download Llama-3.2 3B & 1B draft models
bash scripts/03_tune_threads.sh        # Hardware thread sweep auto-tuning
bash scripts/04_benchmark_baseline.sh  # Baseline benchmark (Port 8001)
bash scripts/05_benchmark_kleidiai.sh  # KleidiAI-only benchmark (Port 8000)
bash scripts/06_benchmark_optimized.sh # KleidiAI + Speculative benchmark (Port 8000)
python benchmark/compare.py            # Comparative metrics & ASCII charts
bash scripts/07_start_dashboard.sh    # FastAPI web UI & playground (:8080)
```

---

*ArmForge v3 · Apache 2.0 · ARM AI Optimization Challenge 2026*
