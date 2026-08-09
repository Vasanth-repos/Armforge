# 🦾 ArmForge — Complete Master Documentation & Build Specification

> **Track:** ARM AI Optimization Challenge 2026 — Cloud & Edge AI (Track 2)  
> **Tagline:** Open-Source ARM64 LLM Inference Suite: KleidiAI + Speculative Decoding + Auto-Tuning on Cloud & Windows ARM.  
> **License:** Apache 2.0 (100% Open Source) | **Cost:** $0.00 | **Python:** 3.12 | **OS:** Ubuntu 22.04 aarch64 / Windows on ARM (WSL2)  

---

## 📑 Table of Contents
1. [Project Overview & Quick Start](#-project-overview--quick-start)
2. [Complete Step-by-Step Operations Guide](#-complete-step-by-step-operations-guide)
   - [Option A: Windows on ARM via WSL2 (Recommended)](#option-a-windows-on-arm-via-wsl2-recommended)
   - [Option B: Oracle Cloud / AWS Graviton / Hetzner Cloud](#option-b-oracle-cloud--aws-graviton--hetzner-cloud)
3. [About the Project (Hackathon Devpost Story)](#-about-the-project)
   - [Inspiration](#inspiration)
   - [What it does](#what-it-does)
   - [How we built it](#how-we-built-it)
   - [Challenges we ran into](#challenges-we-ran-into)
   - [Accomplishments that we're proud of](#accomplishments-that-were-proud-of)
   - [What we learned](#what-we-learned)
   - [What's next for ArmForge](#whats-next-for-armforge)
4. [Technical Build & Architecture Specification (v3)](#-technical-build--architecture-specification-v3)
   - [Corrections & Fixes Applied Matrix](#corrections--fixes-applied-matrix)
   - [Repository Structure](#repository-structure)
5. [Benchmark Suite & Performance Comparison](#-benchmark-suite--performance-comparison)
   - [Performance Comparison Table](#performance-comparison-table)
   - [Visual Throughput & TTFT Bar Charts](#visual-performance-bar-charts)
6. [Redesigned Web Platform & Developer Tools](#-redesigned-web-platform--developer-tools)
7. [Tags & Try-It-Out Links](#-tags--try-it-out-links)

---

## 🚀 Project Overview & Quick Start

ArmForge evaluates and stacks optimizations for Large Language Model (LLM) inference on ARM Neoverse & Snapdragon processors, benchmarked against a clean baseline on identical hardware.

### 1-Command Master Execution
```bash
git clone https://github.com/Vasanth-repos/Armforge.git
cd Armforge
bash scripts/run_all.sh
```

---

## 📖 Complete Step-by-Step Operations Guide

### Option A: Windows on ARM via WSL2 (Recommended)

Windows on ARM devices (Snapdragon X Elite / X Plus, Surface Pro 11, Lenovo Yoga Slim 7x) run ArmForge natively at 100% full hardware speed using WSL2 Ubuntu ARM64.

1. **Install WSL2 Ubuntu ARM64:** Open PowerShell as Administrator and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. **Launch & Run Master Pipeline:**
   ```bash
   wsl
   git clone https://github.com/Vasanth-repos/Armforge.git
   cd Armforge
   bash scripts/run_all.sh
   ```
3. **Launch Web Platform & Playground:**
   ```bash
   bash scripts/07_start_dashboard.sh
   ```
   Open your Windows browser at: **`http://localhost:8080`**

---

### Option B: Oracle Cloud / AWS Graviton / Hetzner Cloud

* **Oracle Cloud A1 Flex:** 2 OCPU, 12 GB RAM (`VM.Standard.A1.Flex`, Ubuntu 22.04 ARM64).
* **AWS Graviton:** `t4g.small` or `c6g.medium` (Free Tier / $300 Credit).
* **Hetzner Cloud:** `CAX11` (2 ARM vCPU, 4 GB RAM — ~$3.50/mo).

Run on SSH instance:
```bash
bash scripts/run_all.sh
```

---

## 💡 About the Project

### Inspiration

Large Language Models (LLMs) are transforming modern application development, but serving them on dedicated cloud GPUs incurs prohibitive operational costs. ARM64 architecture—exemplified by Snapdragon X processors on laptops and ARM Neoverse CPUs on cloud VPS nodes—offers a compelling zero-cost / low-cost alternative.

However, naive CPU LLM deployments often suffer from disappointing performance (4–5 tokens/second) due to memory bandwidth saturation and unoptimized generic matrix algebra routines. We were inspired to answer a fundamental engineering question:

> *Can we stack hardware-level ARM vector accelerations with architectural inference optimizations to deliver sub-500ms TTFT latency and 35+ tok/s generation throughput on accessible ARM hardware?*

This question birthed **ArmForge**—a scientifically isolated, open-source optimization suite designed to extract maximum LLM inference performance from ARM64 hardware.

### What it does

**ArmForge** provides a complete, 100% open-source optimization and benchmarking stack for LLMs running on ARM64 processors. Key capabilities include:

* **Hardware-Accelerated Inference:** Integrates **Arm KleidiAI** quantized matrix multiplication kernels into `llama.cpp`, accelerating generation throughput by **+50% to +80%** (up to **36.6+ tok/s**).
* **Latency-Optimized Speculative Decoding:** Pairs `Llama-3.2-3B-Instruct` with a tokenizer-matched `Llama-3.2-1B-Instruct` draft model to reduce Time-To-First-Token (TTFT) latency by **40% to 45%** (down to **420 ms**).
* **Dynamic Hardware Auto-Tuning:** Dynamic thread-sweep probing automatically determines the optimal thread allocation ($T_{\text{opt}}$) to prevent main memory bandwidth saturation.
* **1-Command Reproducibility (`bash scripts/run_all.sh`):** Automates bootstrap, build, model download, thread tuning, benchmark execution, and visual comparison generation in a single command.
* **Redesigned Developer Platform & Playground (`:8080`):** Features a Linear/Vercel-inspired UI with live resource meters, Chart.js visualizations, recommendation scoring (96/100), multi-format exports (JSON/Markdown/CSV/PDF), and real-time SSE prompt streaming.

### How we built it

ArmForge was engineered layer-by-layer to systematically conquer compute and memory bottlenecks on ARM CPUs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             ArmForge Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Vector Acceleration : Arm KleidiAI (-DGGML_CPU_KLEIDIAI=ON, -b 512)      │
│ 2. Speculative Engine  : Llama-3.2-3B (Verifier) + Llama-3.2-1B (Draft)     │
│ 3. RAM Auto-Tuning    : Thread-Sweep Probe (03_tune_threads.sh)            │
│ 4. Memory & NUMA      : --load-mode mlock + numactl --interleave=all        │
│ 5. Web Platform       : FastAPI SSE Streamer + Linear/Vercel UI + CLI Client│
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **KleidiAI Kernel Build:** We compiled `llama.cpp` from source with `-DGGML_CPU_KLEIDIAI=ON` targeting ARM `dotprod` (`armv8.2-a+dotprod`) and `i8mm` vector paths, forcing matrix operations onto vectorized execution paths with batch size `-b 512`. Bypassed npm UNC path errors by setting `-DLLAMA_BUILD_SERVER_WEBUI=OFF`.
2. **Speculative Decoding Engine:** Paired `Llama-3.2-3B-Instruct` verifier with `Llama-3.2-1B-Instruct` draft model. Using the exact same tokenizer family guarantees 100% mathematical token alignment.
3. **Dynamic Thread-Sweep Auto-Tuning:** Created `03_tune_threads.sh` using `llama-bench` to benchmark thread counts $T \in \{1, 2, 3, 4\}$, probing RAM channel limits in under 20 seconds before saving $T_{\text{opt}}$.
4. **Memory Locking & NUMA Interleaving:** Used `--load-mode mlock` to lock weights in RAM and `numactl --interleave=all` for uniform memory access across CPU cores.
5. **Full-Stack Interface:** Built a FastAPI service (`:8080`) providing live metrics, Server-Sent Events (SSE) streaming (`/api/generate`), in-browser prompt playground, and CLI terminal client (`demo_chat.py`).

### Challenges we ran into

#### 1. CPU vs. GPU Speculative Decoding Realities
On GPUs, verification of $N$ draft tokens occurs in parallel during a single forward pass with near-zero marginal time cost. On CPUs, however, forward passes execute sequentially per step:

$$T_{\text{step}} = \sum_{i=1}^{N} T_{\text{draft\_step}, i} + T_{\text{verifier\_verify}}$$

If token acceptance rate $\alpha$ is low, sequential CPU verification can result in flat or net-negative throughput.  
* **Our Solution:** We correctly framed speculative decoding on CPU as a **Time-To-First-Token (TTFT) latency play** (-40% to -45% TTFT reduction) via draft prefill overlap, while throughput gains were strictly isolated to KleidiAI vector kernels.

#### 2. Memory Bandwidth Saturation on ARM CPUs
Large quantized weights ($Q4\_K\_M$) must be streamed from main memory for every generated token. Throughput follows the saturation curve:

$$\text{Throughput}(T) = \min\left(T \cdot \mu_{\text{core\_bandwidth}}, \mathcal{B}_{\text{RAM\_max}}\right)$$

When thread count $T$ exceeds memory channel capacity, thread synchronization overhead causes throughput degradation.  
* **Our Solution:** Built `03_tune_threads.sh` to sweep threads automatically prior to benchmarking, pinning execution strictly to peak bandwidth $\mathcal{B}_{\text{RAM\_max}}$.

#### 3. Windows UNC Path CMake & npm Build Errors
When compiling `llama.cpp` inside WSL across Windows UNC shares (`\\wsl.localhost`), CMake attempted to invoke `npm run build` in `cmd.exe`, which defaulted to `C:\Windows` and failed.  
* **Our Solution:** Added `-DLLAMA_BUILD_SERVER_WEBUI=OFF` to CMake builds, bypassing the unused npm web UI step and accelerating build times by 3x.

### Accomplishments that we're proud of

* **+80% Throughput Boost & -44% Latency at $0 Cost:** Achieved 420ms TTFT latency and 36.6+ tok/s generation speed on a 3B LLM model running on ARM hardware.
* **Mathematical Improvement Metrics:**
  $$\text{Throughput Gain} = \frac{\text{TPS}_{\text{KleidiAI}} - \text{TPS}_{\text{Baseline}}}{\text{TPS}_{\text{Baseline}}} \times 100\% \approx +50\% \text{ to } +82\%$$
  $$\text{TTFT Reduction} = \frac{\text{TTFT}_{\text{Baseline}} - \text{TTFT}_{\text{Speculative}}}{\text{TTFT}_{\text{Baseline}}} \times 100\% \approx 40\% \text{ to } 45\%$$
* **100% Scientific Metric Isolation:** Created a clean 3-row benchmark methodology separating Baseline vs. KleidiAI-only vs. KleidiAI+Speculative.
* **1-Command Reproducibility:** Single script (`bash scripts/run_all.sh`) executes the entire end-to-end pipeline non-interactively and generates committed Markdown summaries (`results/SUMMARY.md`).

### What we learned

1. **Hardware Vector Extensions Outperform Generic BLAS:** Integrating Arm KleidiAI unlocked +50% to +80% higher throughput without changing model weights or losing precision.
2. **Honest Metric Framing Builds Credibility:** Disambiguating throughput wins (KleidiAI) from latency wins (Speculative decoding) produces transparent, reproducible engineering results.
3. **OS-Level Memory Controls are Critical:** Using `--load-mode mlock` and `numactl --interleave=all` eliminates random latency jitter caused by OS page swapping.

### What's next for ArmForge

* **ARM SVE2 & SME Acceleration:** Extend support to ARM Scalable Vector Extension 2 (SVE2) and Scalable Matrix Extension (SME) architectures found on AWS Graviton3/4 and Apple M-series chips.
* **Multi-Draft Speculative Sampling:** Implement tree-based speculative decoding algorithms ($N > 5$) to increase token acceptance rate $\alpha$.
* **Automated Quantization Pipeline:** Integrate automated INT4 / AWQ / GGUF quantization tuning directly into the ArmForge pipeline to further optimize model footprint.

---

## 🛠️ Technical Build & Architecture Specification (v3)

### Corrections & Fixes Applied Matrix

| # | Issue | Old Approach | Corrected Approach |
|---|---|---|---|
| 1 | Speculative decoding framing | Claimed throughput win on CPU | Framed as TTFT/latency play; isolated from KleidiAI in results |
| 2 | Benchmark warmup | No warmup — first result skewed | Warmup call before recording any measurements |
| 3 | Thread tuning | Fixed `nproc-1` | Sweep 1,2,3,4 threads via `llama-bench` non-interactive |
| 4 | Model locking | Deprecated `--mlock` | Updated to `--load-mode mlock` |
| 5 | GPU layer ambiguity | Implicit | `-ngl 0` explicit on all server calls |
| 6 | KleidiAI batch size | Default batch | `-b 512` added to activate dotprod kernel paths |
| 7 | Windows UNC CMake npm error | Failed on `\\wsl.localhost` | Added `-DLLAMA_BUILD_SERVER_WEBUI=OFF` |
| 8 | Folder path casing | Hardcoded `~/armforge` | Dynamic resolution `ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"` |
| 9 | UI/UX Quality | Generic metrics UI | Redesigned Linear/Vercel developer platform (`:8080`) |
| 10 | Browser favicon 404 | Missing route notice | Added `/favicon.ico` handler returning HTTP 204 |

### Repository Structure

```
armforge/
├── LICENSE
├── README.md                  ← Master repository landing page (this file)
├── ARMFORGE_COMPLETE_DOCS.md  ← Master consolidated documentation
├── TUTORIAL.md                ← Step-by-step operations tutorial
├── ArmForge_Project_Submission_Details.docx
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   ├── run_all.sh             ← 1-command master execution script
│   ├── start_server.sh        ← Background model server helper (:8000)
│   ├── 00_bootstrap.sh
│   ├── 01_build_llamacpp.sh   ← Builds llama.cpp with KleidiAI ON (-DLLAMA_BUILD_SERVER_WEBUI=OFF)
│   ├── 01b_build_baseline.sh  ← Builds llama.cpp baseline (KleidiAI OFF)
│   ├── 02_download_models.sh
│   ├── 03_tune_threads.sh     ← Fast llama-bench thread sweep
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
│       └── index.html         ← Linear/Vercel-inspired UI platform
└── results/                   ← Benchmark JSON outputs & SUMMARY.md
```

---

## 📊 Benchmark Suite & Performance Comparison

### Performance Comparison Table

| Configuration | Throughput | TTFT | vs Baseline (tps) | vs Baseline (ttft) | Primary Focus |
|---|---|---|---|---|---|
| **[1] Baseline (vanilla `llama.cpp`, KleidiAI OFF)** | 20.08 tok/s | 750.0 ms | — | — | Reference baseline |
| **[2] + KleidiAI dotprod kernels** | **36.61 tok/s** | 620.0 ms | **+82%** | −17% | **Throughput Acceleration** |
| **[3] + KleidiAI + Speculative Decoding** | **36.90 tok/s** | **420.0 ms** | **+83%** | **−44%** | **TTFT Latency Reduction** |

### Visual Performance Bar Charts

```text
--- Throughput Comparison (tokens/sec — higher is better) ---
Baseline:     #################### 20.08 tok/s
+KleidiAI:    #################################### 36.61 tok/s (+82%)
+Speculative: ##################################### 36.90 tok/s (+83%)

--- Latency Comparison (TTFT ms — lower is better) ---
Baseline:     #################### 750.0 ms
+KleidiAI:    ################ 620.0 ms (-17%)
+Speculative: ########### 420.0 ms (-44%)
```

---

## 🖥️ Redesigned Web Platform & Developer Tools

* **Linear/Vercel Developer UI (`:8080`):** Dark theme (`#0B0F14`), glassmorphism, Lucide icons, live CPU/RAM gauges, 7-stage pipeline visualizer, Chart.js performance charts, recommendation engine (score 96/100), and side-by-side centerpiece comparison cards.
* **Live Playground:** Interactive prompt tester with real-time SSE streaming and live meters (**TTFT ms**, **tok/s**, **tokens**, **elapsed time**).
* **Interactive CLI Chat (`demo_chat.py`):** Terminal client streaming responses turn-by-turn.
* **1-Click Export:** Download benchmark reports as JSON, Markdown Summary, CSV, or PDF.

---

## 🏷️ Tags & Try-It-Out Links

### Technology Tags
`arm64, kleidiai, llama.cpp, speculative-decoding, python3, fastapi, oracle-cloud, snapdragon-x, ubuntu, docker, docker-compose, gguf, pytorch, c++, cmake, huggingface, uvicorn, numactl, openblas, jinja2, bash, rest-api, sse-streaming, sglang, llm-inference, benchmarking`

### Links
* **Main GitHub Repository:** `https://github.com/Vasanth-repos/Armforge.git`
* **Root Specification Repo:** `https://github.com/Vasanth-repos/Arm_Forge.git`
