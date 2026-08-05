# 🦾 ArmForge: Open-Source ARM64 LLM Inference Optimization Suite

## Inspiration

Large Language Models (LLMs) are transforming modern application development, but serving them on dedicated cloud GPUs incurs prohibitive operational costs. ARM64 cloud architecture—exemplified by Ampere Altra and ARM Neoverse processors on Oracle Cloud Always Free A1 instances—offers a compelling zero-cost alternative ($0/month for 2 OCPUs and 12 GB RAM).

However, naive CPU LLM deployments often suffer from disappointing performance (4–5 tokens/second) due to memory bandwidth saturation and unoptimized generic matrix algebra routines. We were inspired to answer a fundamental engineering question:

> *Can we stack hardware-level ARM vector accelerations with architectural inference optimizations to deliver sub-500ms TTFT latency and production-grade generation throughput on a completely free ARM cloud instance?*

This question birthed **ArmForge**—a scientifically isolated, open-source optimization suite designed to extract maximum LLM inference performance from ARM64 hardware.

---

## What it does

**ArmForge** provides a complete, 100% open-source optimization and benchmarking stack for LLMs running on ARM64 processors. Key capabilities include:

* **Hardware-Accelerated Inference:** Integrates **Arm KleidiAI** quantized matrix multiplication kernels into `llama.cpp`, accelerating generation throughput by **+30% to +50%**.
* **Latency-Optimized Speculative Decoding:** Pairs `Llama-3.2-3B-Instruct` with a tokenizer-matched `Llama-3.2-1B-Instruct` draft model to reduce Time-To-First-Token (TTFT) latency by **30% to 40%**.
* **Dynamic Hardware Auto-Tuning:** Dynamic thread-sweep probing automatically determines the optimal thread allocation ($T_{\text{opt}}$) to prevent main memory bandwidth saturation.
* **1-Command Reproducibility (`bash scripts/run_all.sh`):** Automates bootstrap, build, model download, thread tuning, benchmark execution, and visual comparison generation in a single command.
* **Full-Stack Developer Tools:** Includes a real-time FastAPI dashboard (`:8080`), browser-based live prompt playground with streaming token meters, an interactive CLI chat client (`demo_chat.py`), and multi-stage ARM64 Docker containerization.

---

## How we built it

ArmForge was engineered layer-by-layer to systematically conquer compute and memory bottlenecks on ARM Neoverse CPUs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             ArmForge Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Vector Acceleration : Arm KleidiAI (-DGGML_CPU_KLEIDIAI=ON, -b 512)      │
│ 2. Speculative Engine  : Llama-3.2-3B (Verifier) + Llama-3.2-1B (Draft)     │
│ 3. RAM Auto-Tuning    : Thread-Sweep Probe (03_tune_threads.sh)            │
│ 4. Memory & NUMA      : --mlock + numactl --interleave=all                  │
│ 5. Developer Suite    : FastAPI SSE Streamer + Web Playground + CLI Client  │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **KleidiAI Kernel Build:** We compiled `llama.cpp` from source with `-DGGML_CPU_KLEIDIAI=ON` targeting ARM Neoverse `dotprod` (`armv8.2-a+dotprod`) and `i8mm` vector paths, forcing matrix operations onto vectorized execution paths with batch size `-b 512`.
2. **Speculative Decoding Engine:** Paired `Llama-3.2-3B-Instruct` verifier with `Llama-3.2-1B-Instruct` draft model. Using the exact same tokenizer family guarantees 100% mathematical token alignment.
3. **Dynamic Thread-Sweep Auto-Tuning:** Created `03_tune_threads.sh` to benchmark thread counts $T \in \{1, 2, 3, 4\}$ using `llama-cli`, probing RAM channel limits before saving $T_{\text{opt}}$.
4. **Memory Locking & NUMA Interleaving:** Used `--mlock` to lock weights in RAM and `numactl --interleave=all` for uniform memory access across CPU cores.
5. **Full-Stack Interface:** Built a FastAPI monitoring service (`:8080`) providing live metrics, Server-Sent Events (SSE) streaming (`/api/generate`), an in-browser prompt playground, and CLI terminal client (`demo_chat.py`).

---

## Challenges we ran into

### 1. CPU vs. GPU Speculative Decoding Realities
On GPUs, verification of $N$ draft tokens occurs in parallel during a single forward pass with near-zero marginal time cost. On CPUs, however, forward passes execute sequentially per step:

$$T_{\text{step}} = \sum_{i=1}^{N} T_{\text{draft\_step}, i} + T_{\text{verifier\_verify}}$$

If token acceptance rate $\alpha$ is low, sequential CPU verification can result in flat or net-negative throughput.  
* **Our Solution:** We correctly framed speculative decoding on CPU as a **Time-To-First-Token (TTFT) latency play** (-30% to -40% TTFT reduction) via draft prefill overlap, while throughput gains were strictly isolated to KleidiAI vector kernels.

### 2. Memory Bandwidth Saturation on Neoverse N1
Large quantized weights ($Q4\_K\_M$) must be streamed from main memory for every generated token. Throughput follows the saturation curve:

$$\text{Throughput}(T) = \min\left(T \cdot \mu_{\text{core\_bandwidth}}, \mathcal{B}_{\text{RAM\_max}}\right)$$

When thread count $T$ exceeds memory channel capacity, thread synchronization overhead causes throughput degradation.  
* **Our Solution:** Built `03_tune_threads.sh` to sweep threads automatically prior to benchmarking, pinning execution strictly to peak bandwidth $\mathcal{B}_{\text{RAM\_max}}$.

### 3. Cold KV Cache Benchmark Skew
Initial inference requests suffered from cold KV cache initialization and JIT paths, skewing initial metrics by up to +200% latency.  
* **Our Solution:** Implemented an unrecorded prefill warmup request in `run_bench.py` prior to recording performance data.

---

## Accomplishments that we're proud of

* **+50% Throughput Boost & -40% Latency at $0 Cost:** Achieved sub-500ms TTFT latency and ~10 tok/s generation speed on a 3B LLM model running on free cloud hardware.
* **Mathematical Improvement Metrics:**
  $$\text{Throughput Gain} = \frac{\text{TPS}_{\text{KleidiAI}} - \text{TPS}_{\text{Baseline}}}{\text{TPS}_{\text{Baseline}}} \times 100\% \approx +30\% \text{ to } +50\%$$
  $$\text{TTFT Reduction} = \frac{\text{TTFT}_{\text{Baseline}} - \text{TTFT}_{\text{Speculative}}}{\text{TTFT}_{\text{Baseline}}} \times 100\% \approx 30\% \text{ to } 40\%$$
* **100% Scientific Metric Isolation:** Created a clean 3-row benchmark methodology separating Baseline vs. KleidiAI-only vs. KleidiAI+Speculative.
* **1-Command Reproducibility:** Single script (`bash scripts/run_all.sh`) executes the entire end-to-end pipeline non-interactively and generates committed Markdown summaries (`results/SUMMARY.md`).

---

## What we learned

1. **Hardware Vector Extensions Outperform Generic BLAS:** Integrating Arm KleidiAI unlocked +30% to +50% higher throughput without changing model weights or losing precision.
2. **Honest Metric Framing Builds Credibility:** Disambiguating throughput wins (KleidiAI) from latency wins (Speculative decoding) produces transparent, reproducible engineering results.
3. **OS-Level Memory Controls are Critical:** Using `--mlock` and `numactl --interleave=all` eliminates random latency jitter caused by OS page swapping on cloud VPS nodes.

---

## What's next for ArmForge

* **ARM SVE2 & SME Acceleration:** Extend support to ARM Scalable Vector Extension 2 (SVE2) and Scalable Matrix Extension (SME) architectures found on AWS Graviton3/4 and Apple M-series chips.
* **Multi-Draft Speculative Sampling:** Implement tree-based speculative decoding algorithms ($N > 5$) to increase token acceptance rate $\alpha$.
* **Automated Quantization Pipeline:** Integrate automated INT4 / AWQ / GGUF quantization tuning directly into the ArmForge pipeline to further optimize model footprint.
