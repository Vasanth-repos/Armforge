# 🦾 ArmForge — Complete Step-by-Step Tutorial & Operations Guide

> **Learn how to deploy, auto-tune, benchmark, and run ArmForge on ARM64 cloud servers (Oracle Cloud Always Free A1, AWS Graviton, or local ARM64 hardware).**

---

## 📑 Table of Contents
1. [Prerequisites & Server Setup](#1-prerequisites--server-setup)
   - [Option A: Oracle Cloud Always Free A1 (Recommended)](#option-a-oracle-cloud-always-free-a1-recommended)
   - [Option B: AWS Graviton or Local ARM64 Instance](#option-b-aws-graviton-or-local-arm64-instance)
2. [Method 1: 1-Command Automated Pipeline (Fastest)](#method-1-1-command-automated-pipeline-fastest)
3. [Method 2: Step-by-Step Hands-On Guide](#method-2-step-by-step-hands-on-guide)
   - [Phase 0: Environment Bootstrap & Swap Setup](#phase-0-environment-bootstrap--swap-setup)
   - [Phase 1: Build llama.cpp with KleidiAI ON](#phase-1-build-llamacpp-with-kleidiai-on)
   - [Phase 1b: Build llama.cpp Baseline (KleidiAI OFF)](#phase-1b-build-llamacpp-baseline-kleidiai-off)
   - [Phase 2: Download Main & Draft Models](#phase-2-download-main--draft-models)
   - [Phase 3: Hardware Thread Sweep Auto-Tuning](#phase-3-hardware-thread-sweep-auto-tuning)
   - [Phase 4: Run Baseline Benchmark](#phase-4-run-baseline-benchmark)
   - [Phase 5: Run KleidiAI Throughput Benchmark](#phase-5-run-kleidiai-throughput-benchmark)
   - [Phase 6: Run KleidiAI + Speculative Latency Benchmark](#phase-6-run-kleidiai--speculative-latency-benchmark)
   - [Phase 7: Generate Comparative Breakdown Table & Charts](#phase-7-generate-comparative-breakdown-table--charts)
   - [Phase 8: Launch Web Dashboard & Streaming Playground](#phase-8-launch-web-dashboard--streaming-playground)
4. [Method 3: Docker & Container Deployment](#method-3-docker--container-deployment)
5. [Using Developer Tools](#using-developer-tools)
   - [Web Playground & Streaming Meters](#a-web-playground--streaming-meters)
   - [Interactive CLI Chat Client](#b-interactive-cli-chat-client)
   - [Bonus: SGLang W8A8 CPU Engine](#c-bonus-sglang-w8a8-cpu-engine)
6. [Troubleshooting & FAQs](#6-troubleshooting--faqs)

---

## 1. Prerequisites & Server Setup

### Option A: Oracle Cloud Always Free A1 (Recommended)

1. Sign up at [Oracle Cloud Free Tier](https://cloud.oracle.com/free).
2. Create an instance:
   * **Shape:** `VM.Standard.A1.Flex` (Ampere Altra ARM Processor)
   * **OCPUs:** `2` (set manually — default slider may select 1)
   * **Memory:** `12 GB` (set manually — default slider may select 6)
   * **Image:** `Canonical Ubuntu 22.04 Minimal aarch64`
   * **Boot Volume:** `100 GB`
3. **Network Security List:** Open ingress ports `22` (SSH), `8000` (KleidiAI server), `8001` (Baseline server), `8080` (Dashboard UI), and `30000` (SGLang).
4. SSH into your instance:
   ```bash
   ssh -i your_key.pem ubuntu@YOUR_INSTANCE_PUBLIC_IP
   ```

> 💡 **Region Availability Tip:** If Oracle Cloud shows *"Out of capacity"* in US regions, select `ap-singapore-1`, `eu-frankfurt-1`, or `ap-tokyo-1`.

---

### Option B: AWS Graviton or Local ARM64 Instance

* **AWS Instance:** `t4g.small` or `c6g.medium` running Ubuntu 22.04 ARM64 AMI.
* **Apple Silicon / Local ARM Linux:** Any Linux ARM64 system running Ubuntu 22.04 LTS (aarch64).

---

## Method 1: 1-Command Automated Pipeline (Fastest)

If you want to execute the complete setup, model download, thread tuning, benchmarking, and result generation non-interactively:

```bash
git clone https://github.com/Vasanth-repos/Armforge.git
cd Armforge
bash scripts/run_all.sh
```

---

## Method 2: Step-by-Step Hands-On Guide

### Phase 0: Environment Bootstrap & Swap Setup

Install system build tools, Python 3.12 venv, CPU-only PyTorch, and create a 4 GB swap file to prevent OOM errors during model loading:

```bash
cd ~/Armforge
bash scripts/00_bootstrap.sh
```

**What it does:**
- Verifies `aarch64` architecture.
- Creates `/swapfile` (4 GB) if missing.
- Configures virtualenv `~/armforge_env`.
- Installs CPU-only PyTorch (`torch` CPU build).
- Probes CPU features (`dotprod`, `i8mm`, `sve`).

---

### Phase 1: Build llama.cpp with KleidiAI ON

Compile `llama.cpp` with Arm KleidiAI matrix kernels enabled:

```bash
bash scripts/01_build_llamacpp.sh
```

**What it does:**
- Auto-detects `i8mm` or `dotprod` vector extensions.
- Sets `-DGGML_CPU_KLEIDIAI=ON` and `-DCMAKE_BUILD_TYPE=Release`.
- Compiles `llama-server` in `~/llama.cpp/build/bin/llama-server`.

---

### Phase 1b: Build llama.cpp Baseline (KleidiAI OFF)

Compile a separate baseline binary with KleidiAI disabled for scientific metric comparison:

```bash
bash scripts/01b_build_baseline.sh
```

**What it does:**
- Compiles `llama.cpp` in `~/llama.cpp/build_baseline` with `-DGGML_CPU_KLEIDIAI=OFF`.

---

### Phase 2: Download Main & Draft Models

Download the verifier model ($Llama\text{-}3.2\text{-}3B$) and tokenizer-matched draft model ($Llama\text{-}3.2\text{-}1B$):

```bash
bash scripts/02_download_models.sh
```

**Models Downloaded:**
- Main Model: `bartowski/Llama-3.2-3B-Instruct-GGUF` (`Llama-3.2-3B-Instruct-Q4_K_M.gguf`, ~2.0 GB)
- Draft Model: `bartowski/Llama-3.2-1B-Instruct-GGUF` (`Llama-3.2-1B-Instruct-Q4_K_M.gguf`, ~0.7 GB)

---

### Phase 3: Hardware Thread Sweep Auto-Tuning

Perform an automated thread sweep ($T \in \{1, 2, 3, 4\}$) to determine peak RAM channel throughput before memory bandwidth saturation occurs:

```bash
bash scripts/03_tune_threads.sh
```

**Output:** Saves optimal thread count $T_{\text{opt}}$ to `results/optimal_threads.txt`.

---

### Phase 4: Run Baseline Benchmark

Launch baseline server on port `8001` and execute benchmark suite:

```bash
bash scripts/04_benchmark_baseline.sh
```

**Output:** Saves `results/bench_baseline_<timestamp>.json`.

---

### Phase 5: Run KleidiAI Throughput Benchmark

Launch KleidiAI-accelerated server on port `8000` (`-b 512` batch size) and run throughput benchmark:

```bash
bash scripts/05_benchmark_kleidiai.sh
```

**Output:** Saves `results/bench_kleidiai_<timestamp>.json`.

---

### Phase 6: Run KleidiAI + Speculative Latency Benchmark

Launch server with speculative decoding (`--model-draft`, `--spec-type draft-simple`, `--draft-max 5`):

```bash
bash scripts/06_benchmark_optimized.sh
```

**Output:** Saves `results/bench_optimized_<timestamp>.json`.

---

### Phase 7: Generate Comparative Breakdown Table & Charts

Evaluate benchmark metrics, print terminal ASCII performance bar charts, and export `results/SUMMARY.md`:

```bash
source ~/armforge_env/bin/activate
python benchmark/compare.py
```

---

### Phase 8: Launch Web Dashboard & Streaming Playground

Start the FastAPI web interface on port `8080`:

```bash
bash scripts/07_start_dashboard.sh
```

Open your browser at: `http://YOUR_SERVER_IP:8080`

---

## Method 3: Docker & Container Deployment

If you prefer containerized deployment using Docker:

```bash
git clone https://github.com/Vasanth-repos/Armforge.git
cd Armforge

# Build and start services
docker compose up --build -d

# View logs
docker compose logs -f
```

---

## Using Developer Tools

### A. Web Playground & Streaming Meters
1. Open `http://YOUR_SERVER_IP:8080` in your web browser.
2. Select target port (`:8000` for KleidiAI / Speculative, `:8001` for Baseline).
3. Type any prompt into the text box and click **Generate Stream**.
4. Watch tokens stream live in real time while observing the **TTFT (ms)**, **tok/s**, and **token count** performance meters update live.

---

### B. Interactive CLI Chat Client

Start an interactive streaming chat session directly in your terminal:

```bash
source ~/armforge_env/bin/activate
python inference/demo_chat.py --port 8000
```

---

### C. Bonus: SGLang W8A8 CPU Engine

To benchmark SGLang W8A8 INT8 CPU serving:

```bash
source ~/armforge_env/bin/activate
pip install "sglang[srt]"
python inference/sglang_server.py &
sleep 60
python benchmark/run_bench.py --mode sglang --port 30000
python benchmark/compare.py
```

---

## 6. Troubleshooting & FAQs

### Q1: `llama-server: command not found`
- **Fix:** Ensure you ran `bash scripts/01_build_llamacpp.sh` successfully. Check `~/llama.cpp/build/bin/llama-server`.

### Q2: Port 8000 or 8080 already in use
- **Fix:** Kill existing server processes:
  ```bash
  pkill -f llama-server || true
  pkill -f uvicorn || true
  ```

### Q3: Out of Memory (OOM) during model loading
- **Fix:** Verify swap file allocation:
  ```bash
  free -h
  # If swap is 0 B, run:
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  ```

---

*ArmForge Tutorial · Apache 2.0 License · ARM AI Optimization Challenge 2026*
