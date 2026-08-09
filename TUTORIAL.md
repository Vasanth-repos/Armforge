# 🦾 ArmForge — Complete Step-by-Step Operations Tutorial

> **Learn how to deploy, auto-tune, benchmark, and run ArmForge on ARM64 cloud servers (Oracle Cloud Always Free A1, AWS Graviton, Hetzner Cloud) or Windows on ARM (Snapdragon X Elite / X Plus via WSL2).**

---

## 📑 Table of Contents
1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
   - [Option A: Windows on ARM via WSL2 (Recommended)](#option-a-windows-on-arm-via-wsl2-recommended)
   - [Option B: Oracle Cloud Always Free A1 (Recommended Cloud)](#option-b-oracle-cloud-always-free-a1-recommended-cloud)
   - [Option C: AWS Graviton / Hetzner Cloud](#option-c-aws-graviton--hetzner-cloud)
2. [Method 1: 1-Command Automated Pipeline (Fastest)](#method-1-1-command-automated-pipeline-fastest)
3. [Method 2: Step-by-Step Hands-On Guide](#method-2-step-by-step-hands-on-guide)
   - [Phase 0: Environment Bootstrap & Swap Setup](#phase-0-environment-bootstrap--swap-setup)
   - [Phase 1: Build llama.cpp with KleidiAI ON](#phase-1-build-llamacpp-with-kleidiai-on)
   - [Phase 1b: Build llama.cpp Baseline (KleidiAI OFF)](#phase-1b-build-llamacpp-baseline-kleidiai-off)
   - [Phase 2: Download Main & Draft Models](#phase-2-download-main--draft-models)
   - [Phase 3: Fast Hardware Thread Sweep Auto-Tuning](#phase-3-fast-hardware-thread-sweep-auto-tuning)
   - [Phase 4: Run Baseline Benchmark](#phase-4-run-baseline-benchmark)
   - [Phase 5: Run KleidiAI Throughput Benchmark](#phase-5-run-kleidiai-throughput-benchmark)
   - [Phase 6: Run KleidiAI + Speculative Latency Benchmark](#phase-6-run-kleidiai--speculative-latency-benchmark)
   - [Phase 7: Generate Comparative Breakdown Table & Charts](#phase-7-generate-comparative-breakdown-table--charts)
   - [Phase 8: Launch Redesigned Web Platform & Playground](#phase-8-launch-redesigned-web-platform--playground)
4. [Method 3: Docker & Container Deployment](#method-3-docker--container-deployment)
5. [Using Developer Tools](#using-developer-tools)
   - [Web Playground & Streaming Meters](#a-web-playground--streaming-meters)
   - [Interactive CLI Chat Client](#b-interactive-cli-chat-client)
   - [Bonus: SGLang W8A8 CPU Engine](#c-bonus-sglang-w8a8-cpu-engine)
6. [Troubleshooting & FAQs](#6-troubleshooting--faqs)

---

## 1. Prerequisites & Environment Setup

### Option A: Windows on ARM via WSL2 (Recommended)

Windows on ARM devices (Snapdragon X Elite / X Plus, Surface Pro 11, Lenovo Yoga Slim 7x) run ArmForge natively at 100% full hardware speed using WSL2 Ubuntu ARM64.

1. **Install WSL2 Ubuntu ARM64:** Open PowerShell as Administrator and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. **Launch & Clone:**
   ```bash
   wsl
   git clone https://github.com/Vasanth-repos/Armforge.git
   cd Armforge
   ```

---

### Option B: Oracle Cloud Always Free A1 (Recommended Cloud)

1. Sign up at [Oracle Cloud Free Tier](https://cloud.oracle.com/free).
2. Create an instance:
   * **Shape:** `VM.Standard.A1.Flex` (Ampere Altra ARM Processor)
   * **OCPUs:** `2` (set manually)
   * **Memory:** `12 GB` (set manually)
   * **Image:** `Canonical Ubuntu 22.04 Minimal aarch64`
   * **Boot Volume:** `100 GB`
3. **Network Security List:** Open ingress ports `22` (SSH), `8000` (KleidiAI server), `8001` (Baseline server), `8080` (Dashboard UI), and `30000` (SGLang).
4. SSH into your instance:
   ```bash
   ssh -i your_key.pem ubuntu@YOUR_INSTANCE_PUBLIC_IP
   ```

---

### Option C: AWS Graviton / Hetzner Cloud

* **AWS Instance:** `t4g.small` or `c6g.medium` running Ubuntu 22.04 ARM64 AMI.
* **Hetzner Cloud:** `CAX11` (2 ARM vCPU, 4 GB RAM — ~$3.50/mo).

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

Install system build tools, Python 3.12 venv, CPU-only PyTorch, and create a 4 GB swap file to prevent OOM errors:

```bash
cd ~/Armforge
bash scripts/00_bootstrap.sh
```

---

### Phase 1: Build llama.cpp with KleidiAI ON

Compile `llama.cpp` with Arm KleidiAI matrix kernels enabled (`-DLLAMA_BUILD_SERVER_WEBUI=OFF` to bypass npm UNC path errors):

```bash
bash scripts/01_build_llamacpp.sh
```

---

### Phase 1b: Build llama.cpp Baseline (KleidiAI OFF)

Compile a separate baseline binary with KleidiAI disabled for scientific metric comparison:

```bash
bash scripts/01b_build_baseline.sh
```

---

### Phase 2: Download Main & Draft Models

Download the verifier model ($Llama\text{-}3.2\text{-}3B$) and tokenizer-matched draft model ($Llama\text{-}3.2\text{-}1B$):

```bash
bash scripts/02_download_models.sh
```

---

### Phase 3: Fast Hardware Thread Sweep Auto-Tuning

Perform an automated non-interactive thread sweep ($T \in \{1, 2, 3, 4\}$) using `llama-bench` to determine peak RAM channel throughput in under 20 seconds:

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

---

### Phase 5: Run KleidiAI Throughput Benchmark

Launch KleidiAI-accelerated server on port `8000` (`-b 512` batch size) and run throughput benchmark:

```bash
bash scripts/05_benchmark_kleidiai.sh
```

---

### Phase 6: Run KleidiAI + Speculative Latency Benchmark

Launch server with speculative decoding (`--model-draft`, `--spec-type draft-simple`, `--draft-max 5`):

```bash
bash scripts/06_benchmark_optimized.sh
```

---

### Phase 7: Generate Comparative Breakdown Table & Charts

Evaluate benchmark metrics, print terminal ASCII performance bar charts, and export `results/SUMMARY.md`:

```bash
source ~/armforge_env/bin/activate
python benchmark/compare.py
```

---

### Phase 8: Launch Redesigned Web Platform & Streaming Playground

Start the FastAPI web interface on port `8080`:

```bash
bash scripts/07_start_dashboard.sh
```

Open your browser at: `http://localhost:8080`

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
1. Open `http://localhost:8080` in your web browser.
2. Select target port (`:8000` for KleidiAI / Speculative, `:8001` for Baseline).
3. Type any prompt into the text box and click **Generate Stream**.
4. Watch tokens stream live in real time while observing **TTFT (ms)**, **tok/s**, **token count**, and **elapsed time**.

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

### Q2: Error running inference: Make sure model server is running on port 8000
- **Fix:** Start the background model server helper:
  ```bash
  bash scripts/start_server.sh &
  ```

### Q3: `npm error ENOENT package.json` during CMake
- **Fix:** Update to the latest scripts which include `-DLLAMA_BUILD_SERVER_WEBUI=OFF` in CMake. Run `git pull origin main`.

---

*ArmForge Tutorial · Apache 2.0 License · ARM AI Optimization Challenge 2026*
