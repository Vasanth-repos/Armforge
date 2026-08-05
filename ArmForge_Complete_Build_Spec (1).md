# ArmForge — Complete Build Specification (v3 Corrected)
> **AI Agent Instructions:** Execute every shell block in strict order. All files defined inline. Do not skip sections. Each phase must succeed before proceeding.

---

## Project Summary
**Name:** ArmForge  
**Track:** ARM AI Optimization Challenge 2026 — Cloud AI (Track 2)  
**Tagline:** Open-source ARM64 LLM inference benchmark: KleidiAI + speculative decoding on free Oracle A1  
**License:** Apache 2.0 | **Cost:** $0 | **Python:** 3.12 | **OS:** Ubuntu 22.04 aarch64

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
| 9 | Oracle region note | Missing | Added: use ap-singapore-1 or eu-frankfurt-1 if A1 unavailable |
| 10 | Batch size in benchmark | 128 tokens fixed | Configurable; throughput measured per-token correctly |

---

## Repository Structure
```
armforge/
├── LICENSE
├── README.md
├── AGENT_BUILD_SPEC.md
├── requirements.txt
├── scripts/
│   ├── 00_bootstrap.sh
│   ├── 01_build_llamacpp.sh
│   ├── 01b_build_baseline.sh  ← separate baseline build
│   ├── 02_download_models.sh
│   ├── 03_tune_threads.sh     ← thread sweep before benchmarking
│   ├── 04_benchmark_baseline.sh
│   ├── 05_benchmark_kleidiai.sh    ← KleidiAI only, no speculative
│   ├── 06_benchmark_optimized.sh   ← KleidiAI + speculative
│   └── 07_start_dashboard.sh
├── inference/
│   ├── arm_features.py
│   ├── speculative_server.py
│   └── sglang_server.py
├── benchmark/
│   ├── run_bench.py
│   └── compare.py
├── dashboard/
│   ├── app.py
│   └── templates/index.html
└── results/
```

---

## Infrastructure

### Oracle Cloud A1 (Primary — Always Free)
```
1. https://cloud.oracle.com/free → sign up
2. Create instance:
   Shape:   VM.Standard.A1.Flex
   OCPUs:   2  (set manually — default may give 1)
   Memory:  12 GB (set manually — default may give 6)
   OS:      Canonical Ubuntu 22.04 aarch64
   Storage: 100 GB boot volume
3. Security List: open ports 22, 8000, 8001, 8002, 8080, 30000
4. SSH: ssh -i key.pem ubuntu@YOUR_IP

REGION NOTE: A1 is often out of capacity in US regions.
Try: ap-singapore-1, eu-frankfurt-1, ap-tokyo-1 (in that order).
```

### AWS Graviton2 (Fallback — Free until Dec 31 2026)
```
Instance: t4g.small (2 vCPU, 2 GB RAM)
OS:       Ubuntu 22.04 ARM64 AMI
Limit:    Use Q2_K model. No SGLang. No speculative (OOM risk).
```

---

## Tech Stack
| Layer | Tool | Version | Notes |
|---|---|---|---|
| Primary inference | llama.cpp | latest main | KleidiAI build |
| ARM acceleration | KleidiAI | bundled in llama.cpp | GGML_CPU_KLEIDIAI=ON |
| Secondary (bonus) | SGLang | 0.5.16+ | optional only |
| Main model | Llama-3.2-3B-Instruct-Q4_K_M | ~2.0 GB | bartowski mirror |
| Draft model | Llama-3.2-1B-Instruct-Q4_K_M | ~0.7 GB | same tokenizer family |
| Web framework | FastAPI | 0.111+ | dashboard |
| Build tools | cmake 3.22+, gcc 12+ | system | |

---

## requirements.txt
```
fastapi==0.111.1
uvicorn[standard]==0.29.0
psutil==5.9.8
httpx==0.27.0
huggingface_hub==0.23.4
pandas==2.2.2
tabulate==0.9.0
jinja2==3.1.4
requests==2.32.3
```

---

## LICENSE — Apache 2.0
```
Copyright 2026 ArmForge Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
```

---

## PHASE 0 — Bootstrap

### scripts/00_bootstrap.sh
```bash
#!/bin/bash
set -e
echo "=== ArmForge Bootstrap ==="

[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: Must run on aarch64"; exit 1; }
echo "Platform: aarch64 OK"

sudo apt-get update -qq
sudo apt-get install -y \
  build-essential cmake git wget curl \
  python3.12 python3.12-venv python3-pip \
  libblas-dev liblapack-dev libopenblas-dev \
  pkg-config libnuma-dev numactl htop

# 4 GB swap — prevents OOM during 3B model load
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo "Swap: 4 GB created"
fi

python3.12 -m venv ~/armforge_env
source ~/armforge_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# CPU PyTorch FIRST — prevents CUDA wheel resolution before sglang install
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

python3 -c "import torch; assert not torch.cuda.is_available(); print('PyTorch CPU OK:', torch.__version__)"

echo "=== CPU Features ==="
grep -m1 'Features' /proc/cpuinfo | tr ' ' '\n' | grep -E 'i8mm|sve|asimddp|asimd' || echo "baseline NEON only"
echo "Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "=== Bootstrap done ==="
```

---

## PHASE 1 — Build llama.cpp

### scripts/01_build_llamacpp.sh
```bash
#!/bin/bash
set -e
source ~/armforge_env/bin/activate
echo "=== Building llama.cpp (KleidiAI ON) ==="

cd ~
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git pull

FEATURES=$(grep -m1 'Features' /proc/cpuinfo 2>/dev/null || echo "")
if echo "$FEATURES" | grep -q "i8mm"; then
  ARCH_FLAG="-DGGML_CPU_ARM_ARCH=armv8.2-a+i8mm+dotprod"
  echo "Detected: i8mm → armv8.2-a+i8mm+dotprod"
elif echo "$FEATURES" | grep -q "asimddp"; then
  ARCH_FLAG="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod"
  echo "Detected: dotprod → armv8.2-a+dotprod (Neoverse N1 path)"
else
  ARCH_FLAG="-DGGML_NATIVE=ON"
  echo "Baseline NEON — no dotprod detected"
fi

cmake -B build \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DCMAKE_BUILD_TYPE=Release \
  $ARCH_FLAG

cmake --build build -j$(nproc)
grep -i "KLEIDIAI" build/CMakeCache.txt | head -3
echo "KleidiAI build complete: $(ls ~/llama.cpp/build/bin/llama-server)"
```

### scripts/01b_build_baseline.sh
```bash
#!/bin/bash
set -e
source ~/armforge_env/bin/activate
echo "=== Building llama.cpp (KleidiAI OFF — baseline) ==="

cd ~/llama.cpp
cmake -B build_baseline \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_KLEIDIAI=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_baseline -j$(nproc)
echo "Baseline build complete: $(ls ~/llama.cpp/build_baseline/bin/llama-server)"
```

---

## PHASE 2 — Download Models

### scripts/02_download_models.sh
```bash
#!/bin/bash
set -e
source ~/armforge_env/bin/activate
echo "=== Downloading Models ==="

mkdir -p ~/llama.cpp/models

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os, shutil

MODELS_DIR = os.path.expanduser("~/llama.cpp/models")
downloads = [
    {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "dest": "main_model.gguf",
    },
    {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "dest": "draft_model.gguf",
    },
]
for m in downloads:
    dest = os.path.join(MODELS_DIR, m["dest"])
    if os.path.exists(dest):
        print(f"Already exists: {m['dest']} ({os.path.getsize(dest)/1e9:.2f} GB)")
        continue
    print(f"Downloading {m['file']} ...")
    tmp = hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=MODELS_DIR)
    shutil.move(tmp, dest)
    print(f"  Saved: {m['dest']} ({os.path.getsize(dest)/1e9:.2f} GB)")
print("All models ready.")
EOF
```

---

## PHASE 3 — Thread Sweep (Run Before Any Benchmark)

### scripts/03_tune_threads.sh
```bash
#!/bin/bash
# CRITICAL: Run this before benchmarks. Neoverse N1 memory bandwidth saturates
# at low thread counts for large quantized models. Best thread count is often
# nproc/2, not nproc. This script finds the optimal count automatically.
set -e
source ~/armforge_env/bin/activate

LLAMA_CLI=~/llama.cpp/build/bin/llama-cli
MODEL=~/llama.cpp/models/main_model.gguf
PROMPT="Explain how ARM Neoverse processors accelerate AI inference in detail."
BEST_TPS=0
BEST_T=1
RESULTS_FILE=~/armforge/results/thread_sweep.txt
mkdir -p ~/armforge/results

echo "=== Thread Sweep ===" | tee $RESULTS_FILE
for T in 1 2 3 4; do
  [ $T -gt $(nproc) ] && continue
  echo -n "Threads=$T: "
  OUT=$($LLAMA_CLI \
    -m $MODEL \
    -t $T \
    -ngl 0 \
    --mlock \
    -n 64 \
    -p "$PROMPT" \
    --log-disable 2>&1 | grep "eval time")
  TPS=$(echo "$OUT" | grep -oP '\d+\.\d+ tokens per second' | head -1 | grep -oP '[\d.]+' || echo "0")
  echo "~$TPS tok/s" | tee -a $RESULTS_FILE
  if (( $(echo "$TPS > $BEST_TPS" | bc -l 2>/dev/null || echo 0) )); then
    BEST_TPS=$TPS
    BEST_T=$T
  fi
done

echo "OPTIMAL_THREADS=$BEST_T" | tee -a $RESULTS_FILE
echo "$BEST_T" > ~/armforge/results/optimal_threads.txt
echo "=== Optimal threads: $BEST_T (${BEST_TPS} tok/s) ==="
```

---

## PHASE 4 — Benchmark Scripts

### scripts/04_benchmark_baseline.sh
```bash
#!/bin/bash
set -e
source ~/armforge_env/bin/activate
cd ~/armforge

T=$(cat ~/armforge/results/optimal_threads.txt 2>/dev/null || echo $(nproc))
echo "=== Benchmark: Baseline (KleidiAI OFF, threads=$T) ==="

pkill -f llama-server 2>/dev/null || true; sleep 2

~/llama.cpp/build_baseline/bin/llama-server \
  -m ~/llama.cpp/models/main_model.gguf \
  -t $T \
  -ngl 0 \
  --mlock \
  -c 2048 \
  --host 0.0.0.0 --port 8001 &
SERVER_PID=$!
echo "Waiting 25s for model load..."
sleep 25

python benchmark/run_bench.py --mode baseline --port 8001

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Baseline done."
```

### scripts/05_benchmark_kleidiai.sh
```bash
#!/bin/bash
# KleidiAI only — NO speculative decoding
# Isolates KleidiAI contribution from speculative decoding contribution
set -e
source ~/armforge_env/bin/activate
cd ~/armforge

T=$(cat ~/armforge/results/optimal_threads.txt 2>/dev/null || echo $(nproc))
echo "=== Benchmark: KleidiAI only (threads=$T, batch=512) ==="

pkill -f llama-server 2>/dev/null || true; sleep 2

~/llama.cpp/build/bin/llama-server \
  -m ~/llama.cpp/models/main_model.gguf \
  -t $T \
  -ngl 0 \
  --mlock \
  -b 512 \
  -c 2048 \
  --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
echo "Waiting 25s for model load..."
sleep 25

python benchmark/run_bench.py --mode kleidiai --port 8000

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "KleidiAI benchmark done."
```

### scripts/06_benchmark_optimized.sh
```bash
#!/bin/bash
# KleidiAI + Speculative Decoding
# NOTE: On CPU, speculative decoding primarily reduces TTFT, not throughput.
# Throughput may be flat vs KleidiAI-only — this is expected and correct behavior.
set -e
source ~/armforge_env/bin/activate
cd ~/armforge

T=$(cat ~/armforge/results/optimal_threads.txt 2>/dev/null || echo $(nproc))
echo "=== Benchmark: KleidiAI + Speculative Decoding (threads=$T) ==="
echo "NOTE: Measuring TTFT reduction. Throughput gain vs KleidiAI-only is expected to be flat."

pkill -f llama-server 2>/dev/null || true; sleep 2

~/llama.cpp/build/bin/llama-server \
  -m  ~/llama.cpp/models/main_model.gguf \
  --model-draft ~/llama.cpp/models/draft_model.gguf \
  --spec-type draft-simple \
  --draft-max 5 \
  -t $T \
  -ngl 0 \
  --mlock \
  -b 512 \
  -c 2048 \
  --host 0.0.0.0 --port 8000 \
  --log-format json &
SERVER_PID=$!
echo "Waiting 30s for both models to load..."
sleep 30

python benchmark/run_bench.py --mode optimized --port 8000

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Optimized benchmark done."
```

### scripts/07_start_dashboard.sh
```bash
#!/bin/bash
source ~/armforge_env/bin/activate
cd ~/armforge
echo "Dashboard: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP'):8080"
uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
```

---

## Expected Results (Neoverse N1, 2 OCPU)

| Configuration | Expected tok/s | Expected TTFT |
|---|---|---|
| ① Baseline | 4–7 tok/s | 600–900 ms |
| ② + KleidiAI | 6–10 tok/s (+30–50%) | 500–700 ms |
| ③ + KleidiAI + Speculative | ~same as ② | 300–500 ms (−30–40%) |

**Speculative decoding win on CPU is TTFT, not throughput. This is correct and expected.**

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| `--draft-model` not found | Use `--model-draft` |
| `--draft N` not found | Use `--draft-max N` |
| `--spec-type` missing | Add `--spec-type draft-simple` |
| SGLang CUDA build installed | Install CPU torch first (done in 00_bootstrap.sh) |
| SGLang `--dtype int8` rejected | Use `--quantization w8a8` |
| OOM during model load | 4 GB swap created in 00_bootstrap.sh |
| Port already in use | `pkill -f llama-server && sleep 2` |
| KleidiAI not in CMakeCache | `grep KLEIDIAI ~/llama.cpp/build/CMakeCache.txt` — rebuild if missing |
| Oracle A1 no capacity | Try ap-singapore-1, eu-frankfurt-1, ap-tokyo-1 |
| Neoverse N1 = no i8mm | Has dotprod — script auto-detects, uses armv8.2-a+dotprod |
| Benchmark first result slow | Warmup call added in run_bench.py |
| Thread count not optimal | Run 03_tune_threads.sh before any benchmark |
| Model swapped out mid-bench | `--mlock` added to all server calls |

---

*ArmForge v3 · Apache 2.0 · $0 infrastructure · ARM AI Optimization Challenge 2026*
