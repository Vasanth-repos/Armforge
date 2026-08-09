#!/bin/bash
# ==============================================================================
# ArmForge — Launch Background Baseline Model Server on Port 8001
# ==============================================================================
source ~/armforge_env/bin/activate 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"

T=$(cat "$ARMFORGE_DIR/results/optimal_threads.txt" 2>/dev/null || echo $(nproc))
echo "=== Starting Baseline Model Server on port 8001 (KleidiAI OFF, threads=$T) ==="

pkill -f "port 8001" 2>/dev/null || true; sleep 1

~/llama.cpp/build_baseline/bin/llama-server \
  -m ~/llama.cpp/models/main_model.gguf \
  -t $T \
  -ngl 0 \
  --load-mode mlock \
  -c 2048 \
  --host 0.0.0.0 --port 8001
