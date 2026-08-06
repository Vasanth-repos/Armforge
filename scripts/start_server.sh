#!/bin/bash
# ==============================================================================
# ArmForge — Launch Background Model Server on Port 8000 for Dashboard Playground
# ==============================================================================
source ~/armforge_env/bin/activate 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"

T=$(cat "$ARMFORGE_DIR/results/optimal_threads.txt" 2>/dev/null || echo $(nproc))
echo "=== Starting KleidiAI Model Server on port 8000 (threads=$T, batch=512) ==="

pkill -f llama-server 2>/dev/null || true; sleep 1

~/llama.cpp/build/bin/llama-server \
  -m ~/llama.cpp/models/main_model.gguf \
  -t $T \
  -ngl 0 \
  --load-mode mlock \
  -b 512 \
  -c 2048 \
  --host 0.0.0.0 --port 8000
