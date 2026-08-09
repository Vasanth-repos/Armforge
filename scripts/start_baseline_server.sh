#!/bin/bash
# ==============================================================================
# ArmForge — Launch Background Baseline Model Server on Port 8001
# ==============================================================================
source ~/armforge_env/bin/activate 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"

BASELINE_SERVER=~/llama.cpp/build_baseline/bin/llama-server
[ -f "$BASELINE_SERVER" ] || BASELINE_SERVER=~/llama.cpp/build/bin/llama-server

MODEL="$HOME/armforge/models/Llama-3.2-3B-Instruct-Q8_0.gguf"
[ -f "$MODEL" ] || MODEL="$ARMFORGE_DIR/models/Llama-3.2-3B-Instruct-Q8_0.gguf"
[ -f "$MODEL" ] || MODEL="$HOME/armforge/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
[ -f "$MODEL" ] || MODEL="$ARMFORGE_DIR/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

if [ ! -f "$BASELINE_SERVER" ]; then
  echo "Baseline llama-server binary not found! Building baseline server now..."
  bash "$SCRIPT_DIR/01b_build_baseline.sh"
fi

if [ ! -f "$MODEL" ]; then
  echo "Model file not found! Downloading models now..."
  bash "$SCRIPT_DIR/02_download_models.sh"
fi

T=$(cat "$ARMFORGE_DIR/results/optimal_threads.txt" 2>/dev/null || cat "$ARMFORGE_DIR/results/best_threads.txt" 2>/dev/null || echo 4)
echo "=== Starting Baseline Model Server on port 8001 (threads=$T, model=$(basename $MODEL)) ==="

pkill -f "port 8001" 2>/dev/null || true; sleep 1

exec $BASELINE_SERVER \
  -m "$MODEL" \
  -t "$T" \
  -ngl 0 \
  --load-mode mlock \
  -c 2048 \
  --host 0.0.0.0 --port 8001
