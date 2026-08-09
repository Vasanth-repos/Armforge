#!/bin/bash
# ==============================================================================
# ArmForge — Launch Background KleidiAI Model Server on Port 8000
# ==============================================================================
source ~/armforge_env/bin/activate 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"

KLEIDIAI_SERVER=~/llama.cpp/build/bin/llama-server
[ -f "$KLEIDIAI_SERVER" ] || KLEIDIAI_SERVER=~/llama.cpp/build_kleidiai/bin/llama-server

MODEL="$HOME/armforge/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
[ -f "$MODEL" ] || MODEL="$ARMFORGE_DIR/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

if [ ! -f "$KLEIDIAI_SERVER" ]; then
  echo "KleidiAI llama-server binary not found! Building KleidiAI server now..."
  bash "$SCRIPT_DIR/01_build_llamacpp.sh"
fi

if [ ! -f "$MODEL" ]; then
  echo "Model file not found! Downloading models now..."
  bash "$SCRIPT_DIR/02_download_models.sh"
fi

T=$(cat "$ARMFORGE_DIR/results/optimal_threads.txt" 2>/dev/null || cat "$ARMFORGE_DIR/results/best_threads.txt" 2>/dev/null || echo 4)
echo "=== Starting KleidiAI Model Server on port 8000 (threads=$T, batch=512, model=$(basename $MODEL)) ==="

pkill -f "port 8000" 2>/dev/null || true; sleep 1

exec $KLEIDIAI_SERVER \
  -m "$MODEL" \
  -t "$T" \
  -ngl 0 \
  -b 512 \
  -c 2048 \
  --host 0.0.0.0 --port 8000
