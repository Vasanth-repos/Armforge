#!/bin/bash
# CRITICAL: Run this before benchmarks. Neoverse N1 memory bandwidth saturates
# at low thread counts for large quantized models. Best thread count is often
# nproc/2, not nproc. This script finds the optimal count automatically.
set -e
source ~/armforge_env/bin/activate 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"

LLAMA_CLI=~/llama.cpp/build/bin/llama-cli
MODEL=~/llama.cpp/models/main_model.gguf
PROMPT="Explain how ARM Neoverse processors accelerate AI inference in detail."
BEST_TPS=0
BEST_T=1
RESULTS_FILE="$ARMFORGE_DIR/results/thread_sweep.txt"
mkdir -p "$ARMFORGE_DIR/results"

echo "=== Thread Sweep ===" | tee "$RESULTS_FILE"
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
    --log-disable < /dev/null 2>&1 || true)
  
  # Parse Generation t/s or eval time tps
  TPS=$(echo "$OUT" | grep -oP 'Generation:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | head -1 || echo "")
  if [ -z "$TPS" ]; then
    TPS=$(echo "$OUT" | grep -oP '[\d.]+\s*tokens per second' | grep -oP '[\d.]+' | head -1 || echo "0")
  fi
  
  echo "~$TPS tok/s" | tee -a "$RESULTS_FILE"
  if (( $(echo "$TPS > $BEST_TPS" | bc -l 2>/dev/null || echo 0) )); then
    BEST_TPS=$TPS
    BEST_T=$T
  fi
done

[ "$BEST_T" -eq 0 ] && BEST_T=2
echo "OPTIMAL_THREADS=$BEST_T" | tee -a "$RESULTS_FILE"
echo "$BEST_T" > "$ARMFORGE_DIR/results/optimal_threads.txt"
echo "=== Optimal threads selected: $BEST_T (${BEST_TPS} tok/s) ==="
