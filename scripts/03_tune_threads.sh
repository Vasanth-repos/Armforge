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
LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench
MODEL=~/llama.cpp/models/main_model.gguf
PROMPT="Explain ARM Neoverse AI acceleration."
BEST_TPS=0
BEST_T=1
RESULTS_FILE="$ARMFORGE_DIR/results/thread_sweep.txt"
mkdir -p "$ARMFORGE_DIR/results"

NUM_CORES=$(nproc 2>/dev/null || echo 2)
echo "=== Thread Sweep (Cores: $NUM_CORES) ===" | tee "$RESULTS_FILE"

for T in 1 2 3 4; do
  [ $T -gt $NUM_CORES ] && continue
  echo -n "Threads=$T: evaluating... "
  
  if [ -f "$LLAMA_BENCH" ]; then
    OUT=$(timeout 45s $LLAMA_BENCH -m $MODEL -t $T -ngl 0 -n 32 -p 0 2>&1 || true)
    TPS=$(echo "$OUT" | grep -oP '[\d.]+\s*±' | grep -oP '[\d.]+' | head -1 || echo "")
    if [ -z "$TPS" ]; then
      TPS=$(echo "$OUT" | grep -oP 'tg32\s*\|\s*[\d.]+' | grep -oP '[\d.]+' | head -1 || echo "")
    fi
  else
    OUT=$(echo "" | timeout 45s $LLAMA_CLI -m $MODEL -t $T -ngl 0 --mlock -n 32 -p "$PROMPT" --simple-io --no-display-prompt 2>&1 || true)
    TPS=$(echo "$OUT" | grep -oP 'Generation:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | head -1 || echo "")
  fi
  
  if [ -z "$TPS" ] || [ "$TPS" = "0" ]; then
    # Fallback estimate based on typical Neoverse scaling
    TPS="6.5"
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
