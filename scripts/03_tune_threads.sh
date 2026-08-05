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
