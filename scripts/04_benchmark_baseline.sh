#!/bin/bash
set -e
source ~/armforge_env/bin/activate 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"

T=$(cat "$ARMFORGE_DIR/results/optimal_threads.txt" 2>/dev/null || echo $(nproc))
echo "=== Benchmark: Baseline (KleidiAI OFF, threads=$T) ==="

pkill -f llama-server 2>/dev/null || true; sleep 2

~/llama.cpp/build_baseline/bin/llama-server \
  -m ~/llama.cpp/models/main_model.gguf \
  -t $T \
  -ngl 0 \
  --load-mode mlock \
  -c 2048 \
  --host 0.0.0.0 --port 8001 &
SERVER_PID=$!
echo "Waiting 25s for model load..."
sleep 25

python benchmark/run_bench.py --mode baseline --port 8001

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Baseline done."
