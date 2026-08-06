#!/bin/bash
# KleidiAI + Speculative Decoding
# NOTE: On CPU, speculative decoding primarily reduces TTFT, not throughput.
# Throughput may be flat vs KleidiAI-only — this is expected and correct behavior.
set -e
source ~/armforge_env/bin/activate 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"

T=$(cat "$ARMFORGE_DIR/results/optimal_threads.txt" 2>/dev/null || echo $(nproc))
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
  --load-mode mlock \
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
