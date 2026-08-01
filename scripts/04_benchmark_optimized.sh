#!/bin/bash
set -e
source ~/armforge_env/bin/activate
cd ~/armforge
echo "=== Optimized Benchmark (KleidiAI + Speculative Decoding) ==="

# Start optimized server in background
cd ~/llama.cpp
./build/bin/llama-server \
  -m  models/main_model.gguf \
  --model-draft models/draft_model.gguf \
  --spec-type draft-simple \
  --draft-max 5 \
  -t  $(nproc) \
  -c  2048 \
  --host 0.0.0.0 \
  --port 8000 \
  --log-format json &
SERVER_PID=$!
echo "Waiting 25s for both models to load..."
sleep 25

cd ~/armforge
python benchmark/run_bench.py --mode llama

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Optimized benchmark done."
