#!/bin/bash
set -e
[ -f ~/armforge_env/bin/activate ] && source ~/armforge_env/bin/activate
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
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

cd "$REPO_ROOT"
if [ -f "backend/benchmark/run_bench.py" ]; then
  python3 backend/benchmark/run_bench.py --mode llama
else
  python3 benchmark/run_bench.py --mode llama
fi

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Optimized benchmark done."
