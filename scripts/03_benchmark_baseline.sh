#!/bin/bash
set -e
[ -f ~/armforge_env/bin/activate ] && source ~/armforge_env/bin/activate
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
echo "=== Baseline Benchmark (llama.cpp WITHOUT KleidiAI) ==="

# Build without KleidiAI for fair comparison
cd ~/llama.cpp
cmake -B build_baseline \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_KLEIDIAI=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_baseline -j$(nproc)

# Start baseline server in background
./build_baseline/bin/llama-server \
  -m models/main_model.gguf \
  -t $(nproc) \
  -c 2048 \
  --host 0.0.0.0 \
  --port 8001 &
SERVER_PID=$!
echo "Waiting 20s for model to load..."
sleep 20

cd "$REPO_ROOT"
if [ -f "backend/benchmark/run_bench.py" ]; then
  python3 backend/benchmark/run_bench.py --mode baseline
else
  python3 benchmark/run_bench.py --mode baseline
fi

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Baseline benchmark done. Check results/"
