#!/bin/bash
set -e
source ~/armforge_env/bin/activate
cd ~/armforge
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

cd ~/armforge
python benchmark/run_bench.py --mode baseline

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Baseline benchmark done. Check results/"
