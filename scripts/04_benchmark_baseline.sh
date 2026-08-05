#!/bin/bash
set -e
source ~/armforge_env/bin/activate
cd ~/armforge

T=$(cat ~/armforge/results/optimal_threads.txt 2>/dev/null || echo $(nproc))
echo "=== Benchmark: Baseline (KleidiAI OFF, threads=$T) ==="

pkill -f llama-server 2>/dev/null || true; sleep 2

NUMA_CMD=""
if command -v numactl &> /dev/null; then
  NUMA_CMD="numactl --interleave=all"
fi

$NUMA_CMD ~/llama.cpp/build_baseline/bin/llama-server \
  -m ~/llama.cpp/models/main_model.gguf \
  -t $T \
  -ngl 0 \
  --mlock \
  -c 2048 \
  --host 0.0.0.0 --port 8001 &
SERVER_PID=$!
echo "Waiting 25s for model load..."
sleep 25

python benchmark/run_bench.py --mode baseline --port 8001

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Baseline done."
