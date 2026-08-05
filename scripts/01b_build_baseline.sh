#!/bin/bash
set -e
source ~/armforge_env/bin/activate
echo "=== Building llama.cpp (KleidiAI OFF — baseline) ==="

cd ~/llama.cpp
cmake -B build_baseline \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_KLEIDIAI=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_baseline -j$(nproc)
echo "Baseline build complete: $(ls ~/llama.cpp/build_baseline/bin/llama-server)"
