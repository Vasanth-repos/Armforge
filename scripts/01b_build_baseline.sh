#!/bin/bash
set -e
source ~/armforge_env/bin/activate 2>/dev/null || true
echo "=== Building llama.cpp (KleidiAI OFF — baseline) ==="

cd ~/llama.cpp

IS_WSL=false
grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=true
WEBUI_FLAG=""
$IS_WSL && WEBUI_FLAG="-DLLAMA_BUILD_SERVER_WEBUI=OFF -DGGML_BUILD_SERVER_WEBUI=OFF -DNPM_EXECUTABLE=FALSE"

cmake -B build_baseline \
  -DGGML_NATIVE=OFF \
  -DGGML_KLEIDIAI=OFF \
  -DGGML_CPU_KLEIDIAI=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER_WEBUI=OFF \
  -DGGML_BUILD_SERVER_WEBUI=OFF \
  -DNPM_EXECUTABLE=FALSE \
  $WEBUI_FLAG

cmake --build build_baseline -j$(nproc)
echo "=== Baseline Build Complete ==="
./build_baseline/bin/llama-cli --version 2>&1 || true
