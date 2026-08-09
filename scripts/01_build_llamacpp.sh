#!/bin/bash
set -e
source ~/armforge_env/bin/activate 2>/dev/null || true
echo "=== Building llama.cpp (KleidiAI ON) ==="

cd ~
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git pull

FEATURES=$(grep -m1 'Features' /proc/cpuinfo 2>/dev/null || echo "")
if echo "$FEATURES" | grep -q "i8mm"; then
  ARCH_FLAG="-DGGML_CPU_ARM_ARCH=armv8.2-a+i8mm+dotprod"
  echo "Detected: i8mm → armv8.2-a+i8mm+dotprod"
elif echo "$FEATURES" | grep -q "asimddp"; then
  ARCH_FLAG="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod"
  echo "Detected: dotprod → armv8.2-a+dotprod (Neoverse N1 path)"
else
  ARCH_FLAG="-DGGML_NATIVE=ON"
  echo "Baseline NEON — no dotprod detected"
fi

cmake -B build \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER_WEBUI=OFF \
  $ARCH_FLAG

cmake --build build -j$(nproc)
grep -i "KLEIDIAI" build/CMakeCache.txt | head -3
echo "KleidiAI build complete: $(ls ~/llama.cpp/build/bin/llama-server)"
