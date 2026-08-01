#!/bin/bash
set -e
source ~/armforge_env/bin/activate
echo "=== Building llama.cpp with KleidiAI ==="

cd ~
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git pull

# Detect CPU features and set correct -march flag
FEATURES=$(grep -m1 'Features' /proc/cpuinfo 2>/dev/null || echo "")
ARCH_FLAG=""
if echo "$FEATURES" | grep -q "i8mm"; then
  echo "i8mm detected → armv8.2-a+i8mm+dotprod"
  ARCH_FLAG="-DGGML_CPU_ARM_ARCH=armv8.2-a+i8mm+dotprod"
elif echo "$FEATURES" | grep -q "asimddp"; then
  echo "dotprod detected → armv8.2-a+dotprod"
  ARCH_FLAG="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod"
else
  echo "Baseline NEON — standard GGML_NATIVE build"
  ARCH_FLAG="-DGGML_NATIVE=ON"
fi

cmake -B build \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DCMAKE_BUILD_TYPE=Release \
  $ARCH_FLAG

cmake --build build -j$(nproc)

# Verify KleidiAI was enabled in the build
grep -i "KLEIDIAI" build/CMakeCache.txt | head -5

echo "Build complete:"
ls ~/llama.cpp/build/bin/llama-*
