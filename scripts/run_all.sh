#!/bin/bash
# ==============================================================================
# ArmForge — One-Shot Master Execution & Benchmark Pipeline
# Runs all phases (00..06) sequentially and displays the final comparative report.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"

echo "============================================================"
echo " 🦾 ArmForge — Master Execution & Benchmark Pipeline"
echo "============================================================"
echo "Timestamp: $(date)"
echo "Host:      $(uname -n) ($(uname -m))"
echo "Directory: $ARMFORGE_DIR"
echo "============================================================"

bash scripts/00_bootstrap.sh
bash scripts/01_build_llamacpp.sh
bash scripts/01b_build_baseline.sh
bash scripts/02_download_models.sh

echo ""
echo ">>> Phase 3: Thread Sweep Tuning"
bash scripts/03_tune_threads.sh

echo ""
echo ">>> Phase 4: Baseline Benchmark (vanilla llama.cpp, KleidiAI OFF)"
bash scripts/04_benchmark_baseline.sh

echo ""
echo ">>> Phase 5: KleidiAI Benchmark (KleidiAI dotprod kernels ON)"
bash scripts/05_benchmark_kleidiai.sh

echo ""
echo ">>> Phase 6: KleidiAI + Speculative Decoding Benchmark"
bash scripts/06_benchmark_optimized.sh

echo ""
echo "============================================================"
echo " 🎉 ArmForge Pipeline Execution Complete!"
echo "============================================================"

source ~/armforge_env/bin/activate
python benchmark/compare.py

echo ""
echo "To start the live dashboard: bash scripts/07_start_dashboard.sh"
