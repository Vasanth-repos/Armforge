#!/bin/bash
# ==============================================================================
# ArmForge — Launch Both KleidiAI (:8000) & Baseline (:8001) Servers
# ==============================================================================
source ~/armforge_env/bin/activate 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"

echo "=== Starting KleidiAI Model Server (:8000) & Baseline Model Server (:8001) ==="
bash "$SCRIPT_DIR/start_server.sh" &
bash "$SCRIPT_DIR/start_baseline_server.sh" &
echo "Both model servers launched in background!"
