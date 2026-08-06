#!/bin/bash
# ==============================================================================
# ArmForge — Launch FastAPI Web Dashboard & Live Playground
# ==============================================================================
source ~/armforge_env/bin/activate 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMFORGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARMFORGE_DIR"
export PYTHONPATH="$ARMFORGE_DIR:$PYTHONPATH"

echo "============================================================"
echo " 🦾 ArmForge Web Dashboard & Streaming Playground"
echo "============================================================"
echo "Local Dashboard: http://localhost:8080"
echo "Public Access:   http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP'):8080"
echo "NOTE: Ensure model server is running on port 8000:"
echo "      (Run in background: bash scripts/start_server.sh &)"
echo "============================================================"

uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
