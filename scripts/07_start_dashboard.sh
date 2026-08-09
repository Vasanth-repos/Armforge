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
echo "============================================================"

# Auto-launch baseline server (port 8001) in background if inactive
if ! curl -s http://localhost:8001/health >/dev/null 2>&1; then
  echo "Auto-launching Baseline Model Server (Port 8001) in background..."
  bash "$SCRIPT_DIR/start_baseline_server.sh" >/dev/null 2>&1 &
fi

# Auto-launch KleidiAI server (port 8000) in background if inactive
if ! curl -s http://localhost:8000/health >/dev/null 2>&1; then
  echo "Auto-launching KleidiAI Model Server (Port 8000) in background..."
  bash "$SCRIPT_DIR/start_server.sh" >/dev/null 2>&1 &
fi

uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
