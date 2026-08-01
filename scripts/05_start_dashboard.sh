#!/bin/bash
source ~/armforge_env/bin/activate
cd ~/armforge
echo "Dashboard: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP'):8080"
if [ -f "backend/app.py" ]; then
  uvicorn backend.app:app --host 0.0.0.0 --port 8080 --reload
else
  uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
fi
