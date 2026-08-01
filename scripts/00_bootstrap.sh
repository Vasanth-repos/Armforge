#!/bin/bash
set -e
echo "=== ArmForge Bootstrap ==="

# Verify ARM64
[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: Must run on aarch64"; exit 1; }
echo "Platform: aarch64 OK"

# System packages
sudo apt-get update -qq
sudo apt-get install -y \
  build-essential cmake git wget curl \
  python3.12 python3.12-venv python3-pip \
  libblas-dev liblapack-dev libopenblas-dev \
  pkg-config libnuma-dev numactl htop

# 4 GB swap (prevents OOM during 3B model load on 12 GB instance)
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo "Swap: 4 GB created"
fi

# Python venv
python3.12 -m venv ~/armforge_env
source ~/armforge_env/bin/activate
pip install --upgrade pip

# Install FastAPI stack
pip install -r requirements.txt

# Install CPU-only PyTorch FIRST (critical — prevents CUDA wheel resolution)
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch is CPU build
python3 -c "import torch; assert not torch.cuda.is_available(), 'Got CUDA build!'; print('PyTorch CPU build OK:', torch.__version__)"

# Install SGLang CPU (aarch64 wheel available since Jul 2026)
pip install "sglang[srt]"

echo ""
echo "=== CPU Feature Report ==="
grep -m1 'Features' /proc/cpuinfo | tr ' ' '\n' | grep -E 'i8mm|sve|asimddp|asimd' || echo "none detected"
echo "Cores: $(nproc)"
echo "RAM:   $(free -h | awk '/^Mem:/{print $2}')"
echo "=== Bootstrap done ==="
