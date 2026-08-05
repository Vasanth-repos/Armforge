#!/bin/bash
set -e
echo "=== ArmForge Bootstrap ==="

[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: Must run on aarch64"; exit 1; }
echo "Platform: aarch64 OK"

sudo apt-get update -qq
sudo apt-get install -y \
  build-essential cmake git wget curl \
  python3.12 python3.12-venv python3-pip \
  libblas-dev liblapack-dev libopenblas-dev \
  pkg-config libnuma-dev numactl htop bc

# 4 GB swap — prevents OOM during 3B model load
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo "Swap: 4 GB created"
fi

python3.12 -m venv ~/armforge_env
source ~/armforge_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# CPU PyTorch FIRST — prevents CUDA wheel resolution before sglang install
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

python3 -c "import torch; assert not torch.cuda.is_available(); print('PyTorch CPU OK:', torch.__version__)"

echo "=== CPU Features ==="
grep -m1 'Features' /proc/cpuinfo | tr ' ' '\n' | grep -E 'i8mm|sve|asimddp|asimd' || echo "baseline NEON only"
echo "Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "=== Bootstrap done ==="
