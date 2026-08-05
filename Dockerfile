# Multi-stage ARM64 Dockerfile for ArmForge
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3 python3-pip curl \
    libblas-dev liblapack-dev libopenblas-dev pkg-config libnuma-dev numactl

WORKDIR /app

# Clone and build llama.cpp with KleidiAI
RUN git clone https://github.com/ggml-org/llama.cpp /app/llama.cpp && \
    cd /app/llama.cpp && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc)

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000 8080

CMD ["bash", "scripts/run_all.sh"]
