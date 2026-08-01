"""
Launch SGLang CPU server with W8A8 quantization on ARM64.

Install requirements (done in 00_bootstrap.sh):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install "sglang[srt]"

Verified SGLang flags for CPU + quantization:
  --device cpu          : use CPU backend (no GPU required)
  --quantization w8a8   : W8A8 INT8 quantization (not --dtype int8)
  --dtype float32       : base compute dtype for CPU

NOTE: If SGLang CPU crashes on first request (bug in some 0.5.x builds),
fall back to llama.cpp speculative_server.py — it is the primary benchmark target.
SGLang is a bonus demonstration.

API: OpenAI-compatible at http://localhost:30000
"""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from inference.arm_features import optimal_threads

# Use a smaller open model to avoid HF gated model issues
# Qwen2.5-1.5B-Instruct is ungated and well-tested on CPU
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def start(port=30000):
    threads = optimal_threads()
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path",         MODEL_ID,
        "--device",             "cpu",
        "--dtype",              "float32",   # CPU base dtype
        "--quantization",       "w8a8",      # verified SGLang W8A8 flag
        "--host",               "0.0.0.0",
        "--port",               str(port),
        "--context-length",     "2048",
        "--mem-fraction-static","0.7",
        "--log-level",          "info",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"]      = str(threads)
    env["MKL_NUM_THREADS"]      = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)

    print(f"Starting SGLang W8A8 CPU server on :{port}")
    print(f"Model: {MODEL_ID}")
    print(f"OMP threads: {threads}")
    subprocess.run(cmd, env=env)

if __name__ == '__main__':
    start()
