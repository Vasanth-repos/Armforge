"""
Launch SGLang CPU server with W8A8 quantization on ARM64.
API: OpenAI-compatible at http://localhost:30000
"""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from inference.arm_features import optimal_threads

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def start(port=30000):
    threads = optimal_threads()
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path",         MODEL_ID,
        "--device",             "cpu",
        "--dtype",              "float32",
        "--quantization",       "w8a8",
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
