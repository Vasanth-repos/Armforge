"""
Launch llama.cpp server with speculative decoding.

IMPORTANT: On CPU (no GPU), speculative decoding reduces TTFT via draft token
prefill overlap. Throughput (tok/s) gain vs KleidiAI-only is typically flat
or marginal — this is correct behavior, not a bug. The win is latency.

Verified flags (llama.cpp 2026 master):
  --model-draft (-md)  : draft model path
  --draft-max          : max draft tokens per step
  --spec-type          : must be 'draft-simple' for standalone draft model
  -ngl 0               : explicit CPU-only (no GPU offload)
  --load-mode mlock    : lock model weights in RAM, prevents swap thrash
  -b 512               : batch size — activates KleidiAI dotprod kernel paths
"""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from inference.arm_features import optimal_threads

LLAMA_SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MAIN_MODEL   = os.path.expanduser("~/llama.cpp/models/main_model.gguf")
DRAFT_MODEL  = os.path.expanduser("~/llama.cpp/models/draft_model.gguf")

def start(port=8000, context=2048, draft_max=5):
    threads = optimal_threads()

    for path, name in [(LLAMA_SERVER, "llama-server"),
                       (MAIN_MODEL,   "main_model.gguf"),
                       (DRAFT_MODEL,  "draft_model.gguf")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")

    cmd = [
        LLAMA_SERVER,
        "-m",            MAIN_MODEL,
        "--model-draft", DRAFT_MODEL,
        "--spec-type",   "draft-simple",
        "--draft-max",   str(draft_max),
        "-t",            str(threads),
        "-ngl",          "0",
        "--load-mode",   "mlock",
        "-b",            "512",
        "-c",            str(context),
        "--host",        "0.0.0.0",
        "--port",        str(port),
        "--log-format",  "json",
    ]

    print(f"Starting speculative decoding server on :{port}")
    print(f"Threads: {threads} | Draft max: {draft_max} | load-mode: mlock | batch: 512")
    subprocess.run(cmd)

if __name__ == '__main__':
    start()
