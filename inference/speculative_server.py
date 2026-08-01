"""
Launch llama.cpp server with speculative decoding.

Verified flags (llama.cpp current master, 2026):
  --model-draft  (-md)  : draft model path
  --draft-max           : max draft tokens per step (default 5)
  --draft-min           : min draft tokens (default 0)
  --spec-type           : set to 'draft-simple' for standard draft-model decoding

Both models must be from the same model family (identical tokenizer).
Llama-3.2-3B + Llama-3.2-1B share the same Meta tokenizer — compatible.

API: OpenAI-compatible at http://localhost:8000
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
            raise FileNotFoundError(f"Not found: {path}\nRun scripts/01_build_llamacpp.sh and scripts/02_download_models.sh first.")

    cmd = [
        LLAMA_SERVER,
        "-m",          MAIN_MODEL,
        "--model-draft", DRAFT_MODEL,   # verified flag name (not --draft-model)
        "--spec-type", "draft-simple",  # use standalone draft model
        "--draft-max", str(draft_max),  # tokens drafted per step
        "-t",          str(threads),
        "-c",          str(context),
        "--host",      "0.0.0.0",
        "--port",      str(port),
        "--log-format", "json",
    ]

    print(f"Starting speculative decoding server on :{port}")
    print(f"Threads: {threads} | Draft max tokens: {draft_max}")
    subprocess.run(cmd)

if __name__ == '__main__':
    start()
