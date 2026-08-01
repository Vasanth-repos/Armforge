#!/bin/bash
set -e
source ~/armforge_env/bin/activate
echo "=== Downloading Models ==="

mkdir -p ~/llama.cpp/models

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os, shutil

MODELS_DIR = os.path.expanduser("~/llama.cpp/models")

downloads = [
    # Main model — 3B, Q4_K_M ~2.0 GB
    {
        "repo":     "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file":     "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "dest":     "main_model.gguf",
    },
    # Draft model — 1B, same Llama-3.2 family = identical tokenizer
    # Required for speculative decoding compatibility
    {
        "repo":     "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file":     "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "dest":     "draft_model.gguf",
    },
]

for m in downloads:
    dest_path = os.path.join(MODELS_DIR, m["dest"])
    if os.path.exists(dest_path):
        size_gb = os.path.getsize(dest_path) / 1e9
        print(f"Already exists: {m['dest']} ({size_gb:.2f} GB)")
        continue
    print(f"Downloading {m['file']} ...")
    tmp = hf_hub_download(repo_id=m["repo"], filename=m["file"],
                          local_dir=MODELS_DIR)
    shutil.move(tmp, dest_path)
    size_gb = os.path.getsize(dest_path) / 1e9
    print(f"  Saved: {m['dest']} ({size_gb:.2f} GB)")

print("\nAll models ready.")
print(f"Total: {sum(os.path.getsize(os.path.join(MODELS_DIR, f))/1e9 for f in os.listdir(MODELS_DIR) if f.endswith('.gguf')):.2f} GB")
EOF
