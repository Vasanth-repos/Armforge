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
    {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "dest": "main_model.gguf",
    },
    {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "dest": "draft_model.gguf",
    },
]
for m in downloads:
    dest = os.path.join(MODELS_DIR, m["dest"])
    if os.path.exists(dest):
        print(f"Already exists: {m['dest']} ({os.path.getsize(dest)/1e9:.2f} GB)")
        continue
    print(f"Downloading {m['file']} ...")
    tmp = hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=MODELS_DIR)
    shutil.move(tmp, dest)
    print(f"  Saved: {m['dest']} ({os.path.getsize(dest)/1e9:.2f} GB)")
print("All models ready.")
EOF
