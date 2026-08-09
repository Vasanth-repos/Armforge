#!/bin/bash
# ==============================================================================
# ArmForge — Step 2: Download Model GGUF Files
# Downloads Llama-3.2-3B-Instruct (Q4_K_M & Q8_0) and Llama-3.2-1B-Instruct (Q4_K_M)
# ==============================================================================
set -e
source ~/armforge_env/bin/activate 2>/dev/null || true
echo "=== Phase 2: Downloading Model Weights ==="

mkdir -p ~/armforge/models
mkdir -p ~/llama.cpp/models
mkdir -p models

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os, shutil

target_dirs = [
    os.path.expanduser("~/armforge/models"),
    os.path.expanduser("~/llama.cpp/models"),
    os.path.abspath("models"),
]

downloads = [
    {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "aliases": ["main_model.gguf"]
    },
    {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "aliases": ["q8_model.gguf"]
    },
    {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "aliases": ["draft_model.gguf"]
    },
]

primary_dir = target_dirs[0]

for m in downloads:
    target_path = os.path.join(primary_dir, m["file"])
    if not os.path.exists(target_path) or os.path.getsize(target_path) < 100_000_000:
        print(f"Downloading {m['file']} from HF ({m['repo']})...")
        tmp = hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=primary_dir)
        if tmp != target_path and os.path.exists(tmp):
            shutil.move(tmp, target_path)
    
    print(f"✅ Ready: {m['file']} ({os.path.getsize(target_path)/1e9:.2f} GB)")

    # Create copies/symlinks across target directories and aliases
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, m["file"])
        if not os.path.exists(dst):
            try:
                os.symlink(target_path, dst)
            except Exception:
                try:
                    shutil.copy2(target_path, dst)
                except Exception:
                    pass

        for alias in m.get("aliases", []):
            alias_dst = os.path.join(d, alias)
            if not os.path.exists(alias_dst):
                try:
                    os.symlink(target_path, alias_dst)
                except Exception:
                    try:
                        shutil.copy2(target_path, alias_dst)
                    except Exception:
                        pass

print("All models ready across all model directories.")
EOF
