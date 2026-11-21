"""One-time upload of models/*.pt to a Hugging Face model repository (free).

    huggingface-cli login            # once, with a WRITE token from huggingface.co/settings/tokens
    .venv/Scripts/python scripts/upload_to_hf.py <your-username>/asl-vision-models

The repo is created PRIVATE by default (recommended: ASL Citizen's licence forbids redistributing the
data, and whether trained weights count as a derivative is for the team/supervisor to decide). The
Streamlit app then needs a READ token in its secrets (HF_TOKEN). Pass --public to make it public instead.
Only the 7 weight files are uploaded -- no dataset videos, features or notebooks.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inference.config import MODELS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("repo_id")
p.add_argument("--public", action="store_true")
args = p.parse_args()

api = HfApi()
api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
for spec in MODELS.values():
    assert spec.weights_path.exists(), f"missing {spec.weights_path} -- run scripts/prepare_artifacts.py first"
    print(f"uploading {spec.weights_path.name} ({spec.weights_path.stat().st_size / 1e6:.1f} MB)...")
    api.upload_file(path_or_fileobj=str(spec.weights_path), path_in_repo=spec.weights_path.name,
                    repo_id=args.repo_id, repo_type="model")
print(f"\nDone: https://huggingface.co/{args.repo_id}  (private={not args.public})")
