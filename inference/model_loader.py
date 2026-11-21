"""Build a model, fetch its weights if needed, load strictly, validate, return it in eval mode (CPU)."""
import json
import os
from functools import lru_cache

import torch

from .architectures import DualStreamSLRModel, GeometryOnlySLRModel, STGCN, build_swin3d_t
from .config import HF_REPO_ID, MODELS, ModelSpec

BUILDERS = {
    "swin3d_t": build_swin3d_t,
    "r3d18_dualstream": DualStreamSLRModel,
    "geometry_transformer": GeometryOnlySLRModel,
    "stgcn": STGCN,
}


class ModelUnavailable(RuntimeError):
    pass


def _hf_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


def ensure_weights(spec: ModelSpec):
    """Local file if present, otherwise download that single file from the Hugging Face repo."""
    if spec.weights_path.exists():
        return spec.weights_path
    repo = HF_REPO_ID or _secret("ASL_HF_REPO")
    if not repo:
        raise ModelUnavailable(f"Weights for '{spec.model_id}' are not available locally and no Hugging Face "
                               f"repository is configured (set ASL_HF_REPO).")
    from huggingface_hub import hf_hub_download
    spec.weights_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return hf_hub_download(repo_id=repo, filename=spec.weights_path.name, token=_hf_token(),
                               local_dir=str(spec.weights_path.parent))
    except Exception as e:                       # network, auth (private repo without token), missing file
        raise ModelUnavailable(f"could not download {spec.weights_path.name} from Hugging Face repo '{repo}' "
                               f"({type(e).__name__}). Check ASL_HF_REPO / HF_TOKEN in the app secrets.") from e


def _secret(name):
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


@lru_cache(maxsize=8)
def load_class_names(num_classes):
    spec = next(s for s in MODELS.values() if s.num_classes == num_classes)
    with open(spec.class_list_path, encoding="utf-8") as f:
        data = json.load(f)
    names = data["classes"]
    if len(names) != num_classes:
        raise ModelUnavailable(f"class list for {num_classes} classes has {len(names)} entries")
    return names


def load_model(model_id):
    spec = MODELS[model_id]
    path = ensure_weights(spec)
    # mmap + meta-device construction: weights are never held twice in RAM (no random init copy).
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and "model_state_dict" in state:        # tolerate full checkpoints
        state = state["model_state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}     # tolerate DataParallel prefixes
    with torch.device("meta"):
        model = BUILDERS[spec.architecture](spec.num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise ModelUnavailable(f"checkpoint for '{model_id}' does not match the architecture "
                               f"(missing {list(missing)[:3]}, unexpected {list(unexpected)[:3]})")
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftover:
        raise ModelUnavailable(f"checkpoint for '{model_id}' left tensors uninitialised: {leftover[:3]}")
    model.eval()
    head = model.head[1] if spec.architecture == "swin3d_t" else (model.fc if spec.architecture == "stgcn" else model.word_head)
    if head.out_features != spec.num_classes or len(load_class_names(spec.num_classes)) != spec.num_classes:
        raise ModelUnavailable(f"output size of '{model_id}' does not match its class list")
    return model
