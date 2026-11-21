"""Turn cached-format features into each model's exact input tensor and run inference."""
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import KINETICS_MEAN, KINETICS_STD, MODELS
from .model_loader import load_class_names

SWIN_IMG_SIZE = 224
_MEAN = torch.tensor(KINETICS_MEAN)
_STD = torch.tensor(KINETICS_STD)


def landmark_tensor(landmarks):
    """(16, 116) -> (1, 16, 116); nan_to_num exactly as every training Dataset did."""
    x = torch.from_numpy(np.asarray(landmarks, dtype=np.float32)).unsqueeze(0)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def swin_tensor(appearance):
    """(16, 1, 112, 112, 3) uint8 -> (1, 3, 16, 224, 224): /255, trilinear resize, Kinetics norm
    (eval path of train_1000_swin3d_t.ipynb gpu_preprocess)."""
    x = torch.from_numpy(np.ascontiguousarray(appearance[:, 0])).unsqueeze(0)     # (1, T, H, W, 3)
    x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)                                # (1, 3, T, H, W)
    T = x.shape[2]
    x = F.interpolate(x, size=(T, SWIN_IMG_SIZE, SWIN_IMG_SIZE), mode="trilinear", align_corners=False)
    return (x - _MEAN.view(1, 3, 1, 1, 1)) / _STD.view(1, 3, 1, 1, 1)


def r3d_tensor(appearance):
    """(16, 1, 112, 112, 3) uint8 -> (1, 16, 1, 3, 112, 112): /255, Kinetics norm (v9 Dataset, eval)."""
    x = torch.from_numpy(np.ascontiguousarray(appearance)).float() / 255.0
    x = x.permute(0, 1, 4, 2, 3)
    x = (x - _MEAN.view(1, 1, 3, 1, 1)) / _STD.view(1, 1, 3, 1, 1)
    return x.unsqueeze(0)


@dataclass
class Prediction:
    model_id: str
    top_labels: list
    top_probs: list
    top_indices: list
    inference_s: float

    @property
    def label(self):
        return self.top_labels[0]

    @property
    def confidence(self):
        return self.top_probs[0]


def model_inputs(model_id, features):
    kind = MODELS[model_id].inputs
    if kind == "landmarks":
        return (landmark_tensor(features.landmarks),)
    if kind == "rgb":
        return (swin_tensor(features.appearance),)
    return (r3d_tensor(features.appearance), landmark_tensor(features.landmarks))


def predict(model, model_id, features, k=5):
    spec = MODELS[model_id]
    inputs = model_inputs(model_id, features)
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(*inputs)
    elapsed = time.perf_counter() - t0
    if logits.shape != (1, spec.num_classes):
        raise RuntimeError(f"unexpected output shape {tuple(logits.shape)} for {model_id}")
    probs = torch.softmax(logits.float(), dim=1)[0]
    top = probs.topk(min(k, spec.num_classes))
    names = load_class_names(spec.num_classes)
    idx = top.indices.tolist()
    return Prediction(model_id, [names[i] for i in idx], top.values.tolist(), idx, elapsed)
