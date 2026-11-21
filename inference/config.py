"""Registry of every trained model shown in the app.

Weights live in ``models/<model_id>.pt`` (plain fp32 state_dict). Locally they are produced by
``scripts/prepare_artifacts.py``; in the cloud they are downloaded on demand from the Hugging Face
model repo below. Metrics / class lists / confusion matrices are small and live in ``results/``.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CLASS_LIST_DIR = RESULTS_DIR / "class_lists"

# Hugging Face model repo holding the weights. A *private* repo is recommended (see README:
# ASL Citizen's licence forbids redistributing the data); the app then reads HF_TOKEN from
# Streamlit secrets / the environment. Override with env var ASL_HF_REPO.
HF_REPO_ID = os.environ.get("ASL_HF_REPO", "")

KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    family: str                  # "RGB video" | "RGB + landmarks" | "Landmarks"
    architecture: str            # key understood by model_loader.build_model
    num_classes: int
    inputs: str                  # "rgb" | "rgb+landmarks" | "landmarks"
    description: str
    approx_weights_mb: float
    origin: str                  # which notebook / capstone stage produced it
    extra: dict = field(default_factory=dict)

    @property
    def weights_path(self) -> Path:
        return MODELS_DIR / f"{self.model_id}.pt"

    @property
    def class_list_path(self) -> Path:
        return CLASS_LIST_DIR / f"classes_{self.num_classes}.json"

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.model_id


MODELS = {
    "swin3d_t_1000": ModelSpec(
        "swin3d_t_1000", "Video Swin-T", "RGB video", "swin3d_t", 1000, "rgb",
        "Video Swin Transformer (tiny), Kinetics-400 initialised, fine-tuned end-to-end on 16-frame "
        "signer crops resized to 224x224. Shifted-window self-attention over space and time.",
        129, "Capstone C -- train_1000_swin3d_t.ipynb"),
    "r3d18_dualstream_1000": ModelSpec(
        "r3d18_dualstream_1000", "R3D-18 Dual-Stream", "RGB + landmarks", "r3d18_dualstream", 1000,
        "rgb+landmarks",
        "Capstone B (v9) model: Kinetics-400 R3D-18 3D-CNN on 112x112 signer crops (layer4 fine-tuned) "
        "fused with a MediaPipe landmark stream, followed by a 3-layer Transformer encoder.",
        141, "Capstone B -- r3d_18 dual-stream notebook (v9)"),
    "geometry_transformer_1000": ModelSpec(
        "geometry_transformer_1000", "Geometry Transformer", "Landmarks", "geometry_transformer", 1000,
        "landmarks",
        "v10 geometry-only model: 58 MediaPipe landmarks (hands, arms, eyebrows) + velocity per frame, "
        "linear embedding, sinusoidal positions, 3-layer Transformer encoder, mean pooling.",
        7.6, "Capstone C -- train_1000_landmark_models.ipynb"),
    "stgcn_1000": ModelSpec(
        "stgcn_1000", "ST-GCN", "Landmarks", "stgcn", 1000, "landmarks",
        "Spatial-temporal graph convolutional network over the 58-landmark skeleton (hand bones, arm "
        "chain, eyebrows) plus a learned adjacency; 7 graph/temporal-conv blocks.",
        5.5, "Capstone C -- train_1000_landmark_models.ipynb"),
    "geometry_transformer_300": ModelSpec(
        "geometry_transformer_300", "Geometry Transformer", "Landmarks", "geometry_transformer", 300,
        "landmarks", "v10 geometry-only Transformer trained on the 300 most frequent glosses.",
        6.9, "Capstone C -- asl-300-class-single-stream-model-pilot-run"),
    "geometry_transformer_500": ModelSpec(
        "geometry_transformer_500", "Geometry Transformer", "Landmarks", "geometry_transformer", 500,
        "landmarks", "v10 geometry-only Transformer trained on the 500 most frequent glosses.",
        7.1, "Capstone C -- asl-500-class-single-stream-model-pilot-run"),
    "geometry_transformer_1200": ModelSpec(
        "geometry_transformer_1200", "Geometry Transformer", "Landmarks", "geometry_transformer", 1200,
        "landmarks", "v10 geometry-only Transformer trained on the 1200 most frequent glosses.",
        7.8, "Capstone C -- asl-1200-class-single-stream-model-pilot-run"),
}

# Order used in dropdowns / tables
COMPARISON_1000 = ["swin3d_t_1000", "geometry_transformer_1000", "stgcn_1000", "r3d18_dualstream_1000"]
SCALING_SERIES = ["geometry_transformer_300", "geometry_transformer_500",
                  "geometry_transformer_1000", "geometry_transformer_1200"]


def label(spec: ModelSpec) -> str:
    return f"{spec.display_name} -- {spec.num_classes} classes ({spec.family})"
