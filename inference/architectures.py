"""Exact model definitions for every trained ASL Vision checkpoint.

Each class reproduces the training notebook's module tree, so state_dict keys match and
checkpoints load with strict=True. Only inference-irrelevant training details (backbone
freezing, pretrained-weight download) are omitted.

Sources:
  GeometryOnlySLRModel  -- asl-*-class-single-stream-model-pilot-run.ipynb (v10), Section 5
  DualStreamSLRModel    -- previous_r3d_18 notebook.ipynb (v9), Section 9
  build_swin3d_t        -- training_notebooks/train_1000_swin3d_t.ipynb, Section 5
  STGCN                 -- training_notebooks/train_1000_landmark_models.ipynb, Section 5
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_COUNT = 16
N_HAND_LANDMARKS = 21
POSE_SUBSET_IDX = [11, 12, 13, 14, 15, 16]
FACE_SUBSET_IDX = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
N_POINTS = N_HAND_LANDMARKS * 2 + len(POSE_SUBSET_IDX) + len(FACE_SUBSET_IDX)   # 58
LANDMARK_VECTOR_DIM = N_POINTS * 2                                              # 116

# Shared Transformer dims (identical in v9 dual-stream and v10 geometry-only)
LANDMARK_EMBED_DIM = 128
CNN_EMBED_DIM = 256
D_MODEL = 256
N_HEADS = 4
N_ENCODER_LAYERS = 3
FFN_DIM = 512
DROPOUT = 0.5


# ---------------------------------------------------------------- shared blocks
class GeometryStream(nn.Module):
    """[coords, frame-to-frame velocity] -> Linear -> ReLU -> LayerNorm."""

    def __init__(self, input_dim=LANDMARK_VECTOR_DIM, embed_dim=LANDMARK_EMBED_DIM):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim * 2, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))

    def forward(self, x):
        velocity = torch.zeros_like(x)
        velocity[:, 1:] = x[:, 1:] - x[:, :-1]
        return self.project(torch.cat([x, velocity], dim=-1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=FRAME_COUNT):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def _temporal_encoder():
    layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
                                       dropout=DROPOUT, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=N_ENCODER_LAYERS)


# ---------------------------------------------------------------- v10 geometry-only Transformer
class GeometryOnlySLRModel(nn.Module):
    """Input: landmarks (B, 16, 116). Output: logits (B, num_classes)."""

    def __init__(self, num_classes):
        super().__init__()
        self.geometry_stream = GeometryStream()
        self.to_d_model = nn.Linear(LANDMARK_EMBED_DIM, D_MODEL)
        self.pos_encoding = PositionalEncoding(D_MODEL)
        self.temporal_encoder = _temporal_encoder()
        self.word_head = nn.Linear(D_MODEL, num_classes)

    def forward(self, landmarks):
        fused = self.pos_encoding(self.to_d_model(self.geometry_stream(landmarks)))
        return self.word_head(self.temporal_encoder(fused).mean(dim=1))


# ---------------------------------------------------------------- v9 dual-stream (r3d_18 + geometry)
class AppearanceStream(nn.Module):
    """r3d_18 trunk (stem..layer4), spatial avg-pool, temporal interp back to T, Linear 512->256."""

    def __init__(self, embed_dim=CNN_EMBED_DIM):
        super().__init__()
        from torchvision.models.video import r3d_18
        backbone = r3d_18(weights=None)          # trained weights come from the checkpoint
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.project = nn.Linear(512, embed_dim)

    def forward(self, x):
        B, T, S, C, H, W = x.shape                          # (B, T, streams, C, H, W)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B * S, C, T, H, W)
        feats = self.backbone(x).mean(dim=[-1, -2])         # (B*S, 512, T')
        feats = F.interpolate(feats, size=T, mode="linear", align_corners=False).permute(0, 2, 1)
        feats = self.project(feats)
        return feats.view(B, S, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)


class DualStreamSLRModel(nn.Module):
    """Inputs: appearance (B, 16, 1, 3, 112, 112) Kinetics-normalised, landmarks (B, 16, 116)."""

    def __init__(self, num_classes, num_appearance_streams=1):
        super().__init__()
        self.appearance_stream = AppearanceStream()
        self.geometry_stream = GeometryStream()
        self.fusion = nn.Linear(CNN_EMBED_DIM * num_appearance_streams + LANDMARK_EMBED_DIM, D_MODEL)
        self.pos_encoding = PositionalEncoding(D_MODEL)
        self.temporal_encoder = _temporal_encoder()
        self.word_head = nn.Linear(D_MODEL, num_classes)

    def forward(self, appearance, landmarks):
        fused = torch.cat([self.appearance_stream(appearance), self.geometry_stream(landmarks)], dim=-1)
        fused = self.pos_encoding(self.fusion(fused))
        return self.word_head(self.temporal_encoder(fused).mean(dim=1))


# ---------------------------------------------------------------- Video Swin-T
def build_swin3d_t(num_classes, head_dropout=0.5):
    """Input: (B, 3, 16, 224, 224) Kinetics-normalised."""
    from torchvision.models.video import swin3d_t
    m = swin3d_t(weights=None)
    m.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(m.head.in_features, num_classes))
    return m


# ---------------------------------------------------------------- ST-GCN (adaptive graph)
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def to_points(x, packing="interleaved"):
    if packing == "interleaved":
        return x.reshape(*x.shape[:-1], N_POINTS, 2)
    if packing == "planar":
        return x.reshape(*x.shape[:-1], 2, N_POINTS).transpose(-1, -2)
    raise ValueError(packing)


def skeleton_edges():
    e = []
    for off in (0, 21):
        e += [(i + off, j + off) for i, j in HAND_EDGES]
    p = {idx: 42 + k for k, idx in enumerate(POSE_SUBSET_IDX)}
    e += [(p[11], p[12]), (p[11], p[13]), (p[13], p[15]), (p[12], p[14]), (p[14], p[16])]
    f0 = 42 + len(POSE_SUBSET_IDX)
    e += [(f0 + i, f0 + i + 1) for i in range(4)] + [(f0 + 5 + i, f0 + 6 + i) for i in range(4)]
    return e


def normalized_adjacency(V, edges):
    A = torch.eye(V)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]


class AdaptiveGraphConv(nn.Module):
    def __init__(self, cin, cout, A):
        super().__init__()
        self.register_buffer("A", A)
        self.B = nn.Parameter(torch.zeros_like(A))
        self.conv = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return torch.einsum("nctv,vw->nctw", self.conv(x), self.A + self.B)


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, dropout=0.1, kt=5):
        super().__init__()
        self.gcn = AdaptiveGraphConv(cin, cout, A)
        self.bn1 = nn.BatchNorm2d(cout)
        self.tcn = nn.Sequential(nn.Conv2d(cout, cout, (kt, 1), (stride, 1), ((kt - 1) // 2, 0)),
                                 nn.BatchNorm2d(cout), nn.Dropout(dropout))
        if cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.tcn(F.relu(self.bn1(self.gcn(x)))) + self.res(x))


class STGCN(nn.Module):
    """Input: landmarks (B, 16, 116) -- same tensor as the v10 model."""

    def __init__(self, num_classes, packing="interleaved", in_ch=4, dropout=0.5):
        super().__init__()
        self.packing = packing
        A = normalized_adjacency(N_POINTS, skeleton_edges())
        self.data_bn = nn.BatchNorm1d(in_ch * N_POINTS)
        cfg = [(in_ch, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1)]
        self.blocks = nn.ModuleList(STGCNBlock(i, o, A, s) for i, o, s in cfg)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, landmarks):
        N, T, _ = landmarks.shape
        pts = to_points(landmarks, self.packing)
        vel = torch.zeros_like(pts)
        vel[:, 1:] = pts[:, 1:] - pts[:, :-1]
        x = torch.cat([pts, vel], -1).permute(0, 2, 3, 1).reshape(N, N_POINTS * 4, T)
        x = self.data_bn(x).reshape(N, N_POINTS, 4, T).permute(0, 2, 3, 1)
        for blk in self.blocks:
            x = blk(x)
        return self.fc(self.drop(x.mean(dim=(2, 3))))
