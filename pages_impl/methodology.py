import streamlit as st

from ui import data
from ui.theme import card, flow, html_row, note, page_title


def render():
    page_title("How it works", "Methodology",
               "From a raw video to a predicted sign: the same pipeline is used for training data and for the "
               "live demo, so the models always see inputs prepared exactly as during training.")

    flow([
        ("Video input", "one isolated sign, any length up to 20 s"),
        ("Frame extraction", "16 frames, evenly spaced over the clip"),
        ("Preprocessing", "MediaPipe hands / pose / face; signer crop 112x112"),
        ("Feature extraction", "3D-CNN, video Transformer, or landmark network"),
        ("Classification layer", "linear head + softmax over the vocabulary"),
        ("Predicted sign", "top-1 gloss with confidence, plus top-5"),
    ])

    st.subheader("1. Why video, not single images?")
    st.markdown(
        "An ASL sign is defined by **handshape, location, orientation and movement**. A single frame captures "
        "the first three but not the movement, and many signs differ *only* in how the hands move (for example "
        "direction or repetition). The models therefore classify a short **sequence** of frames, so motion is "
        "part of what they learn.")

    st.subheader("2. Preprocessing (identical in training and in this app)")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            "- **Temporal sampling:** all frames are decoded; 16 indices are chosen with "
            "`linspace(0, n-1, 16)` (short clips repeat their last frame). Temporal order is preserved.\n"
            "- **Landmarks:** Google MediaPipe detects 21 points per hand, the upper-body pose and the face on "
            "each sampled frame. We keep both hands, shoulders / elbows / wrists, and 10 eyebrow points "
            "(non-manual markers) = **58 points**.\n"
            "- **Normalisation:** points are centred on the shoulder midpoint and divided by shoulder width, "
            "making them independent of where the signer stands and how far they are from the camera.")
    with c2:
        st.markdown(
            "- **Signer crop (RGB stream):** a box around the detected body (+25 % margin) is cut out and "
            "resized to **112x112**, removing most of the background.\n"
            "- **RGB normalisation:** pixels are scaled to [0, 1] and normalised with Kinetics-400 mean / std, "
            "the statistics the pretrained video backbones expect. Swin-T upsamples the crop to 224x224.\n"
            "- **Tensor layouts:** R3D-18 `(B, T, 1, 3, 112, 112)`, Swin-T `(B, 3, T, 224, 224)`, "
            "landmark models `(B, T, 116)` with T = 16.")

    st.subheader("3. What a 3D CNN is")
    st.markdown(
        "A 2D convolution slides a *k x k* filter over an image. A **3D convolution** slides a *t x k x k* filter "
        "over a stack of frames, so each filter responds to a pattern in **space and time together** -- e.g. "
        "a hand moving downward. Stacking such layers builds features that describe motion directly, "
        "instead of describing each frame separately.")

    st.subheader("4. The architectures")
    html_row([
        card("R3D-18 (3D ResNet-18)",
             "18-layer residual network in which every convolution is 3D (3x3x3). Stem + 4 residual stages; "
             "the time axis is downsampled from 16 to 2 steps while spatial features grow to 512 channels. "
             "Pretrained on Kinetics-400 human-action videos. In our dual-stream model its per-time-step "
             "features are fused with a landmark stream and passed to a 3-layer Transformer.", tag="3D CNN"),
        card("Video Swin Transformer (Swin3D-T)",
             "Splits the clip into 2x4x4 space-time patches and applies self-attention inside shifted 3D "
             "windows, so every layer can relate distant hand positions across frames. 28 M parameters, "
             "Kinetics-400 pretrained, fine-tuned end-to-end.", tag="VIDEO TRANSFORMER"),
    ])
    st.write("")
    html_row([
        card("Geometry Transformer (v10)",
             "Each frame's 116 landmark coordinates plus their frame-to-frame velocity are embedded to 256 "
             "dimensions, given sinusoidal time positions, and processed by a 3-layer, 4-head Transformer "
             "encoder; the 16 outputs are averaged. Only ~1.9 M parameters.", tag="LANDMARKS"),
        card("ST-GCN (spatial-temporal graph network)",
             "Treats the 58 landmarks as nodes of a skeleton graph (finger bones, arm chain, eyebrows) with an "
             "additional learned adjacency. Graph convolutions mix information between connected joints; "
             "temporal convolutions model each joint's motion.", tag="LANDMARKS"),
    ])

    st.subheader("5. How spatial and temporal features are extracted")
    st.markdown(
        "- **RGB models** learn *spatial* features (handshape, hand position relative to face and body) and "
        "*temporal* features (movement) jointly, inside the 3D convolutions (R3D-18) or the space-time attention "
        "windows (Swin).\n"
        "- **Landmark models** receive geometry directly: spatial structure comes from the coordinates / graph, "
        "temporal structure from velocity features, self-attention over time (Transformer) or temporal "
        "convolutions (ST-GCN).")

    st.subheader("6. From features to a prediction")
    st.markdown(
        "The final feature vector (768-d for Swin, 256-d for the others) is mapped by a **linear "
        "classification layer** to one score (logit) per sign in the vocabulary. **Softmax** turns the scores "
        "into probabilities that sum to 1; the highest is the predicted sign and its probability is the "
        "displayed confidence. Training minimised label-smoothed cross-entropy with class-balanced sampling.")

    st.subheader("7. Evaluation protocol")
    ds = data.dataset()
    st.markdown(
        f"- **Data:** ASL Citizen, official **signer-independent** splits (no person appears in two splits). "
        f"1,000-sign subset: {ds['subset_1000']['train']:,} train / {ds['subset_1000']['val']:,} validation / "
        f"{ds['subset_1000']['test']:,} test clips, 30-32 clips per sign.\n"
        "- **Model selection** on the validation split only; the test split is used **once**, at the end.\n"
        "- **Metrics:** top-1 and top-5 accuracy; precision, recall and F1 macro-averaged over classes; "
        "confusion matrices; CPU latency per clip.")
    note("Every model in this project classifies <b>one isolated sign per clip</b>. Continuous signing "
         "(sentences) and real-time translation are outside the scope of this work.")
