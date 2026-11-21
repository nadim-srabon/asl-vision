import gc
import logging
import hashlib
import os
import tempfile
import time

import numpy as np
import streamlit as st

from inference.config import COMPARISON_1000, MODELS, label
from inference.model_loader import ModelUnavailable, load_model
from inference.predictor import predict
from inference.video_processor import MAX_DURATION_S, FeatureExtractor, VideoError
from ui import charts, data
from ui.theme import MODEL_COLORS, esc, html_row, metric, note, page_title

log = logging.getLogger("asl_vision")
ALLOWED = ["mp4", "mov", "avi", "webm", "mkv", "m4v"]
DEMO_ORDER = COMPARISON_1000 + ["geometry_transformer_300", "geometry_transformer_500", "geometry_transformer_1200"]


@st.cache_resource(show_spinner=False)
def get_extractor():
    return FeatureExtractor(reset_each_video=True)


@st.cache_resource(show_spinner=False, max_entries=1)
def get_model(model_id):
    """Only one model is resident at a time: max_entries=1 evicts the previous one on switch."""
    gc.collect()
    return load_model(model_id)


def _extract(file_bytes, suffix):
    key = hashlib.sha1(file_bytes).hexdigest()
    cache = st.session_state.setdefault("_features", {})
    if key in cache:
        return cache[key]
    fd, path = tempfile.mkstemp(suffix=f".{suffix}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
        t0 = time.perf_counter()
        feats = get_extractor().extract(path)
        feats.extract_s = time.perf_counter() - t0
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    cache.clear()                      # keep only the current upload's features in memory
    cache[key] = feats
    return feats


def _result_card(pred, spec):
    pct = pred.confidence * 100
    st.markdown(f"""
<div class="result">
  <div class="label">Predicted sign &middot; {esc(spec.display_name)} ({spec.num_classes} classes)</div>
  <div class="sign">{esc(pred.label)}</div>
  <div class="conf">Confidence {pct:.1f}% &nbsp;&middot;&nbsp; model inference {pred.inference_s * 1000:.0f} ms on CPU</div>
  <div class="bar"><div style="width:{min(pct, 100):.1f}%"></div></div>
</div>""", unsafe_allow_html=True)


def _quality_warnings(feats):
    msgs = []
    if feats.frames_with_hands == 0:
        msgs.append("No hands were detected in any sampled frame -- landmark-based models are effectively blind "
                    "for this clip. Make sure both hands are visible and well lit.")
    elif feats.frames_with_hands < 6:
        msgs.append(f"Hands were detected in only {feats.frames_with_hands}/16 sampled frames.")
    if feats.frames_with_pose < 8:
        msgs.append(f"The upper body was detected in only {feats.frames_with_pose}/16 frames, so the signer crop "
                    "falls back to the full frame for the rest. Frame the signer from the waist up, facing the camera.")
    for m in msgs:
        st.warning(m)


def render():
    page_title("Interactive demo", "ASL sign recognition",
               "Upload a short video of one isolated ASL sign, choose a trained model, and see what it predicts. "
               "All inference runs on CPU with the team's own checkpoints.")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        up = st.file_uploader("Video of a single sign", type=ALLOWED,
                              help=f"MP4 / MOV / AVI / WEBM / MKV, up to 50 MB and {int(MAX_DURATION_S)} s. "
                                   "Best results: one signer, upper body visible, plain background.")
        model_id = st.selectbox("Model", DEMO_ORDER, format_func=lambda m: label(MODELS[m]),
                                help="The four 1000-class models share the same vocabulary and test split.")
        spec = MODELS[model_id]
        m = data.metrics(model_id)
        if m:
            st.caption(f"{spec.description}  \nTest accuracy: **{m['test']['top1']:.1%} top-1 / "
                       f"{m['test']['top5']:.1%} top-5** on {spec.num_classes} signs.")
        compare_all = st.checkbox("Also run all four 1000-class models on this clip",
                                  help="Models are loaded one after another (never all at once).")
        go = st.button("Predict sign", type="primary", width="stretch", disabled=up is None)
    with right:
        if up is not None:
            if up.name.lower().endswith((".mp4", ".webm", ".m4v", ".mov")):
                st.video(up)
            else:
                st.info("Preview is not available for this container in most browsers; the video will still be "
                        "processed.")
        else:
            note("<b>Tips for a good recording</b><br>&bull; One sign per clip, roughly 1-5 seconds<br>"
                 "&bull; Signer centred, upper body and both hands visible<br>&bull; Even lighting, plain "
                 "background<br>&bull; The models know the most frequent ASL Citizen glosses, e.g. "
                 "<i>HELLO, THANKYOU, LOVE, BATHROOM, EXERCISE</i>")

    if up is None or not go and "_last" not in st.session_state:
        return
    if go:
        file_bytes = up.getvalue()
        suffix = up.name.rsplit(".", 1)[-1].lower()
        try:
            with st.status("Processing video...", expanded=False) as status:
                status.update(label="Sampling 16 frames and extracting MediaPipe landmarks...")
                feats = _extract(file_bytes, suffix)
                status.update(label=f"Loading {spec.display_name}...")
                model = get_model(model_id)
                status.update(label="Running the model...")
                pred = predict(model, model_id, feats)
                others = []
                if compare_all:
                    for mid in COMPARISON_1000:
                        status.update(label=f"Running {MODELS[mid].display_name}...")
                        others.append(pred if mid == model_id else predict(get_model(mid), mid, feats))
                    get_model(model_id)           # leave the selected model resident
                status.update(label="Done", state="complete")
            st.session_state["_last"] = dict(feats=feats, pred=pred, model_id=model_id, others=others,
                                             name=up.name)
        except VideoError as e:
            st.error(str(e))
            return
        except ModelUnavailable as e:
            st.error(f"The model could not be loaded: {e}")
            return
        except Exception as e:                      # no raw traceback for visitors; full one goes to the server log
            log.exception("prediction failed for %s with %s", up.name, model_id)
            st.error(f"Something went wrong while processing this video ({type(e).__name__}). "
                     "Please try another clip (MP4 with H.264 works best).")
            with st.expander("Technical details"):
                st.code(f"{type(e).__name__}: {e}"[:600])
            return

    last = st.session_state["_last"]
    if up is not None and last["name"] != up.name and not go:
        return
    feats, pred, spec = last["feats"], last["pred"], MODELS[last["model_id"]]
    info = feats.info

    st.write("")
    html_row([
        metric("Duration", f"{info.duration_s:.2f} s", f"{info.width}x{info.height}"),
        metric("Frame rate", f"{info.fps:.1f} fps"),
        metric("Total frames", f"{info.total_frames}"),
        metric("Frames used", "16", f"hands found in {feats.frames_with_hands}/16"),
    ])
    st.write("")
    _quality_warnings(feats)

    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        _result_card(pred, spec)
        st.caption(f"Feature extraction (frame sampling + MediaPipe) took {getattr(feats, 'extract_s', 0):.1f} s. "
                   "Confidence is the model's softmax probability, not a guarantee of correctness.")
    with c2:
        st.markdown("**Top-5 predictions**")
        st.plotly_chart(charts.top_k_bar(pred.top_labels, pred.top_probs,
                                         MODEL_COLORS.get(last["model_id"], "#2a78d6")),
                        width="stretch", config={"displayModeBar": False})

    if last["others"]:
        st.markdown("#### All 1000-class models on this clip")
        rows = [{"Model": MODELS[p.model_id].display_name, "Input": MODELS[p.model_id].family,
                 "Top-1 prediction": p.label, "Confidence": f"{p.confidence:.1%}",
                 "Top-5": ", ".join(p.top_labels), "Inference (ms)": round(p.inference_s * 1000)}
                for p in last["others"]]
        st.dataframe(rows, hide_index=True, width="stretch")

    with st.expander("What the model sees -- the 16 sampled signer crops (112x112)"):
        idx = np.linspace(0, 15, 8).astype(int)
        cols = st.columns(8)
        for c, t in zip(cols, idx):
            c.image(feats.appearance[t, 0], caption=f"frame {info.sampled_indices[t]}", width="stretch")
        st.caption("RGB models (Swin, R3D) see these crops; landmark models see the 58 MediaPipe points "
                   "(hands, arms, eyebrows) detected on the same 16 frames.")
    note("This is <b>isolated sign classification</b>: the clip is assigned to one of the model's known signs. "
         "Signs outside that vocabulary, multiple signs in one clip, or very different camera setups from ASL "
         "Citizen's webcam recordings will reduce accuracy.")
