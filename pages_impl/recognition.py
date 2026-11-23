import gc
import hashlib
import logging
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
from ui.theme import esc, panel_title

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


def _result_html(pred, spec):
    pct = pred.confidence * 100
    return f"""
<div class="result">
  <div class="label">Predicted sign &middot; {esc(spec.display_name)}</div>
  <div class="sign">{esc(pred.label)}</div>
  <div class="conf">Confidence {pct:.1f}% &nbsp;&middot;&nbsp; {spec.num_classes} classes</div>
  <div class="bar"><div style="width:{min(pct, 100):.1f}%"></div></div>
</div>"""


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


def _figcaption(text, pill=""):
    p = f'<span class="pill">{esc(pill)}</span>' if pill else ""
    st.markdown(f'<div class="figcaption"><span>{esc(text)}</span>{p}</div>', unsafe_allow_html=True)


def _hint(title, sub):
    st.markdown(f'<div class="stage"><div class="drop-hint"><b>{esc(title)}</b>{esc(sub)}</div></div>',
                unsafe_allow_html=True)


def _hint_text(text):
    st.markdown(f'<div class="hint">{text}</div>', unsafe_allow_html=True)


def _short(mid):
    s = MODELS[mid]
    return f"{s.display_name} · {s.num_classes}"


def _run(up, model_id, compare_all):
    spec = MODELS[model_id]
    file_bytes = up.getvalue()
    suffix = up.name.rsplit(".", 1)[-1].lower()
    box = st.empty()
    try:
        with box.status("Processing video...", expanded=False) as status:
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
    box.empty()                                 # success: the result card replaces the progress box
    st.session_state["_last"] = dict(feats=feats, pred=pred, model_id=model_id, others=others, name=up.name)
    hist = st.session_state.setdefault("_history", [])
    hist.insert(0, dict(sign=pred.label, conf=pred.confidence, model=_short(model_id)))
    del hist[8:]
    st.session_state["_status"] = f"Last run: {pred.label}"
    st.session_state["_status_detail"] = f"{_short(model_id)} · {pred.inference_s * 1000:.0f} ms model"


def _controls():
    with st.container(key="panel_video"):
        panel_title("Video")
        up = st.file_uploader("Clip of a single sign", type=ALLOWED, label_visibility="collapsed",
                              help=f"MP4 / MOV / AVI / WEBM / MKV, up to 50 MB and {int(MAX_DURATION_S)} s.")
        _hint_text("One sign, 1-5 s, upper body and both hands visible.")
    with st.container(key="panel_model"):
        panel_title("Model")
        model_id = st.selectbox("Model", DEMO_ORDER, format_func=lambda m: label(MODELS[m]),
                                label_visibility="collapsed")
        spec = MODELS[model_id]
        m = data.metrics(model_id)
        if m:
            _hint_text(f'{esc(spec.family)} &middot; test accuracy <b style="color:#d8b4fe">{m["test"]["top1"]:.1%}</b>'
                       f' top-1 / {m["test"]["top5"]:.1%} top-5 on {spec.num_classes} signs')
    with st.container(key="panel_options"):
        panel_title("Options")
        compare_all = st.toggle("Compare all four 1000-class models",
                                help="Models are loaded one after another (never all at once).")
        _hint_text("CPU only &middot; one model in memory at a time")
    go = st.button("Recognize sign", type="primary", width="stretch", disabled=up is None)
    with st.container(key="panel_about_model"):
        panel_title("About this model")
        _hint_text(esc(spec.description))
    return up, model_id, compare_all, go


def _documents(up, last):
    d1, d2 = st.columns(2, gap="small")
    with d1:
        with st.container(key="doc_source"):
            _figcaption("Source", up.name[:30] if up is not None else "")
            if up is None:
                _hint("No video yet", "Upload a clip in the Video panel")
            elif up.name.lower().endswith((".mp4", ".webm", ".m4v", ".mov")):
                st.video(up)
            else:
                _hint("Preview unavailable", "This container can't play in the browser, but it will be processed")
            if last:
                i = last["feats"].info
                meta = f"{i.duration_s:.2f} s · {i.fps:.0f} fps · {i.width}x{i.height} · {i.total_frames} frames"
            else:
                meta = "—"
            st.markdown(f'<div class="doc-bar"><span>{esc(meta)}</span><span>16 sampled</span></div>',
                        unsafe_allow_html=True)
    with d2:
        with st.container(key="doc_result"):
            _figcaption("Result", _short(last["model_id"]) if last else "")
            if last:
                st.markdown(f'<div class="stage">{_result_html(last["pred"], MODELS[last["model_id"]])}</div>',
                            unsafe_allow_html=True)
                meta = (f"model {last['pred'].inference_s * 1000:.0f} ms · "
                        f"features {getattr(last['feats'], 'extract_s', 0):.1f} s")
                right_meta = f"hands {last['feats'].frames_with_hands}/16"
            else:
                _hint("Nothing yet", "Run to see the predicted sign")
                meta, right_meta = "—", ""
            st.markdown(f'<div class="doc-bar"><span>{esc(meta)}</span><span>{esc(right_meta)}</span></div>',
                        unsafe_allow_html=True)


def render():
    left, right = st.columns([2.35, 1], gap="medium")
    with right:
        up, model_id, compare_all, go = _controls()

    with left:
        if go and up is not None:
            _run(up, model_id, compare_all)
        last = st.session_state.get("_last")
        if last and (up is None or last["name"] != up.name):
            last = None
        _documents(up, last)

        hist = st.session_state.get("_history", [])
        if hist:
            chips = "".join(f'<div class="chip"><b>{esc(h["sign"])}</b><span>{h["conf"]:.0%} · {esc(h["model"])}'
                            f'</span></div>' for h in hist)
            st.markdown(f'<div class="history"><span class="h-label">History</span>{chips}</div>',
                        unsafe_allow_html=True)
        if not last:
            return

        feats, pred = last["feats"], last["pred"]
        _quality_warnings(feats)
        with st.container(key="panel_top5"):
            panel_title("Top-5 predictions")
            st.plotly_chart(charts.top_k_bar(pred.top_labels, pred.top_probs), width="stretch",
                            config={"displayModeBar": False})
        if last["others"]:
            with st.container(key="panel_compare"):
                panel_title("All 1000-class models on this clip")
                rows = [{"Model": MODELS[p.model_id].display_name, "Input": MODELS[p.model_id].family,
                         "Top-1": p.label, "Confidence": f"{p.confidence:.1%}",
                         "Top-5": ", ".join(p.top_labels), "Model ms": round(p.inference_s * 1000)}
                        for p in last["others"]]
                st.dataframe(rows, hide_index=True, width="stretch")
        with st.container(key="panel_frames"):
            panel_title("What the model sees · signer crops")
            idx = np.linspace(0, 15, 8).astype(int)
            cols = st.columns(8, gap="small")
            for c, t in zip(cols, idx):
                c.image(feats.appearance[t, 0], caption=f"f{feats.info.sampled_indices[t]}", width="stretch")
            _hint_text("RGB models (Swin, R3D) see these 112x112 crops; landmark models see the 58 MediaPipe points "
                       "detected on the same 16 frames. This is isolated sign classification: the clip is mapped "
                       "to one of the model's known signs.")
