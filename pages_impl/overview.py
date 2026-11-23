import streamlit as st

from inference.config import COMPARISON_1000, MODELS
from ui import data
from ui.theme import card, esc, html_row, metric


def render():
    ds = data.dataset()
    best_id = max(COMPARISON_1000, key=lambda m: data.metrics(m)["test"]["top1"])
    best = data.metrics(best_id)

    st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Capstone Project &middot; East West University</div>
  <h1>ASL Vision</h1>
  <p class="sub">Video-Based American Sign Language Recognition Using 3D Convolutional Neural Networks
  and video / skeleton Transformers &mdash; isolated-sign classification on the ASL Citizen dataset.</p>
  <div class="chips"><span>Video Swin-T</span><span>R3D-18 3D-CNN</span><span>Geometry Transformer</span>
  <span>ST-GCN</span><span>MediaPipe landmarks</span><span>{ds['subset_1000']['test']:,} held-out test clips</span></div>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    if c1.button("Open the recognizer  →", type="primary", width="stretch"):
        st.switch_page(st.session_state["_pages"]["recognition"])
    c2.markdown('<div class="small-muted" style="padding-top:.55rem">Upload a short clip of a single sign and '
                'test any of our trained models on CPU.</div>', unsafe_allow_html=True)
    st.write("")

    html_row([
        metric("Best test top-1 (1000 signs)", f"{best['test']['top1']:.1%}", f"{MODELS[best_id].display_name}"),
        metric("Best test top-5 (1000 signs)", f"{best['test']['top5']:.1%}", "correct sign within 5 guesses"),
        metric("Architectures compared", "4", "on the identical 1000-class split"),
        metric("Videos in feature cache", f"{ds['cached_videos']:,}", f"{ds['glosses_in_cache']:,} ASL glosses"),
    ])
    st.write("")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("About the project")
        st.markdown(
            "ASL Vision recognises **isolated American Sign Language signs from short video clips**. Each clip "
            "is reduced to 16 evenly spaced frames; the models then classify the whole clip as one of up to "
            "1,200 sign glosses. This is *video classification of a single sign* &mdash; not continuous, "
            "real-time sign-language translation.\n\n"
            "Across two capstone stages we compared appearance-based models (a 3D-CNN and a video Transformer on "
            "RGB signer crops) with skeleton-based models (a Transformer and a graph network on MediaPipe hand, "
            "arm and face landmarks), all trained and tested on the same signer-independent splits.")
        st.subheader("Research objectives")
        st.markdown(
            "1. Build an isolated ASL sign classifier that scales to **1,000+ signs**.\n"
            "2. Compare **RGB-video**, **landmark-only** and **fused** architectures under one protocol.\n"
            "3. Measure how accuracy changes as the vocabulary grows (300 → 1,200 signs).\n"
            "4. Deploy the trained models in a **zero-cost, CPU-only** web demo.")
    with right:
        st.subheader("Main findings")
        st.markdown(
            f"- **Video Swin-T** is the most accurate model: **{best['test']['top1']:.1%} top-1 / "
            f"{best['test']['top5']:.1%} top-5** on 1,000 signs.\n"
            f"- The **landmark Transformer** reaches {data.metrics('geometry_transformer_1000')['test']['top1']:.1%} "
            "with only 1.9 M parameters and runs in milliseconds on CPU.\n"
            f"- The Capstone-B **R3D-18 dual-stream** model overfitted "
            f"({data.metrics('r3d18_dualstream_1000')['test']['top1']:.1%} test top-1), which motivated the "
            "Capstone-C redesign.\n"
            "- Landmark accuracy degrades gracefully with vocabulary size: "
            f"{data.metrics('geometry_transformer_300')['test']['top1']:.1%} at 300 signs → "
            f"{data.metrics('geometry_transformer_1200')['test']['top1']:.1%} at 1,200.")

    st.write("")
    st.subheader("Experimental configurations")
    html_row([card(MODELS[m].display_name, MODELS[m].description, tag=MODELS[m].family.upper(), accent=True)
              for m in COMPARISON_1000])
    st.write("")
    html_row([
        card("Vocabulary scaling (300 / 500 / 1000 / 1200)",
             "The Geometry Transformer was trained at four vocabulary sizes -- the 300, 500, 1,000 and 1,200 "
             "most frequent glosses -- to study how accuracy scales with the number of classes.", tag="SCALING"),
        card("Dataset: ASL Citizen",
             f"Microsoft Research's crowd-sourced isolated-sign dataset. Our 1,000-sign subset has "
             f"{ds['subset_1000']['videos']:,} videos ({ds['subset_1000']['train']:,} train / "
             f"{ds['subset_1000']['val']:,} validation / {ds['subset_1000']['test']:,} test), with "
             f"signers never shared between splits.", tag="DATA"),
        card("Contributions",
             "A unified preprocessing cache (MediaPipe + signer crops), a four-architecture comparison on one "
             "split, a vocabulary-scaling study, and a reproducible CPU inference pipeline verified against "
             "the training cache.", tag="CONTRIBUTIONS"),
    ])
