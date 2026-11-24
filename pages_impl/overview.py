import streamlit as st

from inference.config import COMPARISON_1000, MODELS
from ui import data
from ui.theme import esc

THESIS_TITLE = ("Building a Word-Level American Sign Language Recognition System: "
                "A Multi-Architecture Design and Evaluation Study")


def _section(eyebrow, title, lead=""):
    lead_html = f'<p class="sec-lead">{esc(lead)}</p>' if lead else ""
    st.markdown(f'<div class="sec-head"><div class="eyebrow">{esc(eyebrow)}</div><h2>{esc(title)}</h2>{lead_html}</div>',
                unsafe_allow_html=True)


def _grid(items, cols=4):
    """items: list of (kicker, title, body) -> centred cards in an even CSS grid."""
    cells = "".join(f'<div class="hcard"><div class="kicker">{esc(k)}</div><h4>{esc(t)}</h4><p>{b}</p></div>'
                    for k, t, b in items)
    st.markdown(f'<div class="hgrid c{cols}">{cells}</div>', unsafe_allow_html=True)


def render():
    ds = data.dataset()
    team = data.team()
    m = {mid: data.metrics(mid) for mid in MODELS}
    best_id = max(COMPARISON_1000, key=lambda k: m[k]["test"]["top1"])
    best = m[best_id]["test"]
    names = " &middot; ".join(esc(p["name"]) for p in team["members"])

    # ---------------- hero
    st.markdown(f"""
<div class="home-hero">
  <div class="eyebrow">Capstone Thesis &middot; {esc(team['department'].replace('Department of ', ''))}</div>
  <div class="thesis" role="heading" aria-level="1">{esc(THESIS_TITLE.split(': ')[0])}<span>{esc(THESIS_TITLE.split(': ')[1])}</span></div>
  <p class="lead">An isolated, word-level ASL recogniser trained on the ASL Citizen dataset. We design and
  compare four architectures &mdash; RGB video, skeleton landmarks and a fusion of both &mdash; under one
  signer-independent evaluation protocol, and deploy them in this free CPU web demo.</p>
  <div class="byline">{names}</div>
  <div class="byline sub">Supervised by {esc(team['supervisor']['name'])} &middot; {esc(team['university'])},
  {esc(team['location'])}</div>
</div>""", unsafe_allow_html=True)

    _, b1, b2, _ = st.columns([1.35, 1, 1, 1.35], gap="small")
    if b1.button("Try the recognizer  →", type="primary", width="stretch"):
        st.switch_page(st.session_state["_pages"]["recognition"])
    if b2.button("See the results", width="stretch"):
        st.switch_page(st.session_state["_pages"]["comparison"])

    # ---------------- key numbers
    stats = [
        (f"{best['top1']:.1%}", "Best top-1 accuracy", f"{MODELS[best_id].display_name}, 1,000 signs"),
        (f"{best['top5']:.1%}", "Best top-5 accuracy", "correct sign within 5 guesses"),
        ("4", "Architectures", "compared on one identical split"),
        (f"{ds['subset_1000']['test']:,}", "Test clips", "signers unseen in training"),
    ]
    cells = "".join(f'<div class="stat"><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div>'
                    f'<div class="n">{esc(n)}</div></div>' for v, l, n in stats)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)

    # ---------------- about
    _section("The project", "What we built",
             "ASL Vision classifies a short video of one American Sign Language sign as one of up to 1,200 "
             "words. Each clip is reduced to 16 evenly spaced frames, and the whole clip is classified at once. "
             "This is word-level (isolated sign) recognition, not continuous sentence translation.")

    # ---------------- objectives
    _section("Research objectives", "What the study sets out to do")
    _grid([
        ("01", "Scale to 1,000+ words", "Build a word-level ASL classifier that works on a large vocabulary."),
        ("02", "Compare architectures", "Evaluate RGB-video, landmark-only and fused models under one protocol."),
        ("03", "Study vocabulary size", "Measure how accuracy changes from 300 to 1,200 signs."),
        ("04", "Deploy for free", "Ship every trained model in a zero-cost, CPU-only web application."),
    ])

    # ---------------- architectures
    _section("Architectures", "Four models, one evaluation protocol",
             "All four 1,000-sign models share the same class list, training split and test clips.")
    _grid([(MODELS[k].family, MODELS[k].display_name,
            f'<b class="acc">{m[k]["test"]["top1"]:.1%}</b> test top-1<br>{esc(MODELS[k].description)}')
           for k in COMPARISON_1000])

    # ---------------- findings
    gt = m["geometry_transformer_1000"]["test"]["top1"]
    _section("Findings", "What we learned")
    _grid([
        ("Accuracy", "Video Swin-T leads",
         f"The video Transformer is the most accurate model: {best['top1']:.1%} top-1 and {best['top5']:.1%} "
         "top-5 on 1,000 signs."),
        ("Efficiency", "Landmarks are light",
         f"The landmark Transformer reaches {gt:.1%} with only 1.9 M parameters and runs in milliseconds on CPU."),
        ("Capstone B", "Fusion overfitted",
         f"The earlier R3D-18 dual-stream model reached {m['r3d18_dualstream_1000']['test']['top1']:.1%}, "
         "which motivated the Capstone C redesign."),
        ("Scaling", "Graceful degradation",
         f"Landmark accuracy falls only from {m['geometry_transformer_300']['test']['top1']:.1%} (300 signs) "
         f"to {m['geometry_transformer_1200']['test']['top1']:.1%} (1,200 signs)."),
    ])

    # ---------------- data & contributions
    _section("Study design", "Data and contributions")
    sub = ds["subset_1000"]
    _grid([
        ("Dataset", "ASL Citizen",
         f"Microsoft Research's crowd-sourced isolated-sign dataset. The 1,000-sign subset has {sub['videos']:,} "
         f"videos: {sub['train']:,} train, {sub['val']:,} validation and {sub['test']:,} test."),
        ("Protocol", "Signer-independent",
         "No signer appears in more than one split. Models are selected on validation and tested once."),
        ("Contributions", "End-to-end pipeline",
         "A shared preprocessing cache, a four-architecture comparison, a vocabulary-scaling study and a CPU "
         "inference pipeline verified against the training cache."),
    ], cols=3)
