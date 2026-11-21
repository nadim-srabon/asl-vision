"""Visual identity: navy / white / subtle blue, plus small HTML components."""
import html

import streamlit as st

NAVY = "#0B1F3A"
BLUE = "#2E6BD8"
SKY = "#EAF1FB"
INK_2 = "#4A5873"
INK_3 = "#7A879C"
LINE = "#DCE4F0"

# Validated categorical palette (dataviz reference, light mode), fixed per entity -- never by rank.
MODEL_COLORS = {
    "swin3d_t_1000": "#2a78d6",
    "geometry_transformer_1000": "#eb6834",
    "stgcn_1000": "#1baf7a",
    "r3d18_dualstream_1000": "#eda100",
}
SERIES_SEQ = ["#2a78d6", "#eb6834", "#1baf7a"]   # for small fixed series (e.g. P / R / F1)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #E6EDF8 !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(255,255,255,0.10); border-radius: 8px; }}

.hero {{
  background: linear-gradient(135deg, {NAVY} 0%, #15325E 60%, #1D4580 100%);
  border-radius: 18px; padding: 2.6rem 2.6rem 2.2rem; color: #fff; margin-bottom: 1.6rem;
  box-shadow: 0 10px 30px rgba(11,31,58,0.18);
}}
.hero .eyebrow {{ text-transform: uppercase; letter-spacing: .14em; font-size: .78rem; color: #9DBDF0; font-weight: 600; }}
.hero h1 {{ color: #fff; font-size: 2.7rem; font-weight: 800; margin: .3rem 0 .4rem; }}
.hero p.sub {{ color: #D5E2F6; font-size: 1.12rem; margin: 0 0 1rem; max-width: 760px; line-height: 1.55; }}
.hero .chips span {{
  display: inline-block; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18);
  color: #fff; border-radius: 999px; padding: .28rem .8rem; margin: .2rem .35rem .2rem 0; font-size: .82rem;
}}

.page-title {{ margin-bottom: 1.2rem; }}
.page-title .eyebrow {{ text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; color: {BLUE}; font-weight: 700; }}
.page-title h1 {{ font-size: 2.1rem; font-weight: 800; margin: .2rem 0 .3rem; }}
.page-title p {{ color: {INK_2}; font-size: 1.02rem; max-width: 820px; margin: 0; }}

.card {{
  background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1.15rem 1.25rem;
  height: 100%; box-shadow: 0 1px 2px rgba(11,31,58,.04);
}}
.card h4 {{ margin: 0 0 .35rem; font-size: 1.02rem; font-weight: 700; }}
.card p, .card li {{ color: {INK_2}; font-size: .93rem; line-height: 1.55; margin: 0; }}
.card .tag {{ display: inline-block; font-size: .72rem; font-weight: 700; color: {BLUE}; background: {SKY};
  border-radius: 6px; padding: .12rem .5rem; margin-bottom: .5rem; letter-spacing: .03em; }}
.card.accent {{ border-left: 4px solid {BLUE}; }}

.metric {{
  background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1rem 1.15rem; height: 100%;
}}
.metric .label {{ color: {INK_3}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }}
.metric .value {{ color: {NAVY}; font-size: 1.85rem; font-weight: 800; line-height: 1.15; margin-top: .25rem; }}
.metric .note {{ color: {INK_2}; font-size: .82rem; margin-top: .2rem; }}

.result {{
  background: linear-gradient(135deg, {NAVY}, #1D4580); color: #fff; border-radius: 16px;
  padding: 1.4rem 1.6rem; box-shadow: 0 8px 24px rgba(11,31,58,.18);
}}
.result .label {{ color: #9DBDF0; text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; font-weight: 700; }}
.result .sign {{ font-size: 2.6rem; font-weight: 800; letter-spacing: .02em; margin: .2rem 0; }}
.result .conf {{ color: #D5E2F6; font-size: 1rem; }}
.result .bar {{ height: 8px; background: rgba(255,255,255,.16); border-radius: 99px; margin-top: .7rem; overflow: hidden; }}
.result .bar div {{ height: 100%; background: #7FB2F0; border-radius: 99px; }}

.flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: .5rem; margin: .4rem 0 1rem; }}
.flow .step {{ flex: 1 1 140px; background: {SKY}; border: 1px solid {LINE}; border-radius: 12px; padding: .8rem .9rem; }}
.flow .step b {{ color: {NAVY}; display: block; font-size: .92rem; }}
.flow .step span {{ color: {INK_2}; font-size: .8rem; }}
.flow .arrow {{ align-self: center; color: {BLUE}; font-weight: 800; font-size: 1.2rem; }}

.note-box {{ background: {SKY}; border: 1px solid {LINE}; border-radius: 12px; padding: .85rem 1rem;
  color: {INK_2}; font-size: .9rem; }}
.small-muted {{ color: {INK_3}; font-size: .82rem; }}
div.stButton > button[kind="primary"] {{ border-radius: 10px; font-weight: 600; padding: .55rem 1.2rem; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def page_title(eyebrow, title, text=""):
    st.markdown(f'<div class="page-title"><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
                f'<p>{esc(text)}</p></div>', unsafe_allow_html=True)


def metric(label, value, note=""):
    return (f'<div class="metric"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
            f'<div class="note">{esc(note)}</div></div>')


def card(title, body, tag="", accent=False):
    t = f'<div class="tag">{esc(tag)}</div>' if tag else ""
    return f'<div class="card{" accent" if accent else ""}">{t}<h4>{esc(title)}</h4><p>{esc(body)}</p></div>'


def html_row(blocks, gap="medium"):
    cols = st.columns(len(blocks), gap=gap)
    for c, b in zip(cols, blocks):
        c.markdown(b, unsafe_allow_html=True)


def flow(steps):
    parts = []
    for i, (title, sub) in enumerate(steps):
        if i:
            parts.append('<div class="arrow">&rarr;</div>')
        parts.append(f'<div class="step"><b>{esc(title)}</b><span>{esc(sub)}</span></div>')
    st.markdown(f'<div class="flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)
