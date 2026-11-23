"""Visual identity: dark 'studio' look -- deep aubergine surfaces, purple accent, hairline borders,
small uppercase panel titles, Inter + IBM Plex Mono. Plus small HTML components."""
import html

import streamlit as st

# ---- design tokens (mirrored in CSS below and in ui/charts.py)
BG = "#120b1b"
PANEL = "#171021"
PANEL2 = "#1d1429"
RAISED = "#251b34"
BORDER = "rgba(74,57,98,0.75)"
HAIRLINE = "rgba(255,255,255,0.08)"
TEXT = "#fafafa"
MUTED = "#a3a3a3"
DIM = "#71717a"
ACCENT = "#9334e9"
ACCENT2 = "#8f35ff"
LAVENDER = "#d8b4fe"
LINK = "#bf8bff"

# Validated categorical palette (dataviz reference, DARK-mode steps), fixed per entity -- never by rank.
MODEL_COLORS = {
    "swin3d_t_1000": "#3987e5",
    "geometry_transformer_1000": "#d95926",
    "stgcn_1000": "#199e70",
    "r3d18_dualstream_1000": "#c98500",
}
SERIES_SEQ = ["#3987e5", "#d95926", "#199e70"]   # small fixed series (e.g. P / R / F1)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
  --bg:#120b1b; --panel:#171021; --panel2:#1d1429; --raised:#251b34;
  --border:rgba(74,57,98,0.75); --hairline:rgba(255,255,255,0.08);
  --text:#fafafa; --muted:#a3a3a3; --dim:#71717a;
  --accent:#9334e9; --accent2:#8f35ff; --accent-soft:rgba(147,52,233,0.14);
  --lavender:#d8b4fe; --link:#bf8bff; --stage-base:#0e0a16; --stage-square:#171221; --radius:10px;
  --mono:'IBM Plex Mono', ui-monospace, monospace;
}
html, body, .stApp, [class*="css"], .stMarkdown, button, input, textarea, select {
  font-family:'Inter', ui-sans-serif, system-ui, sans-serif; -webkit-font-smoothing:antialiased;
}
.stApp {
  background: radial-gradient(ellipse 60% 35% at 50% -5%, rgba(147,52,233,0.18), transparent 55%) no-repeat, var(--bg);
  background-attachment: fixed;
}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbarActions"], [data-testid="stMainMenu"],
[data-testid="stAppDeployButton"] { display:none !important; }
/* custom HTML blocks: cancel Streamlit's -1rem markdown margin so nothing is clipped */
[data-testid="stMarkdownContainer"]:has(> div[class]) { margin-bottom:0 !important; }

/* ---- top bar */
[data-testid="stHeader"] { background:var(--panel); border-bottom:1px solid var(--hairline); height:3.25rem; }
[data-testid="stHeader"] img { max-height:30px; }
.block-container { padding-top:4.6rem !important; padding-bottom:4.5rem !important; max-width:1320px; }

/* segmented-control navigation (top nav links): plain tabs, the active one raised */
[data-testid="stTopNavLinkContainer"], [data-testid="stTopNavSection"] {
  background:transparent !important; border:none !important; padding:0 !important; margin:0 1px !important;
}
[data-testid="stTopNavLink"] { border-radius:7px !important; padding:4px 13px !important; min-height:0 !important;
  border:1px solid transparent !important; background:transparent !important; }
[data-testid="stTopNavLink"] span, [data-testid="stTopNavLink"] p { color:var(--muted) !important; font-weight:550 !important; font-size:.82rem !important; }
[data-testid="stTopNavLink"]:hover { background:rgba(255,255,255,0.04) !important; }
[data-testid="stTopNavLink"][aria-current="page"] {
  background:var(--raised) !important; border-color:var(--border) !important; box-shadow:0 1px 3px rgba(0,0,0,.45);
}
[data-testid="stTopNavLink"][aria-current="page"] span, [data-testid="stTopNavLink"][aria-current="page"] p { color:var(--text) !important; }

/* ---- typography */
h1, h2, h3, h4 { color:var(--text) !important; letter-spacing:-0.02em; }
h3 { font-size:1.05rem !important; font-weight:650 !important; }
p, li { color:#d4d4d8; }
a { color:var(--link) !important; }
.stCaption, [data-testid="stCaptionContainer"] { color:var(--muted) !important; }
code { font-family:var(--mono) !important; background:var(--raised) !important; color:var(--lavender) !important; }

/* ---- panels = bordered containers */
[data-testid="stVerticalBlockBorderWrapper"], div[class*="st-key-panel"], div[class*="st-key-doc"] {
  background:var(--panel2); border:1px solid var(--border) !important; border-radius:var(--radius) !important;
}
div[class*="st-key-doc"] { background:var(--panel); padding:0 !important; gap:0 !important; overflow:hidden; }
div[class*="st-key-panel"] { padding:12px !important; gap:.55rem !important; }
.panel-title {
  font-size:.62rem; font-weight:650; text-transform:uppercase; letter-spacing:.12em; color:var(--dim);
  margin:0 0 .1rem;
}
.panel-title::before { content:"▾ "; color:var(--dim); }
.figcaption {
  display:flex; justify-content:space-between; align-items:center; padding:8px 12px;
  border-bottom:1px solid var(--hairline); font-size:.72rem; font-weight:600; color:var(--muted);
  text-transform:uppercase; letter-spacing:.1em;
}
.figcaption .pill { font-family:var(--mono); font-size:.62rem; color:var(--lavender); border:1px solid var(--border);
  border-radius:5px; padding:1px 6px; letter-spacing:0; text-transform:none; }
.stage {
  min-height:300px; display:flex; align-items:center; justify-content:center; padding:18px;
  background-color:var(--stage-base);
  background-image:linear-gradient(45deg,var(--stage-square) 25%,transparent 25%),
    linear-gradient(-45deg,var(--stage-square) 25%,transparent 25%),
    linear-gradient(45deg,transparent 75%,var(--stage-square) 75%),
    linear-gradient(-45deg,transparent 75%,var(--stage-square) 75%);
  background-size:20px 20px; background-position:0 0,0 10px,10px -10px,-10px 0;
}
.drop-hint { text-align:center; color:var(--muted); font-size:.82rem; line-height:1.6; }
.drop-hint b { display:block; color:var(--text); font-size:.92rem; font-weight:600; }
.doc-bar { display:flex; justify-content:space-between; gap:10px; padding:6px 12px; border-top:1px solid var(--hairline);
  font-family:var(--mono); font-size:.68rem; color:var(--muted); }
div[class*="st-key-doc"] [data-testid="stVideo"], div[class*="st-key-doc"] video {
  border-radius:0; background:var(--stage-base); max-height:420px;
}

/* ---- widgets */
[data-testid="stWidgetLabel"] p { color:var(--muted) !important; font-size:.75rem !important; font-weight:550; }
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, textarea {
  background:var(--panel) !important; border-color:var(--border) !important; border-radius:7px !important;
  color:var(--text) !important; font-size:.85rem !important;
}
[data-testid="stFileUploaderDropzone"] {
  background:var(--raised) !important; border:1px dashed var(--border) !important; border-radius:8px !important;
}
[data-testid="stFileUploaderDropzone"] button { border-color:var(--border) !important; }
div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] {
  background:var(--accent) !important; border:1px solid rgba(147,52,233,.7) !important; border-radius:8px !important;
  box-shadow:inset 0 1px rgba(255,255,255,.2); font-weight:600; font-size:.9rem; padding:.6rem 1rem; color:#fff !important;
}
div.stButton > button[kind="primary"]:hover { filter:brightness(1.12); }
div.stButton > button[kind="primary"]:active { transform:translateY(1px); }
div.stButton > button[kind="primary"]:disabled { opacity:.5; }
div.stButton > button[kind="secondary"] {
  background:transparent !important; border:1px solid var(--border) !important; border-radius:8px !important;
  color:var(--text) !important;
}
div.stButton > button[kind="secondary"]:hover { border-color:var(--accent) !important; background:var(--accent-soft) !important; }
[data-testid="stExpander"] details { background:var(--panel2); border:1px solid var(--border) !important; border-radius:var(--radius) !important; }
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; }
[data-testid="stAlert"] { border-radius:var(--radius) !important; }
[data-testid="stPlotlyChart"] { background:var(--panel2); border:1px solid var(--border); border-radius:var(--radius); padding:6px; }

/* ---- hero / page title */
.hero { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:2.2rem 2.3rem 1.9rem;
  margin-bottom:1.2rem; position:relative; overflow:hidden; }
.hero::after { content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(ellipse 55% 80% at 100% 0%, rgba(147,52,233,.22), transparent 60%); }
.hero .eyebrow, .page-title .eyebrow { display:inline-block; font-size:.62rem; font-weight:650; text-transform:uppercase;
  letter-spacing:.12em; color:var(--lavender); border:1px solid var(--border); background:var(--accent-soft);
  border-radius:999px; padding:2px 10px; }
.hero h1 { font-size:2.5rem; font-weight:700; margin:.7rem 0 .35rem; letter-spacing:-0.03em; }
.hero p.sub { color:var(--muted); font-size:1.02rem; max-width:760px; line-height:1.6; margin:0 0 1rem; }
.hero .chips span { display:inline-block; font-family:var(--mono); font-size:.7rem; color:var(--lavender);
  background:var(--raised); border:1px solid var(--border); border-radius:6px; padding:3px 9px; margin:.2rem .3rem .2rem 0; }
.page-title { margin:.2rem 0 1.1rem; }
.page-title h1 { font-size:1.7rem; font-weight:700; margin:.55rem 0 .3rem; }
.page-title p { color:var(--muted); font-size:.95rem; max-width:860px; margin:0; line-height:1.55; }

/* ---- cards & metrics */
.card { background:var(--panel2); border:1px solid var(--border); border-radius:var(--radius); padding:1rem 1.1rem; height:100%; }
.card h4 { margin:0 0 .35rem; font-size:.95rem; font-weight:650; }
.card p, .card li { color:var(--muted); font-size:.85rem; line-height:1.6; margin:0; }
.card .tag { font-size:.6rem; font-weight:650; color:var(--lavender); text-transform:uppercase; letter-spacing:.12em;
  margin-bottom:.45rem; font-family:var(--mono); }
.card.accent { border-color:rgba(147,52,233,.55); background:linear-gradient(180deg, rgba(147,52,233,.10), transparent 70%), var(--panel2); }
.metric { background:var(--panel2); border:1px solid var(--border); border-radius:var(--radius); padding:.85rem 1rem; height:100%; }
.metric .label { color:var(--dim); font-size:.62rem; text-transform:uppercase; letter-spacing:.12em; font-weight:650; }
.metric .value { color:var(--text); font-size:1.6rem; font-weight:700; line-height:1.2; margin-top:.3rem; letter-spacing:-0.02em; }
.metric .note { color:var(--muted); font-size:.75rem; margin-top:.15rem; }

/* ---- result */
.result { width:100%; max-width:460px; background:rgba(23,16,33,.92); border:1px solid rgba(147,52,233,.55);
  border-radius:var(--radius); padding:1.2rem 1.3rem; box-shadow:0 0 28px rgba(147,52,233,.18); }
.result .label { color:var(--lavender); text-transform:uppercase; letter-spacing:.12em; font-size:.6rem; font-weight:650; }
.result .sign { font-size:2.4rem; font-weight:700; letter-spacing:-0.01em; margin:.25rem 0 .15rem; color:var(--text); word-break:break-word; }
.result .conf { color:var(--muted); font-size:.85rem; }
.result .bar { height:6px; background:rgba(255,255,255,.08); border-radius:99px; margin-top:.75rem; overflow:hidden; }
.result .bar div { height:100%; background:linear-gradient(90deg, var(--accent), var(--lavender)); border-radius:99px; }

/* ---- pipeline flow */
.flow { display:flex; flex-wrap:wrap; align-items:stretch; gap:.45rem; margin:.3rem 0 1rem; }
.flow .step { flex:1 1 140px; background:var(--panel2); border:1px solid var(--border); border-radius:var(--radius); padding:.75rem .85rem; }
.flow .step b { color:var(--text); display:block; font-size:.85rem; font-weight:650; }
.flow .step span { color:var(--muted); font-size:.75rem; }
.flow .arrow { align-self:center; color:var(--accent); font-weight:700; }

.note-box { background:var(--accent-soft); border:1px solid var(--border); border-radius:var(--radius); padding:.8rem 1rem;
  color:#d4d4d8; font-size:.85rem; line-height:1.6; }
.small-muted { color:var(--dim); font-size:.78rem; }
.hint { color:var(--dim); font-size:.72rem; line-height:1.5; }

/* ---- history strip */
.history { display:flex; gap:8px; align-items:center; overflow-x:auto; padding:8px 2px; }
.history .h-label { font-size:.62rem; text-transform:uppercase; letter-spacing:.14em; color:var(--dim); margin-right:4px; }
.history .chip { flex:0 0 auto; background:var(--raised); border:1px solid var(--border); border-radius:6px; padding:5px 9px; }
.history .chip b { display:block; font-size:.78rem; color:var(--text); }
.history .chip span { font-family:var(--mono); font-size:.6rem; color:var(--lavender); }

/* ---- status bar */
.statusbar { position:fixed; left:0; right:0; bottom:0; z-index:999; display:flex; align-items:center; gap:14px;
  background:var(--panel); border-top:1px solid var(--hairline); padding:7px 16px; font-size:.75rem; color:var(--muted); }
.statusbar .pulse { width:8px; height:8px; border-radius:50%; background:#4fd1a5; box-shadow:0 0 6px rgba(79,209,165,.6); }
.statusbar .spacer { flex:1; }
.statusbar a { color:var(--dim) !important; font-size:.72rem; text-decoration:none; }
.statusbar a:hover { color:var(--text) !important; }
.statusbar .mono { font-family:var(--mono); font-size:.7rem; color:var(--dim); }
@media (max-width: 860px) { .statusbar a, .statusbar .mono { display:none; } .hero h1 { font-size:1.9rem; } }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def page_title(eyebrow, title, text=""):
    st.markdown(f'<div class="page-title"><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
                f'<p>{esc(text)}</p></div>', unsafe_allow_html=True)


def panel_title(text):
    st.markdown(f'<div class="panel-title">{esc(text)}</div>', unsafe_allow_html=True)


def metric(label, value, note=""):
    return (f'<div class="metric"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
            f'<div class="note">{esc(note)}</div></div>')


def card(title, body, tag="", accent=False):
    t = f'<div class="tag">{esc(tag)}</div>' if tag else ""
    return f'<div class="card{" accent" if accent else ""}">{t}<h4>{esc(title)}</h4><p>{esc(body)}</p></div>'


def html_row(blocks, gap="small"):
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


def statusbar(status="Ready", detail="CPU inference", github=""):
    gh = f'<a href="{esc(github)}" target="_blank">GitHub</a>' if github else ""
    st.markdown(f'<div class="statusbar"><div class="pulse"></div><span>{esc(status)}</span>'
                f'<span class="mono">{esc(detail)}</span><div class="spacer"></div>'
                f'<span class="mono">ASL Citizen &middot; 16 frames &middot; up to 1,200 signs</span>{gh}</div>',
                unsafe_allow_html=True)
