import streamlit as st

from inference.config import COMPARISON_1000, MODELS
from ui import data
from ui.theme import esc, note, page_title


def _person(p, t):
    show_email = t.get("show_emails")
    email = f'<p style="margin-top:.3rem">{esc(p["email"])}</p>' if show_email and p.get("email") else ""
    return (f'<div class="card"><div class="tag">{esc(p.get("role", "").upper())}</div>'
            f'<h4>{esc(p["name"])}</h4><p>{esc(t["department"])}<br>{esc(t["university"])}</p>{email}</div>')



def render():
    t = data.team()
    ds = data.dataset()
    page_title("About", "About the project",
               f"Capstone project, {t['department']}, {t['university']}, {t['location']}.")

    st.subheader("Research background")
    st.markdown(
        "Sign languages are complete natural languages, and automatic recognition can lower communication "
        "barriers between Deaf and hearing people. Isolated sign recognition -- identifying a single sign from "
        "a short clip -- is the building block for dictionaries, learning tools and, eventually, continuous "
        "translation. It is hard because signs differ in fine handshape and motion details, signers vary "
        "widely, and large vocabularies leave only a few dozen examples per sign.")

    st.subheader("Research objectives")
    st.markdown(
        "- Recognise up to **1,200 isolated ASL signs** from video.\n"
        "- Compare **appearance (RGB)**, **skeleton (landmark)** and **fused** models under one protocol.\n"
        "- Study how accuracy scales with vocabulary size.\n"
        "- Ship a **free, CPU-only** demonstration of the trained models.")

    st.subheader("Dataset")
    st.markdown(
        f"**{ds['name']}** ({ds['publisher']}), a crowd-sourced dataset of isolated ASL signs recorded on "
        f"webcams by Deaf and hard-of-hearing signers. Our preprocessing cache holds {ds['cached_videos']:,} "
        f"videos over {ds['glosses_in_cache']:,} glosses; the 1,000-sign experiments use "
        f"{ds['subset_1000']['videos']:,} of them with the official signer-independent splits.")
    note(f"<b>Licence.</b> {esc(ds['license'])} This app therefore contains <b>no dataset videos or "
         "features</b>; only aggregate evaluation results and the trained model weights are used.")

    st.subheader("Models and experimental configurations")
    rows = [{"Model": MODELS[m].display_name, "Input": MODELS[m].family, "Classes": MODELS[m].num_classes,
             "Origin": MODELS[m].origin} for m in COMPARISON_1000 + ["geometry_transformer_300",
                                                                    "geometry_transformer_500",
                                                                    "geometry_transformer_1200"]]
    st.dataframe(rows, hide_index=True, width="stretch")

    st.subheader("Evaluation methodology")
    st.markdown(
        "Models were selected on the validation split and evaluated once on the held-out test split. We report "
        "top-1 / top-5 accuracy, macro-averaged precision, recall and F1, confusion matrices and CPU latency. "
        "All 1,000-sign models share the exact same class list and test clips, so their numbers are directly "
        "comparable. The app's video pipeline was verified against the training feature cache on 30 held-out "
        "test videos (see the project's testing report).")

    st.subheader("Team")
    cols = st.columns(len(t["members"]), gap="medium")
    for c, p in zip(cols, t["members"]):
        c.markdown(_person(p, t), unsafe_allow_html=True)
    st.write("")
    sup = t["supervisor"]
    st.markdown(f'<div class="card accent" style="max-width:420px"><div class="tag">SUPERVISOR</div>'
                f'<h4>{esc(sup["name"])}</h4><p>{esc(t["department"])}<br>{esc(t["university"])}</p></div>',
                unsafe_allow_html=True)

    st.subheader("Source code")
    if t.get("github_url"):
        st.markdown(f"GitHub repository: [{t['github_url']}]({t['github_url']})")
    else:
        st.caption("Repository link: set `github_url` in config/team.json.")
