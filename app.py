"""ASL Vision -- Streamlit entry point.  Run:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="ASL Vision Studio", page_icon="🤟", layout="wide", initial_sidebar_state="collapsed")

from pages_impl import about, comparison, methodology, overview, recognition  # noqa: E402
from ui import data  # noqa: E402
from ui.theme import inject, statusbar  # noqa: E402

inject()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

pages = {
    "overview": st.Page(overview.render, title="Overview", url_path="overview", default=True),
    "recognition": st.Page(recognition.render, title="Recognize", url_path="recognition"),
    "comparison": st.Page(comparison.render, title="Results", url_path="comparison"),
    "methodology": st.Page(methodology.render, title="Methodology", url_path="methodology"),
    "about": st.Page(about.render, title="About", url_path="about"),
}
st.session_state["_pages"] = pages

nav = st.navigation(list(pages.values()), position="top")
nav.run()
statusbar(st.session_state.get("_status", "Ready"), st.session_state.get("_status_detail", "CPU inference"),
          data.team().get("github_url", ""))
